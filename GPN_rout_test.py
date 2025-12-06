import math
import os
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATv2Conv
import networkx as nx
from torch_geometric.data import Data, Batch
import copy
import time

torch.autograd.set_detect_anomaly(True)

def check_reachability(G, source, demands):
    for target in demands.keys():
        if not nx.has_path(G, source, target):
            return 0
    return 1

def sort_edges(edge_index, edge_attr):
    """
    Sort the edge indices of the graph and simultaneously update the edge attributes.
    
    Parameters:
    - edge_index (torch.Tensor): Edge index tensor of shape (2, num_edges).
    - edge_attr (torch.Tensor): Edge attribute tensor of shape (num_edges, num_features).
    
    Returns:
    - torch.Tensor: Sorted edge index tensor.
    - torch.Tensor: Corresponding sorted edge attribute tensor.
    """
    # Create a sorting key, first by source node, then by target node if sources are the same
    num_edges = edge_index.size(1)
    max_node = edge_index.max() + 1
    sort_key = edge_index[0] * max_node + edge_index[1]  # Generate a unique sorting key

    # Use argsort to get the sorted indices
    sorted_indices = torch.argsort(sort_key)

    # Apply the sorted indices to the original edge_index and edge_attr
    sorted_edge_index = edge_index[:, sorted_indices]
    sorted_edge_attr = edge_attr[sorted_indices]

    return sorted_edge_index, sorted_edge_attr

def generate_graph_with_avg_degree(n, avg_degree):
    # Generate a random graph with a specified average degree
    m = int((avg_degree * n) / 2)
    G = nx.gnm_random_graph(n, m)
    return G

def nx_to_pyg(G):
    # ---------- node ----------
    num_nodes = G.number_of_nodes()
    feat_dim  = len(next(iter(G.nodes(data='x')))[1])
    x = torch.zeros((num_nodes, feat_dim), dtype=torch.float)
    for n, feat in G.nodes(data='x'):
        x[n] = feat

    # ---------- edge ----------
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor([G[u][v]['edge_attr'] for u, v in G.edges()],
                              dtype=torch.float).view(-1, 1)

    if not G.is_directed():
        edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
        edge_attr  = torch.cat([edge_attr,  edge_attr],             dim=0)

    edge_index, edge_attr = sort_edges(edge_index, edge_attr)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def generate_graph_data(p,num_nodes,num_users,MAX_USER, if_avrg_degree = False, if_degree = False):
    """
    Generate data for a single graph, with each edge feature set to 1.

    Parameters:
    - num_nodes: Number of nodes in the graph

    Returns:
    - PyTorch Geometric Data object containing node features, edge information, and edge features
    """
    while True:
        while True:
            if if_avrg_degree:
                G = generate_graph_with_avg_degree(num_nodes, p)
            elif if_degree:
                G = nx.random_regular_graph(p, num_nodes)
            else:
                G = nx.random_regular_graph(p, num_nodes)
            # Check if each node is connected to at least one other node
            if all(len(list(G.neighbors(n))) > 1 for n in range(num_nodes)):
                break

        # Set features for each node
        for node in G.nodes(data=True):
            G.nodes[node[0]]['x'] = torch.tensor([0,0,1,0,0,0], dtype=torch.float)

        # Generate a random permutation from 1 to size-1
        User_id = torch.randperm(num_nodes - 1)[:num_users] + 1
        demand = {}
        for user in range(num_users):
            if user+1 > num_users * 2/3:
                transmission = 0.25
            elif user+1 > num_users * 1/3:
                transmission = 0.5
            else:
                transmission = 1.0
            user_order = (user + 1)/MAX_USER
            G.nodes[User_id[user].item()]['x'] = torch.tensor([0,0,0,1,transmission,user_order], dtype=torch.float)
            demand[User_id[user].item()] = transmission
        G.nodes[0]['x'] = torch.tensor([0,1,0,0,0,0], dtype=torch.float)
        # Set features for each edge
        for u, v in G.edges():
            random_weight = (1 - random.random()) * 0.5
            G[u][v]['edge_attr'] = [1 - random_weight]
            G[u][v]['weight'] = 1 - random_weight

        # Convert networkx graph to PyTorch Geometric Data object
        data = nx_to_pyg(G)
        if_reach = check_reachability(G, 0, demand)
        if if_reach == 1:
            break
        else:
            continue
    return data, User_id, G

def generate_graph_data_virtual_node(num_nodes, p = 0.05, num_users = 3, MAX_USER = 10, if_avrg_degree = False, if_degree = False):
    data, User_id, G = generate_graph_data(p,num_nodes-1,num_users,MAX_USER, if_avrg_degree = if_avrg_degree, if_degree = if_degree)
    # New node feature vector (randomly generated or specified)
    new_node_features = torch.tensor([[1,0,0,0,0,0]], dtype=torch.float)    # Update node features
    data.x = torch.cat([data.x, new_node_features], dim=0)
    
    # Index of the new node
    new_node_index = data.num_nodes - 1

    # Create connections from the new node to all other nodes
    new_edges = torch.tensor([[new_node_index] * new_node_index + list(range(new_node_index)),
                            list(range(new_node_index)) + [new_node_index] * new_node_index])

    # Update edge connections
    data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

    # Set attributes for the new edges to -1.0
    new_edge_attrs = torch.full((new_edges.size(1), 1), float(-1.0))
    # Update edge attributes
    data.edge_attr = torch.cat([data.edge_attr, new_edge_attrs], dim=0)
    p = 1
    return data.cuda(), User_id, G
def generate_batch_data_with_user(batch_size, num_nodes, num_users, MAX_USER, p = 0.05, if_avrg_degree = False, if_degree = False):
    """
    Generate multiple graphs and combine them into a batch.
    """
    graphs, User_id, G = zip(*[generate_graph_data_virtual_node(
                                num_nodes, p = p,num_users = num_users,MAX_USER = MAX_USER, 
                                if_avrg_degree = if_avrg_degree, if_degree = if_degree) 
                                for _ in range(batch_size)]
                            )
    User_id = torch.stack(User_id, dim=0)
    batch_data = Batch.from_data_list(graphs)

    return batch_data, User_id, G

def add_virtual_node(data):
    """
    Add a virtual node to the graph data.
    
    Parameters:
    - data (torch_geometric.data.Data): The graph data object.
    
    Returns:
    - torch_geometric.data.Data: The graph data object with an added virtual node.
    """
    # New node features
    new_node_features = torch.tensor([[1,0,0,0,0,0]], dtype=torch.float)    # Update node features
    data.x = torch.cat([data.x, new_node_features], dim=0)

    # Index of the new node
    new_node_index = data.num_nodes - 1

    # Create connections from the new node to all other nodes
    new_edges = torch.tensor([[new_node_index] * new_node_index + list(range(new_node_index)),
                            list(range(new_node_index)) + [new_node_index] * new_node_index]).cuda(2)

    # Update edge connections
    data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)

    # Set attributes for the new edges to -1.0
    new_edge_attrs = torch.full((new_edges.size(1), 1), float(-1.0)).cuda(2)

    # Update edge attributes
    data.edge_attr = torch.cat([data.edge_attr, new_edge_attrs], dim=0)

    return data

def write_input_from_torch_data_with_virtual_with_user(epoch, data, userid,num_users,filename):
    # Extract basic graph information
    size_V = data[0].num_nodes-1
    size_E = (data[0].edge_index.size(1)-size_V*2) 
    
    # Assuming first node as sender and specific nodes as receivers with fixed properties
    senders = [1]
    receivers = {}
    for user in userid[0]:
        receivers[user.item()+1] = (int(data.x[user][4].item()*4000))

    with open(filename, 'a') as file:
        file.write(f"{size_V} {size_E} {len(senders)} {len(receivers)} {num_users} {epoch}\n")
        
        for i in range(size_E):
            src = data.edge_index[0, i].item() + 1  # converting zero-indexed to one-indexed
            dst = data.edge_index[1, i].item() + 1
            attr = data.edge_attr[i].item()
            file.write(f"{src} {dst} {attr}\n")
        
        # Write sender vertices
        file.write(' '.join(map(str, senders)) + '\n')
        # Write receiver vertices and their properties
        for r, (definition) in receivers.items():
            file.write(f"{r} {definition}\n")

def from_txt_to_graph(num_users, num_nodes, MAX_USER, train_size = False, if_epoch = False, epoch = 0, num_batch = 1,file_path = '/home/GPN_rout/input.txt'):
    graphs = []
    file_path_ex = '/home/GPN_rout/input.txt'
    User_id = torch.tensor([],dtype=torch.int)
    nx_graphs = []
    if if_epoch:
        for epoch in range(10):
            file_path = file_path_ex + str(epoch) + '.txt'
            with open(file_path, 'r') as file:
                G = nx.DiGraph()
                first_line = file.readline().strip()
                
                num_nodes, num_edges, _, num_users, _, _, num_graph = map(int, first_line.split())
                edge_index = []
                edge_attr = []
                edge_G = []
                for _ in range(num_edges):
                    line = file.readline().strip()
                    u, v, w = map(float, line.split())
                    u, v = int(u), int(v)
                    u = u - 1
                    v = v - 1
                    edge_G.append([u, v, w])
                    edge_index.append([u, v])
                    edge_attr.append([w])
                G.add_weighted_edges_from(edge_G)
                nx_graphs.append(G)
                edge_index = torch.tensor(edge_index).t().contiguous()
                edge_index = edge_index.cuda(2)
                edge_attr = torch.tensor(edge_attr).cuda(2)
                edge_index, edge_attr = sort_edges(edge_index, edge_attr)
                line = file.readline().strip()
                ini_server = int(line) - 1
                user_id = torch.empty(num_users,dtype=torch.int)
                
                for i in range(num_users):
                    line = file.readline().strip()
                    user_id[i], _ = map(int, line.split())
                    user_id[i] = user_id[i] - 1
                x = torch.empty(num_nodes, 6, dtype=torch.float)
                for i in range(num_nodes):
                    x[i] = torch.tensor([0,0,1,0], dtype=torch.float).cuda(2)
                x[ini_server] = torch.tensor([0,1,0,0], dtype=torch.float).cuda(2)
                for i in range(num_users):
                    if i+1 > num_users * 2/3:
                        transmission = 0.25
                    elif i+1 > num_users * 1/3:
                        transmission = 0.5
                    else:
                        transmission = 1.0
                    user_order = (i)/MAX_USER
                    x[user_id[i]] = torch.tensor([0,0,0,1], dtype=torch.float).cuda(2)
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                data = add_virtual_node(data)
                graphs.append(data)
                User_id = torch.cat([User_id, user_id.view(1,-1)], dim=0)
        batch_data = Batch.from_data_list(graphs)
        
        return batch_data, User_id, nx_graphs
    else:
        with open(file_path, 'r') as file:
            for b in range(num_batch):
                G = nx.DiGraph()
                first_line = file.readline().strip()
                num_nodes, num_edges, _, num_users, _, num_graph = map(int, first_line.split())
                edge_index = []
                edge_attr = []
                edge_G = []
                for _ in range(num_edges):
                    line = file.readline().strip()
                    u, v, w = map(float, line.split())
                    u, v = int(u), int(v)
                    u = u - 1
                    v = v - 1
                    edge_G.append([u, v, w])
                    edge_index.append([u, v])
                    edge_attr.append([w])
                G.add_weighted_edges_from(edge_G)
                edge_index = torch.tensor(edge_index).t().contiguous()
                edge_index = edge_index.cuda(2)
                edge_attr = torch.tensor(edge_attr).cuda(2)
                edge_index, edge_attr = sort_edges(edge_index, edge_attr)
                line = file.readline().strip()
                ini_server = int(line) - 1
                user_id = torch.empty(num_users,dtype=torch.int)
                
                for i in range(num_users):
                    line = file.readline().strip()
                    user_id[i], _ = map(int, line.split())
                    user_id[i] = user_id[i] - 1
                x = torch.empty(num_nodes, 6, dtype=torch.float)
                for i in range(num_nodes):
                    x[i] = torch.tensor([0,0,1,0,0,0], dtype=torch.float).cuda(2)
                x[ini_server] = torch.tensor([0,1,0,0,0,0], dtype=torch.float).cuda(2)
                for user in range(num_users):
                    if user+1 > num_users * 2/3:
                        transmission = 0.25
                    elif user+1 > num_users * 1/3:
                        transmission = 0.5
                    else:
                        transmission = 1.0
                    user_order = (user + 1)/MAX_USER
                    x[user_id[user]] = torch.tensor([0,0,0,1,transmission,user_order], dtype=torch.float).cuda(2)
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                data = add_virtual_node(data)
                if b == epoch:
                    graphs.append(data)
                    nx_graphs.append(G)
                    User_id = torch.cat([User_id, user_id.view(1,-1)], dim=0)
                    break
        batch_data = Batch.from_data_list(graphs)
        
        return batch_data, User_id, nx_graphs

class Attention(nn.Module):
    def __init__(self, n_hidden):
        super(Attention, self).__init__()
        self.size = 0
        self.batch_size = 0
        self.dim = n_hidden
        
        v  = torch.FloatTensor(n_hidden).cuda()
        self.v  = nn.Parameter(v)
        self.v.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        # parameters for pointer attention
        self.Wref = nn.Linear(n_hidden, n_hidden)
        self.Wq = nn.Linear(n_hidden, n_hidden)
    
    
    def forward(self, q, ref):       # query and reference
        self.batch_size = q.size(0)
        self.size = int(ref.size(0) / self.batch_size)
        q = self.Wq(q)     # (B, dim)
        ref = self.Wref(ref)
        ref = ref.view(self.batch_size, self.size, self.dim)  # (B, size, dim)
        
        q_ex = q.unsqueeze(1).repeat(1, self.size, 1) # (B, size, dim)
        # v_view: (B, dim, 1)
        v_view = self.v.unsqueeze(0).expand(self.batch_size, self.dim).unsqueeze(2)
        
        # (B, size, dim) * (B, dim, 1)
        u = torch.bmm(torch.tanh(q_ex + ref), v_view).squeeze(2)
        
        return u, ref
    
class LSTM(nn.Module):
    def __init__(self, n_hidden):
        super(LSTM, self).__init__()
        
        # parameters for input gate
        self.Wxi = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whi = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wci = nn.Linear(n_hidden, n_hidden)    # w(ct)
        
        # parameters for forget gate
        self.Wxf = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whf = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wcf = nn.Linear(n_hidden, n_hidden)    # w(ct)
        
        # parameters for cell gate
        self.Wxc = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whc = nn.Linear(n_hidden, n_hidden)    # W(ht)
        
        # parameters for forget gate
        self.Wxo = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Who = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wco = nn.Linear(n_hidden, n_hidden)    # w(ct)
    
    
    def forward(self, x, h, c):       # query and reference
        
        # input gate
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h) + self.wci(c))
        # forget gate
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h) + self.wcf(c))
        # cell gate
        c = f * c + i * torch.tanh(self.Wxc(x) + self.Whc(h))
        # output gate
        o = torch.sigmoid(self.Wxo(x) + self.Who(h) + self.wco(c))
        
        h = o * torch.tanh(c)
        
        return h, c

class GPN(nn.Module):
    
    def __init__(self, n_feature, n_hidden,num_heads):
        super(GPN, self).__init__()
        self.node_size = 0
        self.batch_size = 0
        self.dim = n_hidden
        
        # lstm for first turn
        self.lstm0 = nn.LSTM(n_hidden, n_hidden)
        
        # pointer layer
        self.pointer = Attention(n_hidden)
        
        # lstm encoder
        self.encoder = LSTM(n_hidden)
        
        # trainable first hidden input
        h0 = torch.FloatTensor(n_hidden).cuda()
        c0 = torch.FloatTensor(n_hidden).cuda()
        
        # trainable latent variable coefficient
        alpha = torch.ones(1).cuda()
        
        self.h0 = nn.Parameter(h0)
        self.c0 = nn.Parameter(c0)
        
        self.alpha = nn.Parameter(alpha)
        self.h0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        self.c0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        r1 = torch.ones(1).cuda()
        r2 = torch.ones(1).cuda()
        r3 = torch.ones(1).cuda()
        self.r1 = nn.Parameter(r1)
        self.r2 = nn.Parameter(r2)
        self.r3 = nn.Parameter(r3)
        

        
        #GAT embedding
        self.gat1 = GATv2Conv(n_feature, n_hidden,heads = num_heads, edge_dim = 1,concat=True)  # 第一个GAT层
        #self.gat1 = GATConv(6400, n_hidden,heads = 1, edge_dim = 1)  # 第一个GAT层
        self.gat2 = GATv2Conv(n_hidden*num_heads , n_hidden,heads = num_heads, edge_dim = 1,concat=False)  # 第二个GAT层
        self.gat3 = GATv2Conv(n_hidden, n_hidden,heads = num_heads, edge_dim = 1,concat=False)  # 第二个GAT层
        
        
        
    def forward(self, idx, batch_size,node_size,X_all, mask, edge_index, edge_attr, h=None, c=None, latent=None):
        '''
        Inputs

        idx: index of current node (B)
        x: current node index (B, 1)
        X_all: all nodes' feature(B*size)
        mask: mask visited cities
        edge_index: connection of nodes
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent: latent pointer vector from previous layer (B, size, dim)
        
        Outputs
        
        softmax: probability distribution of next city (B, size)
        h: hidden variable (B, dim)
        c: cell gate (B, dim)
        latent_u: latent pointer vector for next layer
        '''
        
        self.batch_size = batch_size
        self.node_size = node_size
        
        context = self.gat1(X_all,edge_index,edge_attr = edge_attr)
        context = self.gat2(context,edge_index,edge_attr = edge_attr)   # x_all embedding
        context = self.gat3(context,edge_index,edge_attr = edge_attr)   # x_all embedding
        context = context.reshape(self.batch_size,self.node_size,-1)
        
        x = context[[i for i in range(self.batch_size)],idx,:]   # x embedding

        
        first_turn = False
        if h is None or c is None:
            first_turn = True
        
        if first_turn:
            # (dim) -> (B, dim)
            
            h0 = self.h0.unsqueeze(0).expand(self.batch_size, self.dim)
            c0 = self.c0.unsqueeze(0).expand(self.batch_size, self.dim)

            h0 = h0.unsqueeze(0).contiguous()
            c0 = c0.unsqueeze(0).contiguous()

            input_context = context.permute(1,0,2).contiguous()
            _, (h_enc, c_enc) = self.lstm0(input_context, (h0, c0))

            # let h0, c0 be the hidden variable of first turn
            h = h_enc.squeeze(0)
            c = c_enc.squeeze(0)
        
        # (B, size, dim)
        context = context.view(-1, self.dim)

        # LSTM encoder
        h, c = self.encoder(x, h, c)
        
        # query vector
        q = h
        
        # pointer
        u, _ = self.pointer(q, context)
        
        latent_u = u.clone()
        
        u = 10 * torch.tanh(u) + mask
        
        if latent is not None:
            u += self.alpha * latent
        F.softmax(u, dim=1)
        return F.softmax(u, dim=1), h, c, latent_u



for size, num_users in [(30,12)]:
    num_range = 12
    if size == 50 and num_users == 12:
        num_range = 9
    size = size + 1
    for n_range in range(num_range):
        
        if_avrg_degree = False
        if_degree = False
        if size == 51 and num_users == 12:
            if n_range == 0:
                continue
        if n_range > 0 and n_range < 5:
            continue
            p = (n_range - 1) % 4 +3
            if p == 3:
                continue
            if_avrg_degree = True
        elif n_range > 4 and n_range < 9:
            
            p = (n_range - 1) % 4 +3
            if p == 3:
                continue
            if_degree = True
        
        learn_rate = 1e-3    # learning rate
        B = 1    # batch_size
        B_val = 1     # validation size
        size_val = size
        num_users_val = num_users
        steps = 1    # training steps
        n_epoch = 100    # epochs
        
        MAX_USER = 20
        virtual_edge_attr = 10
        if if_avrg_degree:
            file_path_root = '/home/GPN_rout/data/avr'+ str(p) + 'n' +  str(size-1) + '/'
            file_path_size = file_path_root +  'avrdegree'+ str(p) + 'size' +  str(size-1) +'.txt'
            file_path_output = file_path_root + 'avrdegree'+ str(p) + 'size' +  str(size-1) + 'output.csv'
            file_path_output_path = file_path_root + 'avrdegree'+ str(p) + 'size' +  str(size-1) + 'output_path.csv'  
        elif if_degree:
            file_path_root = '/home/GPN_rout/data/d'+ str(p) + 'n' +  str(size-1) + '/'
            file_path_size = file_path_root +  'degree'+ str(p) + 'size' +  str(size-1) +'.txt'
            file_path_output = file_path_root + 'degree'+ str(p) + 'size' +  str(size-1) + 'output.csv'
            file_path_output_path = file_path_root + 'degree'+ str(p) + 'size' +  str(size-1) + 'output_path.csv'  
        else:
            file_path_root = '/home/GPN_rout/data/n'+ str(size-1) + 'u' + str(num_users) + '/'
            file_path_size = file_path_root +  'size' + str(size-1) + '_user' + str(num_users) +'.txt'
            file_path_output = file_path_root + 'size' + str(size-1) + '_user' + str(num_users) + 'output.csv'
            file_path_output_path = file_path_root + 'size' + str(size-1) + '_user' + str(num_users) + 'output_path.csv'  
        
        save_root = '/home/GPN_rout/model/gpn_mst_gat2_noshort.pt'

        print('=========================')
        print('prepare to train')
        print('=========================')
        print('Hyperparameters:')
        print('size', size)
        print('learning rate', learn_rate)
        print('batch size', B)
        print('validation size', B_val)
        print('steps', steps)
        print('epoch', n_epoch)
        print('save root:', save_root)
        print('=========================')

        # model = GPN(n_feature=6, n_hidden=128,num_heads=4).cuda()

        # load model
        model = torch.load(save_root).cuda()
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)

        # lr_decay_step = 500
        # lr_decay_rate = 0.96
        # opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step*1000,
        #                                     lr_decay_step), gamma=lr_decay_rate)
        model.eval()
        C = 0     # baseline
        R = torch.zeros(B_val).cuda()     # reward


        val_mean = []
        val_std = []
        first_run = True
        for epoch in range(n_epoch):
            for i in range(steps):
                if i % 50 == 0:
                    passed_edges_tot_final = []
                    # greedy validation
                    
                    tour_len = 0
                    R = torch.zeros(B_val).cuda()
                    logprobs = 0
                    Idx = []
                    reward = 0
                    batch_data_val,User_id, nx_graphs_ori = generate_batch_data_with_user(B_val,size_val,num_users_val,MAX_USER, p = p, if_avrg_degree = if_avrg_degree, if_degree = if_degree)
                    write_input_from_torch_data_with_virtual_with_user(epoch, batch_data_val, User_id, num_users,file_path_size)
                    # batch_data_val,User_id, nx_graphs_ori = from_txt_to_graph(num_users_val,size_val,MAX_USER,train_size = False, if_epoch = False, epoch = epoch, num_batch = n_epoch,file_path = file_path_size)
                    start_time = time.time()
                    batch_data_val = batch_data_val.cuda()
                    batch_data_val_ori = copy.deepcopy(batch_data_val).cuda()
                    final_val = 100000
                    
                    R = torch.zeros(B_val).cuda()
                    batch_data_val = copy.deepcopy(batch_data_val_ori).cuda()
                    nx_graphs = copy.deepcopy(nx_graphs_ori)
                    tot_cost = torch.zeros(B_val).cuda()
                    
                    h = None
                    c = None
                    for user in range(num_users_val):
                        if user+1 > num_users_val * 2/3:
                            transmission = torch.full((B_val,), 0.25).cuda()
                        elif user+1 > num_users_val * 1/3:
                            transmission = torch.full((B_val,), 0.50).cuda()
                        else:
                            transmission = torch.full((B_val,), 1.0).cuda()
                        X = batch_data_val.x
                        X = torch.Tensor(X).cuda()
                        mask = torch.full((B_val,size_val), float('-inf')).cuda()

                        Y = X.view(B_val, size_val, 6)    # to the same batch size
                        idx0_ini = User_id[:,user].cuda()
                        idx0 = idx0_ini.clone().cuda()
                        idx1 = idx0.clone().cuda()
                        idx_pass = torch.tensor([]).cuda()
                        if_continue = torch.ones([B_val]).cuda()
                        first_turn = True
                        idx_pass = torch.cat((idx_pass,idx0_ini.unsqueeze(1)),dim = 1)
                        new_if_continue = if_continue.clone()
                        for k in range(B_val):
                            if batch_data_val[k].x[idx0_ini[k],1] == 1:
                                new_if_continue[k] = 0
                        if_continue = new_if_continue.clone()
                        if torch.sum(if_continue) == 0:
                            continue
                        while True:
                            for n in range(B_val):  #batch
                                connected_nodes = ( batch_data_val[n].edge_index[0].cuda() == idx0[n])   #index
                                neighbors = batch_data_val[n].edge_index[1, connected_nodes].flatten().unsqueeze(0)
                                mask[n,neighbors] = 0.0
                            #let visited nodes be -inf
                            for n in range(B_val):
                                if first_turn == False:
                                    for i in idx_pass[n]:
                                        mask[n,i.int()] = float('-inf')
                            else:
                                first_turn = False
                            #let the last position of mask be 0
                            for n in range(B_val):
                                if idx0[n] != size_val-1:
                                    mask[n,size_val-1] = 0.0
                            for n in range(B_val):
                                if if_continue[n] == 0:
                                    mask[n,0] = 0.0
                            output, h, c, hidden_u = model(idx=idx0, X_all=X,batch_size = B_val,node_size = size_val, 
                                                        h=h, c=c, mask=mask,edge_index = batch_data_val.edge_index,edge_attr = batch_data_val.edge_attr)
                            
                            sampler = torch.distributions.Categorical(output)
                            
                            idx1 = torch.argmax(output, dim=1)
                            Idx.append(idx1.data)
                            idx_pass = torch.cat((idx_pass,(idx1*if_continue).unsqueeze(1)),dim = 1)
                            
                            edge_attr = torch.tensor([]).cuda()
                            for n in range(B_val):
                                if if_continue[n] == 0:
                                    edge_attr = torch.cat((edge_attr,torch.tensor([[0]]).cuda()),dim = 0)
                                else:
                                    edge_position = ((batch_data_val[n].edge_index[0] == idx0[n]) & (batch_data_val[n].edge_index[1] == idx1[n]))
                                    edge_attr =  torch.cat((edge_attr,batch_data_val[n].edge_attr[edge_position]),dim = 0)
                            edge_attr = edge_attr.reshape(shape=[B_val]).cuda()
                            for n in range(B_val):
                                if edge_attr[n] == -1:
                                    edge_attr[n] = virtual_edge_attr
                            Y1 = Y[[i for i in range(B_val)], idx1.data]
                            
                            reward = edge_attr * if_continue * transmission
                            
                            
                            R += reward
                            mask = torch.full((B_val,size_val), float('-inf')).cuda()
                            idx0 = idx1.clone()

                            for m in range(B_val):
                                if if_continue[m] != 0:
                                    if Y1[m,1] == 1:
                                        if_continue[m] = 0
                                        if m == 0:
                                            idx_pass_node0= idx_pass[0].clone()
                            if torch.sum(if_continue) == 0:
                                break
                            
                        new_x = batch_data_val.x.clone()
                        for n in range(B_val):
                            for m in idx_pass[n]:
                                new_x[n*size_val+m.int()] = torch.tensor([0,1,0,0,0,0], dtype=torch.float)
                        #set virtual node feature to 0
                        for n in range(B_val):
                            new_x[n*size_val+size_val-1] = torch.tensor([1,0,0,0,0,0], dtype=torch.float)
                        batch_data_val.x = new_x
                        
                        passed_edges = [(int(idx_pass_node0[i].item()), int(idx_pass_node0[i+1].item())) 
                                        for i in range(len(idx_pass_node0.tolist())-1)]
                        
                        passed_edges_tot_final = passed_edges_tot_final + [(transmission[0].item(),transmission[0].item())] + passed_edges
                        val_mean.append(R.mean().item())
                        val_std.append(R.std().item())
                    end_time = time.time()
                    time_cost = end_time - start_time
                    tot_cost[0] = tot_cost[0]+ R[0].item()
                    final_val = min(final_val, tot_cost[0].item())
                        
                    if first_run:
                        if os.path.exists(file_path_output):
                            os.remove(file_path_output)
                            print(f"{file_path_output} is deleted")
                        else:
                            print(file_path_output + " file does not exist")
                        if os.path.exists(file_path_output_path):
                            os.remove(file_path_output_path)
                            print(f"{file_path_output_path} is deleted")
                        else:
                            print(file_path_output_path + " file does not exist")
                        first_run = False
                    tour_len = R.mean().item()
                    with open(file_path_output, 'a') as f:
                        f.write(str(epoch) + ',' + str(final_val) + ',' + str(time_cost) + '\n')
                    with open(file_path_output_path, 'a') as f:
                        for value in passed_edges_tot_final:
                            f.write(str(epoch) + ',' + str(value[0]) + ',' + str(value[1]) + '\n')