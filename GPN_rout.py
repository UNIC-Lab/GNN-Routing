import math
import os
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
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
                G = nx.gnp_random_graph(num_nodes, p)
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


def get_selected_x(batch_data, idx):
    idx = torch.tensor(idx, device=batch_data.x.device)  # convert idx to tensor on the same device as batch_data.x
    ptr = batch_data.ptr

    idx_global = ptr[:-1] + idx

    selected_x = batch_data.x[idx_global]
    return selected_x

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

