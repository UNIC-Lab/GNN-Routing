from GPN_rout import *

size = 30
learn_rate = 5e-4    # learning rate
B = 16    # batch_size
B_val = 8    # validation size
size_val = 30
steps = 2500    # training steps
n_epoch = 20    # epochs
num_users = 9
num_users_val = 9
MAX_USER = 20
virtual_edge_attr = 10

save_root = '/home/GPN_rout/model/gpn_model.pt'

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

model = GPN(n_feature=6, n_hidden=128,num_heads=4).cuda()

# load model
# model = torch.load(save_root).cuda()
optimizer = optim.Adam(model.parameters(), lr=learn_rate)

lr_decay_step = 500
lr_decay_rate = 0.96
opt_scheduler = lr_scheduler.MultiStepLR(optimizer, range(lr_decay_step, lr_decay_step*1000,
                                     lr_decay_step), gamma=lr_decay_rate)

C = 0     # baseline
R = 0     # reward


val_mean = []
val_std = []

for epoch in range(n_epoch):
    for i in range(steps):
        print("step:",i)
        R = 0
        logprobs = 0
        reward = 0

        C = 0
        baseline = 0
        
        batch_data,User_id, G = generate_batch_data_with_user(B,size,num_users,MAX_USER,p=0.1)
        batch_data = batch_data.cuda()
        
        batch_data_bl = copy.deepcopy(batch_data).cuda()
        h = None
        c = None
    
        for user in range(num_users):
            if user+1 > num_users * 2/3:
                transmission = torch.full((B,), 0.25).cuda()
            elif user+1 > num_users * 1/3:
                transmission = torch.full((B,), 0.50).cuda()
            else:
                transmission = torch.full((B,), 1.0).cuda()
            optimizer.zero_grad()
            h_bl = h
            c_bl = c
            
            X = batch_data.x
            X = torch.Tensor(X).cuda()
            
            mask = torch.full((B,size), float('-inf')).cuda()
            Y = X.view(B,size,6)
            idx0_ini = User_id[:,user].cuda()
            idx0 = idx0_ini.clone().cuda()
            idx1 = idx0.clone().cuda()
            idx_pass = torch.tensor([]).cuda()
            if_continue = torch.ones([B]).cuda()
            first_turn = True
            idx_pass = torch.cat((idx_pass,idx0_ini.unsqueeze(1)),dim = 1)
            batch_data_bl.x = batch_data.x.clone().cuda()
            
            new_if_continue = if_continue.clone()
            
            for k in range(B):
                if batch_data[k].x[idx0_ini[k],1] == 1:
                    new_if_continue[k] = 0
            if_continue = new_if_continue.clone()
            if torch.sum(if_continue) == 0:
                continue

            while True:
                #assign mask
                new_mask = mask.clone()
                for n in range(B):  #batch
                    connected_nodes = ( batch_data[n].edge_index[0].cuda() == idx0[n])   #index
                    neighbors = batch_data[n].edge_index[1, connected_nodes].flatten().unsqueeze(0)
                    new_mask[n,neighbors] = 0.0
                    #mask[n,idx0[n]] = 0.0
                #let visited nodes be -inf
                for n in range(B):
                    if first_turn == False:
                        for m in idx_pass[n]:
                            new_mask[n,m.int()] = float('-inf')
                    else:
                        first_turn = False
                #set the last position of mask to 0
                for n in range(B):
                    if idx0[n] != size-1:
                        new_mask[n,size-1] = 0
                for n in range(B):
                    if if_continue[n] == 0:
                        new_mask[n,0] = 0.0
                mask = new_mask
                output, h, c, _ = model(idx=idx0, X_all=X,batch_size = B,node_size = size, h=h, 
                                        c=c, mask=mask,edge_index = batch_data.edge_index,edge_attr = batch_data.edge_attr)
                
                #if finish, set output to [1,0...0]
                new_output = output.clone()
                sampler = torch.distributions.Categorical(output)


                idx1 = sampler.sample()         # now the idx has B elements [B]
                idx_pass = torch.cat((idx_pass,(idx1*if_continue).unsqueeze(1)),dim = 1)
                #find the edge_attr
                edge_attr = torch.tensor([]).cuda()
                for n in range(B):
                    if if_continue[n] == 0:
                        edge_attr = torch.cat((edge_attr,torch.tensor([[0.]]).cuda()),dim = 0)
                    else:
                        edge_position = ((batch_data[n].edge_index[0] == idx0[n]) & (batch_data[n].edge_index[1] == idx1[n]))
                        edge_attr =  torch.cat((edge_attr,batch_data[n].edge_attr[edge_position]),dim = 0)
                edge_attr = edge_attr.reshape(shape=[B]).cuda()
                for n in range(B):
                    if edge_attr[n] == -1:
                        edge_attr[n] = virtual_edge_attr
                Y1 = Y[[m for m in range(B)], idx1.data].clone()    #每个当前所在节点的特征
                selected_x = get_selected_x(batch_data, idx0_ini)
                reward = edge_attr * if_continue * selected_x[:,4]
                
                R += reward

                TINY = 1e-15
                logprobs += torch.log(output[[i for i in range(B)], idx1.data]+TINY).cuda() * if_continue
                
                mask = torch.full((B,size), float('-inf')).cuda()
                idx0 = idx1.clone()
                
                new_if_continue = if_continue.clone()
                for m in range(B):
                    if new_if_continue[m] != 0:
                        if Y1[m,1] == 1:
                            new_if_continue[m] = 0
                if_continue = new_if_continue.clone()
                if torch.sum(if_continue) == 0:
                    break
                
            new_x = batch_data.x.clone()
            for n in range(B):
                for m in idx_pass[n]:
                    new_x[n*size+m.int()] = torch.tensor([0,1,0,0,0,0], dtype=torch.float)
            #set virtual node feature to -1
            for n in range(B):
                new_x[n*size+size-1] = torch.tensor([1,0,0,0,0,0], dtype=torch.float)
            batch_data.x = new_x
            #R += torch.norm(Y1-Y_ini, dim=1)
            

            # self-critic base line
            mask = torch.full((B,size), float('-inf')).cuda()
            
            Y = X.view(B,size,6)
            
            x = Y[:,0,:]
            
            idx0 = idx0_ini.clone()
            idx1 = idx0.clone()
            idx_pass = torch.tensor([]).cuda()
            idx_pass = torch.cat((idx_pass,idx0_ini.unsqueeze(1)),dim = 1)
            if_continue = torch.ones([B]).cuda()
            first_turn = True
            new_if_continue = if_continue.clone()
            for k in range(B):
                if batch_data_bl[k].x[idx0_ini[k],1] == 1:
                    new_if_continue[k] = 0
            if_continue = new_if_continue.clone()
            if torch.sum(if_continue) == 0:
                continue

            while True:
                new_mask = mask.clone()
                for n in range(B):  #batch
                    connected_nodes = ( batch_data_bl[n].edge_index[0].cuda() == idx0[n])   #index
                    neighbors = batch_data_bl[n].edge_index[1, connected_nodes].flatten().unsqueeze(0)
                    new_mask[n,neighbors] = 0.0
                    
                #let visited nodes be -inf
                for n in range(B):
                    if first_turn == False:
                        for m in idx_pass[n]:
                            new_mask[n,m.int()] = float('-inf')
                else:
                    first_turn = False
                # set the last position of mask to 0
                for n in range(B):
                    if idx0[n] != size-1:
                        new_mask[n,size-1] = 0.0
                for n in range(B):
                    if if_continue[n] == 0:
                        new_mask[n,0] = 0.0
                mask = new_mask
                
                output, h_bl, c_bl, _ = model(idx=idx0, X_all=X,batch_size = B,node_size = size, h=h_bl, c=c_bl, mask=mask,edge_index = batch_data_bl.edge_index,edge_attr = batch_data_bl.edge_attr)
                idx1 = torch.argmax(output, dim=1)    # greedy baseline
                idx_pass = torch.cat((idx_pass,(idx1*if_continue).unsqueeze(1)),dim = 1)
                #find the edge_attr
                edge_attr = torch.tensor([]).cuda()
                for n in range(B):
                    if if_continue[n] == 0:
                        edge_attr = torch.cat((edge_attr,torch.tensor([[0.]]).cuda()),dim = 0)
                    else:
                        edge_position = ((batch_data_bl[n].edge_index[0] == idx0[n]) & (batch_data_bl[n].edge_index[1] == idx1[n]))
                        edge_attr =  torch.cat((edge_attr,batch_data_bl[n].edge_attr[edge_position]),dim = 0)
                edge_attr = edge_attr.reshape(shape=[B]).cuda()
                for n in range(B):
                    if edge_attr[n] == -1:
                        edge_attr[n] = virtual_edge_attr
                Y1 = Y[[m for m in range(B)], idx1.data].clone()
                selected_x = get_selected_x(batch_data_bl, idx0_ini)
                baseline = edge_attr * if_continue * selected_x[:,4]
                C += baseline
                mask = torch.full((B,size), float('-inf')).cuda()
                idx0 = idx1.clone()
                new_if_continue = if_continue.clone()
                for m in range(B):
                    if new_if_continue[m] != 0:
                        if Y1[m,1] == 1:
                            new_if_continue[m] = 0
                if_continue = new_if_continue.clone()
                if torch.sum(if_continue) == 0:
                    break
                
    
        gap = (R-C).mean()
        loss = ((R-C-gap)*logprobs).mean()
        # print(loss)
        loss.backward()
        
        max_grad_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_grad_norm, norm_type=2)
        optimizer.step()
        opt_scheduler.step()
        print('i',i)
        if i % 50 == 0:
            print("epoch:{}, batch:{}/{}, reward:{}"
                .format(epoch, i, steps, R.mean().item()))

            # greedy validation
            
            tour_len = 0
            R = 0
            logprobs = 0
            Idx = []
            reward = 0
            batch_data_val,User_id, nx_graphs_ori = generate_batch_data_with_user(B_val,size_val,num_users_val,MAX_USER, p = 0.08)
            batch_data_val = batch_data_val.cuda()
            batch_data_val_ori = copy.deepcopy(batch_data_val).cuda()
            final_val = 100000
            
            R = 0
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
                    
                    selected_x = get_selected_x(batch_data_val, idx0_ini)
                    reward = edge_attr * if_continue * selected_x[:,4]
                    
                    
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
                passed_edges = [(idx_pass_node0[i].item(), idx_pass_node0[i+1].item()) for i in range(len(idx_pass_node0.tolist())-1)]
                
                val_mean.append(R.mean().item())
                val_std.append(R.std().item())
            tot_cost[0] = tot_cost[0]+ R[0].item()
            final_val = min(final_val, tot_cost[0].item())
                

            tour_len = R.mean().item()
            print('validation tour length:', tour_len)
            # plt.show()
            print('save model to: ', save_root)
            torch.save(model, save_root)
    print('save model to: ', save_root)
    torch.save(model, save_root)
