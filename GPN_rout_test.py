from GPN_rout import *

torch.autograd.set_detect_anomaly(True)

for size, num_users in [(30,12)]:
    save_idx = 0
    num_range = 12
    if size == 50 and num_users == 12:
        num_range = 9
    size = size + 1
    for n_range in range(num_range):
        
        if_avrg_degree = False
        if_degree = False
        if n_range > 0 and n_range < 5:
            save_idx = (n_range - 1) % 4 +3
            if_avrg_degree = True
        elif n_range > 4 and n_range < 9:
            
            save_idx = (n_range - 1) % 4 +3
            if save_idx == 3:
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
            file_path_root = '/home/GPN_rout/data/avr'+ str(save_idx) + 'n' +  str(size-1) + '/'
            file_path_size = file_path_root +  'avrdegree'+ str(save_idx) + 'size' +  str(size-1) +'.txt'
            file_path_output = file_path_root + 'avrdegree'+ str(save_idx) + 'size' +  str(size-1) + 'output.csv'
            file_path_output_path = file_path_root + 'avrdegree'+ str(save_idx) + 'size' +  str(size-1) + 'output_path.csv'  
        elif if_degree:
            file_path_root = '/home/GPN_rout/data/d'+ str(save_idx) + 'n' +  str(size-1) + '/'
            file_path_size = file_path_root +  'degree'+ str(save_idx) + 'size' +  str(size-1) +'.txt'
            file_path_output = file_path_root + 'degree'+ str(save_idx) + 'size' +  str(size-1) + 'output.csv'
            file_path_output_path = file_path_root + 'degree'+ str(save_idx) + 'size' +  str(size-1) + 'output_path.csv'  
        else:
            file_path_root = '/home/GPN_rout/data/n'+ str(size-1) + 'u' + str(num_users) + '/'
            file_path_size = file_path_root +  'size' + str(size-1) + '_user' + str(num_users) +'.txt'
            file_path_output = file_path_root + 'size' + str(size-1) + '_user' + str(num_users) + 'output.csv'
            file_path_output_path = file_path_root + 'size' + str(size-1) + '_user' + str(num_users) + 'output_path.csv'  
        
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

        # model = GPN(n_feature=6, n_hidden=128,num_heads=4).cuda()

        # load model
        model = torch.load(save_root,weights_only=False).cuda()
        optimizer = optim.Adam(model.parameters(), lr=learn_rate)

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
                    # generate validation data(if needed)
                    batch_data_val,User_id, nx_graphs_ori = generate_batch_data_with_user(B_val,size_val,num_users_val,MAX_USER, p = 0.1, if_avrg_degree = if_avrg_degree, if_degree = if_degree)
                    # write data to txt(if needed)
                    write_input_from_torch_data_with_virtual_with_user(epoch, batch_data_val, User_id, num_users,file_path_size)
                    # load data from txt(if needed)
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