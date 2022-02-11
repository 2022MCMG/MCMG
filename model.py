import torch
from torch import nn
from torch.nn import Module, Parameter
import math
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
import datetime
def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable
def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
class PositionEmbedding(nn.Module):
    MODE_ADD = 'MODE_ADD'
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 mode=MODE_ADD):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings 
        self.embedding_dim = embedding_dim
        self.mode = mode
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        batch_size, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)#torch.Size([1, 5, 120])
        if self.mode == self.MODE_ADD:
            return x + embeddings
        raise NotImplementedError('Unknown mode: %s' % self.mode)
    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={}, mode={}'.format(
            self.num_embeddings, self.embedding_dim, self.mode,
        )

class GCN(Module):
    def __init__(self,hidden_size, step,dropout):
        super(GCN, self).__init__()
        self.hidden_size = hidden_size
        self.weight = Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.step = step#
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))#120
        self.dropout=dropout

    def GCNCell(self, A, hidden):        
        support=torch.matmul(hidden, self.weight)#
        output = torch.matmul(A, support)
        output=output + self.b_iah
        output = F.relu(output)
        output = F.dropout(output, self.dropout, training=self.training)
        return output

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GCNCell(A, hidden)
        return hidden

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_size, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) 
        outputs += inputs
        return outputs
class Model(Module):
    def __init__(self,hidden_size,lr,l2,step,n_head,k_blocks,args,POI_n_node, cate_n_node,regi_n_node,time_n_node,POI_dist_n_node,regi_dist_n_node,len_max):#len_max=16
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.lr=lr
        self.l2=l2
        self.len_max = len_max
        self.POI_n_node = POI_n_node
        self.cate_n_node = cate_n_node
        self.regi_n_node = regi_n_node
        self.time_n_node=time_n_node
        self.POI_dist_n_node=POI_dist_n_node
        self.regi_dist_n_node=regi_dist_n_node
        self.batch_size = args.batch_size
        self.step=step
        self.drop_out=args.GCN_drop_out
        self.dropout_fwd=args.SA_drop_out
        self.n_head=n_head
        self.k_blocks=k_blocks
        self.w_p_c_r_g1=Parameter(torch.ones(3))
        self.w_p_c_r_g2=Parameter(torch.ones(3))

        self.w_p=Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.w_p_t=Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.w_p_d=Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.w_c=Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.w_c_t=Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.w_r=Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.w_r_t=Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.w_r_d=Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
     
        self.POI_embedding = nn.Embedding(self.POI_n_node, self.hidden_size)#n_no
        self.cate_embedding=nn.Embedding(self.cate_n_node, self.hidden_size)
        self.regi_embedding=nn.Embedding(self.regi_n_node, self.hidden_size)
        self.time_embedding=nn.Embedding(self.time_n_node, self.hidden_size)
        self.POI_dist_embedding=nn.Embedding(self.POI_dist_n_node, self.hidden_size)
        self.regi_dist_embedding=nn.Embedding(self.regi_dist_n_node, self.hidden_size)
        self.emb_dropout = torch.nn.Dropout(p=self.dropout_fwd)
        self.gcn = GCN(self.hidden_size, self.step,self.drop_out)
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.n_head).cuda()
        self.ffn=PointWiseFeedForward(self.hidden_size,self.dropout_fwd)
        self.pe = PositionEmbedding(len_max, self.hidden_size)
        self.loss_function = nn.CrossEntropyLoss()#loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=50, gamma=0.1)
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    def get_channel_scores(self,POI_hidden, POI_mask,cate_hidden,cate_mask,regi_hidden,regi_mask,group_label_inputs):
        POI_attn_output = POI_hidden
        cate_attn_output = cate_hidden
        regi_attn_output = regi_hidden
        for k in range(self.k_blocks):
            POI_attn_output = POI_attn_output.transpose(0,1)
            POI_attn_output, POI_attn_output_weights = self.multihead_attn(POI_attn_output, POI_attn_output, POI_attn_output)
            POI_attn_output = POI_attn_output.transpose(0,1)
            POI_attn_output = self.ffn(POI_attn_output)
            
            cate_attn_output = cate_attn_output.transpose(0,1)
            cate_attn_output, cate_attn_output_weights = self.multihead_attn(cate_attn_output, cate_attn_output, cate_attn_output)
            cate_attn_output = cate_attn_output.transpose(0,1)
            cate_attn_output = self.ffn(cate_attn_output)
            
            regi_attn_output = regi_attn_output.transpose(0,1)
            regi_attn_output, regi_attn_output_weights = self.multihead_attn(regi_attn_output, regi_attn_output, regi_attn_output)
            regi_attn_output = regi_attn_output.transpose(0,1)
            regi_attn_output = self.ffn(regi_attn_output)
            
        POI_hn = POI_attn_output[torch.arange(POI_mask.shape[0]).long(), torch.sum(POI_mask, 1) - 1]  # use last one as global interest
        cate_hn = cate_attn_output[torch.arange(cate_mask.shape[0]).long(), torch.sum(cate_mask, 1) - 1]
        regi_hn = regi_attn_output[torch.arange(regi_mask.shape[0]).long(), torch.sum(regi_mask, 1) - 1]

        POI_represent=POI_hn
        cate_represent=cate_hn
        regi_represent=regi_hn

        POI_candi = self.POI_embedding.weight[1:]  
        cate_candi = self.cate_embedding.weight[1:]  
        regi_candi = self.regi_embedding.weight[1:]  
       
        cate_scores = torch.matmul(cate_represent, cate_candi.transpose(1, 0))       
        regi_scores = torch.matmul(regi_represent, regi_candi.transpose(1, 0))
        #group 1
        w_p_g1 = torch.exp(self.w_p_c_r_g1[0]) / torch.sum(torch.exp(self.w_p_c_r_g1))
        w_c_g1 = torch.exp(self.w_p_c_r_g1[1]) / torch.sum(torch.exp(self.w_p_c_r_g1))
        w_r_g1 = torch.exp(self.w_p_c_r_g1[2]) / torch.sum(torch.exp(self.w_p_c_r_g1))
        #group 2
        w_p_g2 = torch.exp(self.w_p_c_r_g2[0]) / torch.sum(torch.exp(self.w_p_c_r_g2))
        w_c_g2 = torch.exp(self.w_p_c_r_g2[1]) / torch.sum(torch.exp(self.w_p_c_r_g2))
        w_r_g2 = torch.exp(self.w_p_c_r_g2[2]) / torch.sum(torch.exp(self.w_p_c_r_g2))
        for i in range(len(POI_hidden)):
            if group_label_inputs[i]==1:
                POI_represent[i]=w_p_g1*POI_represent[i].clone()+w_c_g1*cate_represent[i].clone()+w_r_g1*regi_represent[i].clone()
            elif group_label_inputs[i]==2:
                POI_represent[i]=w_p_g2*POI_represent[i].clone()+w_c_g2*cate_represent[i].clone()+w_r_g2*regi_represent[i].clone()
            else:
                raise ValueError(f'Invalid group label')
        POI_scores = torch.matmul(POI_represent,POI_candi.transpose(1, 0))
        return POI_scores,cate_scores,regi_scores

    def forward(self, POI_inputs,POI_A,cate_inputs,regi_inputs,time_inputs,POI_dist_inputs,regi_dist_inputs):
        hidden_POI=self.POI_embedding(POI_inputs)
        hidden_cate=self.cate_embedding(cate_inputs)
        hidden_regi=self.regi_embedding(regi_inputs)
        hidden_time=self.time_embedding(time_inputs)
        hidden_POI_dist=self.POI_dist_embedding(POI_dist_inputs)
        hidden_regi_dist=self.regi_dist_embedding(regi_dist_inputs)
        hidden_POI_gcn = self.gcn(POI_A, hidden_POI)
        return hidden_POI_gcn,hidden_cate,hidden_regi,hidden_time,hidden_POI_dist,hidden_regi_dist

def forward(model, i,POI_adj_matrix,POI_data,cate_data,regi_data,time_data,POI_dist_data,regi_dist_data,group_label):
    POI_mask, POI_groundtruth,POI_inputs = POI_data.get_slice(i)
    cate_mask, cate_groundtruth,cate_inputs = cate_data.get_slice(i)
    regi_mask, regi_groundtruth,regi_inputs = regi_data.get_slice(i)
    time_mask, time_groundtruth,time_inputs = time_data.get_slice(i)
    POI_dist_mask, POI_dist_groundtruth,POI_dist_inputs = POI_dist_data.get_slice(i)
    regi_dist_mask, regi_dist_groundtruth,regi_dist_inputs = regi_dist_data.get_slice(i)
    group_label_inputs = group_label.get_slice(i)
    POI_all_items=[]
    cate_all_items=[]
    regi_all_items=[]
    time_all_items=[]
    POI_dist_all_items=[]
    regi_dist_all_items=[]
    for i in range(model.POI_n_node): 
        POI_all_items.append(i)
    for i in range(model.cate_n_node):  
        cate_all_items.append(i)
    for i in range(model.regi_n_node):  
        regi_all_items.append(i)
    for i in range(model.time_n_node):
        time_all_items.append(i)
    for i in range(model.POI_dist_n_node):
        POI_dist_all_items.append(i)
    for i in range(model.regi_dist_n_node):
        regi_dist_all_items.append(i)
   
    POI_adj_matrix=trans_to_cuda(torch.Tensor(POI_adj_matrix).float())
    POI_mask = trans_to_cuda(torch.Tensor(POI_mask).long())
    POI_all_items=trans_to_cuda(torch.Tensor(POI_all_items).long())
    POI_inputs=trans_to_cuda(torch.Tensor(POI_inputs).long())

    cate_mask = trans_to_cuda(torch.Tensor(cate_mask).long())
    cate_all_items=trans_to_cuda(torch.Tensor(cate_all_items).long())
    cate_inputs=trans_to_cuda(torch.Tensor(cate_inputs).long())

    regi_mask = trans_to_cuda(torch.Tensor(regi_mask).long())
    regi_all_items=trans_to_cuda(torch.Tensor(regi_all_items).long())
    regi_inputs=trans_to_cuda(torch.Tensor(regi_inputs).long())

    time_mask = trans_to_cuda(torch.Tensor(time_mask).long())
    time_all_items=trans_to_cuda(torch.Tensor(time_all_items).long())
    time_inputs=trans_to_cuda(torch.Tensor(time_inputs).long())

    POI_dist_mask = trans_to_cuda(torch.Tensor(POI_dist_mask).long())
    POI_dist_all_items=trans_to_cuda(torch.Tensor(POI_dist_all_items).long())
    POI_dist_inputs=trans_to_cuda(torch.Tensor(POI_dist_inputs).long())

    regi_dist_mask = trans_to_cuda(torch.Tensor(regi_dist_mask).long())
    regi_dist_all_items=trans_to_cuda(torch.Tensor(regi_dist_all_items).long())
    regi_dist_inputs=trans_to_cuda(torch.Tensor(regi_dist_inputs).long())

    POI_hidden,cate_hidden,regi_hidden,time_hidden,POI_dist_hidden,regi_dist_hidden = model(POI_all_items,POI_adj_matrix,cate_all_items,regi_all_items,time_all_items,POI_dist_all_items,regi_dist_all_items)
    POI_seq_hidden=torch.stack([torch.matmul(POI_hidden[i],model.w_p) for i in POI_inputs])+torch.stack([torch.matmul(time_hidden[i],model.w_p_t) for i in time_inputs])+torch.stack([torch.matmul(POI_dist_hidden[i],model.w_p_d) for i in POI_dist_inputs])    
    cate_seq_hidden = torch.stack([torch.matmul(cate_hidden[i],model.w_c) for i in cate_inputs])+torch.stack([torch.matmul(time_hidden[i],model.w_c_t) for i in time_inputs])
    regi_seq_hidden = torch.stack([torch.matmul(regi_hidden[i],model.w_r) for i in regi_inputs])+torch.stack([torch.matmul(time_hidden[i],model.w_r_t) for i in time_inputs])+torch.stack([torch.matmul(regi_dist_hidden[i],model.w_r_d) for i in regi_dist_inputs])

    POI_seq_hidden = model.pe(POI_seq_hidden)
    cate_seq_hidden = model.pe(cate_seq_hidden)
    regi_seq_hidden = model.pe(regi_seq_hidden)
    #add self_attention
    POI_score_result,cate_score_result,regi_score_result=model.get_channel_scores(POI_seq_hidden, POI_mask,cate_seq_hidden, cate_mask,regi_seq_hidden, regi_mask,group_label_inputs)
    return POI_groundtruth, POI_score_result,cate_groundtruth, cate_score_result,regi_groundtruth, regi_score_result,group_label_inputs
def train_test(model,POI_adj_matrix,POI_train_data, POI_test_data,cate_train_data,cate_test_data,regi_train_data,regi_test_data,time_train_data,time_test_data,POI_dist_train_data,POI_dist_test_data,regi_dist_train_data,regi_dist_test_data,group_label_train_valid,group_label_test):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = POI_train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.optimizer.zero_grad()
        POI_groundtruth, POI_scores,cate_groundtruth, cate_scores,regi_groundtruth, regi_scores,group_label_inputs= forward(model, i,POI_adj_matrix,POI_train_data,cate_train_data,regi_train_data,time_train_data,POI_dist_train_data,regi_dist_train_data,group_label_train_valid)
        POI_groundtruth = trans_to_cuda(torch.Tensor(POI_groundtruth).long())
        cate_groundtruth = trans_to_cuda(torch.Tensor(cate_groundtruth).long())
        regi_groundtruth = trans_to_cuda(torch.Tensor(regi_groundtruth).long())
        loss_POI = model.loss_function(POI_scores, POI_groundtruth-1)
        loss_regi = model.loss_function(regi_scores, regi_groundtruth-1)
        loss_cate = model.loss_function(cate_scores, cate_groundtruth-1)
        loss=loss_POI+loss_regi+loss_cate
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    
    print('start predicting: ', datetime.datetime.now())
    
    model.eval()
    g1_POI_NDCG_1,g1_POI_NDCG_5, g1_POI_NDCG_10=[],[],[]
    g1_cate_NDCG_1,g1_cate_NDCG_5, g1_cate_NDCG_10=[],[],[]
    g1_regi_NDCG_1,g1_regi_NDCG_5, g1_regi_NDCG_10=[],[],[]

    g1_POI_HR_1,g1_POI_HR_5, g1_POI_HR_10=[],[],[]
    g1_cate_HR_1,g1_cate_HR_5, g1_cate_HR_10=[],[],[]
    g1_regi_HR_1,g1_regi_HR_5, g1_regi_HR_10=[],[],[]
    

    g2_POI_NDCG_1,g2_POI_NDCG_5, g2_POI_NDCG_10=[],[],[]
    g2_cate_NDCG_1,g2_cate_NDCG_5, g2_cate_NDCG_10=[],[],[]
    g2_regi_NDCG_1,g2_regi_NDCG_5, g2_regi_NDCG_10=[],[],[]
    g2_POI_HR_1,g2_POI_HR_5,g2_POI_HR_10=[],[],[]
    g2_cate_HR_1,g2_cate_HR_5,g2_cate_HR_10=[],[],[]
    g2_regi_HR_1,g2_regi_HR_5,g2_regi_HR_10=[],[],[]

    slices = POI_test_data.generate_batch(model.batch_size)
    for i in slices:
        POI_groundtruth, POI_scores,cate_groundtruth, cate_scores,regi_groundtruth, regi_scores,group_label_inputs = forward(model, i,POI_adj_matrix,POI_test_data,cate_test_data,regi_test_data,time_test_data,POI_dist_test_data,regi_dist_test_data,group_label_test)
        sub_items_1 = POI_scores.topk(1)[1]
        sub_items_5 = POI_scores.topk(5)[1]
        sub_items_10 = POI_scores.topk(10)[1]     

        sub_items_1 = trans_to_cpu(sub_items_1).detach().numpy()
        sub_items_5 = trans_to_cpu(sub_items_5).detach().numpy()
        sub_items_10 = trans_to_cpu(sub_items_10).detach().numpy()

        #category
        cate_sub_items_1 = cate_scores.topk(1)[1]
        cate_sub_items_5 = cate_scores.topk(5)[1]
        cate_sub_items_10 = cate_scores.topk(10)[1]

        cate_sub_items_1 = trans_to_cpu(cate_sub_items_1).detach().numpy()
        cate_sub_items_5 = trans_to_cpu(cate_sub_items_5).detach().numpy()
        cate_sub_items_10 = trans_to_cpu(cate_sub_items_10).detach().numpy()
         #region
        regi_sub_items_1 = regi_scores.topk(1)[1]
        regi_sub_items_5 = regi_scores.topk(5)[1]
        regi_sub_items_9 = regi_scores.topk(9)[1]

        regi_sub_items_1 = trans_to_cpu(regi_sub_items_1).detach().numpy()
        regi_sub_items_5 = trans_to_cpu(regi_sub_items_5).detach().numpy()
        regi_sub_items_9 = trans_to_cpu(regi_sub_items_9).detach().numpy() 

        #group performance
        #top 1
        #POI
        
        for i in range(len(POI_groundtruth)):
            if group_label_inputs[i]==1:
                g1_POI_HR_1.append(np.isin(POI_groundtruth[i]-1, sub_items_1[i][:]))
              
            elif group_label_inputs[i]==2:
                g2_POI_HR_1.append(np.isin(POI_groundtruth[i]-1, sub_items_1[i][:]))

        POI_g1_NDCG_i=0
        POI_g2_NDCG_i=0
        POI_g1_groundtruth_num=0
        cate_g1_groundtruth_num=0
        regi_g1_groundtruth_num=0
        POI_g2_groundtruth_num=0
        cate_g2_groundtruth_num=0
        regi_g2_groundtruth_num=0
        for i in range(len(POI_groundtruth)):
            if group_label_inputs[i]==1:
                POI_g1_groundtruth_num=POI_g1_groundtruth_num+1
                cate_g1_groundtruth_num=cate_g1_groundtruth_num+1
                regi_g1_groundtruth_num=regi_g1_groundtruth_num+1
            else:
                POI_g2_groundtruth_num=POI_g2_groundtruth_num+1
                cate_g2_groundtruth_num=cate_g2_groundtruth_num+1
                regi_g2_groundtruth_num=regi_g2_groundtruth_num+1

        for i in range(len(sub_items_1)):
            for j in range(len(sub_items_1[i])):
                if ((POI_groundtruth[i]-1)==sub_items_1[i][j]) and (group_label_inputs[i]==1): 
                    POI_g1_NDCG_i+=1/(math.log2(1+j+1))
                    break
                if ((POI_groundtruth[i]-1)==sub_items_1[i][j]) and (group_label_inputs[i]==2): 
                    POI_g2_NDCG_i+=1/(math.log2(1+j+1))
                    break
      
        POI_g1_NDCG_i/=POI_g1_groundtruth_num
        g1_POI_NDCG_1.append(POI_g1_NDCG_i)
        POI_g2_NDCG_i/=POI_g2_groundtruth_num
        g2_POI_NDCG_1.append(POI_g2_NDCG_i) 

        #category
        for i in range(len(cate_groundtruth)):
            if group_label_inputs[i]==1:
                g1_cate_HR_1.append(np.isin(cate_groundtruth[i]-1, cate_sub_items_1[i][:]))
              
            elif group_label_inputs[i]==2:
                g2_cate_HR_1.append(np.isin(cate_groundtruth[i]-1, cate_sub_items_1[i][:]))

        cate_g1_NDCG_i=0
        cate_g2_NDCG_i=0
        for i in range(len(cate_sub_items_1)):
            for j in range(len(cate_sub_items_1[i])):
                if ((cate_groundtruth[i]-1)==cate_sub_items_1[i][j]) and (group_label_inputs[i]==1): 
                    cate_g1_NDCG_i+=1/(math.log2(1+j+1))
                    break
                if ((cate_groundtruth[i]-1)==cate_sub_items_1[i][j]) and (group_label_inputs[i]==2): 
                    cate_g2_NDCG_i+=1/(math.log2(1+j+1))
                    break
        cate_g1_NDCG_i/=cate_g1_groundtruth_num
        g1_cate_NDCG_1.append(cate_g1_NDCG_i)  
        cate_g2_NDCG_i/=cate_g2_groundtruth_num
        g2_cate_NDCG_1.append(cate_g2_NDCG_i) 

        #region
        for i in range(len(regi_groundtruth)):
            if group_label_inputs[i]==1:
                g1_regi_HR_1.append(np.isin(regi_groundtruth[i]-1, regi_sub_items_1[i][:]))
              
            elif group_label_inputs[i]==2:
                g2_regi_HR_1.append(np.isin(regi_groundtruth[i]-1, regi_sub_items_1[i][:]))

        regi_g1_NDCG_i=0
        regi_g2_NDCG_i=0

        for i in range(len(regi_sub_items_1)):
            for j in range(len(regi_sub_items_1[i])):
                if ((regi_groundtruth[i]-1)==regi_sub_items_1[i][j]) and (group_label_inputs[i]==1): 
                    regi_g1_NDCG_i+=1/(math.log2(1+j+1))
                    break
                if ((regi_groundtruth[i]-1)==regi_sub_items_1[i][j]) and (group_label_inputs[i]==2): 
                    regi_g2_NDCG_i+=1/(math.log2(1+j+1))
                    break
        regi_g1_NDCG_i/=regi_g1_groundtruth_num
        g1_regi_NDCG_1.append(regi_g1_NDCG_i)  
        regi_g2_NDCG_i/=regi_g2_groundtruth_num
        g2_regi_NDCG_1.append(regi_g2_NDCG_i) 
        ############top 5###########
        #POI
        for i in range(len(POI_groundtruth)):
            if group_label_inputs[i]==1:
                g1_POI_HR_5.append(np.isin(POI_groundtruth[i]-1, sub_items_5[i][:]))
              
            elif group_label_inputs[i]==2:
                g2_POI_HR_5.append(np.isin(POI_groundtruth[i]-1, sub_items_5[i][:]))

        POI_g1_NDCG_i_5=0
        POI_g2_NDCG_i_5=0
        Len_T = len(POI_groundtruth)
        for i in range(len(sub_items_5)):
            for j in range(len(sub_items_5[i])):
                if ((POI_groundtruth[i]-1)==sub_items_5[i][j]) and (group_label_inputs[i]==1): 
                    POI_g1_NDCG_i_5+=1/(math.log2(1+j+1))
                    break
                if ((POI_groundtruth[i]-1)==sub_items_5[i][j]) and (group_label_inputs[i]==2): 
                    POI_g2_NDCG_i_5+=1/(math.log2(1+j+1))
                    break
        POI_g1_NDCG_i_5/=POI_g1_groundtruth_num
        g1_POI_NDCG_5.append(POI_g1_NDCG_i_5)  
        POI_g2_NDCG_i_5/=POI_g2_groundtruth_num
        g2_POI_NDCG_5.append(POI_g2_NDCG_i_5)   

        #category
        for i in range(len(cate_groundtruth)):
            if group_label_inputs[i]==1:
                g1_cate_HR_5.append(np.isin(cate_groundtruth[i]-1, cate_sub_items_5[i][:]))
              
            elif group_label_inputs[i]==2:
                g2_cate_HR_5.append(np.isin(cate_groundtruth[i]-1, cate_sub_items_5[i][:]))

        cate_g1_NDCG_i_5=0
        cate_g2_NDCG_i_5=0
        for i in range(len(cate_sub_items_5)):
            for j in range(len(cate_sub_items_5[i])):
                if ((cate_groundtruth[i]-1)==cate_sub_items_5[i][j]) and (group_label_inputs[i]==1): 
                    cate_g1_NDCG_i_5+=1/(math.log2(1+j+1))
                    break
                if ((cate_groundtruth[i]-1)==cate_sub_items_5[i][j]) and (group_label_inputs[i]==2): 
                    cate_g2_NDCG_i_5+=1/(math.log2(1+j+1))
                    break
        cate_g1_NDCG_i_5/=cate_g1_groundtruth_num
        g1_cate_NDCG_5.append(cate_g1_NDCG_i_5)  
        cate_g2_NDCG_i_5/=cate_g2_groundtruth_num
        g2_cate_NDCG_5.append(cate_g2_NDCG_i_5)     
        #region
        for i in range(len(regi_groundtruth)):
            if group_label_inputs[i]==1:
                g1_regi_HR_5.append(np.isin(regi_groundtruth[i]-1, regi_sub_items_5[i][:]))
              
            elif group_label_inputs[i]==2:
                g2_regi_HR_5.append(np.isin(regi_groundtruth[i]-1, regi_sub_items_5[i][:]))

        regi_g1_NDCG_i_5=0
        regi_g2_NDCG_i_5=0
        for i in range(len(regi_sub_items_5)):
            for j in range(len(regi_sub_items_5[i])):
                if ((regi_groundtruth[i]-1)==regi_sub_items_5[i][j]) and (group_label_inputs[i]==1): 
                    regi_g1_NDCG_i_5+=1/(math.log2(1+j+1))
                    break
                if ((regi_groundtruth[i]-1)==regi_sub_items_5[i][j]) and (group_label_inputs[i]==2): 
                    regi_g2_NDCG_i_5+=1/(math.log2(1+j+1))
                    break
        regi_g1_NDCG_i_5/=regi_g1_groundtruth_num
        g1_regi_NDCG_5.append(regi_g1_NDCG_i_5)  
        regi_g2_NDCG_i_5/=regi_g2_groundtruth_num
        g2_regi_NDCG_5.append(regi_g2_NDCG_i_5)  

        ##############top 10#####################
        #POI
        for i in range(len(POI_groundtruth)):
            if group_label_inputs[i]==1:
                g1_POI_HR_10.append(np.isin(POI_groundtruth[i]-1, sub_items_10[i][:]))
              
            elif group_label_inputs[i]==2:
                g2_POI_HR_10.append(np.isin(POI_groundtruth[i]-1, sub_items_10[i][:]))

        g1_NDCG_i_10=0
        g2_NDCG_i_10=0
        for i in range(len(sub_items_10)):
            for j in range(len(sub_items_10[i])):
                if ((POI_groundtruth[i]-1)==sub_items_10[i][j]) and (group_label_inputs[i]==1): 
                    g1_NDCG_i_10+=1/(math.log2(1+j+1))
                    break
                if ((POI_groundtruth[i]-1)==sub_items_10[i][j]) and (group_label_inputs[i]==2): 
                    g2_NDCG_i_10+=1/(math.log2(1+j+1))
                    break
        g1_NDCG_i_10/=POI_g1_groundtruth_num
        g1_POI_NDCG_10.append(g1_NDCG_i_10)  
        g2_NDCG_i_10/=POI_g2_groundtruth_num
        g2_POI_NDCG_10.append(g2_NDCG_i_10)      

        #category
        for i in range(len(cate_groundtruth)):
            if group_label_inputs[i]==1:
                g1_cate_HR_10.append(np.isin(cate_groundtruth[i]-1, cate_sub_items_10[i][:]))
              
            elif group_label_inputs[i]==2:
                g2_cate_HR_10.append(np.isin(cate_groundtruth[i]-1, cate_sub_items_10[i][:]))

        cate_g1_NDCG_i_10=0
        cate_g2_NDCG_i_10=0
        for i in range(len(cate_sub_items_10)):
            for j in range(len(cate_sub_items_10[i])):
                if ((cate_groundtruth[i]-1)==cate_sub_items_10[i][j]) and (group_label_inputs[i]==1): 
                    cate_g1_NDCG_i_10+=1/(math.log2(1+j+1))
                    break
                if ((cate_groundtruth[i]-1)==cate_sub_items_10[i][j]) and (group_label_inputs[i]==2): 
                    cate_g2_NDCG_i_10+=1/(math.log2(1+j+1))
                    break
        cate_g1_NDCG_i_10/=cate_g1_groundtruth_num
        g1_cate_NDCG_10.append(cate_g1_NDCG_i_10)  
        cate_g2_NDCG_i_10/=cate_g2_groundtruth_num
        g2_cate_NDCG_10.append(cate_g2_NDCG_i_10)     
        #region
        for i in range(len(regi_groundtruth)):
            if group_label_inputs[i]==1:
                g1_regi_HR_10.append(np.isin(regi_groundtruth[i]-1, regi_sub_items_9[i][:]))
              
            elif group_label_inputs[i]==2:
                g2_regi_HR_10.append(np.isin(regi_groundtruth[i]-1, regi_sub_items_9[i][:]))

        regi_g1_NDCG_i_10=0
        regi_g2_NDCG_i_10=0
        for i in range(len(regi_sub_items_9)):
            for j in range(len(regi_sub_items_9[i])):
                if ((regi_groundtruth[i]-1)==regi_sub_items_9[i][j]) and (group_label_inputs[i]==1): 
                    regi_g1_NDCG_i_10+=1/(math.log2(1+j+1))
                    break
                if ((regi_groundtruth[i]-1)==regi_sub_items_9[i][j]) and (group_label_inputs[i]==2): 
                    regi_g2_NDCG_i_10+=1/(math.log2(1+j+1))
                    break
        regi_g1_NDCG_i_10/=regi_g1_groundtruth_num
        g1_regi_NDCG_10.append(regi_g1_NDCG_i_10)  
        regi_g2_NDCG_i_10/=regi_g2_groundtruth_num
        g2_regi_NDCG_10.append(regi_g2_NDCG_i_10)             
                              
 #top1   
    g1_POI_HR_1 = np.mean(g1_POI_HR_1) 
    g1_POI_NDCG_1=np.mean(g1_POI_NDCG_1)

    g1_cate_HR_1 = np.mean(g1_cate_HR_1) 
    g1_cate_NDCG_1=np.mean(g1_cate_NDCG_1)

    g1_regi_HR_1 = np.mean(g1_regi_HR_1) 
    g1_regi_NDCG_1=np.mean(g1_regi_NDCG_1)
    
    g2_POI_HR_1 = np.mean(g2_POI_HR_1) 
    g2_POI_NDCG_1=np.mean(g2_POI_NDCG_1)

    g2_cate_HR_1 = np.mean(g2_cate_HR_1) 
    g2_cate_NDCG_1=np.mean(g2_cate_NDCG_1)

    g2_regi_HR_1 = np.mean(g2_regi_HR_1) 
    g2_regi_NDCG_1=np.mean(g2_regi_NDCG_1)
#top5
    #group 1
    g1_POI_HR_5 = np.mean(g1_POI_HR_5) 
    g1_POI_NDCG_5=np.mean(g1_POI_NDCG_5)

    g1_cate_HR_5 = np.mean(g1_cate_HR_5) 
    g1_cate_NDCG_5=np.mean(g1_cate_NDCG_5)

    g1_regi_HR_5 = np.mean(g1_regi_HR_5) 
    g1_regi_NDCG_5=np.mean(g1_regi_NDCG_5)
    
    #group 2
    g2_POI_HR_5 = np.mean(g2_POI_HR_5) 
    g2_POI_NDCG_5=np.mean(g2_POI_NDCG_5)

    g2_cate_HR_5 = np.mean(g2_cate_HR_5) 
    g2_cate_NDCG_5=np.mean(g2_cate_NDCG_5)

    g2_regi_HR_5 = np.mean(g2_regi_HR_5) 
    g2_regi_NDCG_5=np.mean(g2_regi_NDCG_5)
#top10
    #g1
    g1_POI_HR_10 = np.mean(g1_POI_HR_10) 
    g1_POI_NDCG_10=np.mean(g1_POI_NDCG_10)
    g1_cate_HR_10 = np.mean(g1_cate_HR_10) 
    g1_cate_NDCG_10=np.mean(g1_cate_NDCG_10)
    g1_regi_HR_10 = np.mean(g1_regi_HR_10) 
    g1_regi_NDCG_10=np.mean(g1_regi_NDCG_10)
    #g2
    g2_POI_HR_10 = np.mean(g2_POI_HR_10) 
    g2_POI_NDCG_10=np.mean(g2_POI_NDCG_10)
    g2_cate_HR_10 = np.mean(g2_cate_HR_10) 
    g2_cate_NDCG_10=np.mean(g2_cate_NDCG_10)
    g2_regi_HR_10 = np.mean(g2_regi_HR_10) 
    g2_regi_NDCG_10=np.mean(g2_regi_NDCG_10)

    return g1_POI_HR_1,g1_POI_HR_5,g1_POI_HR_10,g1_POI_NDCG_1,g1_POI_NDCG_5,g1_POI_NDCG_10,g1_cate_HR_1,g1_cate_HR_5,g1_cate_HR_10,g1_cate_NDCG_1,g1_cate_NDCG_5,g1_cate_NDCG_10,g1_regi_HR_1,g1_regi_HR_5,g1_regi_HR_10,g1_regi_NDCG_1,g1_regi_NDCG_5,g1_regi_NDCG_10,g2_POI_HR_1,g2_POI_HR_5,g2_POI_HR_10,g2_POI_NDCG_1,g2_POI_NDCG_5,g2_POI_NDCG_10,g2_cate_HR_1,g2_cate_HR_5,g2_cate_HR_10,g2_cate_NDCG_1,g2_cate_NDCG_5,g2_cate_NDCG_10,g2_regi_HR_1,g2_regi_HR_5,g2_regi_HR_10,g2_regi_NDCG_1,g2_regi_NDCG_5,g2_regi_NDCG_10