import pickle
import os
import argparse
import pickle
import time
from utils import *
from model import *
from parameter_setting import parse_args
args = parse_args()


def main():
    if args.dataset == 'CAL':
        train_data = pickle.load(open('./dataset/train_CAL.txt', 'rb'))
        test_data = pickle.load(open('./dataset/test_CAL.txt', 'rb'))
        group_label_train=pickle.load(open('./dataset/train_group_label.txt', 'rb'))
        group_label_test=pickle.load(open('./dataset/test_group_label.txt', 'rb'))
        POI_adj_matrix = pickle.load(open('./dataset/AdjacentMatrix.txt', 'rb'))
        POI_n_node=579
        cate_n_node=149
        regi_n_node=10
        time_n_node=25
        POI_dist_n_node=31
        regi_dist_n_node=31
    else:
        raise ValueError(f'Invalid dataset')
 
    POI_train_data = Data(train_data[0:2], shuffle=False)
    cate_train_data = Data(train_data[2:4], shuffle=False)
    regi_train_data = Data(train_data[4:6], shuffle=False)
    time_train_data = Data(train_data[6:8], shuffle=False)
    POI_dist_train_data = Data(train_data[8:10], shuffle=False)
    regi_dist_train_data = Data(train_data[10:12], shuffle=False)
    group_label_train= Data_GroupLabel(group_label_train, shuffle=False)
    group_label_test= Data_GroupLabel(group_label_test, shuffle=False)

    POI_test_data = Data(test_data[0:2], shuffle=False)
    cate_test_data = Data(test_data[2:4], shuffle=False)
    regi_test_data = Data(test_data[4:6], shuffle=False)
    time_test_data = Data(test_data[6:8], shuffle=False)
    POI_dist_test_data = Data(test_data[8:10], shuffle=False)
    regi_dist_test_data = Data(test_data[10:12], shuffle=False)
    model=trans_to_cuda(Model(args.hidden_size,args.lr,args.l2,args.step,args.n_head,args.k_blocks,args,POI_n_node,cate_n_node,regi_n_node,time_n_node, POI_dist_n_node,regi_dist_n_node,max(POI_train_data.len_max, POI_test_data.len_max)))
    start = time.time()
    best_result = [0, 0,0,0,0,0]
    best_epoch = [0, 0,0,0,0,0]
    bad_counter = 0
    for epoch in range(args.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        g1_POI_HR_1,g1_POI_HR_5,g1_POI_HR_10,g1_POI_NDCG_1,g1_POI_NDCG_5,g1_POI_NDCG_10,g1_cate_HR_1,g1_cate_HR_5,g1_cate_HR_10,g1_cate_NDCG_1,g1_cate_NDCG_5,g1_cate_NDCG_10,g1_regi_HR_1,g1_regi_HR_5,g1_regi_HR_10,g1_regi_NDCG_1,g1_regi_NDCG_5,g1_regi_NDCG_10,g2_POI_HR_1,g2_POI_HR_5,g2_POI_HR_10,g2_POI_NDCG_1,g2_POI_NDCG_5,g2_POI_NDCG_10,g2_cate_HR_1,g2_cate_HR_5,g2_cate_HR_10,g2_cate_NDCG_1,g2_cate_NDCG_5,g2_cate_NDCG_10,g2_regi_HR_1,g2_regi_HR_5,g2_regi_HR_10,g2_regi_NDCG_1,g2_regi_NDCG_5,g2_regi_NDCG_10= train_test(model, POI_adj_matrix,POI_train_data, POI_test_data,cate_train_data,cate_test_data,regi_train_data,regi_test_data,time_train_data,time_test_data,POI_dist_train_data,POI_dist_test_data,regi_dist_train_data,regi_dist_test_data,group_label_train,group_label_test)
        flag = 0
        if g1_POI_NDCG_1 >= best_result[0]:
            best_result[0] = g1_POI_NDCG_1
            best_epoch[0] = epoch
            flag = 1
        if g1_POI_NDCG_5 >= best_result[1]:
            best_result[1] = g1_POI_NDCG_5
            best_epoch[1] = epoch
            flag = 1
        if g1_POI_NDCG_10 >= best_result[2]:
            best_result[2] = g1_POI_NDCG_10
            best_epoch[2] = epoch
            flag = 1
        if g2_POI_NDCG_1 >= best_result[3]:
            best_result[3] = g2_POI_NDCG_1
            best_epoch[3] = epoch
            flag = 1
        if g2_POI_NDCG_5 >= best_result[4]:
            best_result[4] = g2_POI_NDCG_5
            best_epoch[4] = epoch
            flag = 1
        if g2_POI_NDCG_10 >= best_result[5]:
            best_result[5] = g2_POI_NDCG_10
            best_epoch[5] = epoch
            flag = 1
        tmp_metric = np.array([g1_POI_HR_1,g1_POI_HR_5,g1_POI_HR_10,g1_POI_NDCG_1,g1_POI_NDCG_5,g1_POI_NDCG_10,g1_cate_HR_1,g1_cate_HR_5,g1_cate_HR_10,g1_cate_NDCG_1,g1_cate_NDCG_5,g1_cate_NDCG_10,g1_regi_HR_1,g1_regi_HR_5,g1_regi_HR_10,g1_regi_NDCG_1,g1_regi_NDCG_5,g1_regi_NDCG_10,g2_POI_HR_1,g2_POI_HR_5,g2_POI_HR_10,g2_POI_NDCG_1,g2_POI_NDCG_5,g2_POI_NDCG_10,g2_cate_HR_1,g2_cate_HR_5,g2_cate_HR_10,g2_cate_NDCG_1,g2_cate_NDCG_5,g2_cate_NDCG_10,g2_regi_HR_1,g2_regi_HR_5,g2_regi_HR_10,g2_regi_NDCG_1,g2_regi_NDCG_5,g2_regi_NDCG_10])
        tmp_metric = [f'{mt:.4f}' for mt in tmp_metric]
        line = ','.join(tmp_metric) + '\n'
        f.write(line)
        f.flush()
        print('Best Result:')
        bad_counter += 1 - flag
        if bad_counter >= args.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))

    
if __name__ == '__main__':
    result_log_path = './test_log/'
    if not os.path.exists(result_log_path):
        os.makedirs(result_log_path)
    f = open(result_log_path + f'{args.dataset}_result.csv', 'w', encoding='utf-8')
    f.write('g1_POI_HR_1,g1_POI_HR_5,g1_POI_HR_10,g1_POI_NDCG_1,g1_POI_NDCG_5,g1_POI_NDCG_10,g1_cate_HR_1,g1_cate_HR_5,g1_cate_HR_10,g1_cate_NDCG_1,g1_cate_NDCG_5,g1_cate_NDCG_10,g1_regi_HR_1,g1_regi_HR_5,g1_regi_HR_10,g1_regi_NDCG_1,g1_regi_NDCG_5,g1_regi_NDCG_10,g2_POI_HR_1,g2_POI_HR_5,g2_POI_HR_10,g2_POI_NDCG_1,g2_POI_NDCG_5,g2_POI_NDCG_10,g2_cate_HR_1,g2_cate_HR_5,g2_cate_HR_10,g2_cate_NDCG_1,g2_cate_NDCG_5,g2_cate_NDCG_10,g2_regi_HR_1,g2_regi_HR_5,g2_regi_HR_10,g2_regi_NDCG_1,g2_regi_NDCG_5,g2_regi_NDCG_10' + '\n')
    f.flush()
    main()
    f.close()
