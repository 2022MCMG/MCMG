import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CAL', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--hidden_size', type=int, default=150, help='the size of embeddings')
    parser.add_argument('--epoch', type=int, default=100, help='the number of epochs')
    parser.add_argument('--lr', type=float, default=0.004, help='learning rate')  
    parser.add_argument('--l2', type=float, default=0.0001, help='L2 regularization coefficient')
    parser.add_argument('--step', type=int, default=1, help='the number of GCN layers')
    parser.add_argument('--patience', type=int, default=10, help='the number of epoch to wait before early stop')
    parser.add_argument('--topk', type=int, default=10,help='top number of recommendation list')
    parser.add_argument('--n_head', type=int, default=1,help='the heads of self-attention')
    parser.add_argument('--k_blocks', type=int, default=1,help='the blocks of self-attention')
    parser.add_argument('--GCN_drop_out', type=float, default=0.5,help='the droup out rate of GCN')
    parser.add_argument('--SA_drop_out', type=float, default=0.5,help='the droup out rate of self-attention')
    parser.add_argument('--tune_epochs', type=int, default=50,help='tune epochs')
    parser.add_argument('--score_metric', type=str, default='NDCG10', help='metric used to define hyperopt score')
    args = parser.parse_args()
    return args
