import torch
import torch.nn as nn
import dgl
from dgl.data import register_data_args
import argparse, time
from model import *
from utils import *
from sklearn.metrics import roc_auc_score, recall_score, average_precision_score
from pytorch_memlab import LineProfiler, profile


def train_local(net, graph, feats, opt, args, init=True):
    memo = {}
    labels = graph.ndata['label']
    num_nodes=  graph.num_nodes()

    device = args.gpu
    if device >= 0:
        torch.cuda.set_device(device)
        net = net.to(device)
        labels = labels.cuda()
        feats = feats.cuda()

    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
    
    if init:
        net.apply(init_xavier)
    
    print('train on:', 'cpu' if device<0 else 'gpu {}'.format(device))

    cnt_wait = 0
    best = 999
    dur = []

    for epoch in range(args.local_epochs):
        net.train()
        if epoch >= 3:
            t0 = time.time()

        opt.zero_grad()
        loss, l1, l2 = net(feats)

        loss.backward()
        opt.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        if loss.item() < best:
            best = loss.item()
            torch.save(net.state_dict(), 'best_local_model.pkl')
        
        # 安全计算平均时间，避免空列表
        avg_time = np.mean(dur) if dur else 0.0
        print("Epoch {} | Time(s) {:.4f} | Loss {:.4f} | l1 {:.4f} | l2 {:.4f}"
              .format(epoch+1, avg_time, loss.item(), l1.item(), l2.item()))

    memo['graph'] = graph
    net.load_state_dict(torch.load('best_local_model.pkl'))
    h, mean_h = net.encoder(feats)
    h, mean_h = h.detach(), mean_h.detach()
    memo['h'] = h
    memo['mean_h'] = mean_h

    torch.save(memo, 'memo.pth')


def load_info_from_local(local_net, device):
    if device >= 0:
        torch.cuda.set_device(device)
        local_net = local_net.to(device)

    memo = torch.load('memo.pth')
    local_net.load_state_dict(torch.load('best_local_model.pkl'))
    graph = memo['graph']
    pos = graph.ndata['pos']
    scores = -pos.detach()
    ano_topk = 0.05  # k_ano
    nor_topk = 0.3  # k_nor
    num_nodes = graph.num_nodes()

    num_ano = int(num_nodes * ano_topk)
    _, ano_idx = torch.topk(scores, num_ano)

    num_nor = int(num_nodes * nor_topk)
    _, nor_idx = torch.topk(-scores, num_nor)

    # 检查节点特征字段名，与main函数保持一致
    if 'feat' in graph.ndata:
        feats = graph.ndata['feat']
    elif 'features' in graph.ndata:
        feats = graph.ndata['features']
    elif 'feature' in graph.ndata:
        feats = graph.ndata['feature']
    elif 'x' in graph.ndata:
        feats = graph.ndata['x']
    else:
        # 如果没有特征，报错
        available_fields = list(graph.ndata.keys())
        raise ValueError(f"数据集中没有找到节点特征字段。可用的字段: {available_fields}")

    h, _ = local_net.encoder(feats)

    center = h[nor_idx].mean(dim=0).detach()

    if device >= 0:
        memo = {k: v.to(device) for k, v in memo.items()}
        nor_idx = nor_idx.cuda()
        ano_idx = ano_idx.cuda()
        center = center.cuda()

    return memo, nor_idx, ano_idx, center


def train_global(global_net, opt, graph, args):
    epochs = args.global_epochs

    labels = graph.ndata['label'].cpu().numpy()
    num_nodes=  graph.num_nodes()
    device = args.gpu
    # 检查节点特征字段名，与main函数保持一致
    if 'feat' in graph.ndata:
        feats = graph.ndata['feat']
    elif 'features' in graph.ndata:
        feats = graph.ndata['features']
    elif 'feature' in graph.ndata:
        feats = graph.ndata['feature']
    elif 'x' in graph.ndata:
        feats = graph.ndata['x']
    else:
        # 如果没有特征，报错
        available_fields = list(graph.ndata.keys())
        raise ValueError(f"数据集中没有找到节点特征字段。可用的字段: {available_fields}")
    pos = graph.ndata['pos']

    if device >= 0:
        torch.cuda.set_device(device)
        global_net = global_net.to(device)
        # labels = labels.cuda()
        feats = feats.cuda()

    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
    
    init = True
    if init:
        global_net.apply(init_xavier)
    
    print('train on:', 'cpu' if device<0 else 'gpu {}'.format(device))

    cnt_wait = 0
    best = 999
    dur = []

    pred_labels = np.zeros_like(labels)
    best_metrics = {'mix_auc': 0, 'recall@k': 0, 'ap': 0}
    
    for epoch in range(epochs):
        global_net.train()
        if epoch >= 3:
            t0 = time.time()

        opt.zero_grad()
        loss, scores = global_net(feats, epoch)
        loss.backward()
        opt.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        if loss.item() < best:
            best = loss.item()
            torch.save(global_net.state_dict(), 'best_global_model.pkl')

        mix_score = -(scores + pos)
        mix_score = mix_score.detach().cpu().numpy()

        # 计算三个评估指标
        mix_auc = roc_auc_score(labels, mix_score)
        
        sorted_idx = np.argsort(mix_score)
        k = int(sum(labels))
        topk_idx = sorted_idx[-k:]
        pred_labels[topk_idx] = 1

        recall_k = recall_score(np.ones(k), labels[topk_idx])
        ap = average_precision_score(labels, mix_score)

        # 保存最佳指标
        if mix_auc > best_metrics['mix_auc']:
            best_metrics['mix_auc'] = mix_auc
            best_metrics['recall@k'] = recall_k
            best_metrics['ap'] = ap

        # 安全计算平均时间，避免空列表
        avg_time = np.mean(dur) if dur else 0.0
        print("Epoch {} | Time(s) {:.4f} | Loss {:.4f} | mix_auc {:.4f} | recall@k {:.4f} | ap {:.4f}"
            .format(epoch+1, avg_time, loss.item(), mix_auc, recall_k, ap))
    
    print(f"\n最终最佳结果:")
    print(f"Mix_AUC: {best_metrics['mix_auc']:.4f}")
    print(f"Recall@K: {best_metrics['recall@k']:.4f}")
    print(f"AP: {best_metrics['ap']:.4f}")
    
    return best_metrics['mix_auc'], best_metrics['recall@k'], best_metrics['ap']

def main(args):
    seed_everything(args.seed)

    graph = my_load_data(args.data)
    # graph = graph.add_self_loop() test encoder=GCN
    
    # 检查节点特征字段名
    if 'feat' in graph.ndata:
        feats = graph.ndata['feat']
    elif 'features' in graph.ndata:
        feats = graph.ndata['features']
    elif 'feature' in graph.ndata:
        feats = graph.ndata['feature']
    elif 'x' in graph.ndata:
        feats = graph.ndata['x']
    else:
        # 如果没有特征，报错
        available_fields = list(graph.ndata.keys())
        raise ValueError(f"数据集中没有找到节点特征字段。可用的字段: {available_fields}")

    if args.gpu >= 0:
        graph = graph.to(args.gpu)

    in_feats = feats.shape[1]

    local_net = LocalModel(graph,
                     in_feats,
                     args.out_dim,
                     nn.PReLU(),)

    local_opt = torch.optim.Adam(local_net.parameters(), 
                                 lr=args.local_lr, 
                                 weight_decay=args.weight_decay)
    t1 = time.time()
    train_local(local_net, graph, feats, local_opt, args)
    
    # load information from LIM module
    memo, nor_idx, ano_idx, center = load_info_from_local(local_net, args.gpu)
    t2 = time.time()
    graph = memo['graph']
    global_net = GlobalModel(graph, 
                             in_feats, 
                             args.out_dim, 
                             nn.PReLU(), 
                             nor_idx, 
                             ano_idx, 
                             center)
    opt = torch.optim.Adam(global_net.parameters(), 
                                 lr=args.global_lr, 
                                 weight_decay=args.weight_decay)
    t3 = time.time()
    
    mix_auc, recall_k, ap = train_global(global_net, opt, graph, args)
    t4 = time.time()

    t_all = t2+t4-t1-t3
    print('mean_t:{:.4f}'.format(t_all / (args.local_epochs + args.global_epochs)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model')
    register_data_args(parser)
    parser.add_argument("--data", type=str, default="Cora",
                        help="dataset")
    parser.add_argument("--seed", type=int, default=717,
                        help="random seed")
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu")
    parser.add_argument("--local-lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--global-lr", type=float, default=5e-4,
                        help="learning rate")
    parser.add_argument("--local-epochs", type=int, default=100,
                        help="number of training local model epochs")
    parser.add_argument("--global-epochs", type=int, default=50,
                        help="number of training global model epochs")
    parser.add_argument("--out-dim", type=int, default=64,
                        help="number of hidden gcn units")
    parser.add_argument("--train-ratio", type=float, default=0.05,
                        help="train ratio")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=20,
                        help="early stop patience condition")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=True)

    args = parser.parse_args()
    print(args)
    main(args)
    # multi_run(args)
