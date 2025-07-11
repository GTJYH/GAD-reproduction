import dgl
import torch
import torch.nn.functional as F
import random
import os
import dgl.function as fn
from dgl.data.utils import load_graphs
import numpy as np
import scipy.io as sio
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def t_v_t_split(train_ratio, val_ratio, num_nodes):
    node_idx = np.arange(num_nodes)
    train_num = int(train_ratio * num_nodes)
    val_num = int(val_ratio * num_nodes)

    selected_idx = np.random.choice(node_idx, train_num+val_num, replace=False)

    train_mask = torch.zeros(num_nodes).bool()
    val_mask = torch.zeros(num_nodes).bool()

    train_mask[selected_idx[:train_num]] = True
    val_mask[selected_idx[train_num:]] = True
    test_mask = torch.logical_and(~train_mask, ~val_mask)

    print(torch.sum(train_mask))
    print(torch.sum(val_mask))
    print(torch.sum(test_mask))
    return train_mask, val_mask, test_mask


def idx_sample(idxes):
    num_idx = len(idxes)
    random_add = torch.randint(1, num_idx, (1,)).item()
    idx = torch.arange(0, num_idx)

    shuffled_idx = torch.remainder(idx+random_add, num_idx)

    return shuffled_idx

def row_normalization(feats):
    return F.normalize(feats, p=2, dim=1)


def load_data(dataname, path='./raw_dataset/Flickr'):
    data = sio.loadmat(f'{path}/{dataname}.mat')

    adj = data['Network'].toarray()
    feats = torch.FloatTensor(data['Attributes'].toarray())
    label = torch.LongTensor(data['Label'].reshape(-1))

    graph = dgl.from_scipy(coo_matrix(adj)).remove_self_loop()
    graph.ndata['feat'] = feats
    graph.ndata['label'] = label

    return graph


def my_load_data(dataname, path='./data/'):
    """使用dgl.load_graphs读取数据集，严格按照用户要求"""
    # 首先尝试从data目录读取
    data_dir = os.path.join(path, dataname + '.bin')
    if os.path.exists(data_dir):
        try:
            graphs, _ = dgl.load_graphs(data_dir)
            graph = graphs[0]
            print(f"成功加载数据集 {dataname} 从 {data_dir}")
            return graph
        except Exception as e:
            print(f"从data目录加载失败: {e}")
    
    # 如果data目录没有，直接从datasets目录读取原始文件
    datasets_dir = '../datasets'  # 修正路径：从GADAM目录到datasets目录
    dataset_file = os.path.join(datasets_dir, dataname.lower())
    
    if os.path.exists(dataset_file):
        try:
            # 严格按照用户要求：使用dgl.load_graphs读取
            graphs, _ = dgl.load_graphs(dataset_file)
            graph = graphs[0]
            print(f"成功从原始数据集目录加载 {dataname} 从 {dataset_file}")
            print(f"  节点数: {graph.num_nodes()}")
            print(f"  边数: {graph.num_edges()}")
            if 'feat' in graph.ndata:
                print(f"  特征维度: {graph.ndata['feat'].shape[1]}")
            if 'label' in graph.ndata:
                print(f"  异常节点数: {graph.ndata['label'].sum().item()}")
            return graph
        except Exception as e:
            print(f"从原始数据集目录加载失败: {e}")
    
    raise FileNotFoundError(f"无法找到数据集 {dataname}")


def pyg_to_dgl(pyg_graph):
    # Extract the PyG graph components
    edge_index = pyg_graph.edge_index
    edge_attr = pyg_graph.edge_attr
    num_nodes = pyg_graph.num_nodes
    node_attr = pyg_graph.x
    labels = pyg_graph.y
    # Create a DGL graph
    dgl_graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
    dgl_graph.ndata['feat'] = node_attr
    dgl_graph.ndata['label'] = labels
    # Set edge attributes if they exist
    if edge_attr is not None:
        dgl_graph.edata['edge_attr'] = torch.tensor(edge_attr)

    return dgl_graph

