#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC数据集适配器
支持从统一路径加载DGL格式的数据集
"""

import os
import torch
import numpy as np
import scipy.sparse as sp
import dgl
from dgl.data.utils import load_graphs
import warnings
warnings.filterwarnings("ignore")

# 创建一个简单的Data类，不依赖torch_geometric
class Data:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def load_dgl_dataset(dataset_name, datasets_dir='../datasets'):
    """从统一路径加载DGL格式的数据集"""
    dataset_path = os.path.join(datasets_dir, dataset_name.lower())
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    try:
        # 使用dgl.load_graphs加载数据集
        graphs, _ = dgl.load_graphs(dataset_path)
        graph = graphs[0]
        
        print(f"成功加载数据集: {dataset_name}")
        print(f"  节点数: {graph.num_nodes()}")
        print(f"  边数: {graph.num_edges()}")
        
        # 检查节点特征
        if 'feat' in graph.ndata:
            feat = graph.ndata['feat']
            print(f"  特征维度: {feat.shape[1]}")
        elif 'feature' in graph.ndata:
            feat = graph.ndata['feature']
            print(f"  特征维度: {feat.shape[1]}")
        elif 'features' in graph.ndata:
            feat = graph.ndata['features']
            print(f"  特征维度: {feat.shape[1]}")
        elif 'x' in graph.ndata:
            feat = graph.ndata['x']
            print(f"  特征维度: {feat.shape[1]}")
        else:
            raise ValueError(f"数据集中没有找到节点特征字段。可用字段: {list(graph.ndata.keys())}")
        
        # 检查标签
        if 'label' in graph.ndata:
            labels = graph.ndata['label']
            num_anomalies = labels.sum().item()
            anomaly_ratio = num_anomalies / graph.num_nodes()
            print(f"  异常节点数: {num_anomalies}")
            print(f"  异常比例: {anomaly_ratio:.4f}")
        else:
            raise ValueError(f"数据集中没有找到标签字段。可用字段: {list(graph.ndata.keys())}")
        
        return graph, feat, labels
        
    except Exception as e:
        print(f"加载数据集 {dataset_name} 失败: {e}")
        raise

def dgl_to_arc_format(dgl_graph, feat, labels, target_dim=64):
    """将DGL图转换为ARC格式"""
    # 获取边索引
    edge_index = dgl_graph.edges()
    edge_index = torch.stack(edge_index)
    
    # 创建邻接矩阵
    num_nodes = dgl_graph.num_nodes()
    adj = sp.coo_matrix((np.ones(edge_index.shape[1]), 
                        (edge_index[0].numpy(), edge_index[1].numpy())), 
                       shape=(num_nodes, num_nodes))
    
    # 特征降维到目标维度
    if feat.shape[1] != target_dim:
        print(f"特征降维: {feat.shape[1]} -> {target_dim}")
        # 使用PCA或简单的线性变换进行降维
        if feat.shape[1] > target_dim:
            # 使用SVD进行降维
            U, S, V = torch.svd(feat)
            feat_reduced = torch.mm(feat, V[:, :target_dim])
        else:
            # 如果特征维度小于目标维度，用零填充
            feat_reduced = torch.zeros(feat.shape[0], target_dim, dtype=feat.dtype)
            feat_reduced[:, :feat.shape[1]] = feat
        feat = feat_reduced
    
    # 直接返回DGL图对象，添加必要的属性
    dgl_graph.ndata['x'] = feat
    dgl_graph.ndata['y'] = labels
    dgl_graph.ndata['ano_labels'] = labels.float()  # ARC需要float类型的异常标签
    dgl_graph.adj = adj  # 添加邻接矩阵属性
    
    return dgl_graph, adj

class DatasetAdapter:
    """数据集适配器，兼容ARC的Dataset类接口"""
    
    def __init__(self, dims, name='cora', prefix='../datasets/'):
        self.shot_mask = None
        self.shot_idx = None
        self.graph = None
        self.x_list = None
        self.name = name
        
        # 数据集名称映射（映射到实际文件名）
        dataset_mapping = {
            'reddit': 'reddit',
            'weibo': 'weibo',
            'amazon': 'amazon',
            'yelp': 'yelp',
            'tolokers': 'tolokers',
            'questions': 'questions',
            'tfinance': 'tfinance',
            'elliptic': 'elliptic',
            'dgraphfin': 'dgraphfin',
            # 保留一些原始名称的映射
            'Reddit': 'reddit',
            'Amazon': 'amazon',
            'YelpChi': 'yelp',
            'Tolokers': 'tolokers',
            'Questions': 'questions',
            'T-Finance': 'tfinance',
            'Elliptic': 'elliptic',
            'DGraph-Fin': 'dgraphfin'
        }
        
        # 获取实际的数据集名称
        actual_name = dataset_mapping.get(name, name.lower())
        
        try:
            # 加载DGL格式的数据集
            dgl_graph, feat, labels = load_dgl_dataset(actual_name, prefix)
            
            # 转换为ARC格式（直接使用DGL图对象），使用dims作为目标维度
            graph, adj = dgl_to_arc_format(dgl_graph, feat, labels, target_dim=dims)
            
            # 设置属性
            self.label = labels.numpy()
            self.adj_norm = adj
            self.feat = feat
            
            # 直接使用DGL图对象
            self.graph = graph
            
            print(f"数据集 {name} 加载完成")
            
        except Exception as e:
            print(f"加载数据集 {name} 失败: {e}")
            raise
    
    def few_shot(self, shot=10):
        """设置few-shot学习"""
        if 'ano_labels' in self.graph.ndata:
            y = self.graph.ndata['ano_labels']
        else:
            y = self.graph.ndata['y']
        num_nodes = y.shape[0]
        normal_idx = torch.where(y == 0)[0].tolist()
        import random
        random.shuffle(normal_idx)
        shot_idx = torch.tensor(normal_idx[:shot])
        shot_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # 确保张量在正确的设备上
        if torch.cuda.is_available() and hasattr(self.graph, 'device'):
            shot_idx = shot_idx.to(self.graph.device)
            shot_mask = shot_mask.to(self.graph.device)
        
        self.graph.shot_idx = shot_idx
        shot_mask[shot_idx] = True
        self.graph.shot_mask = shot_mask
    
    def propagated(self, k):
        """特征传播 - 使用DGL的图卷积"""
        x = self.graph.ndata['x']
        if torch.cuda.is_available():
            x = x.cuda()
            self.graph = self.graph.to('cuda')
        
        h_list = [x]
        for _ in range(k):
            # 使用DGL的消息传递进行特征传播
            self.graph.ndata['h'] = h_list[-1]
            self.graph.update_all(
                message_func=dgl.function.copy_u('h', 'm'),
                reduce_func=dgl.function.mean('m', 'h')
            )
            h_list.append(self.graph.ndata['h'])
        
        self.graph.x_list = h_list

# 替换原始的Dataset类
class Dataset(DatasetAdapter):
    """兼容ARC原始代码的Dataset类"""
    pass 