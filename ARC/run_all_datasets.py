#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC模型批量运行脚本
使用GADAM逻辑：逐个数据集运行，清空缓存
"""

import os
import sys
import json
import time
import argparse
import warnings
import gc
import torch
warnings.filterwarnings("ignore")

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def clean_memory():
    """清理内存和GPU缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("内存已清理")

def run_single_dataset(dataset_name, trials=1, shot=10, device='cuda:0'):
    """运行单个数据集的ARC实验"""
    print(f"\n{'='*60}")
    print(f"开始运行数据集: {dataset_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 导入ARC的模块
        from utils import test_eval
        from train_test import ARCDetector
        import numpy as np
        
        # 数据集配置 - 每个数据集单独训练和测试
        datasets_test = [dataset_name]
        datasets_train = [dataset_name]  # 使用同一个数据集进行训练和测试
        
        # 训练配置
        train_config = {
            'device': device,
            'epochs': 40,
            'testdsets': datasets_test,
        }
        
        dims = 64
        
        # 加载数据集
        print("加载训练数据集...")
        data_train = []
        for name in datasets_train:
            try:
                from dataset_adapter import Dataset
                data_train.append(Dataset(dims, name))
                print(f"  ✓ 加载训练数据集: {name}")
            except Exception as e:
                print(f"  ✗ 加载训练数据集失败: {name} - {e}")
        
        if not data_train:
            raise ValueError("没有成功加载任何训练数据集")
        
        print("加载测试数据集...")
        data_test = []
        try:
            from dataset_adapter import Dataset
            data_test.append(Dataset(dims, dataset_name))
            print(f"  ✓ 加载测试数据集: {dataset_name}")
        except Exception as e:
            raise ValueError(f"加载测试数据集失败: {dataset_name} - {e}")
        
        # 模型配置
        model_config = {
            "model": "ARC",
            "lr": 1e-5,
            "drop_rate": 0,
            "h_feats": 1024,
            "num_prompt": 10,
            "num_hops": 2,
            "weight_decay": 5e-5,
            "in_feats": 64,
            "num_layers": 4,
            "activation": "ELU"
        }
        
        # 数据预处理
        print("数据预处理...")
        for tr_data in data_train:
            tr_data.propagated(model_config['num_hops'])
        for te_data in data_test:
            te_data.propagated(model_config['num_hops'])
        
        model_config['model'] = 'ARC'
        model_config['in_feats'] = dims
        
        # 存储结果
        auc_dict = {}
        pre_dict = {}
        recall_dict = {}
        
        # 运行trials
        for t in range(trials):
            seed = t
            print(f"Trial {t+1}/{trials}, Seed: {seed}")
            
            # 设置随机种子
            import random
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                try:
                    # 设置cudnn确定性
                    pass
                except:
                    pass
            
            train_config['seed'] = seed
            
            # 设置few-shot
            for te_data in data_test:
                te_data.few_shot(shot)
            
            # 训练和测试
            data = {'train': data_train, 'test': data_test}
            detector = ARCDetector(train_config, model_config, data)
            test_score_list = detector.train()
            
            # 收集结果
            for test_data_name, test_score in test_score_list.items():
                if test_data_name not in auc_dict:
                    auc_dict[test_data_name] = []
                    pre_dict[test_data_name] = []
                    recall_dict[test_data_name] = []
                auc_dict[test_data_name].append(test_score['AUROC'])
                pre_dict[test_data_name].append(test_score['AUPRC'])
                recall_dict[test_data_name].append(test_score['Recall@K'])
                print(f'Trial {t+1}: {test_data_name} - AUROC: {test_score["AUROC"]:.4f}, AUPRC: {test_score["AUPRC"]:.4f}, Recall@K: {test_score["Recall@K"]:.4f}')
        
        # 计算统计结果
        results = {}
        for test_data_name in auc_dict:
            auc_mean = np.mean(auc_dict[test_data_name])
            auc_std = np.std(auc_dict[test_data_name])
            pre_mean = np.mean(pre_dict[test_data_name])
            pre_std = np.std(pre_dict[test_data_name])
            recall_mean = np.mean(recall_dict[test_data_name])
            recall_std = np.std(recall_dict[test_data_name])
            
            results[test_data_name] = {
                'AUROC_mean': auc_mean,
                'AUROC_std': auc_std,
                'AUPRC_mean': pre_mean,
                'AUPRC_std': pre_std,
                'Recall@K_mean': recall_mean,
                'Recall@K_std': recall_std,
                'AUROC_list': auc_dict[test_data_name],
                'AUPRC_list': pre_dict[test_data_name],
                'Recall@K_list': recall_dict[test_data_name]
            }
            
            print(f'\n{test_data_name} 最终结果:')
            print(f'AUROC: {auc_mean:.4f} ± {auc_std:.4f}')
            print(f'AUPRC: {pre_mean:.4f} ± {pre_std:.4f}')
            print(f'Recall@K: {recall_mean:.4f} ± {recall_std:.4f}')
        
        end_time = time.time()
        total_time = end_time - start_time
        
        return {
            'status': 'success',
            'dataset': dataset_name,
            'time': total_time,
            'results': results,
            'config': {
                'trials': trials,
                'shot': shot,
                'epochs': train_config['epochs'],
                'device': device
            }
        }
        
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"运行数据集 {dataset_name} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'dataset': dataset_name,
            'time': total_time,
            'error': str(e),
            'config': {
                'trials': trials,
                'shot': shot,
                'epochs': 40,
                'device': device
            }
        }

def main():
    parser = argparse.ArgumentParser(description='ARC模型批量运行脚本')
    parser.add_argument('--datasets', nargs='+', default=None, 
                       help='指定要运行的数据集列表')
    parser.add_argument('--trials', type=int, default=1, 
                       help='每个数据集的试验次数')
    parser.add_argument('--shot', type=int, default=10, 
                       help='few-shot学习的样本数')
    parser.add_argument('--device', type=str, default='cuda:0', 
                       help='使用的设备')
    parser.add_argument('--output', type=str, default='arc_results.json', 
                       help='结果输出文件')
    
    args = parser.parse_args()
    
    # 所有10个数据集（包括tsocial）
    all_datasets = [
        'reddit', 'weibo', 'amazon', 'yelp', 'tolokers', 
        'questions', 'tfinance', 'elliptic', 'dgraphfin', 'tsocial'
    ]
    
    # 确定要运行的数据集
    if args.datasets:
        datasets_to_run = args.datasets
    else:
        # 检查数据集文件是否存在
        dataset_dir = '../datasets'
        available_datasets = []
        for dataset in all_datasets:
            if os.path.exists(f'{dataset_dir}/{dataset}'):
                available_datasets.append(dataset)
        
        if not available_datasets:
            print("错误：在 ../datasets 目录下未找到任何数据集文件")
            return
        
        datasets_to_run = available_datasets
        print(f"找到 {len(available_datasets)} 个可用数据集: {available_datasets}")
    
    print(f"将运行 {len(datasets_to_run)} 个数据集（包含所有10个数据集）: {datasets_to_run}")
    print(f"配置: trials={args.trials}, shot={args.shot}, device={args.device}")
    
    # 运行实验
    all_results = []
    total_start_time = time.time()
    
    for i, dataset in enumerate(datasets_to_run, 1):
        print(f"\n进度: {i}/{len(datasets_to_run)}")
        
        # 清理内存
        print("清理内存...")
        clean_memory()
        
        # 运行单个数据集
        result = run_single_dataset(
            dataset_name=dataset,
            trials=args.trials,
            shot=args.shot,
            device=args.device
        )
        
        all_results.append(result)
        
        # 实时保存结果
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        # 再次清理内存
        print("实验完成，清理内存...")
        clean_memory()
        
        # 显示当前进度
        successful_count = len([r for r in all_results if r['status'] == 'success'])
        print(f"当前进度: {successful_count}/{len(all_results)} 成功")
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # 统计结果
    successful = sum(1 for r in all_results if r['status'] == 'success')
    failed = len(all_results) - successful
    
    print(f"\n{'='*60}")
    print("实验完成总结")
    print(f"{'='*60}")
    print(f"总数据集数: {len(datasets_to_run)}")
    print(f"成功完成: {successful}")
    print(f"失败: {failed}")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每个数据集: {total_time/len(datasets_to_run):.2f} 秒")
    
    if successful > 0:
        print(f"\n成功的数据集结果:")
        for result in all_results:
            if result['status'] == 'success':
                dataset = result['dataset']
                for test_name, metrics in result['results'].items():
                    print(f"{dataset}: AUROC={metrics['AUROC_mean']:.4f}±{metrics['AUROC_std']:.4f}, "
                          f"AUPRC={metrics['AUPRC_mean']:.4f}±{metrics['AUPRC_std']:.4f}, "
                          f"Recall@K={metrics['Recall@K_mean']:.4f}±{metrics['Recall@K_std']:.4f}")
    
    if failed > 0:
        print(f"\n失败的数据集:")
        for result in all_results:
            if result['status'] == 'failed':
                print(f"{result['dataset']}: {result['error']}")
    
    print(f"\n结果已保存到: {args.output}")

if __name__ == "__main__":
    main() 