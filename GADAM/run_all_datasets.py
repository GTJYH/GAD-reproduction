import subprocess
import os
import json
import time
from datetime import datetime
import re
import gc
import psutil

# 尝试导入torch，如果失败则忽略
try:
    import torch
except ImportError:
    torch = None

def clean_memory():
    """清理内存"""
    gc.collect()
    if torch is not None and hasattr(torch, 'cuda'):
        torch.cuda.empty_cache()
    print(f"内存使用: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")

def extract_final_metrics(output_text):
    """提取最终的评估指标"""
    metrics = {}
    
    lines = output_text.split('\n')
    
    # 查找最终最佳结果部分
    for i, line in enumerate(lines):
        if '最终最佳结果:' in line:
            # 提取接下来的三行指标
            for j in range(1, 4):
                if i + j < len(lines):
                    metric_line = lines[i + j].strip()
                    if 'Mix_AUC:' in metric_line:
                        auc_match = re.search(r'Mix_AUC: ([\d.]+)', metric_line)
                        if auc_match:
                            metrics['mix_auc'] = float(auc_match.group(1))
                    elif 'Recall@K:' in metric_line:
                        recall_match = re.search(r'Recall@K: ([\d.]+)', metric_line)
                        if recall_match:
                            metrics['recall@k'] = float(recall_match.group(1))
                    elif 'AP:' in metric_line:
                        ap_match = re.search(r'AP: ([\d.]+)', metric_line)
                        if ap_match:
                            metrics['ap'] = float(ap_match.group(1))
            break
    
    # 如果没有找到最终结果，尝试从最后几行中提取
    if not metrics:
        for line in reversed(lines[-10:]):  # 从最后10行中查找
            if 'mix_auc' in line and 'recall@k' in line and 'ap' in line:
                # 使用正则表达式提取数值
                auc_match = re.search(r'mix_auc ([\d.]+)', line)
                recall_match = re.search(r'recall@k ([\d.]+)', line)
                ap_match = re.search(r'ap ([\d.]+)', line)
                
                if auc_match:
                    metrics['mix_auc'] = float(auc_match.group(1))
                if recall_match:
                    metrics['recall@k'] = float(recall_match.group(1))
                if ap_match:
                    metrics['ap'] = float(ap_match.group(1))
                break
    
    return metrics

def run_experiment(dataset_name, dataset_mapping=None, local_lr=1e-3, local_epochs=100, global_lr=5e-4, global_epochs=50, gpu=0):
    """运行单个数据集的实验，参数与README.md要求一致"""
    
    # 使用映射后的数据集名称
    actual_dataset_name = dataset_mapping.get(dataset_name, dataset_name) if dataset_mapping else dataset_name
    
    cmd = [
        'python', 'run.py',
        '--data', actual_dataset_name,
        '--local-lr', str(local_lr),
        '--local-epochs', str(local_epochs),
        '--global-lr', str(global_lr),
        '--global-epochs', str(global_epochs),
        '--gpu', str(gpu),
        '--out-dim', '64',
        '--seed', '717'
    ]
    
    print(f"运行数据集: {dataset_name}")
    print(f"命令: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1小时超时
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"数据集 {dataset_name} 运行成功，耗时: {end_time - start_time:.2f}秒")
            
            # 提取指标
            metrics = extract_final_metrics(result.stdout)
            
            return {
                'dataset': dataset_name,
                'status': 'success',
                'time': end_time - start_time,
                'output': result.stdout,
                'metrics': metrics
            }
        else:
            print(f"数据集 {dataset_name} 运行失败")
            print(f"错误信息:")
            print(result.stderr)
            print(f"标准输出:")
            print(result.stdout)
            return {
                'dataset': dataset_name,
                'status': 'failed',
                'error': result.stderr,
                'output': result.stdout,
                'time': end_time - start_time
            }
    except subprocess.TimeoutExpired:
        print(f"数据集 {dataset_name} 运行超时")
        print(f"超时时间: 3600秒")
        return {
            'dataset': dataset_name,
            'status': 'timeout',
            'time': 3600
        }
    except Exception as e:
        print(f"数据集 {dataset_name} 运行异常: {e}")
        print(f"异常详情:")
        import traceback
        traceback.print_exc()
        return {
            'dataset': dataset_name,
            'status': 'exception',
            'error': str(e)
        }

def run_all_datasets():
    """运行所有10个基准数据集"""
    
    # 数据集名称映射：代码中的名称 -> 实际文件名
    dataset_mapping = {
        'YelpChi': 'yelp',
        'T-Finance': 'tfinance', 
        'DGraph-Fin': 'dgraphfin',
        'T-Social': 'tsocial'
    }
    
    # 使用原来的10个数据集
    # 注意：T-Social和DGraph-Fin是超大数据集，可能需要特殊处理
    datasets = [
        'Reddit', 'Weibo', 'Amazon', 'YelpChi', 'Tolokers',
        'Questions', 'T-Finance', 'Elliptic', 'DGraph-Fin', 'T-Social'
    ]
    
    # 可以选择跳过超大数据集
    skip_large_datasets = True  # 设置为True可以跳过T-Social
    if skip_large_datasets:
        datasets = [d for d in datasets if d not in ['T-Social']]
        print("跳过超大数据集: T-Social")
    
    # 实验配置，与README.md中的示例命令一致
    configs = {
        'local_lr': 1e-3,
        'local_epochs': 100,
        'global_lr': 5e-4,
        'global_epochs': 50,
        'gpu': 0  # 可以根据需要调整
    }
    
    results = []
    start_time = time.time()
    
    print("=" * 60)
    print("开始运行GADAM在所有10个基准数据集上的实验")
    print("按照README.md要求：")
    print("python run.py --data Cora --local-lr 1e-3 --local-epochs 100 --global-lr 5e-4 --global-epochs 50")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    for i, dataset in enumerate(datasets):
        print(f"\n{'='*60}")
        print(f"进度: {i+1}/{len(datasets)} - 处理数据集: {dataset}")
        print(f"{'='*60}")
        
        # 清理内存
        print("清理内存...")
        clean_memory()
        
        # 运行实验
        result = run_experiment(dataset, dataset_mapping=dataset_mapping, **configs)
        results.append(result)
        
        # 再次清理内存
        print("实验完成，清理内存...")
        clean_memory()
        
        # 保存中间结果
        with open('experiment_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 显示当前进度
        successful_count = len([r for r in results if r['status'] == 'success'])
        print(f"当前进度: {successful_count}/{len(results)} 成功")
        
        # 如果失败，询问是否继续
        if result['status'] != 'success':
            print(f"数据集 {dataset} 运行失败，是否继续下一个数据集？(y/n)")
            # 这里可以添加用户交互，暂时自动继续
    
    total_time = time.time() - start_time
    
    # 生成实验报告
    generate_report(results, total_time, configs)
    
    return results

def generate_report(results, total_time, configs):
    """生成实验报告"""
    
    print("\n" + "=" * 60)
    print("实验报告")
    print("=" * 60)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']
    
    print(f"总数据集数: {len(results)}")
    print(f"成功运行: {len(successful)}")
    print(f"运行失败: {len(failed)}")
    print(f"总耗时: {total_time/3600:.2f}小时")
    
    print("\n成功运行的数据集:")
    for result in successful:
        print(f"  - {result['dataset']}: {result['time']:.2f}秒")
        if 'metrics' in result:
            metrics = result['metrics']
            print(f"    Mix_AUC: {metrics.get('mix_auc', 'N/A')}")
            print(f"    Recall@K: {metrics.get('recall@k', 'N/A')}")
            print(f"    AP: {metrics.get('ap', 'N/A')}")
    
    if failed:
        print("\n运行失败的数据集:")
        for result in failed:
            print(f"  - {result['dataset']}: {result['status']}")
            if 'error' in result:
                print(f"    错误: {result['error']}")
            if 'output' in result:
                print(f"    输出: {result['output']}")
    
    # 保存详细报告
    report = {
        'summary': {
            'total_datasets': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'total_time': total_time,
            'configs': configs
        },
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('detailed_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细报告已保存至: detailed_report.json")

if __name__ == "__main__":
    run_all_datasets() 