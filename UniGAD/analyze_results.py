#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniGAD模型结果分析和可视化脚本
筛选每个数据集上NODE和EDGE级别的最佳AUROC结果
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_results():
    """加载所有实验结果"""
    results_dir = './results'
    excel_files = glob.glob(os.path.join(results_dir, '*.xlsx'))
    
    all_results = []
    
    for file_path in excel_files:
        try:
            # 从文件名提取数据集名称
            filename = os.path.basename(file_path)
            if 'dataset_' in filename:
                dataset_name = filename.split('dataset_')[1].split('.')[0]
            else:
                dataset_name = 'unknown'
            
            # 读取Excel文件
            df = pd.read_excel(file_path)
            
            # 解析结果
            for col in df.columns:
                if col == 'Unnamed: 0':
                    continue
                
                # 提取模型名称
                model_name = df.loc[df['Unnamed: 0'] == 'model_name', col].iloc[0] if len(df.loc[df['Unnamed: 0'] == 'model_name', col]) > 0 else 'unknown'
                
                # 提取运行时间
                time_cost = df.loc[df['Unnamed: 0'] == 'time cost', col].iloc[0] if len(df.loc[df['Unnamed: 0'] == 'time cost', col]) > 0 else 0
                
                # 提取Node级别的指标
                node_auroc = df.loc[df['Unnamed: 0'] == 'AUROC Node mean', col].iloc[0] if len(df.loc[df['Unnamed: 0'] == 'AUROC Node mean', col]) > 0 else 0
                node_auprc = df.loc[df['Unnamed: 0'] == 'AUPRC Node mean', col].iloc[0] if len(df.loc[df['Unnamed: 0'] == 'AUPRC Node mean', col]) > 0 else 0
                node_f1 = df.loc[df['Unnamed: 0'] == 'MacroF1 Node mean', col].iloc[0] if len(df.loc[df['Unnamed: 0'] == 'MacroF1 Node mean', col]) > 0 else 0
                
                # 提取Edge级别的指标
                edge_auroc = df.loc[df['Unnamed: 0'] == 'AUROC Edge mean', col].iloc[0] if len(df.loc[df['Unnamed: 0'] == 'AUROC Edge mean', col]) > 0 else 0
                edge_auprc = df.loc[df['Unnamed: 0'] == 'AUPRC Edge mean', col].iloc[0] if len(df.loc[df['Unnamed: 0'] == 'AUPRC Edge mean', col]) > 0 else 0
                edge_f1 = df.loc[df['Unnamed: 0'] == 'MacroF1 Edge mean', col].iloc[0] if len(df.loc[df['Unnamed: 0'] == 'MacroF1 Edge mean', col]) > 0 else 0
                
                result = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'time': time_cost,
                    'node_auroc': node_auroc,
                    'node_auprc': node_auprc,
                    'node_f1': node_f1,
                    'edge_auroc': edge_auroc,
                    'edge_auprc': edge_auprc,
                    'edge_f1': edge_f1
                }
                all_results.append(result)
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue
    
    return all_results

def create_best_results_table(data):
    """创建最佳结果表格，每个数据集只保留NODE和EDGE的最佳AUROC"""
    if not data:
        print("没有找到实验结果")
        return None
    
    # 按数据集分组，找出每个数据集的最佳结果
    best_results = []
    
    # 获取所有唯一的数据集
    datasets = list(set([result['dataset'] for result in data]))
    
    for dataset in datasets:
        dataset_results = [r for r in data if r['dataset'] == dataset]
        
        if not dataset_results:
            continue
        
        # 找出Node级别最佳AUROC
        best_node = max(dataset_results, key=lambda x: x['node_auroc'])
        
        # 找出Edge级别最佳AUROC
        best_edge = max(dataset_results, key=lambda x: x['edge_auroc'])
        
        # 创建结果行
        row = {
            '数据集': dataset,
            'Node最佳AUROC': best_node['node_auroc'],
            'Node最佳模型': best_node['model'],
            'Node最佳AUPRC': best_node['node_auprc'],
            'Node最佳F1': best_node['node_f1'],
            'Edge最佳AUROC': best_edge['edge_auroc'],
            'Edge最佳模型': best_edge['model'],
            'Edge最佳AUPRC': best_edge['edge_auprc'],
            'Edge最佳F1': best_edge['edge_f1'],
            '运行时间(秒)': best_node['time']  # 使用Node结果的时间
        }
        best_results.append(row)
    
    df = pd.DataFrame(best_results)
    return df

def plot_metrics_comparison(df):
    """绘制指标对比图"""
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    # 设置图形大小
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('UniGAD模型在基准数据集上的最佳性能对比', fontsize=16, fontweight='bold')
    
    # 1. Node AUROC对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df['数据集'], df['Node最佳AUROC'], color='skyblue', alpha=0.8)
    ax1.set_title('Node最佳AUROC对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Node AUROC')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars1, df['Node最佳AUROC']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Edge AUROC对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df['数据集'], df['Edge最佳AUROC'], color='lightcoral', alpha=0.8)
    ax2.set_title('Edge最佳AUROC对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Edge AUROC')
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars2, df['Edge最佳AUROC']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Node vs Edge AUROC对比
    ax3 = axes[1, 0]
    x = np.arange(len(df['数据集']))
    width = 0.35
    
    bars3_1 = ax3.bar(x - width/2, df['Node最佳AUROC'], width, label='Node', color='skyblue', alpha=0.8)
    bars3_2 = ax3.bar(x + width/2, df['Edge最佳AUROC'], width, label='Edge', color='lightcoral', alpha=0.8)
    
    ax3.set_title('Node vs Edge AUROC对比', fontsize=14, fontweight='bold')
    ax3.set_ylabel('AUROC')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df['数据集'], rotation=45)
    ax3.legend()
    
    # 4. 运行时间对比
    ax4 = axes[1, 1]
    bars4 = ax4.bar(df['数据集'], df['运行时间(秒)'], color='gold', alpha=0.8)
    ax4.set_title('运行时间对比', fontsize=14, fontweight='bold')
    ax4.set_ylabel('运行时间(秒)')
    ax4.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars4, df['运行时间(秒)']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('UniGAD_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_heatmap(df):
    """绘制热力图"""
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    # 准备热力图数据
    metrics_df = df[['Node最佳AUROC', 'Node最佳AUPRC', 'Node最佳F1', 
                     'Edge最佳AUROC', 'Edge最佳AUPRC', 'Edge最佳F1']].copy()
    metrics_df.index = df['数据集']
    
    # 创建热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(metrics_df.T, annot=True, cmap='YlOrRd', fmt='.3f', 
                cbar_kws={'label': '性能指标值'})
    plt.title('UniGAD模型性能指标热力图', fontsize=16, fontweight='bold')
    plt.xlabel('数据集', fontsize=12)
    plt.ylabel('评估指标', fontsize=12)
    plt.tight_layout()
    plt.savefig('UniGAD_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_radar_chart(df):
    """绘制雷达图"""
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    # 准备雷达图数据
    metrics_df = df[['Node最佳AUROC', 'Node最佳AUPRC', 'Node最佳F1', 
                     'Edge最佳AUROC', 'Edge最佳AUPRC', 'Edge最佳F1']].copy()
    metrics_df.index = df['数据集']
    
    # 标准化数据到0-1范围
    metrics_normalized = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())
    
    # 设置雷达图
    categories = ['Node AUROC', 'Node AUPRC', 'Node F1', 'Edge AUROC', 'Edge AUPRC', 'Edge F1']
    N = len(categories)
    
    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合图形
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # 为每个数据集绘制雷达图
    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_normalized)))
    
    for i, (dataset, row) in enumerate(metrics_normalized.iterrows()):
        values = row.values.tolist()
        values += values[:1]  # 闭合图形
        
        ax.plot(angles, values, 'o-', linewidth=2, label=dataset, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('UniGAD模型在各数据集上的性能雷达图', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('UniGAD_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(df):
    """创建汇总表格"""
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    # 计算统计信息
    summary_data = []
    
    # 数值型列
    numeric_cols = ['Node最佳AUROC', 'Node最佳AUPRC', 'Node最佳F1', 
                   'Edge最佳AUROC', 'Edge最佳AUPRC', 'Edge最佳F1', '运行时间(秒)']
    
    for col in numeric_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce')
            values = values.dropna()
            
            if len(values) > 0:
                summary_data.append({
                    '指标': col,
                    '平均值': f'{values.mean():.3f}',
                    '标准差': f'{values.std():.3f}',
                    '最大值': f'{values.max():.3f}',
                    '最小值': f'{values.min():.3f}',
                    '中位数': f'{values.median():.3f}'
                })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def print_detailed_results(df):
    """打印详细结果"""
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    print("\n" + "="*80)
    print("UniGAD模型实验结果详细报告")
    print("="*80)
    print(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"成功运行的数据集数量: {len(df)}")
    print("\n各数据集最佳性能:")
    print("-"*80)
    
    for _, row in df.iterrows():
        print(f"\n数据集: {row['数据集']}")
        print(f"  Node级别 - AUROC: {row['Node最佳AUROC']:.3f} (模型: {row['Node最佳模型']})")
        print(f"  Node级别 - AUPRC: {row['Node最佳AUPRC']:.3f}")
        print(f"  Node级别 - F1: {row['Node最佳F1']:.3f}")
        print(f"  Edge级别 - AUROC: {row['Edge最佳AUROC']:.3f} (模型: {row['Edge最佳模型']})")
        print(f"  Edge级别 - AUPRC: {row['Edge最佳AUPRC']:.3f}")
        print(f"  Edge级别 - F1: {row['Edge最佳F1']:.3f}")
        print(f"  运行时间: {row['运行时间(秒)']:.1f}秒")
    
    # 打印统计信息
    summary_df = create_summary_table(df)
    if summary_df is not None and not summary_df.empty:
        print("\n" + "="*80)
        print("统计信息汇总:")
        print("="*80)
        print(summary_df.to_string(index=False))
    
    # 找出最佳模型
    print("\n" + "="*80)
    print("最佳模型分析:")
    print("="*80)
    
    # Node级别最佳
    best_node_idx = df['Node最佳AUROC'].idxmax()
    best_node_row = df.loc[best_node_idx]
    print(f"Node级别最佳性能:")
    print(f"  数据集: {best_node_row['数据集']}")
    print(f"  AUROC: {best_node_row['Node最佳AUROC']:.3f}")
    print(f"  模型: {best_node_row['Node最佳模型']}")
    
    # Edge级别最佳
    best_edge_idx = df['Edge最佳AUROC'].idxmax()
    best_edge_row = df.loc[best_edge_idx]
    print(f"\nEdge级别最佳性能:")
    print(f"  数据集: {best_edge_row['数据集']}")
    print(f"  AUROC: {best_edge_row['Edge最佳AUROC']:.3f}")
    print(f"  模型: {best_edge_row['Edge最佳模型']}")

def export_results(df):
    """导出结果到CSV和Excel"""
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    # 导出到CSV
    csv_filename = 'UniGAD_results_table.csv'
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    print(f"\n结果已导出到: {csv_filename}")
    
    # 导出到Excel
    excel_filename = 'UniGAD_results_table.xlsx'
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='最佳结果', index=False)
        
        # 添加统计信息
        summary_df = create_summary_table(df)
        if summary_df is not None and not summary_df.empty:
            summary_df.to_excel(writer, sheet_name='统计信息', index=False)
    
    print(f"结果已导出到: {excel_filename}")

def main():
    """主函数"""
    print("开始分析UniGAD实验结果...")
    
    # 加载结果
    data = load_results()
    if not data:
        print("未找到实验结果文件")
        return
    
    print(f"找到 {len(data)} 个实验结果")
    
    # 创建最佳结果表格
    df = create_best_results_table(data)
    if df is None or df.empty:
        print("无法创建结果表格")
        return
    
    print(f"成功处理 {len(df)} 个数据集")
    
    # 打印详细结果
    print_detailed_results(df)
    
    # 绘制图表
    print("\n正在生成可视化图表...")
    plot_metrics_comparison(df)
    plot_heatmap(df)
    plot_radar_chart(df)
    
    # 导出结果
    export_results(df)
    
    print("\n分析完成！")

if __name__ == "__main__":
    main() 