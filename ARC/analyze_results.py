#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC模型结果分析和可视化脚本
与GADAM保持一致，但适配ARC的三个指标：AUROC、AUPRC、Recall@K
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_results():
    """加载实验结果"""
    try:
        with open('arc_results.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print("未找到arc_results.json文件")
        return None

def create_results_table(data):
    """创建结果表格"""
    if not data:
        print("数据格式错误")
        return None
    
    results = []
    for result in data:
        if result['status'] == 'success' and 'results' in result:
            # 获取第一个测试数据集的结果（ARC通常只有一个测试数据集）
            for test_name, metrics in result['results'].items():
                row = {
                    '数据集': result['dataset'],
                    '运行时间(秒)': result['time'],
                    'AUROC': metrics.get('AUROC_mean', 'N/A'),
                    'AUPRC': metrics.get('AUPRC_mean', 'N/A'),
                    'Recall@K': metrics.get('Recall@K_mean', 'N/A')
                }
                results.append(row)
                break  # 只取第一个结果
    
    df = pd.DataFrame(results)
    return df

def plot_metrics_comparison(df):
    """绘制指标对比图"""
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    # 设置图形大小
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ARC模型在基准数据集上的性能对比', fontsize=16, fontweight='bold')
    
    # 1. AUROC对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df['数据集'], df['AUROC'], color='skyblue', alpha=0.8)
    ax1.set_title('AUROC对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('AUROC')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars1, df['AUROC']):
        if value != 'N/A':
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. AUPRC对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df['数据集'], df['AUPRC'], color='lightcoral', alpha=0.8)
    ax2.set_title('AUPRC对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('AUPRC')
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars2, df['AUPRC']):
        if value != 'N/A':
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Recall@K对比
    ax3 = axes[1, 0]
    bars3 = ax3.bar(df['数据集'], df['Recall@K'], color='lightgreen', alpha=0.8)
    ax3.set_title('Recall@K对比', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Recall@K')
    ax3.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars3, df['Recall@K']):
        if value != 'N/A':
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
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
    plt.savefig('ARC_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_heatmap(df):
    """绘制热力图"""
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    # 准备热力图数据
    metrics_df = df[['AUROC', 'AUPRC', 'Recall@K']].copy()
    metrics_df.index = df['数据集']
    
    # 将'N/A'替换为NaN
    for col in metrics_df.columns:
        metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
    
    # 创建热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics_df.T, annot=True, cmap='YlOrRd', fmt='.3f', 
                cbar_kws={'label': '性能指标值'})
    plt.title('ARC模型性能指标热力图', fontsize=16, fontweight='bold')
    plt.xlabel('数据集', fontsize=12)
    plt.ylabel('评估指标', fontsize=12)
    plt.tight_layout()
    plt.savefig('ARC_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_radar_chart(df):
    """绘制雷达图"""
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    # 准备雷达图数据
    metrics_df = df[['AUROC', 'AUPRC', 'Recall@K']].copy()
    metrics_df.index = df['数据集']
    
    # 将'N/A'替换为NaN
    for col in metrics_df.columns:
        metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
    
    # 标准化数据到0-1范围
    metrics_normalized = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())
    
    # 设置雷达图
    categories = ['AUROC', 'AUPRC', 'Recall@K']
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
    ax.set_title('ARC模型在各数据集上的性能雷达图', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('ARC_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(df):
    """创建汇总表格"""
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    # 计算统计信息
    summary_data = []
    
    # 数值型列
    numeric_cols = ['AUROC', 'AUPRC', 'Recall@K', '运行时间(秒)']
    
    for col in numeric_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce')
            values = values.dropna()
            
            if len(values) > 0:
                summary_data.append({
                    '指标': col,
                    '平均值': f'{values.mean():.4f}',
                    '标准差': f'{values.std():.4f}',
                    '最小值': f'{values.min():.4f}',
                    '最大值': f'{values.max():.4f}',
                    '中位数': f'{values.median():.4f}'
                })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

def print_detailed_results(df):
    """打印详细结果"""
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    print("\n" + "="*80)
    print("ARC模型实验结果详细报告")
    print("="*80)
    
    # 打印基本信息
    print(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"成功完成的数据集数量: {len(df)}")
    print(f"总运行时间: {df['运行时间(秒)'].sum():.2f} 秒")
    print(f"平均运行时间: {df['运行时间(秒)'].mean():.2f} 秒")
    
    # 打印每个数据集的结果
    print("\n各数据集详细结果:")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"数据集: {row['数据集']}")
        print(f"  运行时间: {row['运行时间(秒)']:.2f} 秒")
        print(f"  AUROC: {row['AUROC']:.4f}")
        print(f"  AUPRC: {row['AUPRC']:.4f}")
        print(f"  Recall@K: {row['Recall@K']:.4f}")
        print()
    
    # 打印统计汇总
    summary_df = create_summary_table(df)
    if summary_df is not None and not summary_df.empty:
        print("统计汇总:")
        print("-" * 80)
        print(summary_df.to_string(index=False))
    
    # 找出最佳性能
    print("\n最佳性能:")
    print("-" * 80)
    for metric in ['AUROC', 'AUPRC', 'Recall@K']:
        if metric in df.columns:
            values = pd.to_numeric(df[metric], errors='coerce')
            best_idx = values.idxmax()
            if not pd.isna(best_idx):
                best_dataset = df.loc[best_idx, '数据集']
                best_value = values[best_idx]
                print(f"最佳{metric}: {best_dataset} ({best_value:.4f})")

def export_results(df):
    """导出结果到不同格式"""
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    # 导出到CSV
    df.to_csv('ARC_results.csv', index=False, encoding='utf-8-sig')
    print("结果已导出到 ARC_results.csv")
    
    # 导出到Excel
    try:
        with pd.ExcelWriter('ARC_results.xlsx', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='实验结果', index=False)
            
            # 添加统计汇总表
            summary_df = create_summary_table(df)
            if summary_df is not None and not summary_df.empty:
                summary_df.to_excel(writer, sheet_name='统计汇总', index=False)
        
        print("结果已导出到 ARC_results.xlsx")
    except ImportError:
        print("未安装openpyxl，跳过Excel导出")

def main():
    """主函数"""
    print("开始分析ARC模型实验结果...")
    
    # 加载结果
    data = load_results()
    if data is None:
        return
    
    # 创建结果表格
    df = create_results_table(data)
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    # 打印详细结果
    print_detailed_results(df)
    
    # 创建可视化图表
    print("\n正在生成可视化图表...")
    plot_metrics_comparison(df)
    plot_heatmap(df)
    plot_radar_chart(df)
    
    # 导出结果
    print("\n正在导出结果...")
    export_results(df)
    
    print("\n分析完成！")
    print("生成的文件:")
    print("- ARC_metrics_comparison.png (指标对比图)")
    print("- ARC_heatmap.png (热力图)")
    print("- ARC_radar_chart.png (雷达图)")
    print("- ARC_results.csv (CSV格式结果)")
    print("- ARC_results.xlsx (Excel格式结果)")

if __name__ == "__main__":
    main() 