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
        with open('detailed_report.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print("未找到detailed_report.json文件")
        return None

def create_results_table(data):
    """创建结果表格"""
    if not data or 'results' not in data:
        print("数据格式错误")
        return None
    
    results = []
    for result in data['results']:
        if result['status'] == 'success' and 'metrics' in result:
            row = {
                '数据集': result['dataset'],
                '运行时间(秒)': result['time'],
                'Mix_AUC': result['metrics'].get('mix_auc', 'N/A'),
                'Recall@K': result['metrics'].get('recall@k', 'N/A'),
                'AP': result['metrics'].get('ap', 'N/A')
            }
            results.append(row)
    
    df = pd.DataFrame(results)
    return df

def plot_metrics_comparison(df):
    """绘制指标对比图"""
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    # 设置图形大小
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('GADAM模型在9个基准数据集上的性能对比', fontsize=16, fontweight='bold')
    
    # 1. Mix_AUC对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df['数据集'], df['Mix_AUC'], color='skyblue', alpha=0.8)
    ax1.set_title('Mix_AUC对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mix_AUC')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars1, df['Mix_AUC']):
        if value != 'N/A':
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Recall@K对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df['数据集'], df['Recall@K'], color='lightcoral', alpha=0.8)
    ax2.set_title('Recall@K对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Recall@K')
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars2, df['Recall@K']):
        if value != 'N/A':
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. AP对比
    ax3 = axes[1, 0]
    bars3 = ax3.bar(df['数据集'], df['AP'], color='lightgreen', alpha=0.8)
    ax3.set_title('AP对比', fontsize=14, fontweight='bold')
    ax3.set_ylabel('AP')
    ax3.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars3, df['AP']):
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
    plt.savefig('GADAM_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_heatmap(df):
    """绘制热力图"""
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    # 准备热力图数据
    metrics_df = df[['Mix_AUC', 'Recall@K', 'AP']].copy()
    metrics_df.index = df['数据集']
    
    # 将'N/A'替换为NaN
    for col in metrics_df.columns:
        metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
    
    # 创建热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics_df.T, annot=True, cmap='YlOrRd', fmt='.3f', 
                cbar_kws={'label': '性能指标值'})
    plt.title('GADAM模型性能指标热力图', fontsize=16, fontweight='bold')
    plt.xlabel('数据集', fontsize=12)
    plt.ylabel('评估指标', fontsize=12)
    plt.tight_layout()
    plt.savefig('GADAM_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_radar_chart(df):
    """绘制雷达图"""
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    # 准备雷达图数据
    metrics_df = df[['Mix_AUC', 'Recall@K', 'AP']].copy()
    metrics_df.index = df['数据集']
    
    # 将'N/A'替换为NaN
    for col in metrics_df.columns:
        metrics_df[col] = pd.to_numeric(metrics_df[col], errors='coerce')
    
    # 标准化数据到0-1范围
    metrics_normalized = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())
    
    # 设置雷达图
    categories = ['Mix_AUC', 'Recall@K', 'AP']
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
    ax.set_title('GADAM模型在各数据集上的性能雷达图', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('GADAM_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(df):
    """创建汇总表格"""
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    # 计算统计信息
    summary_data = []
    
    # 数值型列
    numeric_cols = ['Mix_AUC', 'Recall@K', 'AP', '运行时间(秒)']
    
    for col in numeric_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce')
            values = values.dropna()
            
            if len(values) > 0:
                summary_data.append({
                    '指标': col,
                    '平均值': f"{values.mean():.3f}",
                    '标准差': f"{values.std():.3f}",
                    '最大值': f"{values.max():.3f}",
                    '最小值': f"{values.min():.3f}",
                    '中位数': f"{values.median():.3f}"
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # 保存汇总表格
    summary_df.to_csv('GADAM_summary_statistics.csv', index=False, encoding='utf-8-sig')
    
    return summary_df

def print_detailed_results(df):
    """打印详细结果"""
    if df is None or df.empty:
        print("没有成功的数据集结果")
        return
    
    print("\n" + "="*80)
    print("GADAM模型在9个基准数据集上的详细结果")
    print("="*80)
    
    # 按Mix_AUC排序
    df_sorted = df.sort_values('Mix_AUC', ascending=False)
    
    print("\n按Mix_AUC排序的结果:")
    print("-" * 80)
    print(f"{'数据集':<15} {'Mix_AUC':<10} {'Recall@K':<10} {'AP':<10} {'运行时间(秒)':<12}")
    print("-" * 80)
    
    for _, row in df_sorted.iterrows():
        print(f"{row['数据集']:<15} {row['Mix_AUC']:<10.3f} {row['Recall@K']:<10.3f} "
              f"{row['AP']:<10.3f} {row['运行时间(秒)']:<12.1f}")
    
    # 计算平均性能
    print("\n" + "="*80)
    print("平均性能统计:")
    print("="*80)
    
    metrics = ['Mix_AUC', 'Recall@K', 'AP']
    for metric in metrics:
        values = pd.to_numeric(df[metric], errors='coerce')
        values = values.dropna()
        if len(values) > 0:
            print(f"{metric}: 平均值={values.mean():.3f}, 标准差={values.std():.3f}")
    
    # 最佳性能数据集
    print("\n" + "="*80)
    print("最佳性能数据集:")
    print("="*80)
    
    best_auc = df.loc[df['Mix_AUC'].idxmax()]
    best_recall = df.loc[df['Recall@K'].idxmax()]
    best_ap = df.loc[df['AP'].idxmax()]
    
    print(f"最佳Mix_AUC: {best_auc['数据集']} ({best_auc['Mix_AUC']:.3f})")
    print(f"最佳Recall@K: {best_recall['数据集']} ({best_recall['Recall@K']:.3f})")
    print(f"最佳AP: {best_ap['数据集']} ({best_ap['AP']:.3f})")

def main():
    """主函数"""
    print("开始分析GADAM实验结果...")
    
    # 加载数据
    data = load_results()
    if data is None:
        return
    
    # 创建结果表格
    df = create_results_table(data)
    if df is None:
        print("没有成功的数据集结果")
        return
    
    print(f"成功加载 {len(df)} 个数据集的结果")
    
    # 保存结果表格
    df.to_csv('GADAM_results_table.csv', index=False, encoding='utf-8-sig')
    print("结果表格已保存至: GADAM_results_table.csv")
    
    # 打印详细结果
    print_detailed_results(df)
    
    # 创建汇总统计
    summary_df = create_summary_table(df)
    if summary_df is not None:
        print("\n汇总统计已保存至: GADAM_summary_statistics.csv")
    
    # 绘制图表
    print("\n正在生成可视化图表...")
    
    # 1. 指标对比图
    plot_metrics_comparison(df)
    print("指标对比图已保存至: GADAM_metrics_comparison.png")
    
    # 2. 热力图
    plot_heatmap(df)
    print("热力图已保存至: GADAM_heatmap.png")
    
    # 3. 雷达图
    plot_radar_chart(df)
    print("雷达图已保存至: GADAM_radar_chart.png")
    
    print("\n分析完成！所有结果和图表已保存。")

if __name__ == "__main__":
    main() 