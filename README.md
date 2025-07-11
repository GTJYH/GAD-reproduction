# 图异常检测模型复现与对比分析项目

本项目对三种主流的图异常检测（GAD）模型——**ARC**、**GADAM** 和 **UniGAD** 进行了全面的复现、分析与对比。

## 📁 项目结构

```
gad/
├── ARC/                          # ARC模型实现
│   ├── main.py                   # 主训练脚本
│   ├── run_all_datasets.py       # 批量运行脚本 ⭐
│   ├── model.py                  # 模型定义
│   ├── analyze_results.py        # 结果分析脚本 ⭐
│   ├── ARC_results.csv           # 实验结果表格
│   ├── ARC_heatmap.png           # 性能热力图
│   ├── ARC_metrics_comparison.png # 指标对比图
│   └── ARC_radar_chart.png       # 雷达图
│
├── GADAM/                        # GADAM模型实现
│   ├── run.py                    # 主训练脚本
│   ├── run_all_datasets.py       # 批量运行脚本 ⭐
│   ├── model.py                  # 模型定义
│   ├── analyze_results.py        # 结果分析脚本 ⭐
│   ├── GADAM_results_table.csv   # 实验结果表格
│   ├── GADAM_heatmap.png         # 性能热力图
│   ├── GADAM_metrics_comparison.png # 指标对比图
│   └── GADAM_radar_chart.png     # 雷达图
│
├── UniGAD/                       # UniGAD模型实现
│   ├── src/
│   │   ├── main.py               # 主训练脚本
│   │   ├── e2e_models.py         # 端到端模型
│   │   └── predictors.py         # 预测器
│   ├── analyze_results.py        # 结果分析脚本 ⭐
│   ├── results/                  # 实验结果目录
│   ├── UniGAD_heatmap.png        # 性能热力图
│   ├── UniGAD_metrics_comparison.png # 指标对比图
│   └── UniGAD_radar_chart.png    # 雷达图
│
├── datasets/                     # 数据集目录
│   ├── reddit/                   # Reddit社交网络
│   ├── weibo/                    # 微博社交网络
│   ├── amazon/                   # 亚马逊评论
│   ├── yelp/                     # Yelp评论
│   ├── tolokers/                 # Tolokers众包平台
│   ├── questions/                # 问答网络
│   ├── tfinance/                 # 金融交易
│   ├── elliptic/                 # 区块链交易
│   ├── dgraphfin/                # 金融图
│   └── tsocial/                  # 社交网络
│
└── 总实验报告.md                 # 完整实验报告 ⭐
```

## 🚀 快速开始

### 环境要求
- Python 3.9
- PyTorch 1.12.0
- CUDA 11.6
- DGL/PyG
- 其他依赖见各模型目录下的 `requirements.txt`

### 运行实验

#### ARC模型
```bash
cd ARC/
python run_all_datasets.py
```

#### GADAM模型
```bash
cd GADAM/
python run_all_datasets.py
```

#### UniGAD模型
```bash
cd UniGAD/src/
python main.py --datasets 0,1,2,3,4,5,6 --pretrain_model graphmae --kernels bwgnn,gcn --lr 5e-4 --save_model --epoch_pretrain 50 --batch_size 1 --khop 1 --epoch_ft 300 --lr_ft 0.003 --final_mlp_layers 3 --cross_modes ne2ne --metric AUROC --trials 1
```

## 📊 实验结果

### 核心性能对比
在共同数据集（Reddit, Weibo, Tolokers）上的节点AUROC对比：

| 模型 | Reddit | Weibo | Tolokers |
|------|--------|-------|----------|
| ARC | 0.593 | 0.801 | 0.432 |
| GADAM | 0.577 | 0.421 | 0.446 |
| UniGAD | **0.664** | **0.979** | **0.807** |

### 结果文件位置
- **ARC结果**: `ARC/ARC_results.csv`, `ARC/ARC_*.png`
- **GADAM结果**: `GADAM/GADAM_results_table.csv`, `GADAM/GADAM_*.png`
- **UniGAD结果**: `UniGAD/results/`, `UniGAD/UniGAD_*.png`

### 可视化图表
每个模型目录下都包含：
- **性能对比图** (`*_metrics_comparison.png`): 展示各数据集上的AUROC、AUPRC等指标
- **热力图** (`*_heatmap.png`): 直观显示模型在不同数据集上的表现
- **雷达图** (`*_radar_chart.png`): 多维度性能评估

## 📈 分析脚本

### 结果分析
每个模型都有对应的分析脚本：
- `ARC/analyze_results.py`
- `GADAM/analyze_results.py`
- `UniGAD/analyze_results.py`

这些脚本会：
1. 自动解析实验结果
2. 筛选最佳性能指标
3. 生成可视化图表
4. 导出汇总表格

### 运行分析
```bash
# 分析ARC结果
cd ARC/
python analyze_results.py

# 分析GADAM结果
cd GADAM/
python analyze_results.py

# 分析UniGAD结果
cd UniGAD/
python analyze_results.py
```

## 📋 实验报告

完整的实验分析请参考：**`实验报告.md`**

该报告包含：
- 三种模型的深度原理分析
- 代码实现与理论的统一性分析
- 复现过程中的技术挑战与解决方案
- 与原论文结果的详细对比
- 跨模型性能分析与洞察

## ⚠️ 注意事项

1. **显存要求**: UniGAD模型显存消耗较大，建议使用12GB+显存
2. **数据集**: 确保所有数据集已正确放置在 `datasets/` 目录下，文件即为钉钉群里所发的文件
3. **环境配置**: 严格按照版本要求配置环境，避免兼容性问题
4. **批量运行**: 建议先在小数据集上测试，确认环境无误后再批量运行

## 🔧 主要修改

### ARC
- 新增 `run_all_datasets.py` 批量运行脚本
- 修复特征维度不匹配问题（SVD降维）
- 修复交叉注意力模块设备不匹配bug

### GADAM  
- 新增 `run_all_datasets.py` 批量运行脚本
- 优化大规模图的内存管理
- 增强数据加载的兼容性

### UniGAD
- 修复GraphMAE掩码索引错误（batch_size=1）
- 实现边异常标签生成策略
- 新增 `analyze_results.py` 结果分析脚本