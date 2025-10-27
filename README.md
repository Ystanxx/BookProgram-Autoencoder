# Autoencoder NIR光谱降维及特征重构系统

> 基于深度学习的近红外光谱数据无监督降维与特征学习系统

## 项目简介

本项目实现了基于**Autoencoder**的近红外（NIR）光谱降维系统，用于**8种农产品光谱数据的无监督降维和精确特征重构**。

**主要特点**：
- ✅ **无监督学习** - 不依赖标签，自主学习光谱低维表示
- ✅ **高效压缩** - 737维→100维，压缩率7.37倍，节省86.4%存储空间
- ✅ **精确重构** - 从100维潜在特征高质量重构原始737维光谱
- ✅ **多品种支持** - 支持8种农产品（红豆、当归、胡萝卜、大蒜、生姜、人参、大豆、小麦）
- ✅ **产地溯源** - 区分国产vs进口农产品的光谱差异特征
- ✅ **可视化分析** - t-SNE/PCA潜在空间可视化，直观展示降维效果
- ✅ **完整文档** - 详细的中文文档和使用指南

### 模型架构

```
输入层 (737维NIR光谱)
    ↓
编码器 Encoder
    └─ Linear(737 → 100) + SELU激活
    ↓
潜在空间 Latent Space (100维) ← 压缩的低维表示
    ↓
解码器 Decoder
    └─ Linear(100 → 737) + SELU激活
    ↓
输出层 (737维重构光谱)

损失函数: MSE重构误差
优化器: Adam (lr=0.001)
```

## 项目结构

```
自编码器（Autoencoder）用于谱图降维及特征重构/
├── config.py                 # 配置文件
├── train.py                  # 训练脚本
├── dimension_reduction.py    # 降维脚本
├── reconstruct.py            # 重构脚本
├── visualize.py              # 可视化脚本
├── quick_start.py            # 快速演示
├── requirements.txt          # 依赖列表
├── README.md                 # 项目说明（本文件）
├── 环境配置.md               # 环境安装指南
├── 使用指南.md               # 详细使用文档
│
├── models/                   # 模型定义
│   ├── __init__.py
│   └── autoencoder.py        # Autoencoder模型类
│
├── data/                     # 数据处理
│   ├── __init__.py
│   └── data_generator.py     # NIR光谱数据生成器
│
├── utils/                    # 工具函数
│   ├── __init__.py
│   ├── dataset.py            # PyTorch数据集类
│   └── metrics.py            # 评估指标函数
│
├── saved_models/             # 模型文件（训练后生成）
├── results/                  # 结果图表（训练后生成）
└── logs/                     # 日志文件（训练后生成）
```

## 快速开始

### 环境要求

- **Python**: 3.8 - 3.10
- **操作系统**: Windows / Linux / macOS
- **内存**: 建议4GB以上
- **GPU**: 可选（默认CPU模式）

### 安装步骤

**1. 创建环境**

```bash
conda create -n autoencoder_env python=3.9 -y
conda activate autoencoder_env
```

**2. 安装依赖**

```bash
# 使用conda（推荐）
conda install pytorch numpy scikit-learn matplotlib tqdm -y

# 或使用pip
pip install -r requirements.txt
```

**3. 快速测试**

```bash
python quick_start.py
```

### 使用示例

```bash
# 训练模型（约5-10分钟）
python train.py

# 降维：737维 → 100维
python dimension_reduction.py --mode batch

# 重构：100维 → 737维
python reconstruct.py --mode single

# 可视化潜在空间
python visualize.py
```

## 模型性能

| 指标 | 训练集 | 测试集 | 说明 |
|------|--------|--------|------|
| **MSE** | ~0.001 | ~0.002 | 均方误差 |
| **RMSE** | ~0.03 | ~0.04 | 均方根误差 |
| **相关系数** | >0.99 | >0.98 | 重构相关性 |
| **压缩率** | - | 7.37× | 737→100维 |
| **空间节省** | - | 86.4% | 存储优化 |

*基于400个NIR光谱样本的测试结果*

## 详细文档

- 📖 **[环境配置.md](环境配置.md)** - 详细的环境搭建指南
- 📖 **[使用指南.md](使用指南.md)** - 完整的项目使用文档

## 适用场景

- ✅ **数据压缩存储** - 大规模光谱数据库的存储优化
- ✅ **特征提取** - 为下游分类/回归任务提供低维特征
- ✅ **异常检测** - 通过重构误差识别异常样本
- ✅ **可视化分析** - 高维光谱数据的降维可视化
- ✅ **质量控制** - 农产品质量一致性评估
- ✅ **产地溯源** - 国产vs进口产品的光谱差异分析
- ✅ **迁移学习** - 预训练编码器可迁移到其他任务

## 技术栈

- **深度学习框架**: PyTorch 1.10+
- **数值计算**: NumPy 1.21+
- **机器学习**: scikit-learn 1.0+
- **可视化**: Matplotlib 3.4+
- **其他工具**: tqdm（进度条）

## 许可

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，欢迎通过以下方式联系：
- 📧 提交Issue
- 📝 查看详细文档

---

**版本**: v1.0  
**更新日期**: 2025-10-27

