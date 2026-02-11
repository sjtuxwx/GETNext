# 时空解耦双塔模型使用指南

## 快速开始

### 1. 基础训练命令

使用默认参数训练时空解耦双塔模型：

```bash
python train_disentangled.py \
    --data-train dataset/NYC/NYC/NYC_train.csv \
    --data-val dataset/NYC/NYC/NYC_val.csv \
    --data-adj-mtx dataset/NYC/NYC/graph_A.csv \
    --data-node-feats dataset/NYC/NYC/graph_X.csv \
    --batch 16 \
    --epochs 200 \
    --name disentangled_exp1
```

### 2. 完整参数命令（推荐）

包含所有关键参数的完整命令：

```bash
python train_disentangled.py \
    --data-train dataset/NYC/NYC/NYC_train.csv \
    --data-val dataset/NYC/NYC/NYC_val.csv \
    --data-adj-mtx dataset/NYC/NYC/graph_A.csv \
    --data-node-feats dataset/NYC/NYC/graph_X.csv \
    --time-units 48 \
    --time-feature norm_in_day_time \
    --poi-embed-dim 128 \
    --user-embed-dim 128 \
    --time-embed-dim 32 \
    --cat-embed-dim 32 \
    --node-attn-nhid 128 \
    --dual-tower-hidden-dim 256 \
    --num-fourier-freq 10 \
    --num-pref-heads 4 \
    --num-pref-layers 2 \
    --num-spatial-layers 2 \
    --initial-sigma 1.0 \
    --ortho-loss-weight 0.01 \
    --transformer-dropout 0.1 \
    --batch 16 \
    --epochs 200 \
    --lr 0.001 \
    --name disentangled_exp1
```

### 3. 单行命令（复制粘贴使用）

```bash
python train_disentangled.py --data-train dataset/NYC/NYC/NYC_train.csv --data-val dataset/NYC/NYC/NYC_val.csv --data-adj-mtx dataset/NYC/NYC/graph_A.csv --data-node-feats dataset/NYC/NYC/graph_X.csv --time-units 48 --time-feature norm_in_day_time --poi-embed-dim 128 --user-embed-dim 128 --time-embed-dim 32 --cat-embed-dim 32 --dual-tower-hidden-dim 256 --num-fourier-freq 10 --num-pref-heads 4 --num-pref-layers 2 --num-spatial-layers 2 --initial-sigma 1.0 --ortho-loss-weight 0.01 --batch 16 --epochs 200 --name disentangled_exp1
```

---

## 参数说明

### 原始模型参数（保留）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-train` | - | 训练数据路径 |
| `--data-val` | - | 验证数据路径 |
| `--data-adj-mtx` | - | 图邻接矩阵路径 |
| `--data-node-feats` | - | 图节点特征路径 |
| `--time-units` | 48 | 时间单位（0.5小时 × 48 = 24小时） |
| `--time-feature` | norm_in_day_time | 数据中的时间特征列名 |
| `--poi-embed-dim` | 128 | POI嵌入维度 |
| `--user-embed-dim` | 128 | 用户嵌入维度 |
| `--time-embed-dim` | 32 | 时间嵌入维度 |
| `--cat-embed-dim` | 32 | 类别嵌入维度 |
| `--node-attn-nhid` | 128 | 节点注意力隐藏层维度 |
| `--batch` | 20 | 批次大小 |
| `--epochs` | 200 | 训练轮数 |
| `--lr` | 0.001 | 学习率 |
| `--name` | exp | 实验名称 |

### 双塔模型新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dual-tower-hidden-dim` | 256 | 双塔模型的隐藏层维度 |
| `--num-fourier-freq` | 10 | 傅里叶编码的频率阶数 L |
| `--num-pref-heads` | 4 | 语义偏好塔的注意力头数 |
| `--num-pref-layers` | 2 | 语义偏好塔的层数 |
| `--num-spatial-layers` | 2 | 空间感知塔的层数 |
| `--initial-sigma` | 1.0 | 空间感受野σ的初始值 |
| `--ortho-loss-weight` | 0.01 | 正交约束损失的权重λ |

---

## 与原始模型的对比

### 原始模型命令：
```bash
python train.py \
    --data-train dataset/NYC/NYC_train.csv \
    --data-val dataset/NYC/NYC_val.csv \
    --time-units 48 \
    --time-feature norm_in_day_time \
    --poi-embed-dim 128 \
    --user-embed-dim 128 \
    --time-embed-dim 32 \
    --cat-embed-dim 32 \
    --node-attn-nhid 128 \
    --transformer-nhid 1024 \
    --transformer-nlayers 2 \
    --transformer-nhead 2 \
    --batch 16 \
    --epochs 200 \
    --name exp1
```

### 新模型命令（主要区别）：
```bash
python train_disentangled.py \  # ← 使用新的训练脚本
    --data-train dataset/NYC/NYC/NYC_train.csv \  # ← 注意路径多了一层NYC
    --data-val dataset/NYC/NYC/NYC_val.csv \
    --data-adj-mtx dataset/NYC/NYC/graph_A.csv \  # ← 需要显式指定
    --data-node-feats dataset/NYC/NYC/graph_X.csv \  # ← 需要显式指定
    --dual-tower-hidden-dim 256 \  # ← 新参数：双塔隐藏层维度
    --num-fourier-freq 10 \  # ← 新参数：傅里叶频率
    --num-pref-heads 4 \  # ← 新参数：语义塔注意力头数
    --num-pref-layers 2 \  # ← 新参数：语义塔层数
    --num-spatial-layers 2 \  # ← 新参数：空间塔层数
    --initial-sigma 1.0 \  # ← 新参数：空间感受野初始值
    --ortho-loss-weight 0.01 \  # ← 新参数：正交损失权重
    --batch 16 \
    --epochs 200 \
    --name disentangled_exp1
```

---

## 参数调优建议

### 1. 双塔隐藏层维度 (`--dual-tower-hidden-dim`)

- **小数据集**：128-256
- **中等数据集**：256-512（推荐）
- **大数据集**：512-1024

### 2. 傅里叶频率阶数 (`--num-fourier-freq`)

- **城市级数据**（如NYC）：8-12（推荐10）
- **区域级数据**：6-10
- **全球级数据**：12-16

### 3. 空间感受野初始值 (`--initial-sigma`)

- **密集城市**（如NYC）：0.5-1.0（推荐1.0）
- **郊区/乡村**：1.5-3.0
- **混合区域**：1.0-2.0

提示：σ会在训练过程中自动学习调整

### 4. 正交损失权重 (`--ortho-loss-weight`)

- **强解耦**：0.05-0.1
- **平衡**：0.01-0.05（推荐0.01）
- **弱解耦**：0.001-0.01

### 5. 批次大小 (`--batch`)

- **GPU内存 < 8GB**：8-12
- **GPU内存 8-16GB**：16-24（推荐16）
- **GPU内存 > 16GB**：24-32

---

## 实验示例

### 实验1：基础配置（快速验证）
```bash
python train_disentangled.py \
    --batch 16 \
    --epochs 50 \
    --dual-tower-hidden-dim 128 \
    --name quick_test
```

### 实验2：标准配置（论文复现）
```bash
python train_disentangled.py \
    --batch 16 \
    --epochs 200 \
    --dual-tower-hidden-dim 256 \
    --num-fourier-freq 10 \
    --num-pref-heads 4 \
    --num-pref-layers 2 \
    --num-spatial-layers 2 \
    --initial-sigma 1.0 \
    --ortho-loss-weight 0.01 \
    --name standard_config
```

### 实验3：强解耦配置（消融实验）
```bash
python train_disentangled.py \
    --batch 16 \
    --epochs 200 \
    --dual-tower-hidden-dim 256 \
    --ortho-loss-weight 0.1 \
    --name strong_disentangle
```

### 实验4：大模型配置（高性能）
```bash
python train_disentangled.py \
    --batch 24 \
    --epochs 200 \
    --dual-tower-hidden-dim 512 \
    --num-fourier-freq 12 \
    --num-pref-heads 8 \
    --num-pref-layers 3 \
    --num-spatial-layers 3 \
    --lr 0.0005 \
    --name large_model
```

---

## 输出文件

训练完成后，结果保存在 `runs/train/<name>/` 目录下：

```
runs/train/disentangled_exp1/
├── log_training.txt              # 训练日志
├── args.yaml                     # 参数配置
├── code.zip                      # 代码快照
├── metrics-train.txt             # 训练指标
├── metrics-val.txt               # 验证指标
├── one-hot-encoder.pkl           # 类别编码器
└── checkpoints/
    ├── best_epoch.state.pt       # 最佳模型权重
    └── best_epoch.txt            # 最佳epoch指标
```

---

## 监控训练过程

### 关键指标

训练日志中会显示以下关键信息：

1. **预测性能**：
   - `train_move_top1_acc`：Top-1准确率
   - `train_move_top5_acc`：Top-5准确率
   - `train_move_mAP20`：平均精度
   - `train_move_MRR`：平均倒数排名

2. **损失分解**：
   - `train_move_poi_loss`：POI预测损失
   - `train_move_time_loss`：时间预测损失
   - `train_move_ortho_loss`：正交约束损失

3. **模型状态**：
   - `spatial_sigma_values`：每层的空间感受野σ值
   - `gate_mean`：门控值的平均值（0-1之间）
   - `gate_std`：门控值的标准差

### 理想的训练曲线

- **σ值**：应该从初始值（如1.0）逐渐调整到数据集的最优值
- **门控均值**：应该在0.3-0.7之间，表示两个塔都在起作用
- **正交损失**：应该逐渐下降，表示特征解耦越来越好

---

## 常见问题

### Q1: 训练很慢怎么办？
**A**: 尝试以下方法：
- 减小 `--batch` 大小
- 减小 `--dual-tower-hidden-dim`
- 减少 `--num-pref-layers` 和 `--num-spatial-layers`

### Q2: 内存不足怎么办？
**A**: 
- 减小 `--batch` 到 8 或 12
- 减小 `--dual-tower-hidden-dim` 到 128
- 使用 `--no-cuda` 在CPU上训练（会很慢）

### Q3: 如何判断模型是否收敛？
**A**: 观察以下指标：
- `val_loss` 不再下降
- `val_top1_acc` 不再上升
- `ortho_loss` 稳定在较低值

### Q4: 如何调整正交约束强度？
**A**: 
- 如果两个塔学习到相似的特征，增大 `--ortho-loss-weight`
- 如果预测性能下降严重，减小 `--ortho-loss-weight`
- 推荐范围：0.001 - 0.1

### Q5: σ值一直不变怎么办？
**A**: 
- 检查学习率是否过小
- 尝试不同的 `--initial-sigma` 初始值
- 确保坐标数据已正确归一化

---

## 与原始模型的性能对比

运行以下命令进行对比实验：

```bash
# 原始模型
python train.py --batch 16 --epochs 200 --name baseline

# 新模型
python train_disentangled.py --batch 16 --epochs 200 --name disentangled

# 对比结果
cat runs/train/baseline/metrics-val.txt
cat runs/train/disentangled/metrics-val.txt
```

预期改进：
- Top-1准确率：+2-5%
- Top-5准确率：+3-7%
- mAP@20：+5-10%

---

## 技术支持

如有问题，请参考：
- `ARCHITECTURE_COMPARISON.md` - 详细的架构对比
- `disentangled_model.py` - 模型实现代码
- `train_disentangled.py` - 训练脚本

祝训练顺利！🚀
