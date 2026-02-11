# 快速启动指南

## 方法1：使用启动脚本（最简单）

```bash
bash run_disentangled.sh
```

## 方法2：单行命令（推荐）

```bash
python train_disentangled.py --data-train dataset/NYC/NYC/NYC_train.csv --data-val dataset/NYC/NYC/NYC_val.csv --data-adj-mtx dataset/NYC/NYC/graph_A.csv --data-node-feats dataset/NYC/NYC/graph_X.csv --time-units 48 --time-feature norm_in_day_time --poi-embed-dim 128 --user-embed-dim 128 --time-embed-dim 32 --cat-embed-dim 32 --dual-tower-hidden-dim 256 --num-fourier-freq 10 --num-pref-heads 4 --num-pref-layers 2 --num-spatial-layers 2 --initial-sigma 1.0 --ortho-loss-weight 0.01 --batch 16 --epochs 200 --name disentangled_exp1
```

## 方法3：多行命令（易读）

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
    --dual-tower-hidden-dim 256 \
    --num-fourier-freq 10 \
    --num-pref-heads 4 \
    --num-pref-layers 2 \
    --num-spatial-layers 2 \
    --initial-sigma 1.0 \
    --ortho-loss-weight 0.01 \
    --batch 16 \
    --epochs 200 \
    --name disentangled_exp1
```

## 核心参数说明

### 必需参数（数据路径）
- `--data-train`: 训练数据
- `--data-val`: 验证数据
- `--data-adj-mtx`: 图邻接矩阵
- `--data-node-feats`: 节点特征

### 双塔模型特有参数
- `--dual-tower-hidden-dim 256`: 双塔隐藏层维度
- `--num-fourier-freq 10`: 傅里叶频率阶数
- `--num-pref-heads 4`: 语义塔注意力头数
- `--num-pref-layers 2`: 语义塔层数
- `--num-spatial-layers 2`: 空间塔层数
- `--initial-sigma 1.0`: 空间感受野初始值
- `--ortho-loss-weight 0.01`: 正交损失权重

### 训练参数
- `--batch 16`: 批次大小
- `--epochs 200`: 训练轮数
- `--name disentangled_exp1`: 实验名称

## 查看结果

```bash
# 查看训练日志
tail -f runs/train/disentangled_exp1/log_training.txt

# 查看验证指标
cat runs/train/disentangled_exp1/metrics-val.txt
```

## 更多信息

详细文档请参考：
- `README_DISENTANGLED_MODEL.md` - 完整使用指南
- `ARCHITECTURE_COMPARISON.md` - 架构对比说明
