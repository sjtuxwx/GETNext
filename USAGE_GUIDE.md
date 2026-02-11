# 深度融合 RoTAN 时间旋转到 GETNext - 使用指南

## 快速开始

### 1. 环境检查

确保已安装所需依赖:
```bash
cd /data/xwx/code/GETNext
pip install -r requirements.txt
```

### 2. 测试安装

运行维度测试确保所有组件正常工作:
```bash
python test_dimensions.py
```

预期输出:
```
============================================================
✓ 所有测试通过! 维度匹配正确!
============================================================
```

### 3. 训练模型

使用 NYC 数据集训练:
```bash
python train.py \
    --data-train dataset/NYC/NYC_train.csv \
    --data-val dataset/NYC/NYC_val.csv \
    --data-adj-mtx dataset/NYC/graph_A.csv \
    --data-node-feats dataset/NYC/graph_X.csv \
    --epochs 200 \
    --batch 20 \
    --lr 0.001 \
    --time-embed-dim 128 \
    --transformer-nlayers 2 \
    --transformer-nhead 2 \
    --name temporal_rotary_exp
```

## 关键参数说明

### 新增/修改的参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--time-embed-dim` | 128 | 时间嵌入维度 (从32增大到128) |
| `--temporal-rotation-theta` | 10000.0 | Temporal Rotary Encoding 基础频率 |
| `--target-time-embed-dim` | 128 | 目标时间嵌入维度 |

### 维度相关参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--user-embed-dim` | 128 | 用户嵌入维度 |
| `--poi-embed-dim` | 128 | POI 嵌入维度 |
| `--cat-embed-dim` | 32 | 类别嵌入维度 |
| `seq_input_embed` | 288 | 序列输入维度 (自动计算: 128+128+32) |

### Transformer 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--transformer-nlayers` | 2 | Temporal Rotary Encoder 层数 |
| `--transformer-nhead` | 2 | 多头注意力头数 |
| `--transformer-nhid` | 1024 | FFN 隐藏层维度 |
| `--transformer-dropout` | 0.3 | Dropout 率 |

## 模型对比

### 原始 GETNext vs. Temporal Rotary GETNext

| 特性 | 原始 GETNext | Temporal Rotary GETNext |
|------|-------------|------------------------|
| 时间注入方式 | Time+Cat 拼接 | 门控时间旋转 |
| 注意力机制 | 标准 Transformer | Temporal Rotary Attention |
| 解码器 | Linear | Target-Time Cross-Attention |
| 时间感知 | 仅输入层 | 全链路 (输入+注意力+解码) |
| 序列输入维度 | 320d | 288d |
| 参数量 | 较少 | 稍多 (门控网络+交叉注意力) |

## 训练技巧

### 1. 学习率调整

如果损失不下降，尝试降低学习率:
```bash
--lr 0.0005  # 或 0.0001
```

### 2. 批量大小

根据 GPU 内存调整:
```bash
--batch 16  # 如果 OOM，降低到 16 或 12
```

### 3. 时间嵌入维度

可以尝试不同的时间嵌入维度:
```bash
--time-embed-dim 64   # 更轻量
--time-embed-dim 256  # 更强表达力
```

### 4. Temporal Rotation Theta

调整旋转频率基底:
```bash
--temporal-rotation-theta 5000.0   # 更短周期
--temporal-rotation-theta 20000.0  # 更长周期
```

## 评估指标

训练过程中会输出以下指标:

### POI 预测
- `train_top1_acc`: Top-1 准确率
- `train_top5_acc`: Top-5 准确率
- `train_top10_acc`: Top-10 准确率
- `train_top20_acc`: Top-20 准确率
- `train_mAP20`: Mean Average Precision @ 20
- `train_mrr`: Mean Reciprocal Rank

### 辅助任务
- `train_time_loss`: 时间预测损失
- `train_cat_loss`: 类别预测损失

## 输出文件

训练完成后，结果保存在 `runs/train/temporal_rotary_exp/`:

```
runs/train/temporal_rotary_exp/
├── args.yaml                    # 训练参数
├── code.zip                     # 代码快照
├── log_training.txt             # 训练日志
├── metrics-train.txt            # 训练指标
├── metrics-val.txt              # 验证指标
├── checkpoints/
│   ├── best_epoch.state.pt      # 最佳模型
│   └── best_epoch.txt           # 最佳 epoch 指标
└── embeddings/                  # (如果 --save-embeds)
    ├── saved_poi_embeddings.npy
    ├── saved_user_embeddings.npy
    ├── saved_cat_embeddings.npy
    └── saved_time_embeddings.npy
```

## 推理示例

加载训练好的模型进行推理:

```python
import torch
from model import TemporalRotaryTransformerModel

# 加载模型
checkpoint = torch.load('runs/train/temporal_rotary_exp/checkpoints/best_epoch.state.pt')
args = checkpoint['args']

model = TemporalRotaryTransformerModel(
    num_poi=args.num_poi,
    num_cat=args.num_cat,
    embed_size=args.seq_input_embed,
    nhead=args.transformer_nhead,
    nhid=args.transformer_nhid,
    nlayers=args.transformer_nlayers,
    time_embed_dim=args.time_embed_dim,
    device=args.device,
    dropout=args.transformer_dropout
)
model.load_state_dict(checkpoint['seq_model_state_dict'])
model.eval()

# 推理
with torch.no_grad():
    out_poi, out_time, out_cat = model(
        src, src_mask, seq_times, target_times=None  # 推理模式
    )
```

## 消融实验

### 1. 移除门控旋转 (固定旋转)

在 `GatedTemporalRotation.forward()` 中修改:
```python
# return gate * rotated + (1 - gate) * embed
return rotated  # 完全旋转，无门控
```

### 2. 移除 Temporal Rotary Attention (标准 Transformer)

在 `train.py` 中使用原始 `TransformerModel`:
```python
seq_model = TransformerModel(...)  # 替代 TemporalRotaryTransformerModel
```

### 3. 移除 Cross-Attention (简单解码)

在 `TemporalRotaryTransformerModel.forward()` 中修改:
```python
# out_poi = self.poi_decoder(x, target_t_proj, src_mask)
out_poi = nn.Linear(embed_size, num_poi)(x)  # 简单线性解码
```

## 故障排除

### 1. 维度不匹配错误

运行测试脚本诊断:
```bash
python test_dimensions.py
```

### 2. OOM (Out of Memory)

- 降低批量大小: `--batch 12`
- 降低序列长度: 修改数据预处理
- 减少 Transformer 层数: `--transformer-nlayers 1`

### 3. 损失 NaN

- 降低学习率: `--lr 0.0001`
- 添加梯度裁剪 (在 `train.py` 中):
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```

### 4. 训练不收敛

- 检查数据归一化
- 尝试预热学习率
- 增加训练轮数: `--epochs 300`

## 性能基准

### 预期性能 (NYC 数据集)

| 指标 | 原始 GETNext | Temporal Rotary GETNext |
|------|-------------|------------------------|
| Top-1 Acc | ~0.15 | ~0.18 (预期提升) |
| Top-5 Acc | ~0.30 | ~0.35 (预期提升) |
| Top-10 Acc | ~0.40 | ~0.45 (预期提升) |
| mAP@20 | ~0.25 | ~0.28 (预期提升) |

*注: 实际性能取决于超参数调优*

## 可视化分析

### 1. 门控值分布

```python
# 在训练过程中记录门控值
gates = model.gated_rotation.gate_net(...)
# 绘制不同 POI 类型的门控值分布
```

### 2. 注意力热力图

```python
# 提取注意力权重
attn_weights = model.layers[0].forward(..., return_attention=True)
# 可视化时间距离 vs 注意力权重
```

### 3. 交叉注意力可视化

```python
# 提取交叉注意力权重
_, cross_attn = model.poi_decoder.cross_attn(...)
# 可视化目标时间对历史签到的关注度
```

## 引用

如果使用本实现，请引用:

```bibtex
@article{getnext,
  title={GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation},
  journal={NeurIPS},
  year={2022}
}

@inproceedings{rotan,
  title={RoTAN: Rotation-based Time-aware Attention Network for Next POI Recommendation},
  booktitle={CIKM},
  year={2023}
}
```

## 技术支持

遇到问题？
1. 查看 `IMPLEMENTATION_SUMMARY.md` 了解实现细节
2. 运行 `test_dimensions.py` 诊断维度问题
3. 检查 `log_training.txt` 查看详细训练日志
