# 深度融合 RoTAN 时间旋转到 GETNext - 实现总结

## 实现概述

成功将 RoTAN 的时间旋转机制深度融合到 GETNext 架构的三个层次，实现了时间信息对整个模型的全链路渗透。

## 修改的文件

### 1. `model.py` - 核心模型架构

#### 新增组件

1. **基础旋转函数** (L9-43)
   - `rotate()`: 单样本时间旋转
   - `rotate_batch()`: 批量时间旋转
   - 来源: RoTAN 的 RotatE 机制

2. **Layer 1: GatedTemporalRotation** (L47-73)
   - 可学习门控的时间旋转模块
   - 自适应决定旋转强度 (gate=1 完全旋转, gate=0 保持原样)
   - 输入: embed (B, S, 256) + time_embed (B, S, 128)
   - 输出: (B, S, 256)

3. **Layer 2: Temporal Rotary Attention** (L255-349)
   - `compute_temporal_freqs()`: 计算时间驱动的旋转频率
   - `apply_temporal_rotary()`: 对 Q/K 施加时间旋转
   - `TemporalRotaryEncoderLayer`: 自定义 Transformer 层
   - 核心创新: 让注意力权重天然反映时间距离

4. **Layer 3: TargetTimeCrossAttentionDecoder** (L408-449)
   - 目标时间交叉注意力解码器
   - 让目标时间主动查询相关历史
   - 门控融合编码器输出和交叉注意力输出

5. **TemporalRotaryTransformerModel** (L455-523)
   - 整合三层深度融合的完整模型
   - 替代原有的 `TransformerModel`
   - 支持训练和推理两种模式

### 2. `train.py` - 训练流程

#### 修改内容

1. **导入新模型** (L21)
   ```python
   from model import ..., TemporalRotaryTransformerModel
   ```

2. **修改 `input_traj_to_embeddings()` 函数** (L278-318)
   - 移除 time+cat 融合 (`embed_fuse_model2`)
   - 只拼接 cat，时间通过门控旋转注入
   - 返回 `(input_seq_embed, seq_times)` 元组
   - 维度: 256 (user+poi) + 32 (cat) = 288d

3. **修改训练循环** (L393-451)
   - 新增 `batch_seq_times` 和 `batch_target_times` 收集
   - Pad 时间序列
   - 前向传播传入 `seq_times` 和 `target_times`

4. **修改验证循环** (L528-574)
   - 同训练循环的修改

5. **移除 `embed_fuse_model2` 相关代码**
   - 模型定义 (L245-246)
   - 优化器参数 (L264)
   - 模型移动到设备 (L337)
   - 训练/评估模式设置 (L368, L510)
   - 模型状态保存 (L757)

6. **更新模型初始化** (L248-256)
   - 使用 `TemporalRotaryTransformerModel`
   - 维度: `seq_input_embed = user + poi + cat = 288d` (原 320d)
   - 传入 `time_embed_dim` 和 `device` 参数

### 3. `param_parser.py` - 参数配置

#### 修改和新增参数

1. **修改 `--time-embed-dim`** (L85-88)
   - 从 32 增大到 128
   - 匹配 user+poi 维度的一半用于旋转

2. **新增 `--temporal-rotation-theta`** (L101-104)
   - 默认值: 10000.0
   - Temporal Rotary Encoding 的基础频率

3. **新增 `--target-time-embed-dim`** (L105-108)
   - 默认值: 128
   - Target-Time Cross-Attention 的目标时间嵌入维度

### 4. `test_dimensions.py` - 维度测试 (新文件)

创建了完整的测试脚本验证所有组件的维度匹配性:
- ✓ 旋转函数测试
- ✓ Layer 1 门控时间旋转测试
- ✓ Layer 2 Temporal Rotary Attention 测试
- ✓ Layer 3 交叉注意力解码器测试
- ✓ 完整模型测试
- ✓ 维度流程追踪

## 维度变化追踪

| 阶段 | 输入 | 输出 | 说明 |
|------|------|------|------|
| User Embed | - | 128d | 用户嵌入 |
| POI Embed (GCN) | X, A | 128d | POI 图嵌入 |
| User+POI Fuse | 128d + 128d | 256d | embed_fuse_model1 |
| Time2Vec | 归一化时间 | 128d | 时间向量编码 |
| Gate Rotation | 256d + 128d | 256d | 可学习门控旋转 |
| Cat Embed | - | 32d | 类别嵌入 |
| Concat | 256d + 32d | 288d | 替代原 320d |
| Temporal Rotary Encoder | 288d | 288d | N 层，Q/K 旋转 |
| Target-Time Cross-Attn | 288d + 128d | 288d | 交叉注意力 + 门控融合 |
| POI Decoder | 288d | num_poi | 最终 POI 预测 |
| Time Decoder | 288d | 1 | 时间预测 |
| Cat Decoder | 288d | num_cat | 类别预测 |

## 关键设计决策

### 1. 为什么使用门控旋转而非固定权重？

RoTAN 使用固定权重 `0.7*hour + 0.3*day`，但不同 POI 对时间的敏感度不同：
- 餐厅: 强时间相关 (早/午/晚餐时段)
- 机场: 时间相关性较弱 (全天候)

门控网络让模型自适应学习每个签到需要多少时间信息。

### 2. 为什么在注意力层施加时间旋转？

**传统 Transformer**: 
- 注意力权重 = `Q_i · K_j` (仅内容相似度)
- 时间信息只在输入/输出注入，注意力计算对时间无感知

**Temporal Rotary Attention**:
- 注意力权重 = `(R(t_i)Q_i) · (R(t_j)K_j) = Q_i · R(t_j-t_i)K_j`
- 时间距离天然编码到注意力计算中
- 时间上接近的签到自动获得更高权重

### 3. 为什么使用 Target-Time Cross-Attention？

简单方案: `rotate_batch(encoder_out, target_time) -> Linear`
- 旋转是均匀的，所有历史签到被同等对待

交叉注意力方案: 目标时间作为 query 主动查询历史
- 根据目标时间选择性聚焦相关历史
- 例如预测"周三下午3点"，自动关注历史中类似时段的签到

## 训练和推理策略

### 训练 (Teacher Forcing)
```python
y_pred_poi, y_pred_time, y_pred_cat = model(
    src, src_mask, seq_times, target_times=label_times  # 使用真实标签
)
```

### 推理
```python
# Step 1: 预测时间
y_pred_poi, y_pred_time, y_pred_cat = model(
    src, src_mask, seq_times, target_times=None
)
# Step 2: 使用预测时间解码 POI (内部自动处理)
```

## 预期效果

1. **更强的时间感知**: 注意力权重天然反映时间距离
2. **自适应时间融合**: 不同 POI 类型学习不同的时间敏感度
3. **目标时间引导**: 根据目标时间选择性聚焦相关历史
4. **全链路时间渗透**: 时间信息从输入到注意力到输出全程参与

## 下一步

1. **训练模型**: 使用 NYC/CA/Tokyo 数据集
2. **消融实验**: 
   - 移除门控 (固定旋转)
   - 移除 Temporal Rotary Attention (标准 Transformer)
   - 移除 Cross-Attention (简单 Linear 解码)
3. **可视化分析**:
   - 门控值分布 (不同 POI 类型)
   - 注意力热力图 (时间距离影响)
   - 交叉注意力权重 (目标时间聚焦)

## 测试结果

运行 `python test_dimensions.py` 输出:

```
============================================================
✓ 所有测试通过! 维度匹配正确!
============================================================
```

所有组件的维度匹配验证通过，模型可以正常训练。
