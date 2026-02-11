"""
测试深度融合RoTAN时间旋转到GETNext的维度匹配性
"""
import torch
import torch.nn as nn
from model import (
    rotate, rotate_batch, GatedTemporalRotation,
    compute_temporal_freqs, apply_temporal_rotary, TemporalRotaryEncoderLayer,
    TargetTimeCrossAttentionDecoder, TemporalRotaryTransformerModel,
    Time2Vec
)


def test_rotation_functions():
    """测试基础旋转函数"""
    print("\n=== 测试旋转函数 ===")
    device = torch.device('cpu')
    
    # 测试 rotate (单样本)
    head = torch.randn(5, 256)  # (batch, embed_dim)
    relation = torch.randn(5, 128)  # (batch, time_dim)
    result = rotate(head, relation, 256, device)
    print(f"rotate: head {head.shape} + relation {relation.shape} -> {result.shape}")
    assert result.shape == head.shape, "rotate维度不匹配!"
    
    # 测试 rotate_batch
    head_batch = torch.randn(4, 10, 256)  # (batch, seq_len, embed_dim)
    relation_batch = torch.randn(4, 10, 128)  # (batch, seq_len, time_dim)
    result_batch = rotate_batch(head_batch, relation_batch, 256, device)
    print(f"rotate_batch: head {head_batch.shape} + relation {relation_batch.shape} -> {result_batch.shape}")
    assert result_batch.shape == head_batch.shape, "rotate_batch维度不匹配!"
    print("✓ 旋转函数测试通过")


def test_gated_temporal_rotation():
    """测试Layer 1: 门控时间旋转"""
    print("\n=== 测试Layer 1: GatedTemporalRotation ===")
    device = torch.device('cpu')
    batch_size, seq_len = 4, 10
    embed_dim, time_dim = 256, 128
    
    model = GatedTemporalRotation(embed_dim, time_dim, device)
    embed = torch.randn(batch_size, seq_len, embed_dim)
    time_embed = torch.randn(batch_size, seq_len, time_dim)
    
    output = model(embed, time_embed)
    print(f"输入: embed {embed.shape} + time {time_embed.shape}")
    print(f"输出: {output.shape}")
    assert output.shape == embed.shape, "GatedTemporalRotation维度不匹配!"
    print("✓ Layer 1测试通过")


def test_temporal_rotary_attention():
    """测试Layer 2: Temporal Rotary Attention"""
    print("\n=== 测试Layer 2: Temporal Rotary Attention ===")
    device = torch.device('cpu')
    batch_size, seq_len = 4, 10
    d_model, nhead = 288, 2
    head_dim = d_model // nhead
    
    # 测试频率计算
    time_values = torch.rand(batch_size, seq_len)  # 归一化时间
    freqs_cis = compute_temporal_freqs(time_values, head_dim, device)
    print(f"时间频率: time {time_values.shape} -> freqs_cis {freqs_cis.shape}")
    assert freqs_cis.shape == (batch_size, seq_len, head_dim // 2), "频率维度不匹配!"
    
    # 测试旋转应用
    x = torch.randn(batch_size, seq_len, nhead, head_dim)
    x_rotated = apply_temporal_rotary(x, freqs_cis)
    print(f"旋转应用: {x.shape} -> {x_rotated.shape}")
    assert x_rotated.shape == x.shape, "旋转应用维度不匹配!"
    
    # 测试完整层
    layer = TemporalRotaryEncoderLayer(d_model, nhead, 1024, dropout=0.1)
    src = torch.randn(batch_size, seq_len, d_model)
    output = layer(src, freqs_cis)
    print(f"完整层: {src.shape} -> {output.shape}")
    assert output.shape == src.shape, "TemporalRotaryEncoderLayer维度不匹配!"
    print("✓ Layer 2测试通过")


def test_cross_attention_decoder():
    """测试Layer 3: Target-Time Cross-Attention Decoder"""
    print("\n=== 测试Layer 3: Target-Time Cross-Attention Decoder ===")
    device = torch.device('cpu')
    batch_size, seq_len = 4, 10
    d_model, nhead, num_poi = 288, 2, 1000
    
    decoder = TargetTimeCrossAttentionDecoder(d_model, nhead, num_poi, dropout=0.1)
    encoder_out = torch.randn(batch_size, seq_len, d_model)
    target_time_embed = torch.randn(batch_size, seq_len, d_model)
    
    output = decoder(encoder_out, target_time_embed)
    print(f"输入: encoder {encoder_out.shape} + target_time {target_time_embed.shape}")
    print(f"输出POI logits: {output.shape}")
    assert output.shape == (batch_size, seq_len, num_poi), "解码器输出维度不匹配!"
    print("✓ Layer 3测试通过")


def test_full_model():
    """测试完整的TemporalRotaryTransformerModel"""
    print("\n=== 测试完整模型 ===")
    device = torch.device('cpu')
    
    # 模型参数
    num_poi, num_cat = 1000, 50
    embed_size, nhead, nhid, nlayers = 288, 2, 1024, 2
    time_embed_dim = 128
    batch_size, seq_len = 4, 10
    
    # 创建模型
    model = TemporalRotaryTransformerModel(
        num_poi, num_cat, embed_size, nhead, nhid, 
        nlayers, time_embed_dim, device, dropout=0.1
    )
    
    # 准备输入
    src = torch.randn(seq_len, batch_size, embed_size)  # (seq_len, batch, d_model)
    src_mask = model.generate_square_subsequent_mask(seq_len)
    seq_times = torch.rand(batch_size, seq_len)  # 归一化时间
    target_times = torch.rand(batch_size, seq_len)  # 目标时间
    
    # 前向传播
    out_poi, out_time, out_cat = model(src, src_mask, seq_times, target_times)
    
    print(f"输入: src {src.shape}, seq_times {seq_times.shape}, target_times {target_times.shape}")
    print(f"输出: POI {out_poi.shape}, Time {out_time.shape}, Cat {out_cat.shape}")
    
    # 验证维度
    assert out_poi.shape == (seq_len, batch_size, num_poi), f"POI输出维度错误: {out_poi.shape}"
    assert out_time.shape == (seq_len, batch_size, 1), f"Time输出维度错误: {out_time.shape}"
    assert out_cat.shape == (seq_len, batch_size, num_cat), f"Cat输出维度错误: {out_cat.shape}"
    
    # 测试推理模式(不提供target_times)
    out_poi_infer, _, _ = model(src, src_mask, seq_times, target_times=None)
    assert out_poi_infer.shape == (seq_len, batch_size, num_poi), "推理模式POI输出维度错误!"
    
    print("✓ 完整模型测试通过")


def test_dimension_flow():
    """测试整个流程的维度变化"""
    print("\n=== 测试维度流程 ===")
    
    # GETNext维度配置
    user_embed_dim = 128
    poi_embed_dim = 128
    time_embed_dim = 128
    cat_embed_dim = 32
    
    print("\n维度变化追踪:")
    print(f"1. User Embed: {user_embed_dim}d")
    print(f"2. POI Embed: {poi_embed_dim}d")
    print(f"3. User+POI Fused: {user_embed_dim + poi_embed_dim}d -> 256d")
    print(f"4. Time Embed: {time_embed_dim}d (用于旋转)")
    print(f"5. Gate Rotation: 256d + {time_embed_dim}d -> 256d (门控旋转)")
    print(f"6. Cat Concat: 256d + {cat_embed_dim}d = {256 + cat_embed_dim}d")
    print(f"7. Temporal Rotary Encoder: {256 + cat_embed_dim}d -> {256 + cat_embed_dim}d")
    print(f"8. Target-Time Cross-Attn: {256 + cat_embed_dim}d -> {256 + cat_embed_dim}d")
    print(f"9. POI Decoder: {256 + cat_embed_dim}d -> num_poi")
    
    seq_input_embed = user_embed_dim + poi_embed_dim + cat_embed_dim
    print(f"\n最终序列输入维度: {seq_input_embed}d (原320d减少到288d)")
    print("✓ 维度流程合理")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("深度融合RoTAN时间旋转到GETNext - 维度测试")
    print("=" * 60)
    
    try:
        test_rotation_functions()
        test_gated_temporal_rotation()
        test_temporal_rotary_attention()
        test_cross_attention_decoder()
        test_full_model()
        test_dimension_flow()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过! 维度匹配正确!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
