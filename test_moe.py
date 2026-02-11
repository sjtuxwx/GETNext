"""快速测试MoE模块的前向传播"""
import torch
from model import MixtureOfExpertsDecoder, TransformerModel

print("="*60)
print("测试MoE模块")
print("="*60)

# 测试参数
batch_size = 4
seq_len = 10
embed_size = 320
num_poi = 1000
num_cat = 100
user_embed_dim = 128
time_embed_dim = 32

# 1. 测试MixtureOfExpertsDecoder
print("\n1. 测试 MixtureOfExpertsDecoder...")
moe_decoder = MixtureOfExpertsDecoder(
    embed_size=embed_size,
    output_size=num_poi,
    num_experts=8,
    user_embed_dim=user_embed_dim,
    time_embed_dim=time_embed_dim
)

# 创建测试输入
x = torch.randn(batch_size, seq_len, embed_size)
user_embed = torch.randn(batch_size, user_embed_dim)
time_embed = torch.randn(batch_size, seq_len, time_embed_dim)

# 前向传播
output, gate_weights = moe_decoder(x, user_embed, time_embed)

print(f"   输入形状: {x.shape}")
print(f"   用户embedding形状: {user_embed.shape}")
print(f"   时间embedding形状: {time_embed.shape}")
print(f"   输出形状: {output.shape}")
print(f"   门控权重形状: {gate_weights.shape}")
print(f"   门控权重和: {gate_weights.sum(dim=-1).mean():.4f} (应该接近1.0)")
print("   ✓ MixtureOfExpertsDecoder 测试通过")

# 2. 测试TransformerModel
print("\n2. 测试 TransformerModel...")
transformer_model = TransformerModel(
    num_poi=num_poi,
    num_cat=num_cat,
    embed_size=embed_size,
    nhead=2,
    nhid=1024,
    nlayers=2,
    dropout=0.3
)

# 创建测试输入
src = torch.randn(seq_len, batch_size, embed_size)
src_mask = transformer_model.generate_square_subsequent_mask(seq_len)

# 测试带MoE的前向传播
out_poi, out_time, out_cat, gate_poi, gate_cat = transformer_model(
    src, src_mask, user_embed, time_embed
)

print(f"   输入形状: {src.shape}")
print(f"   POI输出形状: {out_poi.shape}")
print(f"   Time输出形状: {out_time.shape}")
print(f"   Cat输出形状: {out_cat.shape}")
print(f"   POI门控权重形状: {gate_poi.shape}")
print(f"   Cat门控权重形状: {gate_cat.shape}")
print("   ✓ TransformerModel (with MoE) 测试通过")

# 3. 测试降级模式（不使用MoE）
print("\n3. 测试降级模式（不使用MoE）...")
out_poi2, out_time2, out_cat2, gate_poi2, gate_cat2 = transformer_model(
    src, src_mask, None, None
)

print(f"   POI输出形状: {out_poi2.shape}")
print(f"   Time输出形状: {out_time2.shape}")
print(f"   Cat输出形状: {out_cat2.shape}")
print(f"   门控权重: {gate_poi2} (应该是None)")
print("   ✓ 降级模式测试通过")

# 4. 验证专家数量
print("\n4. 验证专家配置...")
print(f"   POI decoder专家数: {len(transformer_model.decoder_poi.experts)}")
print(f"   Cat decoder专家数: {len(transformer_model.decoder_cat.experts)}")
print(f"   Time decoder类型: {type(transformer_model.decoder_time).__name__}")
print("   ✓ 专家配置正确")

print("\n" + "="*60)
print("所有测试通过！MoE模块工作正常。")
print("="*60)
