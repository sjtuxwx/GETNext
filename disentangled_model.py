"""
时空解耦双塔模型 (Spatio-Temporal Disentangled Dual-Tower Model)
实现论文方法论中的所有核心模块
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierFeatures(nn.Module):
    """
    傅里叶位置编码 (Fourier Features)
    将原始坐标映射到高频空间以解决MLP的低频偏差问题
    
    输入: (batch_size, seq_len, 2) - 经纬度坐标
    输出: (batch_size, seq_len, 2 * L) - 傅里叶编码后的特征
    """
    def __init__(self, num_frequencies=10):
        """
        Args:
            num_frequencies: 频率阶数 L
        """
        super(FourierFeatures, self).__init__()
        self.num_frequencies = num_frequencies
        
    def forward(self, coords):
        """
        Args:
            coords: shape (batch_size, seq_len, 2) 或 (seq_len, 2)
        Returns:
            fourier_features: shape (batch_size, seq_len, 4*L) 或 (seq_len, 4*L)
        """
        # 确保coords是3D张量
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)  # (1, seq_len, 2)
        
        batch_size, seq_len, coord_dim = coords.shape
        assert coord_dim == 2, "坐标维度必须是2 (经度, 纬度)"
        
        # 生成频率系数: [1, 2, 4, 8, ..., 2^(L-1)]
        frequencies = 2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32, device=coords.device)
        # frequencies shape: (L,)
        
        # 扩展维度以便广播
        # coords: (batch_size, seq_len, 2, 1)
        # frequencies: (1, 1, 1, L)
        coords_expanded = coords.unsqueeze(-1)  # (batch_size, seq_len, 2, 1)
        frequencies_expanded = frequencies.view(1, 1, 1, -1)  # (1, 1, 1, L)
        
        # 计算 2π * freq * coord
        angles = 2 * math.pi * coords_expanded * frequencies_expanded  # (batch_size, seq_len, 2, L)
        
        # 计算sin和cos
        sin_features = torch.sin(angles)  # (batch_size, seq_len, 2, L)
        cos_features = torch.cos(angles)  # (batch_size, seq_len, 2, L)
        
        # 拼接sin和cos特征
        fourier_features = torch.cat([sin_features, cos_features], dim=-1)  # (batch_size, seq_len, 2, 2*L)
        
        # 展平最后两个维度
        fourier_features = fourier_features.reshape(batch_size, seq_len, -1)  # (batch_size, seq_len, 4*L)
        
        return fourier_features


class AdaptiveGaussianKernelAttention(nn.Module):
    """
    自适应高斯核注意力 (Adaptive Gaussian Kernel Attention)
    使用物理距离而非语义相似度计算注意力权重
    
    关键特性:
    - 可学习的空间感受野参数 σ
    - 基于欧氏距离的高斯核
    - 完全个性化、动态计算
    """
    def __init__(self, hidden_dim, initial_sigma=1.0):
        """
        Args:
            hidden_dim: 隐藏层维度
            initial_sigma: σ的初始值
        """
        super(AdaptiveGaussianKernelAttention, self).__init__()
        
        # 可学习的空间感受野参数 σ (标量)
        self.log_sigma = nn.Parameter(torch.tensor(math.log(initial_sigma)))
        
        # 用于特征变换的线性层
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, spatial_features, coords):
        """
        Args:
            spatial_features: shape (batch_size, seq_len, hidden_dim)
            coords: shape (batch_size, seq_len, 2) - 经纬度坐标
        Returns:
            output: shape (batch_size, seq_len, hidden_dim)
            attention_weights: shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, hidden_dim = spatial_features.shape
        
        # 计算σ (确保为正)
        sigma = torch.exp(self.log_sigma)
        
        # 计算距离矩阵 d_ij
        # coords: (batch_size, seq_len, 2)
        # 使用广播计算所有点对之间的欧氏距离
        coords_i = coords.unsqueeze(2)  # (batch_size, seq_len, 1, 2)
        coords_j = coords.unsqueeze(1)  # (batch_size, 1, seq_len, 2)
        
        # 欧氏距离: sqrt(sum((xi - xj)^2))
        distance_matrix = torch.sqrt(torch.sum((coords_i - coords_j) ** 2, dim=-1) + 1e-8)
        # distance_matrix shape: (batch_size, seq_len, seq_len)
        
        # 计算高斯核注意力权重
        # α_ij = exp(-d_ij^2 / (2*σ^2))
        attention_logits = -distance_matrix ** 2 / (2 * sigma ** 2)
        
        # Softmax归一化
        attention_weights = F.softmax(attention_logits, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # 计算value
        values = self.value_proj(spatial_features)  # (batch_size, seq_len, hidden_dim)
        
        # 加权聚合
        output = torch.bmm(attention_weights, values)  # (batch_size, seq_len, hidden_dim)
        
        return output, attention_weights


class PreferenceTower(nn.Module):
    """
    语义偏好塔 (Preference Tower)
    使用标准的多头自注意力机制建模POI之间的语义依赖
    """
    def __init__(self, hidden_dim, num_heads=4, num_layers=2, dropout=0.1):
        """
        Args:
            hidden_dim: 隐藏层维度
            num_heads: 多头注意力的头数
            num_layers: Transformer层数
            dropout: Dropout比率
        """
        super(PreferenceTower, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        # 多层Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, preference_features, mask=None):
        """
        Args:
            preference_features: shape (batch_size, seq_len, hidden_dim)
            mask: shape (seq_len, seq_len) - causal mask for autoregressive modeling
        Returns:
            output: shape (batch_size, seq_len, hidden_dim)
        """
        # Transformer编码
        output = self.transformer_encoder(preference_features, mask=mask)
        
        # Layer Normalization
        output = self.layer_norm(output)
        
        return output


class SpatialTower(nn.Module):
    """
    空间感知塔 (Spatial Tower)
    使用自适应高斯核注意力建模物理距离的衰减效应
    """
    def __init__(self, hidden_dim, num_layers=2, initial_sigma=1.0, dropout=0.1):
        """
        Args:
            hidden_dim: 隐藏层维度
            num_layers: 空间注意力层数
            initial_sigma: σ的初始值
            dropout: Dropout比率
        """
        super(SpatialTower, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 多层自适应高斯核注意力
        self.spatial_attention_layers = nn.ModuleList([
            AdaptiveGaussianKernelAttention(hidden_dim, initial_sigma)
            for _ in range(num_layers)
        ])
        
        # Feed-forward网络
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Layer Normalization
        self.layer_norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.layer_norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        
    def forward(self, spatial_features, coords):
        """
        Args:
            spatial_features: shape (batch_size, seq_len, hidden_dim)
            coords: shape (batch_size, seq_len, 2)
        Returns:
            output: shape (batch_size, seq_len, hidden_dim)
            all_attention_weights: list of attention weights from each layer
        """
        output = spatial_features
        all_attention_weights = []
        
        for i in range(self.num_layers):
            # 自适应高斯核注意力 + 残差连接
            attn_output, attn_weights = self.spatial_attention_layers[i](output, coords)
            output = self.layer_norms1[i](output + attn_output)
            all_attention_weights.append(attn_weights)
            
            # Feed-forward + 残差连接
            ffn_output = self.ffn_layers[i](output)
            output = self.layer_norms2[i](output + ffn_output)
        
        return output, all_attention_weights


class AdaptiveContextGating(nn.Module):
    """
    自适应门控融合 (Adaptive Context Gating)
    根据时间上下文动态平衡语义偏好和空间距离的权重
    """
    def __init__(self, hidden_dim, time_embed_dim):
        """
        Args:
            hidden_dim: 隐藏层维度
            time_embed_dim: 时间嵌入维度
        """
        super(AdaptiveContextGating, self).__init__()
        
        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, preference_features, spatial_features, time_embeddings):
        """
        Args:
            preference_features: shape (batch_size, seq_len, hidden_dim)
            spatial_features: shape (batch_size, seq_len, hidden_dim)
            time_embeddings: shape (batch_size, seq_len, time_embed_dim)
        Returns:
            fused_features: shape (batch_size, seq_len, hidden_dim)
            gate_values: shape (batch_size, seq_len, 1)
        """
        # 计算门控系数 g_t
        gate_values = self.gate_network(time_embeddings)  # (batch_size, seq_len, 1)
        
        # 融合特征
        # h_t = g_t * h_pref + (1 - g_t) * h_spatial
        fused_features = gate_values * preference_features + (1 - gate_values) * spatial_features
        
        return fused_features, gate_values


class DisentangledDualTowerModel(nn.Module):
    """
    时空解耦双塔模型 (Spatio-Temporal Disentangled Dual-Tower Model)
    完整实现论文方法论中的所有模块
    """
    def __init__(
        self,
        num_pois,
        num_cats,
        poi_embed_dim=128,
        cat_embed_dim=32,
        time_embed_dim=32,
        user_embed_dim=128,
        gcn_poi_embed_dim=128,
        hidden_dim=256,
        num_fourier_freq=10,
        num_pref_heads=4,
        num_pref_layers=2,
        num_spatial_layers=2,
        initial_sigma=1.0,
        dropout=0.1
    ):
        """
        Args:
            num_pois: POI总数
            num_cats: 类别总数
            poi_embed_dim: POI嵌入维度
            cat_embed_dim: 类别嵌入维度
            time_embed_dim: 时间嵌入维度
            user_embed_dim: 用户嵌入维度
            gcn_poi_embed_dim: GCN预训练的POI嵌入维度
            hidden_dim: 双塔的隐藏层维度
            num_fourier_freq: 傅里叶编码的频率阶数
            num_pref_heads: 语义偏好塔的注意力头数
            num_pref_layers: 语义偏好塔的层数
            num_spatial_layers: 空间感知塔的层数
            initial_sigma: 高斯核的初始σ值
            dropout: Dropout比率
        """
        super(DisentangledDualTowerModel, self).__init__()
        
        self.num_pois = num_pois
        self.num_cats = num_cats
        self.hidden_dim = hidden_dim
        
        # ============= 3.2 多模态输入嵌入 =============
        
        # 1. 语义偏好嵌入
        self.poi_embedding = nn.Embedding(num_pois, poi_embed_dim)
        self.cat_embedding = nn.Embedding(num_cats, cat_embed_dim)
        
        # 语义特征投影层 (将POI + Category + GCN嵌入映射到hidden_dim)
        preference_input_dim = poi_embed_dim + cat_embed_dim + gcn_poi_embed_dim
        self.preference_proj = nn.Sequential(
            nn.Linear(preference_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 空间感知嵌入
        self.fourier_features = FourierFeatures(num_frequencies=num_fourier_freq)
        
        # 空间特征投影层 (将傅里叶特征映射到hidden_dim)
        spatial_input_dim = 4 * num_fourier_freq  # 2个坐标 * 2(sin+cos) * L
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # ============= 3.3 时空解耦双塔编码器 =============
        
        # 塔一: 语义偏好塔
        self.preference_tower = PreferenceTower(
            hidden_dim=hidden_dim,
            num_heads=num_pref_heads,
            num_layers=num_pref_layers,
            dropout=dropout
        )
        
        # 塔二: 空间感知塔
        self.spatial_tower = SpatialTower(
            hidden_dim=hidden_dim,
            num_layers=num_spatial_layers,
            initial_sigma=initial_sigma,
            dropout=dropout
        )
        
        # ============= 3.4 自适应门控融合 =============
        
        self.adaptive_gating = AdaptiveContextGating(
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim
        )
        
        # ============= 输出层 =============
        
        # POI预测头
        self.poi_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_pois)
        )
        
        # 时间预测头
        self.time_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 类别预测头
        self.cat_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_cats)
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
    
    def generate_square_subsequent_mask(self, sz, device):
        """生成因果mask用于自回归建模"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float('-inf'))
        return mask
    
    def forward(
        self,
        poi_indices,
        cat_indices,
        coords,
        time_embeddings,
        gcn_poi_embeddings,
        mask=None
    ):
        """
        Args:
            poi_indices: shape (batch_size, seq_len) - POI索引
            cat_indices: shape (batch_size, seq_len) - 类别索引
            coords: shape (batch_size, seq_len, 2) - 经纬度坐标
            time_embeddings: shape (batch_size, seq_len, time_embed_dim) - 时间嵌入
            gcn_poi_embeddings: shape (batch_size, seq_len, gcn_poi_embed_dim) - GCN预训练的POI嵌入
            mask: shape (seq_len, seq_len) - 因果mask
        Returns:
            poi_logits: shape (batch_size, seq_len, num_pois)
            time_pred: shape (batch_size, seq_len, 1)
            cat_logits: shape (batch_size, seq_len, num_cats)
            preference_features: shape (batch_size, seq_len, hidden_dim)
            spatial_features: shape (batch_size, seq_len, hidden_dim)
            gate_values: shape (batch_size, seq_len, 1)
        """
        batch_size, seq_len = poi_indices.shape
        
        # ============= 3.2 多模态输入嵌入 =============
        
        # 1. 语义偏好嵌入
        # 处理padding值: 将-1替换为0（有效索引），后续会用mask过滤
        poi_indices_safe = torch.where(poi_indices >= 0, poi_indices, torch.zeros_like(poi_indices))
        cat_indices_safe = torch.where(cat_indices >= 0, cat_indices, torch.zeros_like(cat_indices))
        
        poi_embeds = self.poi_embedding(poi_indices_safe)  # (batch_size, seq_len, poi_embed_dim)
        cat_embeds = self.cat_embedding(cat_indices_safe)  # (batch_size, seq_len, cat_embed_dim)
        
        # 拼接语义特征: POI + Category + GCN
        preference_input = torch.cat([poi_embeds, cat_embeds, gcn_poi_embeddings], dim=-1)
        preference_features = self.preference_proj(preference_input)  # (batch_size, seq_len, hidden_dim)
        
        # 2. 空间感知嵌入
        fourier_coords = self.fourier_features(coords)  # (batch_size, seq_len, 4*L)
        spatial_features = self.spatial_proj(fourier_coords)  # (batch_size, seq_len, hidden_dim)
        
        # ============= 3.3 时空解耦双塔编码器 =============
        
        # 塔一: 语义偏好塔
        preference_output = self.preference_tower(preference_features, mask=mask)
        # preference_output: (batch_size, seq_len, hidden_dim)
        
        # 塔二: 空间感知塔
        spatial_output, spatial_attn_weights = self.spatial_tower(spatial_features, coords)
        # spatial_output: (batch_size, seq_len, hidden_dim)
        
        # ============= 3.4 自适应门控融合 =============
        
        fused_features, gate_values = self.adaptive_gating(
            preference_output,
            spatial_output,
            time_embeddings
        )
        # fused_features: (batch_size, seq_len, hidden_dim)
        # gate_values: (batch_size, seq_len, 1)
        
        # ============= 输出预测 =============
        
        poi_logits = self.poi_decoder(fused_features)  # (batch_size, seq_len, num_pois)
        time_pred = self.time_decoder(fused_features)  # (batch_size, seq_len, 1)
        cat_logits = self.cat_decoder(fused_features)  # (batch_size, seq_len, num_cats)
        
        return (
            poi_logits,
            time_pred,
            cat_logits,
            preference_output,  # 用于计算正交损失
            spatial_output,     # 用于计算正交损失
            gate_values         # 用于分析
        )


def orthogonality_loss(preference_features, spatial_features):
    """
    正交约束损失 (Orthogonality Constraint Loss) - 归一化版本
    L_ortho = ||H^pref · (H^spatial)^T||_F^2 / (seq_len^2)
    
    归一化确保损失不会随序列长度平方增长，使其在不同长度的序列上具有可比性。
    
    Args:
        preference_features: shape (batch_size, seq_len, hidden_dim)
        spatial_features: shape (batch_size, seq_len, hidden_dim)
    Returns:
        loss: 标量，归一化后的正交损失
    """
    batch_size, seq_len, hidden_dim = preference_features.shape
    
    # 批量计算矩阵乘积: (batch_size, seq_len, hidden_dim) @ (batch_size, hidden_dim, seq_len)
    # 结果: (batch_size, seq_len, seq_len)
    product = torch.bmm(preference_features, spatial_features.transpose(1, 2))
    
    # 计算每个样本的Frobenius范数平方，并除以元素数量进行归一化
    # 这样损失不会随序列长度增长
    loss_per_sample = torch.sum(product ** 2, dim=(1, 2)) / (seq_len * seq_len)  # (batch_size,)
    
    # 对batch取平均
    return loss_per_sample.mean()


if __name__ == "__main__":
    # 测试代码
    print("=" * 50)
    print("测试时空解耦双塔模型")
    print("=" * 50)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 模型参数
    num_pois = 1000
    num_cats = 50
    batch_size = 4
    seq_len = 10
    
    # 创建模型
    model = DisentangledDualTowerModel(
        num_pois=num_pois,
        num_cats=num_cats,
        poi_embed_dim=64,
        cat_embed_dim=16,
        time_embed_dim=16,
        user_embed_dim=64,
        gcn_poi_embed_dim=64,
        hidden_dim=128,
        num_fourier_freq=8,
        num_pref_heads=4,
        num_pref_layers=2,
        num_spatial_layers=2,
        initial_sigma=1.0,
        dropout=0.1
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试数据
    poi_indices = torch.randint(0, num_pois, (batch_size, seq_len))
    cat_indices = torch.randint(0, num_cats, (batch_size, seq_len))
    coords = torch.randn(batch_size, seq_len, 2) * 0.1 + torch.tensor([40.7, -74.0])  # 纽约附近
    time_embeddings = torch.randn(batch_size, seq_len, 16)
    gcn_poi_embeddings = torch.randn(batch_size, seq_len, 64)
    
    # 生成因果mask
    mask = model.generate_square_subsequent_mask(seq_len, device='cpu')
    
    # 前向传播
    print("\n前向传播测试...")
    poi_logits, time_pred, cat_logits, pref_feat, spatial_feat, gate_vals = model(
        poi_indices,
        cat_indices,
        coords,
        time_embeddings,
        gcn_poi_embeddings,
        mask=mask
    )
    
    print(f"POI logits shape: {poi_logits.shape}")
    print(f"Time prediction shape: {time_pred.shape}")
    print(f"Category logits shape: {cat_logits.shape}")
    print(f"Preference features shape: {pref_feat.shape}")
    print(f"Spatial features shape: {spatial_feat.shape}")
    print(f"Gate values shape: {gate_vals.shape}")
    
    # 测试正交损失
    print("\n正交损失测试...")
    ortho_loss = orthogonality_loss(pref_feat, spatial_feat)
    print(f"正交损失: {ortho_loss.item():.4f}")
    
    # 测试门控值的统计
    print("\n门控值统计:")
    print(f"  均值: {gate_vals.mean().item():.4f}")
    print(f"  标准差: {gate_vals.std().item():.4f}")
    print(f"  最小值: {gate_vals.min().item():.4f}")
    print(f"  最大值: {gate_vals.max().item():.4f}")
    
    # 测试可学习的σ
    print("\n空间感受野参数 σ:")
    for i, layer in enumerate(model.spatial_tower.spatial_attention_layers):
        sigma = torch.exp(layer.log_sigma).item()
        print(f"  Layer {i}: σ = {sigma:.4f}")
    
    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
