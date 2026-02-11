import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# ==================== RoTAN 时间旋转核心函数 ====================
def rotate(head, relation, hidden, device):
    """单样本时间旋转 (from RoTAN)"""
    pi = 3.14159265358979323846
    re_head, im_head = torch.chunk(head, 2, dim=1)
    embedding_range = nn.Parameter(
        torch.Tensor([(24.0 + 2.0) / hidden]), 
        requires_grad=False
    ).to(device)
    phase_relation = relation / (embedding_range / pi)
    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)
    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation
    return torch.cat([re_score, im_score], dim=1)


def rotate_batch(head, relation, hidden, device):
    """批量时间旋转 (from RoTAN)"""
    pi = 3.14159265358979323846
    re_head, im_head = torch.chunk(head, 2, dim=2)
    embedding_range = nn.Parameter(
        torch.Tensor([(24.0 + 2.0) / hidden]), 
        requires_grad=False
    ).to(device)
    phase_relation = relation / (embedding_range / pi)
    re_relation = torch.cos(phase_relation)
    im_relation = torch.sin(phase_relation)
    re_score = re_head * re_relation - im_head * im_relation
    im_score = re_head * im_relation + im_head * re_relation
    return torch.cat([re_score, im_score], dim=2)


# ==================== Layer 1: 门控时间旋转模块 ====================
class GatedTemporalRotation(nn.Module):
    """可学习门控的时间旋转模块 - 自适应决定旋转强度"""
    def __init__(self, embed_dim, time_dim, device, dropout=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.device = device
        
        # 时间信息融合网络 (替代原来的旋转)
        self.time_fusion = nn.Sequential(
            nn.Linear(embed_dim + time_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 门控网络: 根据内容决定需要多少时间信息
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim + time_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, embed, time_embed):
        """
        embed: (batch, seq_len, embed_dim) 用户+POI融合嵌入
        time_embed: (batch, seq_len, time_dim) 时间嵌入
        """
        # 计算门控值 (0-1之间，决定时间信息的重要性)
        gate = self.gate_net(torch.cat([embed, time_embed], dim=-1))
        
        # 时间信息融合 (通过网络学习如何融合时间)
        time_fused = self.time_fusion(torch.cat([embed, time_embed], dim=-1))
        
        # 门控融合: gate=1完全使用时间融合, gate=0保持原样
        return gate * time_fused + (1 - gate) * embed


class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A

        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x


class UserEmbeddings(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddings, self).__init__()

        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
        )

    def forward(self, user_idx):
        embed = self.user_embedding(user_idx)
        return embed


class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed


class FuseEmbeddings(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        x = self.leaky_relu(x)
        return x


def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, out_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, out_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, out_dim)

    def forward(self, x):
        x = self.l1(x)
        return x


# ==================== Layer 2: Temporal Rotary Attention 核心 ====================
def compute_temporal_freqs(time_values, head_dim, device, theta=10000.0):
    """
    计算时间驱动的旋转频率 (类RoPE机制,但用真实时间值)
    time_values: (batch, seq_len) 归一化时间值
    head_dim: 注意力头维度
    """
    # RoPE要求head_dim必须是偶数，确保维度正确
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"
    
    # 多频率基底: 捕获不同时间尺度，生成 head_dim//2 个频率
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = freqs.to(device)
    
    # time作为"位置",生成旋转角度
    angles = time_values.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    
    return freqs_cis


def apply_temporal_rotary(x, freqs_cis):
    """
    对Q/K施加时间旋转
    x: (batch, seq_len, nhead, head_dim)
    freqs_cis: (batch, seq_len, head_dim//2) 复数旋转向量
    """
    # 确保head_dim是偶数
    assert x.shape[-1] % 2 == 0, f"head_dim must be even, got {x.shape[-1]}"
    
    # x shape: (batch, seq_len, nhead, head_dim)
    # 转换为复数: (batch, seq_len, nhead, head_dim//2)
    x_complex = torch.view_as_complex(
        x.float().reshape(*x.shape[:-1], -1, 2))
    
    # freqs_cis shape: (batch, seq_len, head_dim//2)
    # 广播到nhead维度: (batch, seq_len, 1, head_dim//2)
    x_rotated = torch.view_as_real(
        x_complex * freqs_cis.unsqueeze(2)).flatten(-2)
    return x_rotated.type_as(x)


class TemporalRotaryEncoderLayer(nn.Module):
    """自定义Transformer层 - Q/K嵌入时间旋转"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        
        # Q/K/V投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, time_freqs_cis, src_mask=None):
        """
        src: (batch, seq_len, d_model)
        time_freqs_cis: (batch, seq_len, head_dim//2) 时间旋转频率
        """
        B, S, D = src.shape
        
        Q = self.W_q(src).view(B, S, self.nhead, self.head_dim)
        K = self.W_k(src).view(B, S, self.nhead, self.head_dim)
        V = self.W_v(src).view(B, S, self.nhead, self.head_dim)
        
        # 关键: 对Q/K施加时间旋转
        Q = apply_temporal_rotary(Q, time_freqs_cis)
        K = apply_temporal_rotary(K, time_freqs_cis)
        
        # 标准缩放点积注意力
        Q = Q.transpose(1, 2)  # (B, nhead, S, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if src_mask is not None:
            attn = attn + src_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ V).transpose(1, 2).reshape(B, S, D)
        out = self.W_o(out)
        
        # 残差+LayerNorm+FFN
        src = self.norm1(src + self.dropout(out))
        src = self.norm2(src + self.ffn(src))
        return src


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# ==================== Layer 3: Target-Time Cross-Attention 解码器 ====================
class TargetTimeCrossAttentionDecoder(nn.Module):
    """目标时间交叉注意力解码器 - 让目标时间主动查询相关历史"""
    def __init__(self, d_model, nhead, num_poi, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        # 目标时间投影
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        
        # 融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        self.decoder_poi = nn.Linear(d_model, num_poi)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, encoder_out, target_time_embed, src_mask=None):
        """
        encoder_out: (batch, seq_len, d_model) 编码器输出
        target_time_embed: (batch, seq_len, d_model) 目标时间嵌入
        """
        query = self.time_proj(target_time_embed)
        
        # 交叉注意力: 目标时间query, 历史序列key/value
        cross_out, attn_weights = self.cross_attn(
            query, encoder_out, encoder_out, 
            attn_mask=src_mask)
        cross_out = self.norm(cross_out)
        
        # 门控融合
        gate = self.fusion_gate(
            torch.cat([encoder_out, cross_out], dim=-1))
        fused = gate * cross_out + (1 - gate) * encoder_out
        
        # 最后输出前 dropout
        fused = self.dropout_layer(fused)
        
        return self.decoder_poi(fused)


class TransformerModel(nn.Module):
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(num_poi, embed_size)
        self.embed_size = embed_size
        self.decoder_poi = nn.Linear(embed_size, num_poi)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder_poi.bias.data.zero_()
        self.decoder_poi.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        x = self.transformer_encoder(src, src_mask)
        out_poi = self.decoder_poi(x)
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        return out_poi, out_time, out_cat


# ==================== 深度融合时间旋转的Transformer模型 ====================
class TemporalRotaryTransformerModel(nn.Module):
    """深度融合时间旋转的Transformer模型"""
    def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, 
                 nlayers, time_embed_dim, device, dropout=0.5):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        
        # Layer 1: 门控时间旋转
        self.gated_rotation = GatedTemporalRotation(
            embed_size, time_embed_dim, device, dropout)
        
        # Layer 2: Temporal Rotary Encoder
        self.layers = nn.ModuleList([
            TemporalRotaryEncoderLayer(embed_size, nhead, nhid, dropout)
            for _ in range(nlayers)
        ])
        
        # Layer 3: Target-Time Cross-Attention Decoder
        self.poi_decoder = TargetTimeCrossAttentionDecoder(
            embed_size, nhead, num_poi, dropout)
        
        # 时间和类别解码器(不需要目标时间)
        self.decoder_time = nn.Linear(embed_size, 1)
        self.decoder_cat = nn.Linear(embed_size, num_cat)
        
        # 目标时间编码
        self.target_time_embed = Time2Vec('sin', time_embed_dim)
        self.target_time_proj = nn.Linear(time_embed_dim, embed_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask, seq_times, target_times=None):
        """
        src: (seq_len, batch, embed_size) 输入序列嵌入
        src_mask: 注意力mask
        seq_times: (batch, seq_len) 每个签到的归一化时间
        target_times: (batch, seq_len) 目标时间(训练时=标签时间)
        """
        src = src * math.sqrt(self.embed_size)
        
        # 通过Temporal Rotary Encoder
        # src is already (seq_len, batch, d_model), transpose to (batch, seq_len, d_model)
        x = src.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # 🔥 关键修复: 应用门控时间旋转 (Layer 1)
        # 首先对时间做embedding
        B, S = seq_times.shape
        time_flat = seq_times.reshape(-1, 1)  # (B*S, 1)
        time_embed = self.target_time_embed(time_flat)
        time_embed = time_embed.reshape(B, S, -1)  # (B, S, time_embed_dim)
        
        # 应用门控旋转: 根据内容和时间自适应决定旋转强度
        x = self.gated_rotation(x, time_embed)
        
        # 🔥 Layer 2: Temporal Rotary Attention
        # 计算temporal rotary频率
        head_dim = self.embed_size // self.layers[0].nhead
        time_freqs_cis = compute_temporal_freqs(
            seq_times, head_dim, self.device)
        
        for layer in self.layers:
            x = layer(x, time_freqs_cis, src_mask)
        
        # Time + Cat 解码
        out_time = self.decoder_time(x)
        out_cat = self.decoder_cat(x)
        
        # POI解码(Target-Time Cross-Attention)
        if target_times is not None:
            # 训练时: 使用ground truth
            # Time2Vec需要(N, 1)输入,所以先reshape
            B, S = target_times.shape
            target_times_flat = target_times.reshape(-1, 1)  # (B*S, 1)
            target_t_embed = self.target_time_embed(target_times_flat)
            target_t_embed = target_t_embed.reshape(B, S, -1)  # (B, S, time_dim)
            target_t_proj = self.target_time_proj(target_t_embed)
            out_poi = self.poi_decoder(x, target_t_proj, src_mask)
        else:
            # 推理时: 先预测时间,再用预测时间解码POI
            pred_time = out_time.squeeze(-1)
            B, S = pred_time.shape
            pred_time_flat = pred_time.reshape(-1, 1)  # (B*S, 1)
            target_t_embed = self.target_time_embed(pred_time_flat)
            target_t_embed = target_t_embed.reshape(B, S, -1)  # (B, S, time_dim)
            target_t_proj = self.target_time_proj(target_t_embed)
            out_poi = self.poi_decoder(x, target_t_proj, src_mask)
        
        # out_poi, out_time, out_cat are already (batch, seq_len, *)
        # No need to transpose back
        return out_poi, out_time, out_cat
