import torch
from torch import nn
from .clip import clip
# from .BLIP2_T5 import *
import torch.nn.functional as F


class PretrainedTextEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        if args.load_model == 'CLIP_B32':
            self.clip_model, _ = clip.load("ViT-B/32", device)
        elif args.load_model == 'CLIP_B16':
            self.clip_model, _ = clip.load("ViT-B/16", device)
        elif args.load_model == 'CLIP_L14':
            self.clip_model, _ = clip.load("ViT-L/14", device)

    def forward(self, prompt):
        text_tokenized = clip.tokenize(prompt, context_length=77, truncate=True).to('cuda')

        text_features = self.clip_model.encode_text(text_tokenized)
        text_features = text_features.float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

class SparseAttnPooling(nn.Module):
    def __init__(self, dim, temperature=1.0, sparse_reg="entropy"):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))  # 可学习查询向量
        self.temperature = temperature
        assert sparse_reg in ["entropy", "gini"]
        self.sparse_reg = sparse_reg
        self.ln = nn.LayerNorm(dim)  # 稍微稳一点

    def forward(self, x):
        # x: [B, L, D]
        x = self.ln(x)
        # 打分: [B, L]
        scores = torch.matmul(x, self.query) / (self.temperature * (x.size(-1) ** 0.5))
        attn = F.softmax(scores, dim=1)
        # 聚合
        z = torch.einsum("bl, bld -> bd", attn, x)

        # 稀疏正则: 最小化熵 或 Gini impurity
        if self.sparse_reg == "entropy":
            # 熵: -sum p log p, 我们要“最小化熵”(让分布更尖)，所以 loss = +entropy
            eps = 1e-8
            entropy = -(attn * (attn + eps).log()).sum(dim=1).mean()
            sparsity_loss = entropy
        else:
            # Gini: sum p(1-p)，越小越稀疏；直接最小化即可
            gini = (attn * (1 - attn)).sum(dim=1).mean()
            sparsity_loss = gini

        return z, attn, sparsity_loss

class VQCodeTransformer(nn.Module):
    def __init__(self, args, classnum=3, seq_len=16, dim=512, depth=2, heads=8, mlp_dim=512, dropout=0.2):
        super().__init__()

        self.token_embed = nn.Linear(512, dim)

        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, dim))

        self.sparse_pool = SparseAttnPooling(dim=dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.clip_proj = nn.Sequential(
            nn.Linear(dim, 768),
        )

        self.cls_proj = nn.Sequential(         # 对整个序列求 CLS 表达
            nn.Linear(dim, classnum)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, indices, text):  # [B, 16]

        B = indices.size(0) # B, 16
        x = self.token_embed(indices)  # [B, 16, dim]

        x = x + self.pos_embed  # 加位置编码

        x = self.transformer(x)  # [B, 17, dim]

        z, attn, sparse_loss = self.sparse_pool(x)  # SparseAttnPooling 或 MultiQuerySparsePooling

        clip_embedding = self.clip_proj(z)
        image_features = clip_embedding / clip_embedding.norm(dim=-1, keepdim=True)

        class_results = self.cls_proj(z)

        # 基于类别prompts生成embeddings, 并计算
        if text is not None:
            text_tokenized = clip.tokenize(text, context_length=77, truncate=True).to('cuda')

            text_features = self.clip_model.encode_text(text_tokenized)
            text_features = text_features.float()
            # text_features = self.projection_head(text_features)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.clip_model.logit_scale.exp()

            return class_results, image_features, text_features, logit_scale, sparse_loss#, attn
        else:
            return class_results, image_features, None, None, sparse_loss#, attn


