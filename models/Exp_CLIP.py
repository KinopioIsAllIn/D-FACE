import torch
from torch import nn
from .clip import clip
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
        self.query = nn.Parameter(torch.randn(dim)) 
        self.temperature = temperature
        assert sparse_reg in ["entropy", "gini"]
        self.sparse_reg = sparse_reg
        self.ln = nn.LayerNorm(dim) 

    def forward(self, x):
        # x: [B, L, D]
        x = self.ln(x)
        scores = torch.matmul(x, self.query) / (self.temperature * (x.size(-1) ** 0.5))
        attn = F.softmax(scores, dim=1)
        z = torch.einsum("bl, bld -> bd", attn, x)

        if self.sparse_reg == "entropy":
            eps = 1e-8
            entropy = -(attn * (attn + eps).log()).sum(dim=1).mean()
            sparsity_loss = entropy
        else:
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

        self.cls_proj = nn.Sequential(  
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

    def forward(self, indices, text):

        B = indices.size(0)
        x = self.token_embed(indices)

        x = x + self.pos_embed 

        x = self.transformer(x) 

        z, attn, sparse_loss = self.sparse_pool(x)

        clip_embedding = self.clip_proj(z)
        image_features = clip_embedding / clip_embedding.norm(dim=-1, keepdim=True)

        class_results = self.cls_proj(z)
        
        return class_results, image_features, None, None, sparse_loss#, attn


