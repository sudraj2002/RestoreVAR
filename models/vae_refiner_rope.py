import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttention(nn.Module):  # CrossAttention with last block of var
    def __init__(self, dim=768, heads=12):
        super().__init__()
        dim_head = dim // heads
        self.heads = heads
        self.dim = dim
        self.scale = dim_head ** -0.5
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(dim, dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, clean_tokens, condition):
        x = self.norm_q(clean_tokens)
        y = self.norm_kv(condition)

        # Query
        q = self.to_q(x)
        # Key, value
        kv = self.to_kv(y).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.heads), kv)
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.heads)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h l d -> b l (h d)')

        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout),
                CrossAttention(dim, heads)
            ]))

    def forward(self, x, last_block):
        for attn, ff, ca in self.layers:
            x = attn(x) + x
            x = ff(x) + x
            x = ca(x, last_block) + x
        x = self.norm(x)
        return x


class Refiner(nn.Module):
    """
    Refiner code
    """

    def __init__(
            self,
            seq_len=256,
            seq_len_cond=2240,  # last block length
            in_dim=32,
            in_dim_cond=1024,  # Last block channels
            out_dim=32,
            dim=1024,  # transformer internal dim
            depth=12,
            heads=16,
            mlp_dim=2048,
            pool='cls',
            dropout=0.,
    ):
        super().__init__()
        assert pool in {'cls', 'mean'}, "pool must be either 'cls' or 'mean'"

        self.seq_len = seq_len
        self.pool = pool

        # Linear projection from in_dim to dim
        self.proj_in = nn.Linear(in_dim, dim, bias=False)
        self.proj_in_cond = nn.Linear(in_dim_cond, dim, bias=False)

        # Positional embedding for the entire sequence at dimension=1024
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, dim))
        self.pos_embedding_cond = nn.Parameter(torch.randn(1, seq_len_cond, dim))

        nn.init.trunc_normal_(self.pos_embedding, mean=0, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding_cond, mean=0, std=0.02)
        # Transformer
        dim_head = dim // heads
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.proj_out = nn.Linear(dim, out_dim, bias=False)

        self.norm_in = nn.LayerNorm(in_dim)
        self.norm_in_cond = nn.LayerNorm(in_dim_cond)

    def forward(self, x, last_block=None):
        # Original x for final residual. x is the f_hat or discrete latent
        # last block is like a continuous guidance to predict the correct residual to add to f_hat to
        # get the gt continuous residual
        inp_clone = x.clone()

        B, n, d_in = x.shape
        b, n_cond, d_cond = last_block.shape
        assert n == self.seq_len, f"Expected seq_len={self.seq_len}, got {n} instead."

        x = self.norm_in(x)
        last_block = self.norm_in_cond(last_block)

        # Project x and last block up to transformer dim
        x = self.proj_in(x)
        last_block = self.proj_in_cond(last_block)

        # Add positional embedding
        x = x + self.pos_embedding[:, :n]
        last_block = last_block + self.pos_embedding_cond[:, :n_cond]

        # Pass through the Transformer
        x = self.transformer(x, last_block=last_block)

        x = self.proj_out(x)

        x = x + inp_clone

        return x