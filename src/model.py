import torch
import torch.nn as nn
import torch.functional as F
from jaxtyping import Float, jaxtyped, Bool
from beartype import beartype as typechecker
from einops import rearrange
from math import sqrt
import copy

class MultiHeadedAttention(nn.Module):
    """
    Implementação do mecanismo de MultiHead Attention do Transformer Original
        Métodos:
            split_heads: divide o tensor de entrada para as cabeças de atenção

            forward: Implementação do fluxo de cálculo da atenção para cada cabeça:
                - divisão
                - cálculo da métrica
                - masking e padding
                - cálculo dos pesos resultantes
    """

    def __init__(self, num_heads, embed_dim, dropout = 0.1):
        
        super().__init__()
        self.d = embed_dim // num_heads
        self.h = num_heads
        self.q_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)
        self.k_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)
        self.v_proj: nn.Linear = nn.Linear(embed_dim, embed_dim)
        self.out_proj: nn.Linear = nn.Linear(embed_dim, embed_dim) # junta os resultados em uma camada única
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    @jaxtyped(typechecker=typechecker)
    def split_heads(self, 
            X: Float[torch.Tensor, "batch seq embed_dim"]
        )-> Float[torch.Tensor, "batch heads seq dim"]:

        return rearrange(X, 'batch seq (heads dim)-> batch heads seq dim', heads=self.h)

    @jaxtyped(typechecker=typechecker)
    def forward(self,
                query: Float[torch.Tensor, 'batch q_len embed_dim'], 
                key: Float[torch.Tensor, 'batch kv_len embed_dim'], 
                value: Float[torch.Tensor, 'batch kv_len embed_dim'],
                attn_mask: Bool[torch.Tensor, "q_len kv_len"] | None = None,
                key_padding_mask: Bool[torch.tensor, "batch kv_len"] | None = None
        ):
        q = self.split_heads(self.q_proj(query))
        k = self.split_heads(self.k_proj(value))
        v = self.split_heads(self.q_proj(key))

        k_t = rearrange(k, 'batch heads kb_len dim -> batch heads dim kv_len')

        scores: Float[torch.Tensor, "batch heads q_len kv_len"] = q @ k_t / self.d**0.5 # cáculo da atenção (sem softmax)

        if attn_mask is not None: # prevent overfitting
            scores = scores.masked_fill(attn_mask, -torch.inf) # zerar depois da aplicação do softmax 

        if key_padding_mask is not None: # deal with diferent size sentences
            mask = rearrange(key_padding_mask, 'batch kv_len -> batch 1 1 kv_len')
            scores = scores.masked_fill(mask, -torch.inf)

        weights: Float[torch.Tensor, 'batch heads q_len kv_len'] = scores.softmax(dim=-1)

        Z: Float[torch.Tensor, "batch heads q_len kv_len"] = self.dropout(weights) @ v
        Z_merged: Float[torch.Tensor, "batch q_len embed_dim"] = rearrange(Z, "batch heads q_len dim -> batch q_len (heads dim)")

        return (self.out_proj(Z_merged), weights)


class PositionalEmbedding(nn.Module):

    def __init__(self, max_length, embed_dim, dropout):
        super().__init__()

        self.pos_embed: Float[nn.Parameter, "max_length embed_dim"] = nn.Parameter(torch.randn(max_length, embed_dim) * 0.2)
        self.dropout = nn.Dropout(dropout)

    @jaxtyped(typechecker=typechecker)
    def forward(self, 
                X: Float[torch.Tensor, "batch seq_len embed_dim"]
    )-> Float[torch.Tensor, "batch seq_len embed_dim"]:
        
        return self.dropout(X + self.pos_embed[:X.size(1)])
        

class LayerNorm(nn.Module):
    
    def __init__(self,features_d ,eps=1e-6):
        super(nn.LayerNorm, self).__init__()

        self.a_1: nn.Parameter = nn.Parameter(torch.ones(features_d))
        self.b_2: nn.Parameter = nn.Parameter(torch.zeros(features_d))
        self.features_d  = features_d
        self.eps = eps
    
    @jaxtyped(typechecker=typechecker)
    def forward(self, X: Float[torch.Tensor, "batch seq_len features"]):
        mean = X.mean()
        std = X.std()
        norm_x = (X - mean) / self.eps + std
        
        return self.a_1 * norm_x + self.b_2


def clone(module: nn.Linear, N: int)-> nn.ModuleList:
    return nn.ModuleList([copy.deep_copy(module) for _ in range(N)])


class TransformerEncoderLayer(nn.Module):
    pass


class TransformerDecoderLayer(nn.Module):
    pass

class NmtTransformer(nn.Module):
    pass



