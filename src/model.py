import torch
import torch.nn as nn
import torch.functional as F
from jaxtyping import Float, jaxtyped, Bool, Int
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

class Embeddings(nn.Module):
    def __init__(self, vocab: int, d_model: int):
        super(Embeddings, self).__init__()

        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, 
        x: Int[torch.Tensor, "batch seq_len"]
    )-> Float[torch.Tensor, "batch seq_len d_model"]:
        return self.lut(x) * sqrt(self.d_model)
    


def clones(module: nn.Linear, N: int)-> nn.ModuleList:
    return nn.ModuleList([copy.deep_copy(module) for _ in range(N)])

class TransformerEncoderLayer(nn.Module):

    def __init__(self,n_heads: int, d_model: int,d_ff: int ,dropout:float = 0.1 ):
        super().__init__()

        self.dropout = dropout
        
        self.self_attn = MultiHeadedAttention(num_heads=n_heads, embed_dim=d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    @jaxtyped(typechecker=typechecker)
    def forward( 
            self, 
            x: Float[torch.Tensor, "batch seq_len d_model"],
            mask: Float[torch.Tensor, "batch seq_len seq_len"] | None = None
        ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm,x_norm,x_norm, attn_mask=mask)
        x = x + self.dropout1(attn_out)

        x_norm2 = self.norm2(x)
        ff_out = self.feed_forward(x_norm2)
        x = x + self.dropout2(ff_out)

        return x

class TransformerEncoder(nn.Module):
    
    def __init__(self,layer: nn.Module, n_layer:int, d_model, int):
        super().__init__()

        self.layers = clones(layer,n_layer)
        self.norm = LayerNorm(d_model)

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq_len d_model"], 
        mask: Float[torch.Tensor, "batch seq_len seq_len"] | None = None
    )->Float[torch.Tensor, "batch seq_len d_model"]:    
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class TransformerDecoderLayer(nn.Module):

    def __init__(self, n_heads: int, d_model: int, d_ff: int, dropout: float = 0.1):

        super().__init__()
        
        self.norm_1 = LayerNorm(d_model)
        self.norm_2 = LayerNorm(d_model)
        self.norm_3 = LayerNorm(d_model)

        self.self_attn = MultiHeadedAttention(n_heads, d_model, dropout)
        self.cross_attn = MultiHeadedAttention(n_heads, d_model, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.size = d_model

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x: Float[torch.Tensor, "batch tgt_len d_model"],
        src_mask: Bool[torch.Tensor, 'batch src_len src_len'], 
        tgt_mask: Bool[torch.Tensor, 'batch tgt_len tgt_len'], 
        memory: Float[torch.Tensor, 'batch src_len d_model']
    )->Float[torch.Tensor, "batch tgt_len d_model"]:

        x_norm = self.norm_1(x)
        attn_out_1, _ = self.self_attn(x_norm,x_norm,x_norm, tgt_mask)
        x = x + self.dropout1(attn_out_1)

        x_norm2 = self.norm_2(x)
        attn_out_2, _ = self.cross_attn(x_norm2, memory, memory, src_mask)
        x = x + self.dropout2(attn_out_2)

        x_norm3 = self.norm_3(x)
        ff = self.feed_forward(x_norm3)
        x = x + self.dropout3(ff)

        return x

class TransformerDecoder(nn.Module):

    def __init__(self, layer: nn.Module, n_layers: int):

        super().__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    @jaxtyped(typechecker=typechecker)
    def forward(self,
        x: Float[torch.Tensor, "batch tgt_len d_model"],
        src_mask: Bool[torch.Tensor, "batch 1 src_len src_len"] | None,
        tgt_mask: Bool[torch.Tensor, "batch 1 tgt_len tgt_len"] | None,
        memory: Float[torch.Tensor, "batch src_len d_model"]
    )->Float[torch.Tensor, "batch tgt_len d_model"]:
        for layer in self.layers:
            x = layer(x, src_mask, tgt_mask, memory)

        return self.norm(x)

class Generator(nn.Module):

    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, X):
        logits = self.proj(X)
        return logits 
    
class NmtTransformer(nn.Module):

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        src_embed:nn.Module,
        tgt_embed: nn.Module,
        generator: nn.Module
    ):
        super().__init__()

        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def encode(self, src, src_mask):
        src_vectors = self.src_embed(src)
        return self.encoder(src_vectors, src_mask)
 
    def decode(self, tgt, memory, src_mask, tgt_mask):
        tgt_vectors = self.tgt_embed(tgt)
        return self.decoder( tgt_vectors, src_mask, tgt_mask,memory)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.generator(decoder_output)

        
