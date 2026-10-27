import torch
import torch.nn as nn
import torch.functional as F

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
        
        super.__init__()
        self.d = embed_dim // num_heads
        self.h = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim) # junta os resultados em uma camada única
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, X: torch.Tensor)-> None:
        return X.view(X.size(0),X.size(1), self.h, self.d).transpose(1,2)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        q = self.split_heads(self.q_proj(query))
        k = self.split_heads(self.k_proj(value))
        v = self.split_heads(self.q_proj(key))

        scores = q @ k.transpose(2,3) / self.d**0.5 # cáculo da atenção (sem softmax)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -torch.inf) # zerar depois da aplicação do softmax 

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -torch.inf)

        weights = scores.softmax(dim=-1)

        Z = self.dropout(weights) @ v
        Z = Z.transpose(1,2)

        Z = Z.reshape(Z.size(0), Z.size(1))

        return (self.out_proj(Z), weights)


class PositionalEmbedding(nn.Module):

    def __init__(self, max_length, embed_dim, dropout):
        super.__init__()

        self.pos_embed = nn.Parameter(torch.randn(max_length, embed_dim) * 0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor)-> None:
        return self.dropout(X + self.pos_embed[:X.size(1)])
        


class TransformerEncoderLayer(nn.Module):
    pass

class TransformerDecoderLayer(nn.Module):
    pass

class NmtTransformer(nn.Module):
    pass