import torch.nn as nn
import torch
import numpy as np
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int, heads_dim: int):

        super().__init__()

        self.sofmax = nn.Softmax(dim=3)
        self.embed_size = embed_size
        self.heads = heads
        self.heads_dim = heads_dim
        self.attention_dim = heads_dim * heads
        self.keys = nn.Linear(embed_size, self.attention_dim)
        self.queries = nn.Linear(embed_size, self.attention_dim)
        self.values = nn.Linear(embed_size, self.attention_dim)
        self.head_projection = nn.Linear(self.attention_dim, embed_size)

    def forward(self, Q, K, V, mask):
        # 1) Get Queries/Keys/Values
        Q = self.queries(Q)
        K = self.keys(K)
        V = self.values(V)
        # (N, seq_len ,heads*head_dim)
        # If decoder then possibility of Q!=K=V else Q=K=V
        N, q_len, _ = Q.shape
        N, k_len, _ = K.shape
        Q = Q.reshape(N, q_len, self.heads, self.heads_dim)
        K = K.reshape(N, k_len, self.heads, self.heads_dim)
        V = V.reshape(N, k_len, self.heads, self.heads_dim)

        # 2) Compute Attention
        attention = torch.einsum("nqhd,nkhd->nhqk", [Q, K])
        # attention = (N, heads, q_len, k_len)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, float("-inf"))

        attention = self.sofmax(attention / self.embed_size**1 / 2)
        # V = (N, k_len, heads, heads_dim)

        attention = torch.einsum("nhqk, nkhd->nqhd", [attention, V])
        # attention = (N, q_len, heads, heads_dim * heads)

        # 3) Concat + project
        attention = attention.reshape(N, q_len, self.attention_dim)
        projected = self.head_projection(attention)
        # projected = (N, q_len, embed_dim)

        return projected


class ExpandFF(nn.Module):
    def __init__(self, embed_size: int, expand_linear_dim: int):
        super().__init__()
        self.expand_fn = nn.Linear(embed_size, expand_linear_dim)
        self.relu = nn.ReLU()
        self.reduce_fn = nn.Linear(expand_linear_dim, embed_size)

    def forward(self, x):
        out = self.expand_fn(x)
        out = self.relu(out)
        out = self.reduce_fn(out)
        return out


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embed_size: int,
        expand_linear_dim: int,
        heads: int,
        heads_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, heads, heads_dim)
        self.lnorm1 = nn.LayerNorm(embed_size)
        self.lnorm2 = nn.LayerNorm(embed_size)
        self.ff = ExpandFF(embed_size, expand_linear_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        # Self attention
        attention = self.attention(x, x, x, mask)
        # Residual
        attention = self.dropout(self.lnorm1(attention + x))

        # FF
        ff = self.ff(attention)
        ff = self.dropout(self.lnorm2(ff + attention))
        return ff


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_size: int,
        expand_linear_dim: int,
        heads: int,
        heads_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(embed_size, heads, heads_dim)
        self.enc_attention = MultiHeadAttention(embed_size, heads, heads_dim)
        self.lnorm1 = nn.LayerNorm(embed_size)
        self.lnorm2 = nn.LayerNorm(embed_size)
        self.lnorm3 = nn.LayerNorm(embed_size)
        self.ff = ExpandFF(embed_size, expand_linear_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, keys, values, self_mask, enc_mask):

        # Self attention
        self_attention = self.self_attention(x, x, x, self_mask)
        # Residual
        self_attention = self.dropout(self.lnorm1(self_attention + x))

        # Encoder attention
        attention = self.enc_attention(self_attention, keys, values, enc_mask)
        # Residual
        attention = self.dropout(self.lnorm2(attention + self_attention))

        # FF
        ff = self.ff(attention)
        ff = self.dropout(self.lnorm3(ff + attention))
        return ff


class TransfomerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_size: int,
        expand_linear_dim: int,
        heads: int,
        heads_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    self.embed_size, expand_linear_dim, heads, heads_dim, dropout
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransfomerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_size: int,
        expand_linear_dim: int,
        heads: int,
        heads_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    self.embed_size, expand_linear_dim, heads, heads_dim, dropout
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, enc_out, self_mask, target_mask):
        for layer in self.layers:
            # Changed order
            x = layer(x, enc_out, enc_out, target_mask, self_mask)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", self.calculate(self.d_model, self.max_len))

    def calculate(self, d_model: int, max_len: int = 5000):
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        # Pe(pos, 2i) = sin(pos/10000^(2i/d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        # Pe(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        s_len = x.shape[1]
        if s_len > self.max_len:
            self.max_len = s_len
            self.register_buffer("pe", self.calculate(self.d_model, self.max_len))

        x = x + self.pe[:, :s_len, :]
        return self.dropout(x)


class Transfomer(nn.Module):
    def __init__(
        self,
        enc: TransfomerEncoder,
        dec: TransfomerDecoder,
        trg_size,
    ):
        super().__init__()
        self.enc = enc
        self.dec = dec
        assert enc.embed_size == dec.embed_size
        self.trg_linear = nn.Linear(dec.embed_size, trg_size)

    def forward(self, enc_input, dec_input, src_mask, trg_mask):
        enc_out = self.enc(enc_input, src_mask)
        dec_out = self.dec(dec_input, enc_out, src_mask, trg_mask)
        dec_out = self.trg_linear(dec_out)
        return dec_out

    @staticmethod
    def make_src_mask(src, pad_enc):
        src_mask = (src != pad_enc).unsqueeze(1).unsqueeze(2)
        return src_mask

    @staticmethod
    def make_trg_mask(trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask


# Tests
def testMultiHeadSelfAttention():
    # Just check if no problem with dimensions
    heads = 8
    heads_dim = 2
    embed_size = 4
    MSA = MultiHeadAttention(embed_size, heads, heads_dim)
    # Input 4x3 aka 3 words
    x = torch.rand(1, 3, 4)
    # Self attention
    mask = torch.ones(1, 1, 1, 3)
    encoded = MSA(x, x, x, mask)

    x = torch.rand(1, 10, 4)
    mask = torch.ones(1, 1, 10, 3)
    decoded = MSA(x, encoded, encoded, mask)


def testPosEncoding():
    x = torch.rand([2, 8, 4])
    enc = PositionalEncoding(4, 1)
    enc(x)


def testTransfomer():
    # 1 batch of sent_len = 5 of d_model = 4
    x = torch.rand(4, 5, 4)
    y = torch.rand(4, 10, 4)
    encoder = TransfomerEncoder(1, 4, 10, 4, 4, 0.1)
    decoder = TransfomerDecoder(1, 4, 10, 8, 4, 0.1)
    trans = Transfomer(encoder, decoder, 10)
    src = torch.randint(0, 2, [4, 5])
    trg = torch.randint(0, 2, [4, 10])

    trans(x, y, Transfomer.make_src_mask(src, 0), Transfomer.make_trg_mask(trg))


if __name__ == "__main__":
    testMultiHeadSelfAttention()
    testPosEncoding()
    testTransfomer()
