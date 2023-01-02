import torch.nn as nn
import torch
from transfomer import (
    TransfomerEncoder,
    Transfomer,
    TransfomerDecoder,
    PositionalEncoding,
)


class WMTModel(nn.Module):
    def __init__(self, DE_size, EN_size, d_model,device=torch.device("cpu")):
        super().__init__()
        self.DE_embedding = nn.Embedding(DE_size, d_model)
        self.EN_embedding = nn.Embedding(EN_size, d_model)
        # 6 layers, 512 embedding size, 2048 expand linear dim, 8 heads, 64 heads dim, 0.1 dropout
        num_layers = 6
        heads = 8
        heads_dim = d_model // heads
        expand_dim = 4*d_model
        dropout = 0.1
        encoder = TransfomerEncoder(num_layers=num_layers, embed_size=d_model, expand_linear_dim=expand_dim, heads=heads, heads_dim=heads_dim, dropout=dropout) 
        decoder = TransfomerDecoder(num_layers=num_layers, embed_size=d_model, expand_linear_dim=expand_dim, heads=heads, heads_dim=heads_dim, dropout=dropout) 
        self.transformer = Transfomer(encoder, decoder, DE_size)
        self.pos_encoding = PositionalEncoding(d_model, 100, dropout)
        self.device = device

    def forward(self, src, trg, src_mask, trg_mask):
        src = src.to(self.device)
        trg = trg.to(self.device)
        src = self.pos_encoding(self.EN_embedding(src))
        trg = self.pos_encoding(self.DE_embedding(trg))
        return self.transformer(src, trg, src_mask, trg_mask)


def test_pass():
    model = WMTModel(100, 200, 512)
    x = torch.randint(0, 50, (10, 20))
    y = torch.randint(0, 100, (10, 20))
    print(
        model(x, y, Transfomer.make_src_mask(x, 0), Transfomer.make_trg_mask(y)).shape
    )


if __name__ == "__main__":
    test_pass()
