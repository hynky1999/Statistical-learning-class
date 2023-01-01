import torch.nn as nn
import torch
from transfomer import (
    TransfomerEncoder,
    Transfomer,
    TransfomerDecoder,
    PositionalEncoding,
)


class WMTModel(nn.Module):
    def __init__(self, DE_size, EN_size, d_model):
        super().__init__()
        self.DE_embedding = nn.Embedding(DE_size, d_model)
        self.EN_embedding = nn.Embedding(EN_size, d_model)
        # 6 layers, 512 embedding size, 2048 expand linear dim, 8 heads, 64 heads dim, 0.1 dropout
        encoder = TransfomerEncoder(6, d_model, 2048, 8, 64, 0.1)
        decoder = TransfomerDecoder(6, d_model, 2048, 8, 64, 0.1)
        self.transformer = Transfomer(encoder, decoder, DE_size)
        self.pos_encoding = PositionalEncoding(d_model, 100, 0.1)

    def forward(self, src, trg, src_mask, trg_mask):
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
