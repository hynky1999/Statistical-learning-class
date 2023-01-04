import torch.nn as nn
import torch
from transfomer import (
    TransfomerEncoder,
    Transformer,
    TransfomerDecoder,
    PositionalEncoding,
)


class WMTModel(nn.Module):
    def __init__(
        self,
        EN_size,
        DE_size,
        d_model,
        device=torch.device("cpu"),
        num_layers=6,
        heads=8,
        expand=4,
        dropout=0.1,
    ):
        super().__init__()
        self.DE_embedding = nn.Embedding(DE_size, d_model)
        self.EN_embedding = nn.Embedding(EN_size, d_model)
        # 6 layers, 512 embedding size, 2048 expand linear dim, 8 heads, 64 heads dim, 0.1 dropout
        num_layers = num_layers
        heads = heads
        heads_dim = d_model // heads
        expand_dim = expand * d_model
        dropout = dropout
        encoder = TransfomerEncoder(
            num_layers=num_layers,
            embed_size=d_model,
            expand_linear_dim=expand_dim,
            heads=heads,
            heads_dim=heads_dim,
            dropout=dropout,
        )
        decoder = TransfomerDecoder(
            num_layers=num_layers,
            embed_size=d_model,
            expand_linear_dim=expand_dim,
            heads=heads,
            heads_dim=heads_dim,
            dropout=dropout,
        )
        self.transformer = Transformer(encoder, decoder, DE_size)
        self.pos_encoding = PositionalEncoding(d_model, 100, dropout, device=device)
        self.device = device

    def forward(self, src, trg, src_mask, trg_mask):
        src = src.to(self.device)
        trg = trg.to(self.device)
        src_mask = src_mask.to(self.device)
        trg_mask = trg_mask.to(self.device)

        src = self.pos_encoding(self.EN_embedding(src))
        trg = self.pos_encoding(self.DE_embedding(trg))
        return self.transformer(src, trg, src_mask, trg_mask)

    def predict(self, src, src_mask, start_token, end_token, max_len=100):
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)

        src = self.pos_encoding(self.EN_embedding(src))
        enc_output = self.transformer.encode(src, src_mask)
        trg = torch.zeros((src.shape[0], 1)).long().to(self.device)
        trg[:, 0] = start_token
        trg_embedding = self.pos_encoding(self.DE_embedding(trg)).to(self.device)
        ended_idx = set()
        for i in range(max_len):
            trg_mask = Transformer.make_trg_mask(
                torch.ones([trg.shape[0], 1, 1, trg.shape[1]]) == 1
            ).to(self.device)
            out = self.transformer.decode(trg_embedding, enc_output, trg_mask, src_mask)
            pred = out[:, -1].argmax(1).unsqueeze(1)
            pred_embedding = self.pos_encoding(self.DE_embedding(pred))
            trg = torch.cat((trg, pred), dim=1)
            trg_embedding = torch.cat((trg_embedding, pred_embedding), dim=1)

            ended_idx.union((pred == end_token).nonzero())
            if len(ended_idx) == src.shape[0]:
                break

        return trg


def test_pass():
    model = WMTModel(100, 200, 512)
    x = torch.randint(0, 50, (10, 20))
    y = torch.randint(0, 100, (10, 20))
    print(
        model(x, y, Transformer.make_src_mask(x, 0), Transformer.make_trg_mask(y)).shape
    )


if __name__ == "__main__":
    test_pass()
