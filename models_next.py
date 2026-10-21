import math
import torch
import torch.nn as nn


class DeepLogModel(nn.Module):
    """DeepLog: stacked unidirectional LSTM, next-event prediction."""
    def __init__(self, n, embed=64, h=128, layers=2):
        super().__init__()
        self.emb = nn.Embedding(n, embed)
        self.lstm = nn.LSTM(embed, h, layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(h, n)

    def forward(self, x):
        o, _ = self.lstm(self.emb(x.long()))
        return self.fc(o[:, -1])


class LogAnomalyModel(nn.Module):
    """LogAnomaly: LSTM with additive self-attention over the sequence."""
    def __init__(self, n, embed=64, h=128):
        super().__init__()
        self.h = h
        self.emb = nn.Embedding(n, embed)
        self.lstm = nn.LSTM(embed, h, 1, batch_first=True)
        self.w = nn.Parameter(torch.empty(h, h))
        self.u = nn.Parameter(torch.zeros(h))
        nn.init.xavier_uniform_(self.w.unsqueeze(0))
        self.fc = nn.Linear(h * 2, n)

    def forward(self, x):
        B, L = x.shape
        o, _ = self.lstm(self.emb(x.long()))          # [B,L,h]
        last = o[:, -1]                                # [B,h]
        score = (torch.tanh(o.reshape(-1, self.h) @ self.w) @ self.u).view(B, L)
        wts = torch.softmax(score, 1).unsqueeze(-1)
        ctx = (wts * o).sum(1)                         # [B,h]
        return self.fc(torch.cat([last, ctx], 1))


class Positional(nn.Module):
    def __init__(self, maxlen, d):
        super().__init__()
        pe = torch.zeros(maxlen, d)
        pos = torch.arange(maxlen).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class NeuralLogModel(nn.Module):
    """NeuralLog: causal Transformer, next-event from the last position."""
    def __init__(self, n, seqlen=10, embed=64, heads=4, ff=256, layers=2):
        super().__init__()
        self.emb = nn.Embedding(n, embed)
        self.pos = Positional(seqlen, embed)
        layer = nn.TransformerEncoderLayer(embed, heads, ff, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.fc = nn.Linear(embed, n)

    def forward(self, x):
        L = x.size(1)
        mask = torch.triu(torch.ones(L, L, device=x.device), 1).bool()
        h = self.enc(self.pos(self.emb(x.long())), mask=mask)
        return self.fc(h[:, -1])


class OneLogModel(nn.Module):
    """OneLog: character-level hierarchical CNN (message then sequence),
    unified under next-event prediction. Characters are encoded directly from
    raw messages (no parser for features); Drain ids are used only as labels."""
    def __init__(self, n, num_chars=130, msg_len=40, dims=32):
        super().__init__()
        self.char_embedding = nn.Embedding(num_chars, dims, padding_idx=0)
        self.message_embedding = nn.Sequential(
            nn.Conv2d(dims, dims, (1, 3), stride=(1, 2), bias=False),
            nn.BatchNorm2d(dims), nn.ReLU(),
            nn.Conv2d(dims, dims, (1, 3), stride=(1, 2), bias=False),
            nn.BatchNorm2d(dims), nn.ReLU(),
            nn.Conv2d(dims, dims * 2, (1, 3), stride=(1, 2), bias=False),
            nn.BatchNorm2d(dims * 2), nn.ReLU(),
        )
        self.sequence_embedding = nn.Sequential(
            nn.Conv1d(dims * 2, dims * 2, 1, bias=False),
            nn.BatchNorm1d(dims * 2), nn.ReLU(),
            nn.Conv1d(dims * 2, dims * 3, 1, bias=False),
            nn.BatchNorm1d(dims * 3), nn.ReLU(),
            nn.Conv1d(dims * 3, dims * 4, 1, bias=False),
            nn.BatchNorm1d(dims * 4), nn.ReLU(),
        )
        self.classification = nn.Sequential(
            nn.Linear(dims * 4, dims * 2), nn.ReLU(),
            nn.Linear(dims * 2, dims), nn.ReLU(),
            nn.Linear(dims, n),
        )

    def forward(self, x):
        # x: [B, HIST, MSG_LEN] character indices
        h = self.char_embedding(x.long()).permute(0, 3, 1, 2)
        h = self.message_embedding(h)
        h, _ = torch.max(h, dim=-1)                 # [B, dims*2, HIST]
        h = self.sequence_embedding(h)
        h, _ = torch.max(h, dim=-1)                 # [B, dims*4]
        return self.classification(h)


def build_model(name, n, num_chars=130, msg_len=40):
    if name == "deeplog":
        return DeepLogModel(n)
    if name == "loganomaly":
        return LogAnomalyModel(n)
    if name == "neurallog":
        return NeuralLogModel(n)
    if name == "onelog":
        return OneLogModel(n, num_chars, msg_len)
    raise ValueError(name)
