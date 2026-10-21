"""R5 Soft Prompt (prefix tuning) variants, kept separate from models_next.py
so the released main model file stays clean.

Soft prompting is a Transformer technique (learnable prefix embeddings); it
applies to NeuralLog, the next-event counterpart of the LogFed Transformer.
The RNN (DeepLog/LogAnomaly) and character-CNN (OneLog) models have no
analogous prefix mechanism and are returned unchanged.
"""
import torch
import torch.nn as nn

from models_next import NeuralLogModel, Positional, build_model as _build_base


class PromptNeuralLog(NeuralLogModel):
    """NeuralLog with P learnable soft-prompt vectors prepended to the
    embedded event sequence. Reuses the standard emb/encoder/fc; only the
    positional encoding is extended and a prompt parameter is added."""
    def __init__(self, n, prompt_len=5, seqlen=10, embed=64, heads=4,
                 ff=256, layers=2):
        super().__init__(n, seqlen=seqlen, embed=embed, heads=heads,
                         ff=ff, layers=layers)
        self.P = prompt_len
        self.pos = Positional(seqlen + prompt_len + 1, embed)
        if prompt_len > 0:
            self.prompt = nn.Parameter(torch.randn(prompt_len, embed) * 0.02)

    def forward(self, x):
        e = self.emb(x.long())
        if self.P > 0:
            p = self.prompt.unsqueeze(0).expand(x.size(0), -1, -1)
            e = torch.cat([p, e], dim=1)
        T = e.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        h = self.enc(self.pos(e), mask=mask)
        return self.fc(h[:, -1])


def build_model_pl(name, n, num_chars=130, msg_len=40, prompt_len=0):
    if name == "neurallog":
        return PromptNeuralLog(n, prompt_len=prompt_len)
    return _build_base(name, n, num_chars, msg_len)
