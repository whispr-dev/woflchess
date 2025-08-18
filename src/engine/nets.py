import torch
import torch.nn as nn

class CNNNet(nn.Module):
    def __init__(self, in_planes=18, channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.head = nn.Linear(channels, 512)  # latent out

    def forward(self, x):  # x: (B,18,8,8)
        z = self.net(x).flatten(1)
        return self.head(z)  # (B,512)

class RNNNet(nn.Module):
    def __init__(self, vocab=64*64*5, emb=64, hidden=128, layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.rnn = nn.GRU(emb, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 512)

    def forward(self, tokens):  # (B,T)
        e = self.emb(tokens)
        out, _ = self.rnn(e)
        h = out[:, -1, :]  # last state
        return self.head(h)  # (B,512)

class LSTMNet(nn.Module):
    def __init__(self, vocab=64*64*5, emb=64, hidden=128, layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 512)

    def forward(self, tokens):  # (B,T)
        e = self.emb(tokens)
        out, _ = self.lstm(e)
        h = out[:, -1, :]
        return self.head(h)

class Discriminator(nn.Module):
    """
    Scores (board, move) pairs. Higher = looks like oracle.
    """
    def __init__(self, board_planes=18, hist_dim=512):
        super().__init__()
        self.board_conv = nn.Sequential(
            nn.Conv2d(board_planes, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Sequential(
            nn.Linear(64 + hist_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, board_planes, fused_hist):  # (B,18,8,8), (B,512)
        b = self.board_conv(board_planes).flatten(1)
        x = torch.cat([b, fused_hist], dim=1)
        return self.fc(x)  # (B,1)

class PolicyHead(nn.Module):
    """
    Maps fused 512-d latent to per-move score over a candidate set.
    We score provided legal moves (masking).
    """
    def __init__(self, hidden=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1)  # per-move scorer applied to each candidate
        )

    def forward(self, fused_latent, candidate_feats):  # fused: (B,512), feats: (B,N,512)
        # broadcast fused to (B,N,512)
        B, N, _ = candidate_feats.shape
        fused = fused_latent.unsqueeze(1).expand(B, N, fused_latent.shape[-1])
        z = fused + candidate_feats  # simple interaction
        scores = self.fc(z).squeeze(-1)  # (B,N)
        return scores
