import torch
import torch.nn as nn
import torch.optim as optim
import random
import chess
from .encoding import encode_board, move_to_token
from .nets import CNNNet, RNNNet, LSTMNet, Discriminator, PolicyHead
from .ganglion import Ganglion

class Oracle:
    """
    Optional Stockfish-backed oracle; falls back to a simple material+mobility heuristic.
    """
    def __init__(self):
        self.use_stockfish = False
        try:
            from stockfish import Stockfish  # optional
            self.sf = Stockfish(path=None)   # auto-find in PATH if available
            self.use_stockfish = self.sf.is_alive()
        except Exception:
            self.sf = None

    def pick(self, board: chess.Board, moves):
        if self.use_stockfish:
            self.sf.set_fen_position(board.fen())
            best = self.sf.get_best_move()
            if best:
                try:
                    m = chess.Move.from_uci(best)
                    if m in moves:
                        return m
                except Exception:
                    pass
        # fallback heuristic: prefer captures and checks, else first
        captures = [m for m in moves if board.is_capture(m)]
        if captures:
            return captures[0]
        checks = []
        for m in moves:
            board.push(m)
            if board.is_check():
                checks.append(m)
            board.pop()
        if checks:
            return checks[0]
        return moves[0]

class TriGANBrain(nn.Module):
    """
    Three generators (CNN/RNN/LSTM) + one discriminator.
    Fused via ganglion+CA bus. Produces policy scores for legal moves.
    """
    def __init__(self, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.cnn = CNNNet().to(self.device)
        self.rnn = RNNNet().to(self.device)
        self.lstm = LSTMNet().to(self.device)
        self.disc = Discriminator().to(self.device)
        self.ph = PolicyHead().to(self.device)
        self.ganglion = Ganglion(device=self.device)

    def forward_latents(self, board_planes, hist_tokens):
        z_cnn = self.cnn(board_planes)
        z_rnn = self.rnn(hist_tokens)
        z_lstm = self.lstm(hist_tokens)

        # Write each latent into CA on separate channels then tick+read:
        self.ganglion.tick()
        def vec_to_patch(z):
            # map 512 vector to 8x8 patch by reshape (fill row-major)
            patch = z.view(-1, 8, 8)[:1]  # use first 64 dims; if <64, pad
            if patch.numel() < 64:
                pad = torch.zeros(1, 8, 8, device=z.device)
                pad.view(-1)[:z.numel()] = z
                patch = pad
            return patch[0]

        self.ganglion.write_signal(0, vec_to_patch(z_cnn[0]).detach(), 0, 0)
        self.ganglion.write_signal(1, vec_to_patch(z_rnn[0]).detach(), 4, 4)
        self.ganglion.write_signal(2, vec_to_patch(z_lstm[0]).detach(), 8 % self.ganglion.bus.H, 8 % self.ganglion.bus.W)

        ca_read = self.ganglion.read_signal(0,0,8,8)  # (C,h,w)
        ca_pool = ca_read.mean(dim=(1,2))             # (C,)
        # fuse: simple weighted sum with bleed awareness
        bleed = self.ganglion.bleed_level()
        fused = (1-bleed) * (z_cnn + z_rnn + z_lstm)/3.0 + bleed * ca_pool[:512].unsqueeze(0).expand_as(z_cnn)
        return z_cnn, z_rnn, z_lstm, fused

    def score_moves(self, fused_latent, cand_latents):
        return self.ph(fused_latent, cand_latents)

class TriGANTrainer:
    def __init__(self, brain: TriGANBrain, lr=1e-3, device=None):
        self.brain = brain
        self.device = device or brain.device
        params = list(brain.cnn.parameters()) + list(brain.rnn.parameters()) + list(brain.lstm.parameters()) + list(brain.ph.parameters())
        self.opt_g = optim.Adam(params, lr=lr)
        self.opt_d = optim.Adam(brain.disc.parameters(), lr=lr)
        self.bce = nn.BCEWithLogitsLoss()
        self.oracle = Oracle()

    def _candidate_latents(self, z_cnn, z_rnn, z_lstm, board, legal_moves):
        # Represent candidates by concatenating per-net latents projected to 512 dims
        # Here: simple re-use (could add move-conditioned features later)
        B = 1
        N = len(legal_moves)
        # broadcast latents and add a tiny move-id embedding so moves are distinguishable
        ids = torch.arange(N, device=self.device).float().unsqueeze(1)  # (N,1)
        proj = torch.cat([z_cnn, z_rnn, z_lstm], dim=1)  # (B,1536)
        W = torch.nn.functional.normalize(torch.randn(1536, 512, device=self.device), dim=0)
        base = proj @ W  # (B,512)
        base = base.repeat(N, 1) + torch.tanh(ids)  # (N,512)
        return base.unsqueeze(0)  # (B,N,512)

    def train_step(self, board):
        board_planes = encode_board(board).unsqueeze(0).to(self.device)  # (1,18,8,8)
        # build tiny history tokens: use last 8 ply from stack
        hist = list(board.move_stack)[-8:]
        import chess
        tokens = [0]*max(1, 8 - len(hist))
        for m in hist:
            tokens.append((m.from_square * 64 * 5) + (m.to_square * 5))
        hist_tokens = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

        z_cnn, z_rnn, z_lstm, fused = self.brain.forward_latents(board_planes, hist_tokens)
        legal = list(board.legal_moves)
        if not legal:
            return 0.0, 0.0

        cand_lat = self._candidate_latents(z_cnn, z_rnn, z_lstm, board, legal)
        scores = self.brain.score_moves(fused, cand_lat)  # (1,N)
        # generator chooses top
        top_idx = int(torch.argmax(scores, dim=1).item())
        gen_move = legal[top_idx]

        # oracle label
        oracle_move = self.oracle.pick(board, legal)
        is_oracle_like = torch.tensor([1.0 if gen_move == oracle_move else 0.0], device=self.device)

        # Discriminator: input fused and board; label 1 for oracle move context, 0 for generated
        # Build fused_hist for both samples
        fused_hist = fused.detach()
        # D(gen)
        d_gen = self.brain.disc(board_planes, fused_hist)
        # D(oracle): perturb fused towards oracle by a small delta so D sees a slightly different point
        fused_oracle = fused_hist + 0.05 * torch.randn_like(fused_hist)
        d_oracle = self.brain.disc(board_planes, fused_oracle)

        # Train D
        self.opt_d.zero_grad()
        loss_d = self.bce(d_oracle, torch.ones_like(d_oracle)) + self.bce(d_gen, torch.zeros_like(d_gen))
        loss_d.backward()
        self.opt_d.step()

        # Train G (three nets + policy head): make D think gen is oracle
        self.opt_g.zero_grad()
        z_cnn2, z_rnn2, z_lstm2, fused2 = self.brain.forward_latents(board_planes, hist_tokens)
        cand_lat2 = self._candidate_latents(z_cnn2, z_rnn2, z_lstm2, board, legal)
        scores2 = self.brain.score_moves(fused2, cand_lat2)
        top_idx2 = int(torch.argmax(scores2, dim=1).item())
        # Encourage picking oracle move via margin; if oracle not in top, push its score up
        oracle_idx = legal.index(oracle_move)
        margin = 0.5
        gen_score = scores2[0, top_idx2]
        oracle_score = scores2[0, oracle_idx]
        # hinge-style:
        loss_policy = torch.relu(margin - (oracle_score - gen_score))

        # fool D:
        d_gen2 = self.brain.disc(board_planes, fused2)
        loss_adv = self.bce(d_gen2, torch.ones_like(d_gen2))

        (loss_policy + loss_adv).backward()
        self.opt_g.step()

        return float(loss_d.item()), float((loss_policy + loss_adv).item())
