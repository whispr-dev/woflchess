import torch
import chess
from .encoding import encode_board
from .tri_gan import TriGANBrain
from .utils import device_select

class Agent:
    def __init__(self, weights_path=None, temperature=0.9):
        self.device = device_select()
        self.brain = TriGANBrain(device=self.device)
        self.temperature = temperature
        if weights_path:
            self.brain.load_state_dict(torch.load(weights_path, map_location=self.device))

    def select_move(self, board: chess.Board) -> chess.Move:
        board_planes = encode_board(board).unsqueeze(0).to(self.device)
        # hist tokens
        hist = list(board.move_stack)[-8:]
        tokens = [0]*max(1, 8 - len(hist))
        for m in hist:
            tokens.append((m.from_square * 64 * 5) + (m.to_square * 5))
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)

        z_cnn, z_rnn, z_lstm, fused = self.brain.forward_latents(board_planes, tokens)
        legal = list(board.legal_moves)
        if not legal:
            raise RuntimeError("No legal moves")

        # simplistic candidate latents same as trainer’s helper (without importing it)
        B = 1
        N = len(legal)
        ids = torch.arange(N, device=self.device).float().unsqueeze(1)
        proj = torch.cat([z_cnn, z_rnn, z_lstm], dim=1)
        W = torch.nn.functional.normalize(torch.randn(1536, 512, device=self.device), dim=0)
        base = (proj @ W).repeat(N, 1) + torch.tanh(ids)
        cand_lat = base.unsqueeze(0)

        scores = self.brain.score_moves(fused, cand_lat).squeeze(0)  # (N,)
        # softmax sample with temperature
        probs = torch.softmax(scores / max(1e-3, self.temperature), dim=0).detach().cpu().numpy()
        import numpy as np
        idx = int(np.random.choice(len(legal), p=probs))
        return legal[idx]
