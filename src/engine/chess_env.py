import chess
import chess.pgn
import random
from typing import List, Optional

class ChessEnv:
    """
    Minimal chess environment around python-chess.
    Provides legal move lists and applies moves.
    """
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()

    def fen(self) -> str:
        return self.board.fen()

    def legal_moves(self) -> List[chess.Move]:
        return list(self.board.legal_moves)

    def push(self, move: chess.Move):
        self.board.push(move)

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def result(self) -> str:
        return self.board.result()

    def random_legal(self) -> chess.Move:
        return random.choice(self.legal_moves())

    def move_from_uci(self, u: str) -> Optional[chess.Move]:
        try:
            m = chess.Move.from_uci(u)
            if m in self.board.legal_moves:
                return m
            return None
        except Exception:
            return None
