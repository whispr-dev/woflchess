import chess
import torch

# Board encoding: 12 piece planes + 1 stm + 4 castling + 1 ep = 18 planes (8x8)
# Planes order: [P,N,B,R,Q,K, p,n,b,r,q,k] then stm, castling KQkq, ep
def encode_board(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros(18, 8, 8, dtype=torch.float32)
    piece_map = [
        (chess.PAWN,   0),
        (chess.KNIGHT, 1),
        (chess.BISHOP, 2),
        (chess.ROOK,   3),
        (chess.QUEEN,  4),
        (chess.KING,   5),
    ]
    for piece, idx in piece_map:
        for sq in board.pieces(piece, chess.WHITE):
            r, c = divmod(sq, 8)
            planes[idx, 7 - r, c] = 1.0
        for sq in board.pieces(piece, chess.BLACK):
            r, c = divmod(sq, 8)
            planes[idx + 6, 7 - r, c] = 1.0

    # side to move
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # castling rights
    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # ep file (if any)
    if board.ep_square is not None:
        r, c = divmod(board.ep_square, 8)
        planes[17, 7 - r, c] = 1.0

    return planes  # (18,8,8)

# Simple move tokenization: uci string to integer id in [0, 4096*73) approx
# We compress (from,to,promo) to a single id deterministically.
_PROMO_MAP = {None:0, chess.QUEEN:1, chess.ROOK:2, chess.BISHOP:3, chess.KNIGHT:4}

def move_to_token(move: chess.Move) -> int:
    frm = move.from_square
    to = move.to_square
    promo = _PROMO_MAP.get(move.promotion, 0)
    return frm * 64 * 5 + to * 5 + promo

def token_to_move(token: int) -> chess.Move:
    frm = token // (64*5)
    rem = token % (64*5)
    to = rem // 5
    promo_idx = rem % 5
    promo = None
    for k,v in _PROMO_MAP.items():
        if v == promo_idx:
            promo = k
            break
    return chess.Move(frm, to, promotion=promo)
