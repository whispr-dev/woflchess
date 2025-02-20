//////////////////////////
// game.rs
//////////////////////////

use std::fmt;
use std::sync::{Arc, Mutex};

use crate::types::*;
use crate::neural::ChessNeuralEngine;

#[derive(Clone)]
pub struct GameState {
    pub board: Board,
    pub current_turn: Color,
    pub white_can_castle_kingside: bool,
    pub white_can_castle_queenside: bool,
    pub black_can_castle_kingside: bool,
    pub black_can_castle_queenside: bool,
    pub last_pawn_double_move: Option<(usize, usize)>,
    pub neural_engine: Option<Arc<Mutex<ChessNeuralEngine>>>,
    pub white_king_pos: (usize, usize),
    pub black_king_pos: (usize, usize),
    pub move_history: Vec<Move>,
}

impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "\n  a b c d e f g h")?;
        writeln!(f, "  ---------------")?;
        // ranks from top (0) to bottom (7) but displayed 8 down to 1
        for rank in 0..8 {
            write!(f, "{} ", 8 - rank)?;
            for file in 0..8 {
                let piece = match &self.board[file][rank] {
                    Some(p) => match (p.piece_type, p.color) {
                        (PieceType::Pawn, Color::White) => "♙",
                        (PieceType::Pawn, Color::Black) => "♟",
                        (PieceType::Knight, Color::White) => "♘",
                        (PieceType::Knight, Color::Black) => "♞",
                        (PieceType::Bishop, Color::White) => "♗",
                        (PieceType::Bishop, Color::Black) => "♝",
                        (PieceType::Rook, Color::White) => "♖",
                        (PieceType::Rook, Color::Black) => "♜",
                        (PieceType::Queen, Color::White) => "♕",
                        (PieceType::Queen, Color::Black) => "♛",
                        (PieceType::King, Color::White) => "♔",
                        (PieceType::King, Color::Black) => "♚",
                    },
                    None => "·",
                };
                write!(f, "{} ", piece)?;
            }
            writeln!(f, "{}", 8 - rank)?;
        }
        writeln!(f, "  ---------------")?;
        writeln!(f, "  a b c d e f g h")?;
        write!(f, "\nTurn: {}", self.current_turn)
    }
}

impl GameState {
    pub fn new() -> Self {
        let mut new_state = GameState {
            board: [[None; 8]; 8],
            current_turn: Color::White,
            white_can_castle_kingside: true,
            white_can_castle_queenside: true,
            black_can_castle_kingside: true,
            black_can_castle_queenside: true,
            last_pawn_double_move: None,
            neural_engine: Some(Arc::new(Mutex::new(ChessNeuralEngine::new()))),
            white_king_pos: (4, 7),
            black_king_pos: (4, 0),
            move_history: Vec::new(),
        };
        new_state.setup_initial_position();
        new_state
    }

    pub fn setup_initial_position(&mut self) {
        let create_piece = |pt, color| Some(Piece { piece_type: pt, color });

        // Black pieces (top: ranks 0,1)
        self.board[0][0] = create_piece(PieceType::Rook, Color::Black);
        self.board[1][0] = create_piece(PieceType::Knight, Color::Black);
        self.board[2][0] = create_piece(PieceType::Bishop, Color::Black);
        self.board[3][0] = create_piece(PieceType::Queen, Color::Black);
        self.board[4][0] = create_piece(PieceType::King, Color::Black);
        self.board[5][0] = create_piece(PieceType::Bishop, Color::Black);
        self.board[6][0] = create_piece(PieceType::Knight, Color::Black);
        self.board[7][0] = create_piece(PieceType::Rook, Color::Black);

        for i in 0..8 {
            self.board[i][1] = create_piece(PieceType::Pawn, Color::Black);
        }

        // White pieces (bottom: ranks 6,7)
        self.board[0][7] = create_piece(PieceType::Rook, Color::White);
        self.board[1][7] = create_piece(PieceType::Knight, Color::White);
        self.board[2][7] = create_piece(PieceType::Bishop, Color::White);
        self.board[3][7] = create_piece(PieceType::Queen, Color::White);
        self.board[4][7] = create_piece(PieceType::King, Color::White);
        self.board[5][7] = create_piece(PieceType::Bishop, Color::White);
        self.board[6][7] = create_piece(PieceType::Knight, Color::White);
        self.board[7][7] = create_piece(PieceType::Rook, Color::White);

        for i in 0..8 {
            self.board[i][6] = create_piece(PieceType::Pawn, Color::White);
        }

        // Middle is empty
        for i in 0..8 {
            for j in 2..6 {
                self.board[i][j] = None;
            }
        }

        // King positions
        self.black_king_pos = (4, 0);
        self.white_king_pos = (4, 7);
    }

    pub fn validate_board_state(&self) -> Result<(), String> {
        let mut white_kings = 0;
        let mut black_kings = 0;

        for file in 0..8 {
            for rank in 0..8 {
                if let Some(piece) = self.board[file][rank] {
                    if piece.piece_type == PieceType::King {
                        match piece.color {
                            Color::White => white_kings += 1,
                            Color::Black => black_kings += 1,
                        }
                    }
                }
            }
        }
        if white_kings != 1 || black_kings != 1 {
            return Err(format!(
                "Invalid number of kings: white={}, black={}",
                white_kings, black_kings
            ));
        }
        Ok(())
    }

    pub fn evaluate_position(&self) -> i32 {
        let pawn_value = 100;
        let knight_value = 320;
        let bishop_value = 330;
        let rook_value = 500;
        let queen_value = 900;
        let king_value = 20000;

        let mut score = 0;
        for file in 0..8 {
            for rank in 0..8 {
                if let Some(piece) = self.board[file][rank] {
                    let piece_val = match piece.piece_type {
                        PieceType::Pawn => pawn_value,
                        PieceType::Knight => knight_value,
                        PieceType::Bishop => bishop_value,
                        PieceType::Rook => rook_value,
                        PieceType::Queen => queen_value,
                        PieceType::King => king_value,
                    };
                    let sign = if piece.color == Color::White { 1 } else { -1 };
                    score += sign * piece_val;
                }
            }
        }
        score
    }

    pub fn is_valid_move(&self, mv: &Move) -> Result<(), MoveError> {
        // Basic checks
        self.is_valid_basic_move(mv)?;

        // Make a temporary copy for deeper checks
        let mut test_state = self.clone();
        test_state.make_move_without_validation(mv);

        // If the king is still safe
        let king_pos = if mv.piece_moved.piece_type == PieceType::King {
            mv.to
        } else if mv.piece_moved.color == Color::White {
            test_state.white_king_pos
        } else {
            test_state.black_king_pos
        };
        // Check if king is attacked
        if test_state.is_square_attacked(king_pos, mv.piece_moved.color.opposite()) {
            return Err(MoveError::WouldCauseCheck);
        }
        Ok(())
    }

    fn is_valid_basic_move(&self, mv: &Move) -> Result<(), MoveError> {
        let (fx, fy) = mv.from;
        let (tx, ty) = mv.to;
        if fx >= 8 || fy >= 8 || tx >= 8 || ty >= 8 {
            return Err(MoveError::OutOfBounds);
        }
        let piece = match self.board[fx][fy] {
            Some(p) => p,
            None => return Err(MoveError::NoPieceAtSource),
        };
        if piece.color != self.current_turn {
            return Err(MoveError::WrongColor);
        }

        match piece.piece_type {
            PieceType::Pawn => self.is_valid_pawn_move(mv, piece.color),
            PieceType::Knight => self.is_valid_knight_move(mv),
            PieceType::Bishop => self.is_valid_bishop_move(mv),
            PieceType::Rook => self.is_valid_rook_move(mv),
            PieceType::Queen => self.is_valid_queen_move(mv),
            PieceType::King => self.is_valid_king_move(mv),
        }
    }

    pub fn is_valid_pawn_move(&self, mv: &Move, color: Color) -> Result<(), MoveError> {
        let (fx, fy) = mv.from;
        let (tx, ty) = mv.to;
        let direction = if color == Color::White { -1i32 } else { 1i32 };
        let start_rank = if color == Color::White { 6 } else { 1 };

        let dx = tx as i32 - fx as i32;
        let dy = ty as i32 - fy as i32;

        // Single step
        if dx == 0 && dy == direction {
            if self.get_piece_at(mv.to).is_none() {
                return Ok(());
            }
            return Err(MoveError::InvalidPieceMove("Pawn path blocked".to_string()));
        }
        // Double step
        if dx == 0 && dy == 2 * direction && fy == start_rank {
            let between = (fx, (fy as i32 + direction) as usize);
            if self.get_piece_at(mv.to).is_none() && self.get_piece_at(between).is_none() {
                return Ok(());
            }
            return Err(MoveError::InvalidPieceMove("Pawn double-step blocked".to_string()));
        }
        // Captures
        if dy == direction && dx.abs() == 1 {
            if let Some(captured_piece) = self.get_piece_at((tx, ty)) {
                if captured_piece.color != color {
                    return Ok(());
                } else {
                    return Err(MoveError::InvalidPieceMove(
                        "Cannot capture your own piece".to_string(),
                    ));
                }
            }
            // En passant
            if mv.is_en_passant {
                if let Some(last_move) = self.last_pawn_double_move {
                    if tx == last_move.0 && fy == last_move.1 {
                        return Ok(());
                    }
                }
            }
        }

        Err(MoveError::InvalidPieceMove("Invalid pawn move".to_string()))
    }

    fn is_valid_knight_move(&self, mv: &Move) -> Result<(), MoveError> {
        let dx = (mv.to.0 as i32 - mv.from.0 as i32).abs();
        let dy = (mv.to.1 as i32 - mv.from.1 as i32).abs();
        if (dx == 2 && dy == 1) || (dx == 1 && dy == 2) {
            Ok(())
        } else {
            Err(MoveError::InvalidPieceMove("Invalid knight move".to_string()))
        }
    }

    fn is_valid_bishop_move(&self, mv: &Move) -> Result<(), MoveError> {
        let (fx, fy) = mv.from;
        let (tx, ty) = mv.to;
        let dx = (tx as i32 - fx as i32).abs();
        let dy = (ty as i32 - fy as i32).abs();
        if dx == dy && dx > 0 {
            if self.is_path_clear(mv.from, mv.to) {
                if let Some(dest_piece) = self.board[tx][ty] {
                    if dest_piece.color == mv.piece_moved.color {
                        return Err(MoveError::InvalidPieceMove(
                            "Cannot capture your own piece".to_string(),
                        ));
                    }
                }
                Ok(())
            } else {
                Err(MoveError::BlockedPath)
            }
        } else {
            Err(MoveError::InvalidPieceMove("Bishop must move diagonally".to_string()))
        }
    }

    fn is_valid_rook_move(&self, mv: &Move) -> Result<(), MoveError> {
        let dx = (mv.to.0 as i32 - mv.from.0 as i32).abs();
        let dy = (mv.to.1 as i32 - mv.from.1 as i32).abs();
        if (dx == 0 && dy != 0) || (dy == 0 && dx != 0) {
            if self.is_path_clear(mv.from, mv.to) {
                if let Some(dest_piece) = self.board[mv.to.0][mv.to.1] {
                    if dest_piece.color == mv.piece_moved.color {
                        return Err(MoveError::InvalidPieceMove(
                            "Cannot capture your own piece".to_string(),
                        ));
                    }
                }
                Ok(())
            } else {
                Err(MoveError::BlockedPath)
            }
        } else {
            Err(MoveError::InvalidPieceMove("Invalid rook move".to_string()))
        }
    }

    fn is_valid_queen_move(&self, mv: &Move) -> Result<(), MoveError> {
        // Queen can move as bishop or rook
        if self.is_valid_bishop_move(mv).is_ok() || self.is_valid_rook_move(mv).is_ok() {
            Ok(())
        } else {
            Err(MoveError::InvalidPieceMove("Invalid queen move".to_string()))
        }
    }

    fn is_valid_king_move(&self, mv: &Move) -> Result<(), MoveError> {
        let dx = (mv.to.0 as i32 - mv.from.0 as i32).abs();
        let dy = (mv.to.1 as i32 - mv.from.1 as i32).abs();

        // Normal king move
        if dx <= 1 && dy <= 1 {
            if let Some(dest_piece) = self.get_piece_at(mv.to) {
                if dest_piece.color == mv.piece_moved.color {
                    return Err(MoveError::InvalidPieceMove(
                        "Cannot capture your own piece".to_string(),
                    ));
                }
            }
            return Ok(());
        }

        // Castling
        if mv.is_castling && dy == 0 && dx == 2 {
            let (orig_pos, can_castle_kingside, can_castle_queenside) = match mv.piece_moved.color {
                Color::White => (
                    (4, 7),
                    self.white_can_castle_kingside,
                    self.white_can_castle_queenside,
                ),
                Color::Black => (
                    (4, 0),
                    self.black_can_castle_kingside,
                    self.black_can_castle_queenside,
                ),
            };
            if mv.from != orig_pos {
                return Err(MoveError::KingNotInOriginalPosition);
            }
            // Not in check
            if self.is_in_check(mv.piece_moved.color) {
                return Err(MoveError::CastlingInCheck);
            }

            match mv.piece_moved.color {
                Color::White => {
                    if mv.to == (6, 7) {
                        // White kingside
                        if !can_castle_kingside {
                            return Err(MoveError::CastlingRightsLost);
                        }
                        if !self.is_path_clear((4, 7), (7, 7)) {
                            return Err(MoveError::CastlingPathBlocked);
                        }
                        if self.board[7][7] != Some(Piece { piece_type: PieceType::Rook, color: Color::White }) {
                            return Err(MoveError::CastlingRookMissing);
                        }
                        for x in 4..=6 {
                            if self.is_square_attacked((x, 7), Color::Black) {
                                return Err(MoveError::CastlingThroughCheck);
                            }
                        }
                    } else if mv.to == (2, 7) {
                        // White queenside
                        if !can_castle_queenside {
                            return Err(MoveError::CastlingRightsLost);
                        }
                        if !self.is_path_clear((4, 7), (0, 7)) {
                            return Err(MoveError::CastlingPathBlocked);
                        }
                        if self.board[0][7] != Some(Piece { piece_type: PieceType::Rook, color: Color::White }) {
                            return Err(MoveError::CastlingRookMissing);
                        }
                        for x in 2..=4 {
                            if self.is_square_attacked((x, 7), Color::Black) {
                                return Err(MoveError::CastlingThroughCheck);
                            }
                        }
                    }
                }
                Color::Black => {
                    if mv.to == (6, 0) {
                        // Black kingside
                        if !can_castle_kingside {
                            return Err(MoveError::CastlingRightsLost);
                        }
                        if !self.is_path_clear((4, 0), (7, 0)) {
                            return Err(MoveError::CastlingPathBlocked);
                        }
                        if self.board[7][0] != Some(Piece { piece_type: PieceType::Rook, color: Color::Black }) {
                            return Err(MoveError::CastlingRookMissing);
                        }
                        for x in 4..=6 {
                            if self.is_square_attacked((x, 0), Color::White) {
                                return Err(MoveError::CastlingThroughCheck);
                            }
                        }
                    } else if mv.to == (2, 0) {
                        // Black queenside
                        if !can_castle_queenside {
                            return Err(MoveError::CastlingRightsLost);
                        }
                        if !self.is_path_clear((4, 0), (0, 0)) {
                            return Err(MoveError::CastlingPathBlocked);
                        }
                        if self.board[0][0] != Some(Piece { piece_type: PieceType::Rook, color: Color::Black }) {
                            return Err(MoveError::CastlingRookMissing);
                        }
                        for x in 2..=4 {
                            if self.is_square_attacked((x, 0), Color::White) {
                                return Err(MoveError::CastlingThroughCheck);
                            }
                        }
                    }
                }
            }
            return Ok(());
        }

        Err(MoveError::InvalidPieceMove("Invalid king move".to_string()))
    }

    fn is_path_clear(&self, from: (usize, usize), to: (usize, usize)) -> bool {
        let (fx, fy) = (from.0 as i32, from.1 as i32);
        let (tx, ty) = (to.0 as i32, to.1 as i32);

        let dx = (tx - fx).signum();
        let dy = (ty - fy).signum();

        let mut x = fx + dx;
        let mut y = fy + dy;

        while (x, y) != (tx, ty) {
            if x < 0 || x >= 8 || y < 0 || y >= 8 {
                return false;
            }
            if self.board[x as usize][y as usize].is_some() {
                return false;
            }
            x += dx;
            y += dy;
        }
        true
    }

    pub fn get_piece_at(&self, pos: (usize, usize)) -> Option<&Piece> {
        if Self::is_within_bounds(pos) {
            self.board[pos.0][pos.1].as_ref()
        } else {
            None
        }
    }

    pub fn is_within_bounds(pos: (usize, usize)) -> bool {
        pos.0 < 8 && pos.1 < 8
    }

    pub fn is_in_check(&self, color: Color) -> bool {
        let king_pos = if color == Color::White {
            self.white_king_pos
        } else {
            self.black_king_pos
        };
        self.is_square_attacked(king_pos, color.opposite())
    }

    pub fn is_square_attacked(&self, pos: (usize, usize), by_color: Color) -> bool {
        for file in 0..8 {
            for rank in 0..8 {
                if let Some(piece) = self.board[file][rank] {
                    if piece.color == by_color {
                        let test_move = Move {
                            from: (file, rank),
                            to: pos,
                            piece_moved: piece,
                            piece_captured: self.board[pos.0][pos.1],
                            is_castling: false,
                            is_en_passant: false,
                            promotion: None,
                        };
                        // Check only basic move validity (no recursion)
                        if self.is_valid_basic_move(&test_move).is_ok()
                            && self.is_path_clear((file, rank), pos)
                        {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    pub fn make_move_without_validation(&mut self, mv: &Move) {
        let (fx, fy) = mv.from;
        let (tx, ty) = mv.to;

        // Move piece
        self.board[tx][ty] = self.board[fx][fy];
        self.board[fx][fy] = None;

        // Update king pos
        if mv.piece_moved.piece_type == PieceType::King {
            if mv.piece_moved.color == Color::White {
                self.white_king_pos = (tx, ty);
            } else {
                self.black_king_pos = (tx, ty);
            }
            // Handle rook movement on castling
            if mv.is_castling {
                // White
                if mv.piece_moved.color == Color::White {
                    if tx == 6 {
                        // kingside
                        self.board[5][7] = self.board[7][7];
                        self.board[7][7] = None;
                    } else if tx == 2 {
                        // queenside
                        self.board[3][7] = self.board[0][7];
                        self.board[0][7] = None;
                    }
                    self.white_can_castle_kingside = false;
                    self.white_can_castle_queenside = false;
                } else {
                    // Black
                    if ty == 0 && tx == 6 {
                        self.board[5][0] = self.board[7][0];
                        self.board[7][0] = None;
                    } else if ty == 0 && tx == 2 {
                        self.board[3][0] = self.board[0][0];
                        self.board[0][0] = None;
                    }
                    self.black_can_castle_kingside = false;
                    self.black_can_castle_queenside = false;
                }
            }
        }
        // Pawn double move
        if mv.piece_moved.piece_type == PieceType::Pawn {
            let dy = ty as i32 - fy as i32;
            let direction = if mv.piece_moved.color == Color::White { -1 } else { 1 };
            if dy == 2 * direction {
                self.last_pawn_double_move = Some((tx, ty));
            } else {
                self.last_pawn_double_move = None;
            }

            // En passant
            if mv.is_en_passant {
                self.board[tx][fy] = None;
            }
        }

        // Switch turns
        self.current_turn = self.current_turn.opposite();
        self.move_history.push(mv.clone());
    }

    pub fn make_move_from_str(&mut self, mv_str: &str) -> Result<(), MoveError> {
        // For example, "e2e4" => e2->(4,6), e4->(4,4)
        if mv_str.len() < 4 {
            return Err(MoveError::InvalidPieceMove("Invalid move format".to_string()));
        }
        // Parse
        let file_from = mv_str.chars().nth(0).unwrap();
        let rank_from = mv_str.chars().nth(1).unwrap();
        let file_to = mv_str.chars().nth(2).unwrap();
        let rank_to = mv_str.chars().nth(3).unwrap();

        let fx = (file_from as u8 - b'a') as usize;
        let fy = 8 - (rank_from as u8 - b'0') as usize; // '1' -> 7, '2'->6, etc.
        let tx = (file_to as u8 - b'a') as usize;
        let ty = 8 - (rank_to as u8 - b'0') as usize;

        let piece = match self.board[fx][fy] {
            Some(p) => p,
            None => return Err(MoveError::NoPieceAtSource),
        };

        let is_castling = piece.piece_type == PieceType::King
            && (tx as i32 - fx as i32).abs() == 2
            && fy == ty;
        // If it might be en passant, check that
        let is_en_passant = if piece.piece_type == PieceType::Pawn && fx != tx && self.board[tx][ty].is_none() {
            true
        } else {
            false
        };

        let move_obj = Move {
            from: (fx, fy),
            to: (tx, ty),
            piece_moved: piece,
            piece_captured: self.board[tx][ty],
            is_castling,
            is_en_passant,
            promotion: None,
        };
        self.is_valid_move(&move_obj)?;
        self.make_move_without_validation(&move_obj);
        Ok(())
    }
}
