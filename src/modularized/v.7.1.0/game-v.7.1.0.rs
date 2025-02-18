//////////////////////////
// game.rs
//////////////////////////

use std::sync::{Arc, Mutex};
use std::fmt;
use crate::types::*;
use crate::neural::ChessNeuralEngine;


// The main GameState
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
        for rank in (0..8).rev() {
            write!(f, "{} ", rank + 1)?;
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
            writeln!(f, "{}", rank + 1)?;
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
            white_king_pos: (4, 0),
            black_king_pos: (4, 7),
            move_history: Vec::new(),
        };
        new_state.setup_initial_position();
        new_state
    }

    pub fn setup_initial_position(&mut self) {
        let create_piece = |pt, color| Some(Piece { piece_type: pt, color });
        
        // White pieces
        self.board[0][0] = create_piece(PieceType::Rook, Color::White);
        self.board[7][0] = create_piece(PieceType::Rook, Color::White);
        self.board[1][0] = create_piece(PieceType::Knight, Color::White);
        self.board[6][0] = create_piece(PieceType::Knight, Color::White);
        self.board[2][0] = create_piece(PieceType::Bishop, Color::White);
        self.board[5][0] = create_piece(PieceType::Bishop, Color::White);
        self.board[3][0] = create_piece(PieceType::Queen, Color::White);
        self.board[4][0] = create_piece(PieceType::King, Color::White);
        for i in 0..8 {
            self.board[i][1] = create_piece(PieceType::Pawn, Color::White);
        }

        // Black pieces
        self.board[0][7] = create_piece(PieceType::Rook, Color::Black);
        self.board[7][7] = create_piece(PieceType::Rook, Color::Black);
        self.board[1][7] = create_piece(PieceType::Knight, Color::Black);
        self.board[6][7] = create_piece(PieceType::Knight, Color::Black);
        self.board[2][7] = create_piece(PieceType::Bishop, Color::Black);
        self.board[5][7] = create_piece(PieceType::Bishop, Color::Black);
        self.board[3][7] = create_piece(PieceType::Queen, Color::Black);
        self.board[4][7] = create_piece(PieceType::King, Color::Black);
        for i in 0..8 {
            self.board[i][6] = create_piece(PieceType::Pawn, Color::Black);
        }
    }

    pub fn board_to_neural_input(&self) -> Vec<f32> {
        let mut input = Vec::with_capacity(64);
        for rank in 0..8 {
            for file in 0..8 {
                if let Some(piece) = self.board[file][rank] {
                    let value = if piece.color == Color::White { 1.0 } else { -1.0 };
                    input.push(value);
                } else {
                    input.push(0.0);
                }
            }
        }
        input
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
                    let multiplier = if piece.color == Color::White { 1 } else { -1 };
                    let piece_value = match piece.piece_type {
                        PieceType::Pawn => pawn_value,
                        PieceType::Knight => knight_value,
                        PieceType::Bishop => bishop_value,
                        PieceType::Rook => rook_value,
                        PieceType::Queen => queen_value,
                        PieceType::King => king_value,
                    };
                    score += multiplier * piece_value;
                }
            }
        }
        score
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

    pub fn is_valid_move(&self, mv: &Move) -> Result<(), MoveError> {
        self.is_valid_basic_move(mv)?;

        let mut test_state = self.clone();
        test_state.make_move_without_validation(mv);

        if test_state.is_in_check(mv.piece_moved.color) {
            return Err(MoveError::WouldCauseCheck);
        }
        Ok(())
    }

    fn is_valid_basic_move(&self, mv: &Move) -> Result<(), MoveError> {
        if !Self::is_within_bounds(mv.from) || !Self::is_within_bounds(mv.to) {
            return Err(MoveError::OutOfBounds);
        }

        let piece = self.get_piece_at(mv.from).ok_or(MoveError::NoPieceAtSource)?;

        if let Some(dest) = self.get_piece_at(mv.to) {
            if dest.color == piece.color {
                return Err(MoveError::InvalidPieceMove("Cannot capture your own piece".to_string()));
            }
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
        let direction = match color {
            Color::White => 1,
            Color::Black => -1,
        };
        let start_rank = match color {
            Color::White => 1,
            Color::Black => 6,
        };

        let f_one = ((fy as i32) + direction) as usize;
        let f_two = ((fy as i32) + 2 * direction) as usize;

        // Single push
        if tx == fx && ty == f_one && self.get_piece_at(mv.to).is_none() {
            return Ok(());
        }
        // Double push
        if fy == start_rank && tx == fx && ty == f_two {
            if self.get_piece_at(mv.to).is_none()
                && self.get_piece_at((fx, f_one)).is_none()
            {
                return Ok(());
            }
        }
        // Capture
        if (ty as i32 - fy as i32) == direction && (tx as i32 - fx as i32).abs() == 1 {
            if self.get_piece_at(mv.to).is_some() {
                return Ok(());
            }
        }
        // En passant
        if mv.is_en_passant {
            if let Some(last) = self.last_pawn_double_move {
                if (ty as i32 - fy as i32) == direction
                    && (tx as i32 - fx as i32).abs() == 1
                    && tx == last.0
                    && fy == last.1
                {
                    return Ok(());
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
        let dx = (mv.to.0 as i32 - mv.from.0 as i32).abs();
        let dy = (mv.to.1 as i32 - mv.from.1 as i32).abs();
        if dx == dy && dx > 0 {
            if self.is_path_clear(mv.from, mv.to) {
                Ok(())
            } else {
                Err(MoveError::BlockedPath)
            }
        } else {
            Err(MoveError::InvalidPieceMove("Invalid bishop move".to_string()))
        }
    }

    fn is_valid_rook_move(&self, mv: &Move) -> Result<(), MoveError> {
        let dx = mv.to.0 as i32 - mv.from.0 as i32;  // Remove parentheses
        let dy = mv.to.1 as i32 - mv.from.1 as i32;  // Remove parentheses
        if (dx == 0 && dy != 0) || (dy == 0 && dx != 0) {
            if self.is_path_clear(mv.from, mv.to) {
                Ok(())
            } else {
                Err(MoveError::BlockedPath)
            }
        } else {
            Err(MoveError::InvalidPieceMove("Invalid rook move".to_string()))
        }
    }

    fn is_valid_queen_move(&self, mv: &Move) -> Result<(), MoveError> {
        // Queen = rook or bishop
        if self.is_valid_bishop_move(mv).is_ok() {
            Ok(())
        } else if self.is_valid_rook_move(mv).is_ok() {
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
            return Ok(());
        }

        // Castling attempt
        if mv.is_castling && dy == 0 && dx == 2 {
            let (orig_pos, can_castle_kingside, can_castle_queenside) = match mv.piece_moved.color {
                Color::White => ((4, 0), self.white_can_castle_kingside, self.white_can_castle_queenside),
                Color::Black => ((4, 7), self.black_can_castle_kingside, self.black_can_castle_queenside),
            };

            // King must be in original position
            if mv.from != orig_pos {
                return Err(MoveError::KingNotInOriginalPosition);
            }

            // Check if king is in check
            if self.is_in_check(mv.piece_moved.color) {
                return Err(MoveError::CastlingInCheck);
            }

            match mv.piece_moved.color {
                Color::White => {
                    // White kingside: e1 -> g1
                    if mv.to == (6, 0) {
                        if !can_castle_kingside {
                            return Err(MoveError::CastlingRightsLost);
                        }
                        if !self.is_path_clear((4,0), (7,0)) {
                            return Err(MoveError::CastlingPathBlocked);
                        }
                        if self.board[7][0] != Some(Piece { piece_type: PieceType::Rook, color: Color::White }) {
                            return Err(MoveError::CastlingRookMissing);
                        }
                        for x in 4..=6 {
                            if self.is_square_attacked((x,0), Color::Black) {
                                return Err(MoveError::CastlingThroughCheck);
                            }
                        }
                    } else if mv.to == (2, 0) {
                        // White queenside: e1 -> c1
                        if !can_castle_queenside {
                            return Err(MoveError::CastlingRightsLost);
                        }
                        if !self.is_path_clear((4,0), (0,0)) {
                            return Err(MoveError::CastlingPathBlocked);
                        }
                        if self.board[0][0] != Some(Piece { piece_type: PieceType::Rook, color: Color::White }) {
                            return Err(MoveError::CastlingRookMissing);
                        }
                        for x in 2..=4 {
                            if self.is_square_attacked((x,0), Color::Black) {
                                return Err(MoveError::CastlingThroughCheck);
                            }
                        }
                    }
                }
                Color::Black => {
                    // Black kingside: e8 -> g8
                    if mv.to == (6, 7) {
                        if !can_castle_kingside {
                            return Err(MoveError::CastlingRightsLost);
                        }
                        if !self.is_path_clear((4,7), (7,7)) {
                            return Err(MoveError::CastlingPathBlocked);
                        }
                        if self.board[7][7] != Some(Piece { piece_type: PieceType::Rook, color: Color::Black }) {
                            return Err(MoveError::CastlingRookMissing);
                        }
                        for x in 4..=6 {
                            if self.is_square_attacked((x,7), Color::White) {
                                return Err(MoveError::CastlingThroughCheck);
                            }
                        }
                    } else if mv.to == (2, 7) {
                        // Black queenside: e8 -> c8
                        if !can_castle_queenside {
                            return Err(MoveError::CastlingRightsLost);
                        }
                        if !self.is_path_clear((4,7), (0,7)) {
                            return Err(MoveError::CastlingPathBlocked);
                        }
                        if self.board[0][7] != Some(Piece { piece_type: PieceType::Rook, color: Color::Black }) {
                            return Err(MoveError::CastlingRookMissing);
                        }
                        for x in 2..=4 {
                            if self.is_square_attacked((x,7), Color::White) {
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

    pub fn is_path_clear(&self, from: (usize, usize), to: (usize, usize)) -> bool {
        let (dx, dy) = (
            (to.0 as i32 - from.0 as i32).signum(),
            (to.1 as i32 - from.1 as i32).signum()
        );
        let mut x = from.0 as i32 + dx;
        let mut y = from.1 as i32 + dy;
        while (x, y) != (to.0 as i32, to.1 as i32) {
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

    fn would_move_cause_check(&self, mv: &Move) -> bool {
        let mut test_state = self.clone();
        test_state.make_move_without_validation(mv);
        test_state.is_in_check(mv.piece_moved.color)
    }

    pub fn is_in_check(&self, color: Color) -> bool {
        let king_pos = if color == Color::White {
            self.white_king_pos
        } else {
            self.black_king_pos
        };
        self.is_square_attacked(king_pos, color.opposite())
    }

    fn is_square_attacked(&self, pos: (usize, usize), by_color: Color) -> bool {
        // Check if enemy king is adjacent
        let enemy_king_pos = if by_color == Color::White {
            self.white_king_pos
        } else {
            self.black_king_pos
        };
        let dx = (pos.0 as i32 - enemy_king_pos.0 as i32).abs();
        let dy = (pos.1 as i32 - enemy_king_pos.1 as i32).abs();
        if dx <= 1 && dy <= 1 {
            return true;
        }

        // Check other pieces' moves
        for file in 0..8 {
            for rank in 0..8 {
                if let Some(piece) = self.board[file][rank] {
                    if piece.color == by_color {
                        let attack_mv = Move {
                            from: (file, rank),
                            to: pos,
                            piece_moved: piece,
                            piece_captured: self.board[pos.0][pos.1],
                            is_castling: false,
                            is_en_passant: false,
                            promotion: None,
                        };
                        if self.is_valid_basic_move(&attack_mv).is_ok() {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    pub fn make_move(&mut self, mv: Move) -> Result<(), MoveError> {
        self.is_valid_move(&mv)?;
        self.make_move_without_validation(&mv);
        self.move_history.push(mv.clone());
        self.current_turn = self.current_turn.opposite();
        Ok(())
    }

    fn make_move_without_validation(&mut self, mv: &Move) {
        // Clear the destination
        self.board[mv.to.0][mv.to.1] = None;

        // Place the piece (or promoted piece)
        if let Some(promote) = mv.promotion {
            let promoted = Piece {
                piece_type: promote,
                color: mv.piece_moved.color,
            };
            self.board[mv.to.0][mv.to.1] = Some(promoted);
        } else {
            self.board[mv.to.0][mv.to.1] = Some(mv.piece_moved);
        }
        // Clear the source
        self.board[mv.from.0][mv.from.1] = None;

        // Handle castling
        if mv.is_castling {
            match (mv.from, mv.to) {
                ((4,0), (6,0)) => {
                    self.board[5][0] = self.board[7][0].take();
                    self.board[7][0] = None;
                },
                ((4,0), (2,0)) => {
                    self.board[3][0] = self.board[0][0].take();
                    self.board[0][0] = None;
                },
                ((4,7), (6,7)) => {
                    self.board[5][7] = self.board[7][7].take();
                    self.board[7][7] = None;
                },
                ((4,7), (2,7)) => {
                    self.board[3][7] = self.board[0][7].take();
                    self.board[0][7] = None;
                },
                _ => {}
            }
        }

        // Handle pawn double‐move / en passant
        if mv.piece_moved.piece_type == PieceType::Pawn {
            let dy = (mv.to.1 as i32 - mv.from.1 as i32).abs();
            if dy == 2 {
                self.last_pawn_double_move = Some(mv.to);
            } else {
                self.last_pawn_double_move = None;
            }
            if mv.is_en_passant {
                let capture_y = mv.from.1;
                self.board[mv.to.0][capture_y] = None;
            }
        } else {
            self.last_pawn_double_move = None;
        }

        // If the king moved, update castling flags
        if mv.piece_moved.piece_type == PieceType::King {
            match mv.piece_moved.color {
                Color::White => {
                    self.white_king_pos = mv.to;
                    self.white_can_castle_kingside = false;
                    self.white_can_castle_queenside = false;
                },
                Color::Black => {
                    self.black_king_pos = mv.to;
                    self.black_can_castle_kingside = false;
                    self.black_can_castle_queenside = false;
                },
            }
        }

        // If a rook moved from its original square, lose castling rights
        match (mv.from, mv.piece_moved.piece_type) {
            ((0,0), PieceType::Rook) => self.white_can_castle_queenside = false,
            ((7,0), PieceType::Rook) => self.white_can_castle_kingside = false,
            ((0,7), PieceType::Rook) => self.black_can_castle_queenside = false,
            ((7,7), PieceType::Rook) => self.black_can_castle_kingside = false,
            _ => {}
        }
    }

    pub fn make_move_from_str(&mut self, input: &str) -> Result<(), String> {
        // Minimal coordinate-based parse: e.g. "e2e4"
        if input.len() < 4 {
            return Err("Move too short (e.g. 'g1f3')".to_string());
        }
        let from = self.parse_square(&input[0..2]).ok_or("Invalid 'from' square")?;
        let to = self.parse_square(&input[2..4]).ok_or("Invalid 'to' square")?;
        let piece = *self.get_piece_at(from).ok_or("No piece at source")?;
        if piece.color != self.current_turn {
            return Err("Not your turn!".to_string());
        }
        let mv = Move {
            from,
            to,
            piece_moved: piece,
            piece_captured: self.board[to.0][to.1],
            is_castling: false,
            is_en_passant: false,
            promotion: None,
        };
        self.make_move(mv).map_err(|e| e.to_string())
    }

    fn parse_square(&self, s: &str) -> Option<(usize, usize)> {
        if s.len() != 2 { return None; }
        let file = (s.chars().next()? as u8).wrapping_sub(b'a') as usize;
        let rank = (s.chars().nth(1)? as u8).wrapping_sub(b'1') as usize;
        if file < 8 && rank < 8 {
            Some((file, rank))
        } else {
            None
        }
    }
}
