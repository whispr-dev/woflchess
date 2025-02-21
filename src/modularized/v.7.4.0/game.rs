//////////////////////////
// game.rs
//////////////////////////

use std::fmt;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::hash::Hash;

use crate::types::*;
use crate::neural::ChessNeuralEngine;

#[derive(Clone)]
pub struct GameState {
    pub board: Board,
    pub current_turn: Color,
    pub castling_rights: CastlingRights,
    pub last_pawn_double_move: Option<(usize, usize)>,
    pub neural_engine: Option<Arc<Mutex<ChessNeuralEngine>>>,
    pub white_king_pos: (usize, usize),
    pub black_king_pos: (usize, usize),
    pub move_history: Vec<Move>,
    pub position_history: Vec<PositionKey>,  // For threefold repetition checking
}

#[derive(Clone, Hash, Eq, PartialEq)]
struct PositionKey {
    board: [[Option<Piece>; 8]; 8],
    castling_rights: CastlingRights,
    en_passant: Option<(usize, usize)>,
    side_to_move: Color,
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
            castling_rights: CastlingRights {
                white_kingside: true,
                white_queenside: true,
                black_kingside: true,
                black_queenside: true,
            },
            last_pawn_double_move: None,
            neural_engine: Some(Arc::new(Mutex::new(ChessNeuralEngine::new()))),
            white_king_pos: (4, 7),
            black_king_pos: (4, 0),
            move_history: Vec::new(),
            position_history: Vec::new(),
        };
        new_state.setup_initial_position();
        new_state.position_history.push(new_state.get_position_key());
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

    pub fn is_insufficient_material(&self) -> bool {
        let mut piece_counts: HashMap<(PieceType, Color), u8> = HashMap::new();
        let mut bishop_colors: HashMap<Color, (u8, u8)> = HashMap::new(); // (light, dark) squares

        for file in 0..8 {
            for rank in 0..8 {
                if let Some(piece) = self.board[file][rank] {
                    piece_counts.entry((piece.piece_type, piece.color))
                        .and_modify(|e| *e += 1)
                        .or_insert(1);

                    // Track bishop square colors
                    if piece.piece_type == PieceType::Bishop {
                        let square_color = (file + rank) % 2;
                        bishop_colors.entry(piece.color)
                            .and_modify(|(light, dark)| {
                                if square_color == 0 { *light += 1 } else { *dark += 1 }
                            })
                            .or_insert(if square_color == 0 { (1, 0) } else { (0, 1) });
                    }
                }
            }
        }

        // Only kings
        if piece_counts.len() == 2 {
            return true;
        }

        // King and bishop vs king
        if piece_counts.len() == 3 && bishop_colors.len() == 1 {
            return true;
        }

        // King and knight vs king
        if piece_counts.len() == 3 
            && piece_counts.iter().any(|((pt, _), count)| 
                *pt == PieceType::Knight && *count == 1) {
            return true;
        }

        // Kings and bishops on same colored squares
        if piece_counts.iter()
        .filter(|((pt, _), _)| *pt != PieceType::King)
        .all(|((pt, _), _)| *pt == PieceType::Bishop) {
        let all_light = bishop_colors.values()
            .all(|(_light, dark)| *dark == 0);
        let all_dark = bishop_colors.values()
            .all(|(light, _dark)| *light == 0);
        if all_light || all_dark {
            return true;
        }
        }
        
        false
    }

    fn get_position_key(&self) -> PositionKey {
        PositionKey {
            board: self.board,
            castling_rights: self.castling_rights,
            en_passant: self.last_pawn_double_move,
            side_to_move: self.current_turn,
        }
    }

    pub fn is_threefold_repetition(&self) -> bool {
        let mut position_counts: HashMap<PositionKey, u8> = HashMap::new();
        
        // Count current position
        position_counts.insert(self.get_position_key(), 1);

        // Replay move history to count positions
        let mut past_state = self.clone();
        for mv in self.move_history.iter().rev() {
            past_state.undo_move(mv);
            let key = past_state.get_position_key();
            let count = position_counts.entry(key).or_insert(0);
            *count += 1;
            if *count >= 3 {
                return true;
            }
        }
        
        false
    }

    pub fn is_valid_move(&mut self, mv: &Move) -> Result<(), MoveError> {
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

    fn is_valid_basic_move(&mut self, mv: &Move) -> Result<(), MoveError> {
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
        // Captures including en passant
        if dy == direction && dx.abs() == 1 {
            if let Some(captured_piece) = self.get_piece_at((tx, ty)) {
                if captured_piece.color != color {
                    return Ok(());
                }
                return Err(MoveError::InvalidPieceMove("Cannot capture your own piece".to_string()));
            }
            
            // En passant validation
            if mv.is_en_passant {
                if let Some((last_file, last_rank)) = self.last_pawn_double_move {
                    if tx == last_file && fy == last_rank {
                        // Verify the pawn to be captured exists
                        if let Some(piece) = self.get_piece_at((last_file, last_rank)) {
                            if piece.piece_type == PieceType::Pawn && piece.color != color {
                                return Ok(());
                            }
                        }
                    }
                }
                return Err(MoveError::InvalidPieceMove("Invalid en passant capture".to_string()));
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

    fn is_valid_king_move(&mut self, mv: &Move) -> Result<(), MoveError> {
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
            let rank = if mv.piece_moved.color == Color::White { 7 } else { 0 };
            let orig_x = 4;
            let orig_y = rank;
            
            if mv.from != (orig_x, orig_y) {
                return Err(MoveError::KingNotInOriginalPosition);
            }

            if self.is_in_check(mv.piece_moved.color) {
                return Err(MoveError::CastlingInCheck);
            }

            match mv.piece_moved.color {
                Color::White => {
                    if mv.to == (6, 7) {  // Kingside
                        if !self.castling_rights.white_kingside {
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
                    } else if mv.to == (2, 7) {  // Queenside
                        if !self.castling_rights.white_queenside {
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
                    if mv.to == (6, 0) {  // Kingside
                        if !self.castling_rights.black_kingside {
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
                    } else if mv.to == (2, 0) {  // Queenside
                        if !self.castling_rights.black_queenside {
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
            
            // Handle castling
            if mv.is_castling {
                let rank = if mv.piece_moved.color == Color::White { 7 } else { 0 };
                match tx {
                    6 => { // Kingside
                        self.board[5][rank] = self.board[7][rank];
                        self.board[7][rank] = None;
                    },
                    2 => { // Queenside
                        self.board[3][rank] = self.board[0][rank];
                        self.board[0][rank] = None;
                    },
                    _ => panic!("Invalid castling move"),
                }
            }
        }

        // Update castling rights
        self.update_castling_rights(mv);

        // Handle en passant capture
        if mv.is_en_passant {
            self.board[tx][fy] = None;  // Remove captured pawn
        }

        // Track pawn double moves for en passant
        if mv.piece_moved.piece_type == PieceType::Pawn 
            && (ty as i32 - fy as i32).abs() == 2 {
            self.last_pawn_double_move = Some((tx, ty));
        } else {
            self.last_pawn_double_move = None;
        }

        // Handle promotion
        if let Some(promotion_type) = mv.promotion {
            self.board[tx][ty] = Some(Piece {
                piece_type: promotion_type,
                color: mv.piece_moved.color,
            });
        }

        // Update turn
    s   elf.current_turn = self.current_turn.opposite();

        // Update histories
        self.move_history.push(mv.clone());
        self.position_history.push(self.get_position_key());
    }

        // Rook moves lose specific castling rights
        if mv.piece_moved.piece_type == PieceType::Rook {
            match (mv.from, mv.piece_moved.color) {
                ((0, 7), Color::White) => self.castling_rights.white_queenside = false,
                ((7, 7), Color::White) => self.castling_rights.white_kingside = false,
                ((0, 0), Color::Black) => self.castling_rights.black_queenside = false,
                ((7, 0), Color::Black) => self.castling_rights.black_kingside = false,
                _ => {}
            }
        }

        // Rook captures lose castling rights
        if let Some(captured) = mv.piece_captured {
            if captured.piece_type == PieceType::Rook {
                match (mv.to, captured.color) {
                    ((0, 7), Color::White) => self.castling_rights.white_queenside = false,
                    ((7, 7), Color::White) => self.castling_rights.white_kingside = false,
                    ((0, 0), Color::Black) => self.castling_rights.black_queenside = false,
                    ((7, 0), Color::Black) => self.castling_rights.black_kingside = false,
                    _ => {}
                }
            }
        }
    }

    pub fn generate_piece_moves(&self, piece: Piece, from: (usize, usize)) -> Vec<(usize, usize)> {
        let mut moves = Vec::new();
        let (x, y) = from;

        match piece.piece_type {
            PieceType::Pawn => {
                let direction = if piece.color == Color::White { -1i32 } else { 1i32 };
                let start_rank = if piece.color == Color::White { 6 } else { 1 };

                // Single push
                if let Some(new_y) = (y as i32 + direction).try_into().ok() {
                    if new_y < 8 {
                        moves.push((x, new_y));
                    }
                }

                // Double push from start
                if y == start_rank {
                    if let Some(double_y) = (y as i32 + 2 * direction).try_into().ok() {
                        if double_y < 8 {
                            moves.push((x, double_y));
                        }
                    }
                }

                // Captures (including potential en passant)
                for dx in [-1, 1] {
                    let new_x = x as i32 + dx;
                    let new_y = y as i32 + direction;
                    if new_x >= 0 && new_x < 8 && new_y >= 0 && new_y < 8 {
                        moves.push((new_x as usize, new_y as usize));
                    }
                }
            }
            PieceType::Knight => {
                let knight_moves = [
                    (-2, -1), (-2, 1),
                    (-1, -2), (-1, 2),
                    (1, -2), (1, 2),
                    (2, -1), (2, 1)
                ];
                for (dx, dy) in knight_moves.iter() {
                    let new_x = x as i32 + dx;
                    let new_y = y as i32 + dy;
                    if new_x >= 0 && new_x < 8 && new_y >= 0 && new_y < 8 {
                        moves.push((new_x as usize, new_y as usize));
                    }
                }
            }
            PieceType::Bishop => {
                let directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
                for (dx, dy) in directions.iter() {
                    let mut dist = 1;
                    loop {
                        let new_x = x as i32 + dx * dist;
                        let new_y = y as i32 + dy * dist;
                        if new_x < 0 || new_x >= 8 || new_y < 0 || new_y >= 8 {
                            break;
                        }
                        moves.push((new_x as usize, new_y as usize));
                        dist += 1;
                    }
                }
            }
            PieceType::Rook => {
                let directions = [(0, -1), (0, 1), (-1, 0), (1, 0)];
                for (dx, dy) in directions.iter() {
                    let mut dist = 1;
                    loop {
                        let new_x = x as i32 + dx * dist;
                        let new_y = y as i32 + dy * dist;
                        if new_x < 0 || new_x >= 8 || new_y < 0 || new_y >= 8 {
                            break;
                        }
                        moves.push((new_x as usize, new_y as usize));
                        dist += 1;
                    }
                }
            }
            PieceType::Queen => {
                // Combine bishop and rook moves
                let directions = [
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1), (0, 1),
                    (1, -1), (1, 0), (1, 1)
                ];
                for (dx, dy) in directions.iter() {
                    let mut dist = 1;
                    loop {
                        let new_x = x as i32 + dx * dist;
                        let new_y = y as i32 + dy * dist;
                        if new_x < 0 || new_x >= 8 || new_y < 0 || new_y >= 8 {
                            break;
                        }
                        moves.push((new_x as usize, new_y as usize));
                        dist += 1;
                    }
                }
            }
            PieceType::King => {
                // Normal moves
                let directions = [
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1), (0, 1),
                    (1, -1), (1, 0), (1, 1)
                ];
                for (dx, dy) in directions.iter() {
                    let new_x = x as i32 + dx;
                    let new_y = y as i32 + dy;
                    if new_x >= 0 && new_x < 8 && new_y >= 0 && new_y < 8 {
                        moves.push((new_x as usize, new_y as usize));
                    }
                }

                // Castling
                let rank = if piece.color == Color::White { 7 } else { 0 };
                if x == 4 && y == rank {
                    // Kingside
                    moves.push((6, rank));
                    // Queenside
                    moves.push((2, rank));
                }
            }
        }
        moves
    }

    pub fn undo_move(&mut self, mv: &Move) {
        let (fx, fy) = mv.from;
        let (tx, ty) = mv.to;

        // Restore piece to original position
        self.board[fx][fy] = self.board[tx][ty];
        self.board[tx][ty] = mv.piece_captured;

        // Restore king position
        if mv.piece_moved.piece_type == PieceType::King {
            if mv.piece_moved.color == Color::White {
                self.white_king_pos = (fx, fy);
            } else {
                self.black_king_pos = (fx, fy);
            }

            // Undo castling rook move
            if mv.is_castling {
                let rank = if mv.piece_moved.color == Color::White { 7 } else { 0 };
                match tx {
                    6 => { // Kingside
                        self.board[7][rank] = self.board[5][rank];
                        self.board[5][rank] = None;
                    },
                    2 => { // Queenside
                        self.board[0][rank] = self.board[3][rank];
                        self.board[3][rank] = None;
                    },
                    _ => panic!("Invalid castling move in undo"),
                }
            }
        }

        // Restore en passant captured pawn
        if mv.is_en_passant {
            self.board[tx][fy] = Some(Piece {
                piece_type: PieceType::Pawn,
                color: mv.piece_moved.color.opposite(),
            });
        }

        // Restore turn
        self.current_turn = self.current_turn.opposite();

        // Remove from history
        self.move_history.pop();
        self.position_history.pop();
    }    

    pub fn is_in_check(&mut self, color: Color) -> bool {
        // Verify king position tracking
        let king_pos = if color == Color::White {
            self.white_king_pos
        } else {
            self.black_king_pos
        };

        // Validate king position
        match self.get_piece_at(king_pos) {
            Some(piece) if piece.piece_type == PieceType::King && piece.color == color => {},
            _ => {
                // Attempt to locate king
                for file in 0..8 {
                    for rank in 0..8 {
                        if let Some(piece) = self.board[file][rank] {
                            if piece.piece_type == PieceType::King && piece.color == color {
                                // Update tracking
                                if color == Color::White {
                                    self.white_king_pos = (file, rank);
                                } else {
                                    self.black_king_pos = (file, rank);
                                }
                                return self.is_square_attacked((file, rank), color.opposite());
                            }
                        }
                    }
                }
                panic!("No king found on board!");
            }
        }

        self.is_square_attacked(king_pos, color.opposite())
    }

    pub fn validate_en_passant(&self, mv: &Move) -> Result<(), MoveError> {
        if mv.is_en_passant {
            if let Some((last_file, last_rank)) = self.last_pawn_double_move {
                let expected_rank = if mv.piece_moved.color == Color::White { 3 } else { 4 };
                if mv.from.1 != expected_rank || last_rank != expected_rank {
                    return Err(MoveError::InvalidPieceMove("Invalid en passant position".to_string()));
                }
                
                // Verify target pawn exists
                if let Some(piece) = self.get_piece_at((last_file, last_rank)) {
                    if piece.piece_type != PieceType::Pawn || piece.color == mv.piece_moved.color {
                        return Err(MoveError::InvalidPieceMove("Invalid en passant target".to_string()));
                    }
                } else {
                    return Err(MoveError::InvalidPieceMove("No pawn to capture".to_string()));
                }
            } else {
                return Err(MoveError::InvalidPieceMove("No en passant target available".to_string()));
            }
        }
        Ok(())
    }

    pub fn validate_promotion(&self, mv: &Move) -> Result<(), MoveError> {
        if let Some(promotion) = mv.promotion {
            // Must be a pawn
            if mv.piece_moved.piece_type != PieceType::Pawn {
                return Err(MoveError::InvalidPromotion("Only pawns can promote".into()));
            }

            // Must reach the back rank
            let promotion_rank = if mv.piece_moved.color == Color::White { 0 } else { 7 };
            if mv.to.1 != promotion_rank {
                return Err(MoveError::InvalidPromotion("Not a promotion square".into()));
            }

            // Validate promotion piece type
            match promotion {
                PieceType::Pawn | PieceType::King => {
                    return Err(MoveError::InvalidPromotion("Invalid promotion piece".into()));
                }
                _ => Ok(())
            }
        } else {
            // If pawn reaches back rank, must specify promotion
            let back_rank = if mv.piece_moved.color == Color::White { 0 } else { 7 };
            if mv.piece_moved.piece_type == PieceType::Pawn && mv.to.1 == back_rank {
                return Err(MoveError::InvalidPromotion("Must specify promotion piece".into()));
            }
            Ok(())
        }
    }

    pub fn handle_promotion(&mut self, mv: &Move) {
        if let Some(promotion) = mv.promotion {
            self.board[mv.to.0][mv.to.1] = Some(Piece {
                piece_type: promotion,
                color: mv.piece_moved.color,
            });
        }
    }

    fn is_path_clear(&self, from: (usize, usize), to: (usize, usize)) -> bool {
        let (fx, fy) = (from.0 as i32, from.1 as i32);
        let (tx, ty) = (to.0 as i32, to.1 as i32);
    
        let dx = (tx - fx).signum();
        let dy = (ty - fy).signum();
    
        let mut x = fx + dx;
        let mut y = fy + dy;
    
        // Check all squares except the destination
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
        
        // Destination square can contain enemy piece (capture) but not friendly piece
        if let Some(dest_piece) = self.board[tx as usize][ty as usize] {
            if dest_piece.color == self.current_turn {
                return false;
            }
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

    pub fn is_checkmate(&mut self) -> bool {
        if !self.is_in_check(self.current_turn) {
            return false;
        }
        self.get_all_legal_moves().is_empty()
    }
    
    pub fn is_stalemate(&mut self) -> bool {
        if self.is_in_check(self.current_turn) {
            return false;
        }
        self.get_all_legal_moves().is_empty()
    }

    pub fn get_all_legal_moves(&mut self) -> Vec<Move> {
        let mut legal_moves = Vec::new();

        for from_file in 0..8 {
            for from_rank in 0..8 {
                if let Some(piece) = self.board[from_file][from_rank] {
                    if piece.color == self.current_turn {
                        let potential_moves = self.generate_piece_moves(piece, (from_file, from_rank));
                        
                        for to_pos in potential_moves {
                            let mv = Move {
                                from: (from_file, from_rank),
                                to: to_pos,
                                piece_moved: piece,
                                piece_captured: self.board[to_pos.0][to_pos.1],
                                is_castling: piece.piece_type == PieceType::King 
                                    && (to_pos.0 as i32 - from_file as i32).abs() == 2,
                                is_en_passant: piece.piece_type == PieceType::Pawn 
                                    && (to_pos.0 as i32 - from_file as i32).abs() == 1 
                                    && self.board[to_pos.0][to_pos.1].is_none(),
                                promotion: if piece.piece_type == PieceType::Pawn 
                                    && (to_pos.1 == 0 || to_pos.1 == 7) {
                                        Some(PieceType::Queen)
                                    } else {
                                        None
                                    },
                            };

                            // Create a clone to test the move
                            let mut test_state = self.clone();
                            if test_state.is_valid_move(&mv).is_ok() {
                                legal_moves.push(mv);
                            }
                        }
                    }
                }
            }
        }

        legal_moves
    }

    pub fn is_square_attacked(&mut self, pos: (usize, usize), by_color: Color) -> bool {
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
                        if self.is_path_clear((file, rank), pos) 
                            && self.is_valid_basic_move(&test_move).is_ok()
                        {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    pub fn make_move_from_str(&mut self, mv_str: &str) -> Result<(), MoveError> {
        if mv_str.len() < 4 {
            return Err(MoveError::InvalidPieceMove("Invalid move format".to_string()));
        }

        let file_from = mv_str.chars().nth(0).unwrap();
        let rank_from = mv_str.chars().nth(1).unwrap();
        let file_to = mv_str.chars().nth(2).unwrap();
        let rank_to = mv_str.chars().nth(3).unwrap();

        let fx = (file_from as u8 - b'a') as usize;
        let fy = 8 - (rank_from as u8 - b'0') as usize;
        let tx = (file_to as u8 - b'a') as usize;
        let ty = 8 - (rank_to as u8 - b'0') as usize;

        let piece = match self.board[fx][fy] {
            Some(p) => p,
            None => return Err(MoveError::NoPieceAtSource),
        };

        let is_castling = piece.piece_type == PieceType::King
            && (tx as i32 - fx as i32).abs() == 2
            && fy == ty;

        let is_en_passant = piece.piece_type == PieceType::Pawn 
            && fx != tx 
            && self.board[tx][ty].is_none();

        let promotion = if piece.piece_type == PieceType::Pawn 
            && (ty == 0 || ty == 7) {
                Some(PieceType::Queen)  // Default to queen promotion
            } else {
                None
            };

        let move_obj = Move {
            from: (fx, fy),
            to: (tx, ty),
            piece_moved: piece,
            piece_captured: self.board[tx][ty],
            is_castling,
            is_en_passant,
            promotion,
        };

        self.is_valid_move(&move_obj)?;
        self.make_move_without_validation(&move_obj);

        Ok(())
}
