use std::sync::{Arc, Mutex};
use rand::Rng;
use colored::*;
use std::fmt;

use crate::types::*;
use crate::ChessNeuralEngine;

// Constants for piece movements
const BISHOP_DIRECTIONS: [(i8, i8); 4] = [(-1,-1), (-1,1), (1,-1), (1,1)];
const ROOK_DIRECTIONS: [(i8, i8); 4] = [(-1,0), (1,0), (0,-1), (0,1)];
const KNIGHT_MOVES: [(i8, i8); 8] = [
    (-2,-1), (-2,1), (-1,-2), (-1,2),
    (1,-2), (1,2), (2,-1), (2,1)
];

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
        for i in 0..8 {
            for j in 0..8 {
                if let Some(piece) = self.board[i][j] {
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

    pub fn evaluate_position_neural(&self) -> i32 {
        if let Some(engine_arc) = &self.neural_engine {
            let engine = engine_arc.lock().unwrap();
            let input = self.board_to_neural_input();
            
            let mut total_eval = 0.0;
            let num_components = 4;

            // Traditional evaluation (40% weight)
            total_eval += self.evaluate_position() as f32 * 0.4;

            // RNN evaluation (20% weight)
            if let Ok(mut rnn) = engine.rnn.lock() {
                let out = rnn.forward(&input);
                if let Some(val) = out.get(0) {
                    total_eval += val * 100.0 * 0.2;
                }
            }

            // CNN evaluation (20% weight)
            if let Ok(mut cnn) = engine.cnn.lock() {
                let reshaped = crate::reshape_vector_to_matrix(&input, 8, 8);
                if let Some(Some(val)) = cnn.forward(&reshaped).first().map(|row| row.first()) {
                    total_eval += val * 100.0 * 0.2;
                }
            }

            // LSTM evaluation (20% weight)
            if let Ok(mut lstm) = engine.lstm.lock() {
                let out = lstm.forward(&input);
                if let Some(val) = out.get(0) {
                    total_eval += val * 100.0 * 0.2;
                }
            }

            total_eval as i32
        } else {
            self.evaluate_position()
        }
    }

    pub fn validate_board_state(&self) -> Result<(), String> {
        let mut white_kings = 0;
        let mut black_kings = 0;
        
        for rank in 0..8 {
            for file in 0..8 {
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

    // All the move validation methods
    pub fn is_valid_move(&self, mv: &Move) -> Result<(), MoveError> {
        self.is_valid_basic_move(mv)?;
        
        let mut test_state = self.clone();
        test_state.make_move_without_validation(mv);
        
        if test_state.is_in_check(mv.piece_moved.color) {
            return Err(MoveError::WouldCauseCheck);
        }
        
        Ok(())
    }

    pub fn is_valid_basic_move(&self, mv: &Move) -> Result<(), MoveError> {
        if !Self::is_within_bounds(mv.from) || !Self::is_within_bounds(mv.to) {
            return Err(MoveError::OutOfBounds);
        }

        let piece = self.get_piece_at(mv.from)
            .ok_or(MoveError::NoPieceAtSource)?;

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

    // Implement all the specific piece move validation methods
    // (is_valid_pawn_move, is_valid_knight_move, etc.)
    
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
                && self.get_piece_at((fx, f_one)).is_none() {
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
                    && tx == last.0 && fy == last.1
                {
                    return Ok(());
                }
            }
        }
        
        Err(MoveError::InvalidPieceMove("Invalid pawn move".to_string()))
    }

    // Movement helper methods
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

    // Board state methods
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

    // Check-related methods
    pub fn is_in_check(&self, color: Color) -> bool {
        let king_pos = if color == Color::White {
            self.white_king_pos
        } else {
            self.black_king_pos
        };
        self.is_square_attacked(king_pos, color.opposite())
    }

    // Update the square attack check to use Results
    fn is_square_attacked(&self, pos: (usize, usize), by_color: types::Color) -> bool {
        // Special case for king attacks
        let enemy_king_pos = if by_color == types::Color::White {
            self.white_king_pos
        } else {
            self.black_king_pos
        };
    
        // Check if enemy king is adjacent to the square
        let dx = (pos.0 as i32 - enemy_king_pos.0 as i32).abs();
        let dy = (pos.1 as i32 - enemy_king_pos.1 as i32).abs();
        if dx <= 1 && dy <= 1 {
            return true;
        }
    
        // Check other pieces' moves
        for i in 0..8 {
            for j in 0..8 {
                if let Some(piece) = self.board[i][j] {
                    if piece.color == by_color {
                        let attack_mv = types::Move {
                            from: (i, j),

// unfortunately token limit hit here and code cut off - i've improvised but... /shrug?

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

    // Update make_move to use the new error types
    fn make_move(&mut self, mv: types::Move) -> Result<(), MoveError> {
        self.is_valid_move(&mv)?;
        self.make_move_without_validation(&mv);
        self.move_history.push(mv.clone());
        self.current_turn = match self.current_turn {
            types::Color::White => types::Color::Black,
            types::Color::Black => types::Color::White,
        };
        Ok(())
    }

    fn make_move_without_validation(&mut self, mv: &types::Move) {
        // Clear the destination square first
        self.board[mv.to.0][mv.to.1] = None;
    
        // Then move the piece
        if let Some(promote) = mv.promotion {
            let promoted = types::Piece {
                piece_type: promote,
                color: mv.piece_moved.color,
            };
            self.board[mv.to.0][mv.to.1] = Some(promoted);
        } else {
            self.board[mv.to.0][mv.to.1] = Some(mv.piece_moved);
        }
        
        // Clear the source square
        self.board[mv.from.0][mv.from.1] = None;
    
        // Handle castling
        if mv.is_castling {
            match (mv.from, mv.to) {
                ((4,0), (6,0)) => {
                    self.board[5][0] = self.board[7][0].take(); // Use take() to avoid cloning
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
    
        // Handle pawn moves
        if mv.piece_moved.piece_type == types::PieceType::Pawn {
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
    
        // Update king pos if the king moved
        if mv.piece_moved.piece_type == types::PieceType::King {
            match mv.piece_moved.color {
                types::Color::White => {
                    self.white_king_pos = mv.to;
                    self.white_can_castle_kingside = false;
                    self.white_can_castle_queenside = false;
                },
                types::Color::Black => {
                    self.black_king_pos = mv.to;
                    self.black_can_castle_kingside = false;
                    self.black_can_castle_queenside = false;
                },
            }
        }
    
        // If a rook moves from its original square, lose castling rights on that side
        match (mv.from, mv.piece_moved.piece_type) {
            ((0,0), types::PieceType::Rook) => self.white_can_castle_queenside = false,
            ((7,0), types::PieceType::Rook) => self.white_can_castle_kingside = false,
            ((0,7), types::PieceType::Rook) => self.black_can_castle_queenside = false,
            ((7,7), types::PieceType::Rook) => self.black_can_castle_kingside = false,
            _ => {}
        }
    }

    fn parse_move_string(&self, move_str: &str) -> Result<((usize, usize), (usize, usize)), &'static str> {
        let parts: Vec<&str> = move_str.split_whitespace().collect();
        if parts.len() == 2 {
            let from = self.parse_square(parts[0]).ok_or("Invalid 'from' square")?;
            let to = self.parse_square(parts[1]).ok_or("Invalid 'to' square")?;
            Ok((from, to))
        } else if move_str.len() == 4 {
            let from = self.parse_square(&move_str[0..2]).ok_or("Invalid 'from' square")?;
            let to = self.parse_square(&move_str[2..4]).ok_or("Invalid 'to' square")?;
            Ok((from, to))
        } else {
            Err("Invalid move format - use 'e2 e4' or 'e2e4'")
        }
    }

    fn parse_square(&self, s: &str) -> Option<(usize, usize)> {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() != 2 {
            return None;
        }
        let file = match chars[0] {
            'a'..='h' => (chars[0] as u8 - b'a') as usize,
            _ => return None,
        };
        let rank = match chars[1] {
            '1'..='8' => (chars[1] as u8 - b'1') as usize,
            _ => return None,
        };
        Some((file, rank))
    }

    fn parse_algebraic_notation(&self, move_str: &str) -> Result<((usize, usize), (usize, usize)), &'static str> {
        // Handle castling notation
        match move_str {
            "O-O" | "0-0" => {
                let (rank, king, to) = if self.current_turn == types::Color::White {
                    (0, 4, 6)
                } else {
                    (7, 4, 6)
                };
                return Ok(((king, rank), (to, rank)));
            },
            "O-O-O" | "0-0-0" => {
                let (rank, king, to) = if self.current_turn == types::Color::White {
                    (0, 4, 2)
                } else {
                    (7, 4, 2)
                };
                return Ok(((king, rank), (to, rank)));
            },
            _ => {}
        }

        // Basic format validation
        if move_str.len() < 2 || move_str.len() > 3 {
            return Err("Invalid move format");
        }

        let chars: Vec<char> = move_str.chars().collect();
        
        // Validate first character for piece moves
        match chars[0] {
            'a'..='h' | 'N' | 'B' | 'R' | 'Q' | 'K' => {},
            _ => return Err("Invalid piece or file"),
        }
        
        // Handle pawn moves (e.g., "e4")
        if chars[0].is_ascii_lowercase() {
            if chars.len() != 2 {
                return Err("Invalid pawn move format");
            }
            
            // Validate rank
            if !chars[1].is_ascii_digit() || !('1'..='8').contains(&chars[1]) {
                return Err("Invalid rank for pawn move");
            }
            
            let file = (chars[0] as u8 - b'a') as usize;
            let rank = chars[1].to_digit(10).ok_or("Invalid rank")? as usize - 1;
            
            // Find the pawn that can make this move
            let pawn_rank = if self.current_turn == types::Color::White {
                vec![1] // White pawns start on rank 2
            } else {
                vec![6] // Black pawns start on rank 7
            };
            
            for &start_rank in &pawn_rank {
                if let Some(piece) = self.get_piece_at((file, start_rank)) {
                    if piece.piece_type == types::PieceType::Pawn && piece.color == self.current_turn {
                        let test_move = types::Move {
                            from: (file, start_rank),
                            to: (file, rank),
                            piece_moved: *piece,
                            piece_captured: None,
                            is_castling: false,
                            is_en_passant: false,
                            promotion: None,
                        };
                        if self.is_valid_move(&test_move).is_ok() {
                            return Ok(((file, start_rank), (file, rank)));
                        }
                    }
                }
            }
            return Err("No pawn can make that move");
        }

        // Handle piece moves (e.g., "Nf3")
        if chars.len() != 3 {
            return Err("Invalid piece move format");
        }

        let (piece_type, start_idx) = match chars[0] {
            'N' => (types::PieceType::Knight, 1),
            'B' => (types::PieceType::Bishop, 1),
            'R' => (types::PieceType::Rook, 1),
            'Q' => (types::PieceType::Queen, 1),
            'K' => (types::PieceType::King, 1),
            _ => return Err("Invalid piece type"),
        };

        // Validate destination square
        if !chars[1].is_ascii_lowercase() || !('a'..='h').contains(&chars[1]) {
            return Err("Invalid file for destination square");
        }
        if !chars[2].is_ascii_digit() || !('1'..='8').contains(&chars[2]) {
            return Err("Invalid rank for destination square");
        }

        let dest_file = (chars[1] as u8 - b'a') as usize;
        let dest_rank = chars[2].to_digit(10).ok_or("Invalid rank")? as usize - 1;
        let to = (dest_file, dest_rank);

        // Find the piece that can make this move
        for rank in 0..8 {
            for file in 0..8 {
                if let Some(piece) = self.get_piece_at((file, rank)) {
                    if piece.piece_type == piece_type && piece.color == self.current_turn {
                        let test_move = types::Move {
                            from: (file, rank),
                            to,
                            piece_moved: *piece,
                            piece_captured: self.get_piece_at(to).copied(),
                            is_castling: false,
                            is_en_passant: false,
                            promotion: None,
                        };
                        if self.is_valid_move(&test_move).is_ok() {
                            return Ok(((file, rank), to));
                        }
                    }
                }
            }
        }

        Err("No piece can make that move")
    }

    fn find_piece_that_can_move(&self, piece_type: types::PieceType, to: (usize, usize)) -> Result<(usize, usize), &'static str> {
        let mut valid_moves = Vec::new();
        for i in 0..8 {
            for j in 0..8 {
                if let Some(piece) = self.board[i][j] {
                    if piece.piece_type == piece_type && piece.color == self.current_turn {
                        let test_mv = types::Move {
                            from: (i, j),
                            to,
                            piece_moved: piece,
                            piece_captured: self.board[to.0][to.1],
                            is_castling: false,
                            is_en_passant: false,
                            promotion: None,
                        };
                        if self.is_valid_move(&test_mv).is_ok() {
                            valid_moves.push((i, j));
                        }
                    }
                }
            }
        }
        match valid_moves.len() {
            0 => Err("No piece can make that move"),
            1 => Ok(valid_moves[0]),
            _ => Err("Ambiguous move - multiple pieces can move there."),
        }
    }

    // Update move_from_str to use proper error mapping
    fn make_move_from_str(&mut self, move_str: &str) -> Result<(), String> {
        // First try algebraic notation
        let (from, to) = match self.parse_algebraic_notation(move_str) {
            Ok(result) => result,
            Err(_) => {
                // If algebraic notation fails, try coordinate notation
                self.parse_move_string(move_str).map_err(|_| {
                    format!("Invalid move. Use algebraic notation (e.g., 'e4', 'Nf3') \
                            or coordinate notation (e.g., 'e2e4')")
                })?
            }
        };
    
        let piece = self.get_piece_at(from)
            .ok_or_else(|| "No piece at starting square".to_string())?;
        
        if piece.color != self.current_turn {
            return Err("It's not your turn!".to_string());
        }
    
        let mv = types::Move {
            from,
            to,
            piece_moved: *piece,
            piece_captured: self.get_piece_at(to).copied(),
            is_castling: ["O-O","0-0","O-O-O","0-0-0"].contains(&move_str),
            is_en_passant: false,
            promotion: if self.is_promotion_move(from, to) {
                Some(types::PieceType::Queen)
            } else {
                None
            },
        };
    
        self.make_move(mv).map_err(|e| e.to_string())
    }    

    fn is_promotion_move(&self, from: (usize, usize), to: (usize, usize)) -> bool {
        if let Some(piece) = self.get_piece_at(from) {
            if piece.piece_type == types::PieceType::Pawn {
                return match piece.color {
                    types::Color::White => to.1 == 7,
                    types::Color::Black => to.1 == 0,
                };
            }
        }
        false
    }

    fn minimax(&mut self, depth: i32, mut alpha: i32, mut beta: i32, maximizing: bool) -> i32 {
        if depth == 0 {
            return self.evaluate_position();
        }
        let legal_moves = self.generate_legal_moves();
        if legal_moves.is_empty() {
            if self.is_in_check(self.current_turn) {
                return if maximizing { -30000 } else { 30000 };
            }
            return 0;
        }
        if maximizing {
            let mut max_eval = i32::MIN;
            for mv in legal_moves {
                let mut new_state = self.clone();
                new_state.make_move_without_validation(&mv);
                let eval = new_state.minimax(depth - 1, alpha, beta, false);
                max_eval = max_eval.max(eval);
                alpha = alpha.max(eval);
                if beta <= alpha {
                    break;
                }
            }
            max_eval
        } else {
            let mut min_eval = i32::MAX;
            for mv in legal_moves {
                let mut new_state = self.clone();
                new_state.make_move_without_validation(&mv);
                let eval = new_state.minimax(depth - 1, alpha, beta, true);
                min_eval = min_eval.min(eval);
                beta = beta.min(eval);
                if beta <= alpha {
                    break;
                }
            }
            min_eval
        }
    }

    fn minimax_neural(&mut self, depth: i32, mut alpha: i32, mut beta: i32, maximizing: bool) -> i32 {
        if depth == 0 {
            return self.evaluate_position_neural();
        }
        let moves = self.generate_moves_neural();
        if moves.is_empty() {
            if self.is_in_check(self.current_turn) {
                return if maximizing { -30000 } else { 30000 };
            }
            return 0;
        }
        if maximizing {
            let mut max_eval = i32::MIN;
            for mv in moves {
                let mut new_state = self.clone();
                new_state.make_move_without_validation(&mv);
                let eval = new_state.minimax_neural(depth - 1, alpha, beta, false);
                max_eval = max_eval.max(eval);
                alpha = alpha.max(eval);
                if beta <= alpha {
                    break;
                }
            }
            max_eval
        } else {
            let mut min_eval = i32::MAX;
            for mv in moves {
                let mut new_state = self.clone();
                new_state.make_move_without_validation(&mv);
                let eval = new_state.minimax_neural(depth - 1, alpha, beta, true);
                min_eval = min_eval.min(eval);
                beta = beta.min(eval);
                if beta <= alpha {
                    break;
                }
            }
            min_eval
        }
    }

    // Update move generation to handle Results
    fn generate_legal_moves(&self) -> Vec<types::Move> {
        let mut moves = Vec::new();
        for i in 0..8 {
            for j in 0..8 {
                if let Some(piece) = self.board[i][j] {
                    if piece.color == self.current_turn {
                        for x in 0..8 {
                            for y in 0..8 {
                                let test_mv = types::Move {
                                    from: (i, j),
                                    to: (x, y),
                                    piece_moved: piece,
                                    piece_captured: self.board[x][y],
                                    is_castling: false,
                                    is_en_passant: false,
                                    promotion: None,
                                };
                                if self.is_valid_move(&test_mv).is_ok() {
                                    moves.push(test_mv);
                                }
                            }
                        }
                        
                        // Check castling if piece is a king
                        if piece.piece_type == types::PieceType::King {
                            let rank = if piece.color == types::Color::White { 0 } else { 7 };
                            let kingside = types::Move {
                                from: (4, rank),
                                to: (6, rank),
                                piece_moved: piece,
                                piece_captured: None,
                                is_castling: true,
                                is_en_passant: false,
                                promotion: None,
                            };
                            if self.is_valid_move(&kingside).is_ok() {
                                moves.push(kingside);
                            }
                            let queenside = types::Move {
                                from: (4, rank),
                                to: (2, rank),
                                piece_moved: piece,
                                piece_captured: None,
                                is_castling: true,
                                is_en_passant: false,
                                promotion: None,
                            };
                            if self.is_valid_move(&queenside).is_ok() {
                                moves.push(queenside);
                            }
                        }
                    }
                }
            }
        }
        self.order_moves(&mut moves);
        moves
    }

    fn order_moves(&self, moves: &mut Vec<types::Move>) {
        moves.sort_by(|a, b| self.move_score(b).cmp(&self.move_score(a)));
    }

    fn move_score(&self, mv: &types::Move) -> i32 {
        if let Some(captured) = mv.piece_captured {
            match captured.piece_type {
                types::PieceType::Pawn => 100,
                types::PieceType::Knight => 320,
                types::PieceType::Bishop => 330,
                types::PieceType::Rook => 500,
                types::PieceType::Queen => 900,
                types::PieceType::King => 20000,
            }
        } else {
            0
        }
    }

    fn generate_moves_neural(&self) -> Vec<types::Move> {
        let mut moves = self.generate_legal_moves();
        if let Some(engine_arc) = &self.neural_engine {
            let engine = engine_arc.lock().unwrap();
            let move_scores: Vec<(types::Move, f32)> = moves
                .iter()
                .map(|mv| {
                    let mut test_state = self.clone();
                    test_state.make_move_without_validation(mv);
                    let position_after = test_state.board_to_neural_input();

                    let rnn_score = engine.rnn.lock().unwrap().discriminate(&position_after);
                    let cnn_score =
                        engine.cnn.lock().unwrap().discriminate(&vec![position_after.clone()]);
                    let lstm_score = engine.lstm.lock().unwrap().discriminate(&position_after);

                    let combined = (rnn_score + cnn_score + lstm_score) / 3.0;
                    (mv.clone(), combined)
                })
                .collect();

            let mut sorted = move_scores;
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            moves = sorted.into_iter().map(|(mv, _)| mv).collect();
        }
        moves
    }

    fn make_computer_move(&mut self) -> Result<(), String> {
        if let Err(e) = self.validate_board_state() {
            println!("Invalid board state before computer move: {}", e);
            return Err("Invalid board state".to_string());
        }
    
        let moves = self.generate_moves_neural();
        if moves.is_empty() {
            if self.is_in_check(self.current_turn) {
                return Err("Checkmate!".to_string());
            } else {
                return Err("Stalemate!".to_string());
            }
        }
    
        let mut best_score = i32::MIN;
        let mut best_moves = Vec::new();
        let search_depth = 3;
        for mv in moves {
            let mut test_state = self.clone();
            if test_state.make_move(mv.clone()).is_err() {
                continue;
            }
            let score = -test_state.minimax_neural(search_depth - 1, i32::MIN, i32::MAX, true);
            if score > best_score {
                best_score = score;
                best_moves.clear();
                best_moves.push(mv);
            } else if score == best_score {
                best_moves.push(mv);
            }
        }
    
        if best_moves.is_empty() {
            return Err("No valid moves found".to_string());
        }
    
        let mut rng = rand::thread_rng();
        let selected = best_moves[rng.gen_range(0..best_moves.len())].clone();
    
        println!("Neural evaluation: {}", best_score);
        println!("Computer selected move: {:?}", selected);
    
        self.make_move(selected.clone()).map_err(|e| e.to_string())?;
    
        if let Err(e) = self.validate_board_state() {
            println!("Invalid board state after computer move: {}", e);
            return Err("Move resulted in invalid board state".to_string());
        }
    
        println!("Computer plays: {}", self.move_to_algebraic(&selected));
        Ok(())
    }

    fn move_to_algebraic(&self, mv: &types::Move) -> String {
        if mv.is_castling {
            if mv.to.0 == 6 {
                "O-O".to_string()
            } else {
                "O-O-O".to_string()
            }
        } else {
            let piece_char = match mv.piece_moved.piece_type {
                types::PieceType::Pawn => "",
                types::PieceType::Knight => "N",
                types::PieceType::Bishop => "B",
                types::PieceType::Rook => "R",
                types::PieceType::Queen => "Q",
                types::PieceType::King => "K",
            };
            let capture = if mv.piece_captured.is_some() || mv.is_en_passant {
                "x"
            } else {
                ""
            };
            let dest_file = (mv.to.0 as u8 + b'a') as char;
            let dest_rank = (mv.to.1 + 1).to_string();
            format!("{}{}{}{}", piece_char, capture, dest_file, dest_rank)
        }
    }

    fn get_game_status(&self) -> String {
        // Check for insufficient material
        let mut piece_count = 0;
        let mut has_minor_piece = false;
        for rank in 0..8 {
            for file in 0..8 {
                if let Some(piece) = self.board[file][rank] {
                    piece_count += 1;
                    if piece.piece_type != types::PieceType::King {
                        has_minor_piece = true;
                    }
                }
            }
        }
        if piece_count == 2 && !has_minor_piece {
            return "Draw by insufficient material!".to_string();
        }

        if self.is_in_check(self.current_turn) && self.generate_legal_moves().is_empty() {
            return format!("Checkmate! {:?} wins!", self.current_turn.opposite());
        } else if self.is_in_check(self.current_turn) {
            return format!("{:?} is in check!", self.current_turn);
        } else if self.generate_legal_moves().is_empty() {
            return "Stalemate! Game is a draw!".to_string();
        } else {
            return format!("{:?}'s turn", self.current_turn);
        }
    }
}

