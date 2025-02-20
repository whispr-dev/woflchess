//////////////////////////
// game.rs
//////////////////////////


// The main GameState
use std::fmt;
use std::sync::{Arc, Mutex};
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
        // Iterate ranks from top (8) to bottom (1)
        for rank in 0..8 {
            write!(f, "{} ", 8 - rank)?;  // Print rank number
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
            writeln!(f, "{}", 8 - rank)?;  // Print rank number
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
        
        // Black pieces (on top rows 0,1)
        self.board[0][0] = create_piece(PieceType::Rook, Color::Black);
        self.board[1][0] = create_piece(PieceType::Knight, Color::Black);
        self.board[2][0] = create_piece(PieceType::Bishop, Color::Black);
        self.board[3][0] = create_piece(PieceType::Queen, Color::Black);
        self.board[4][0] = create_piece(PieceType::King, Color::Black);
        self.board[5][0] = create_piece(PieceType::Bishop, Color::Black);
        self.board[6][0] = create_piece(PieceType::Knight, Color::Black);
        self.board[7][0] = create_piece(PieceType::Rook, Color::Black);
        
        // Black pawns on rank 1
        for i in 0..8 {
            self.board[i][1] = create_piece(PieceType::Pawn, Color::Black);
        }
    
        // White pieces (on bottom rows 6,7)
        self.board[0][7] = create_piece(PieceType::Rook, Color::White);
        self.board[1][7] = create_piece(PieceType::Knight, Color::White);
        self.board[2][7] = create_piece(PieceType::Bishop, Color::White);
        self.board[3][7] = create_piece(PieceType::Queen, Color::White);
        self.board[4][7] = create_piece(PieceType::King, Color::White);
        self.board[5][7] = create_piece(PieceType::Bishop, Color::White);
        self.board[6][7] = create_piece(PieceType::Knight, Color::White);
        self.board[7][7] = create_piece(PieceType::Rook, Color::White);
        
        // White pawns on rank 6
        for i in 0..8 {
            self.board[i][6] = create_piece(PieceType::Pawn, Color::White);
        }
        
        // Clear middle ranks (2-5)
        for i in 0..8 {
            for j in 2..6 {
                self.board[i][j] = None;
            }
        }
        
        // Initialize king positions
        self.black_king_pos = (4, 0);
        self.white_king_pos = (4, 7);
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
        // First check basic move validity
        self.is_valid_basic_move(mv)?;
    
        // Create a test board and make the move
        let mut test_state = self.clone();
        test_state.make_move_without_validation(mv);
    
        // Check if moving piece would leave or put own king in check
        let king_pos = if mv.piece_moved.piece_type == PieceType::King {
            // If moving the king, use the destination square
            mv.to
        } else if mv.piece_moved.color == Color::White {
            test_state.white_king_pos
        } else {
            test_state.black_king_pos
        };
    
        // Check if the king would be in check after the move
        let mut would_be_in_check = false;
        for file in 0..8 {
            for rank in 0..8 {
                if let Some(piece) = test_state.board[file][rank] {
                    if piece.color != mv.piece_moved.color {
                        let attacking_move = Move {
                            from: (file, rank),
                            to: king_pos,
                            piece_moved: piece,
                            piece_captured: test_state.board[king_pos.0][king_pos.1],
                            is_castling: false,
                            is_en_passant: false,
                            promotion: None,
                        };
                        
                        // Only check basic move validity to avoid recursion
                        if test_state.is_valid_basic_move(&attacking_move).is_ok() 
                            && test_state.is_path_clear((file, rank), king_pos) {
                            would_be_in_check = true;
                            break;
                        }
                    }
                }
            }
            if would_be_in_check {
                break;
            }
        }
    
        if would_be_in_check {
            return Err(MoveError::WouldCauseCheck);
        }
    
        Ok(())
    }
    
        println!("DEBUG: Validating {:?} move from {:?} to {:?}", piece.piece_type, mv.from, mv.to);
    
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
        
        println!("DEBUG: Validating pawn move from ({},{}) to ({},{})", fx, fy, tx, ty);
        println!("DEBUG: Pawn color: {:?}", color);
        
        // Direction is negative for White (moving up the board) and positive for Black (moving down)
        let direction = if color == Color::White { -1i32 } else { 1i32 };
        // Starting rank is 6 for White (2nd rank from bottom) and 1 for Black (2nd rank from top)
        let start_rank = if color == Color::White { 6 } else { 1 };
        
        println!("DEBUG: Direction: {}, Start rank: {}", direction, start_rank);
        
        // Convert positions to signed integers for safe arithmetic
        let dx = tx as i32 - fx as i32;
        let dy = ty as i32 - fy as i32;
        
        // Regular forward move
        if dx == 0 && dy == direction {
            if self.get_piece_at(mv.to).is_none() {
                return Ok(());
            }
            return Err(MoveError::InvalidPieceMove("Pawn's path is blocked".to_string()));
        }
        
        // Double move from starting position
        if dx == 0 && dy == 2 * direction && fy == start_rank {
            let between = (fx, (fy as i32 + direction) as usize);
            if self.get_piece_at(mv.to).is_none() && self.get_piece_at(between).is_none() {
                return Ok(());
            }
            return Err(MoveError::InvalidPieceMove("Pawn's double move path is blocked".to_string()));
        }
        
        // Captures (including en passant)
        if dy == direction && dx.abs() == 1 {
            // Regular capture
            if let Some(piece) = self.get_piece_at(mv.to) {
                if piece.color != color {
                    return Ok(());
                }
                return Err(MoveError::InvalidPieceMove("Cannot capture own piece".to_string()));
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
        
        Err(MoveError::InvalidPieceMove("Invalid pawn move pattern".to_string()))
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
        println!("DEBUG: Validating bishop move from {:?} to {:?}", mv.from, mv.to);
        let (fx, fy) = mv.from;
        let (tx, ty) = mv.to;
        
        // Calculate the absolute differences in x and y
        let dx = (tx as i32 - fx as i32).abs();
        let dy = (ty as i32 - fy as i32).abs();
        
        // For a valid bishop move, dx and dy must be equal (diagonal movement)
        if dx == dy && dx > 0 {
            // Calculate the direction of movement
            let x_dir = if tx > fx { 1 } else { -1 };
            let y_dir = if ty > fy { 1 } else { -1 };
            
            // Check each square along the diagonal path
            let mut x = fx as i32 + x_dir;
            let mut y = fy as i32 + y_dir;
            
            while x != tx as i32 || y != ty as i32 {
                if x < 0 || x >= 8 || y < 0 || y >= 8 {
                    return Err(MoveError::OutOfBounds);
                }
                
                if self.board[x as usize][y as usize].is_some() {
                    println!("DEBUG: Bishop path is blocked");
                    return Err(MoveError::BlockedPath);
                }
                
                x += x_dir;
                y += y_dir;
            }
            
            // Check destination square - can't capture own piece
            if let Some(dest_piece) = &self.board[tx][ty] {
                if dest_piece.color == mv.piece_moved.color {
                    return Err(MoveError::InvalidPieceMove("Cannot capture own piece".to_string()));
                }
            }
    
            println!("DEBUG: Bishop path is clear");        
            Ok(())
        } else {
            Err(MoveError::InvalidPieceMove("Bishop must move diagonally".to_string()))
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
        println!("DEBUG: Validating king move from {:?} to {:?}", mv.from, mv.to);
        
        let dx = (mv.to.0 as i32 - mv.from.0 as i32).abs();
        let dy = (mv.to.1 as i32 - mv.from.1 as i32).abs();

        // Normal king move (one square in any direction)
        if dx <= 1 && dy <= 1 {
            // If there's a piece at the destination, it must be an enemy piece
            if let Some(dest_piece) = self.get_piece_at(mv.to) {
                if dest_piece.color == mv.piece_moved.color {
                    return Err(MoveError::InvalidPieceMove("Cannot capture your own piece".to_string()));
                }
                // Capturing an enemy piece is allowed
                return Ok(());
            }
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

    fn is_path_clear(&self, from: (usize, usize), to: (usize, usize)) -> bool {
        println!("DEBUG: Checking path from {:?} to {:?}", from, to);
        
        // Convert to signed integers for safe arithmetic
        let (fx, fy) = (from.0 as i32, from.1 as i32);
        let (tx, ty) = (to.0 as i32, to.1 as i32);
        
        // Calculate direction of movement
        let dx = (tx - fx).signum();
        let dy = (ty - fy).signum();
        
        // Start at first square after origin
        let mut x = fx + dx;
        let mut y = fy + dy;
        
        // Check each square until we reach the destination
        while (x, y) != (tx, ty) {
            if x < 0 || x >= 8 || y < 0 || y >= 8 {
                println!("DEBUG: Path out of bounds at ({},{})", x, y);
                return false;
            }
            
            if self.board[x as usize][y as usize].is_some() {
                println!("DEBUG: Path blocked at ({},{})", x, y);
                return false;
            }
            
            x += dx;
            y += dy;
        }
        
        println!("DEBUG: Path is clear");
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
    
        // Look for any enemy pieces that could capture the king
        for file in 0..8 {
            for rank in 0..8 {
                if let Some(piece) = self.board[file][rank] {
                    if piece.color != color {
                        let test_move = Move {
                            from: (file, rank),
                            to: king_pos,
                            piece_moved: piece,
                            piece_captured: self.board[king_pos.0][king_pos.1],
                            is_castling: false,
                            is_en_passant: false,
                            promotion: None,
                        };
                        
                        // Only check basic move validity to avoid recursion
                        if self.is_valid_basic_move(&test_move).is_ok() 
                            && self.is_path_clear((file, rank), king_pos) {
                            return true;
                        }
                    }
                }
            }
        }
        
        false
    }

    fn would_move_cause_check(&self, mv: &Move) -> bool {
        println!("DEBUG: Checking if move from {:?} to {:?} would cause check", mv.from, mv.to);
        
        let mut test_state = self.clone();
        test_state.make_move_without_validation(mv);
        
        let king_pos = if mv.piece_moved.piece_type == PieceType::King {
            mv.to  // If moving the king, use the destination square
        } else if mv.piece_moved.color == Color::White {
            test_state.white_king_pos
        } else {
            test_state.black_king_pos
        };
        
        println!("DEBUG: Testing king safety at position {:?}", king_pos);
        let result = test_state.is_in_check(mv.piece_moved.color);
        println!("DEBUG: Move would{} result in check", if result { "" } else { " not" });
        result
    }

    pub fn is_square_attacked(&self, pos: (usize, usize), by_color: Color) -> bool {
        println!("DEBUG: Checking if square {:?} is attacked by {:?}", pos, by_color);
        
        for file in 0..8 {
            for rank in 0..8 {
                if let Some(piece) = &self.board[file][rank] {
                    if piece.color == by_color {
                        let from = (file, rank);
                        println!("DEBUG: Found {:?} {:?} at {:?}", piece.color, piece.piece_type, from);
                        
                        // Special case for knights (they can jump)
                        if piece.piece_type == PieceType::Knight {
                            let dx = (pos.0 as i32 - file as i32).abs();
                            let dy = (pos.1 as i32 - rank as i32).abs();
                            if (dx == 2 && dy == 1) || (dx == 1 && dy == 2) {
                                println!("DEBUG: Knight can attack square");
                                return true;
                            }
                            continue;
                        }
                        
                        // Special case for pawns (they capture diagonally)
                        if piece.piece_type == PieceType::Pawn {
                            let direction = if piece.color == Color::White { -1 } else { 1 };
                            if (pos.1 as i32 - rank as i32) == direction 
                                && (pos.0 as i32 - file as i32).abs() == 1 {
                                println!("DEBUG: Pawn can attack square");
                                return true;
                            }
                            continue;
                        }
                        
                        // Test move for all other pieces
                        let mv = Move {
                            from,
                            to: pos,
                            piece_moved: *piece,
                            piece_captured: self.board[pos.0][pos.1],
                            is_castling: false,
                            is_en_passant: false,
                            promotion: None,
                        };
                        
                        // First check basic move validity
                        if self.is_valid_basic_move(&mv).is_ok() {
                            // Then check path
                            if self.is_path_clear(from, pos) {
                                println!("DEBUG: Square can be attacked from {:?}", from);
                                return true;
                            }
                        }
                    }
                }
            }
        }
        
        println!("DEBUG: Square is not attacked");
        false
    }

    pub fn board_to_neural_input(&self) -> Vec<f32> {
        let mut input = Vec::with_capacity(64);
        for rank in 0..8 {
            for file in 0..8 {
                if let Some(piece) = self.board[file][rank] {
                    let base_value = match piece.piece_type {
                        PieceType::Pawn => 1.0,
                        PieceType::Knight => 3.0,
                        PieceType::Bishop => 3.0,
                        PieceType::Rook => 5.0,
                        PieceType::Queen => 9.0,
                        PieceType::King => 100.0,
                    };
                    let value = if piece.color == Color::White {
                        base_value
                    } else {
                        -base_value
                    };
                    input.push(value);
                } else {
                    input.push(0.0);
                }
            }
        }
        input
    }

    pub fn make_move(&mut self, mv: Move) -> Result<(), MoveError> {
        println!("DEBUG: Attempting move from {:?} to {:?}", mv.from, mv.to);
        
        self.is_valid_move(&mv)?;
        
        // Clear source square
        self.board[mv.from.0][mv.from.1] = None;
        println!("DEBUG: Cleared source square");
        
        // Place piece at destination
        self.board[mv.to.0][mv.to.1] = Some(mv.piece_moved);
        println!("DEBUG: Placed piece at destination");
        
        // Update last move
        self.move_history.push(mv.clone());
        
        // Change turn
        self.current_turn = self.current_turn.opposite();
        println!("DEBUG: Changed turn to {:?}", self.current_turn);
        
        Ok(())
    }

    pub fn make_move_without_validation(&mut self, mv: &Move) {
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
    }

    pub fn make_move_from_str(&mut self, input: &str) -> Result<(), String> {
        // Minimal coordinate-based parse: e.g. "e2e4"
        if input.len() < 4 {
            return Err("Move too short (e.g. 'g1f3')".to_string());
        }
        let from = self.parse_square(&input[0..2]).ok_or("Invalid 'from' square")?;
        let to = self.parse_square(&input[2..4]).ok_or("Invalid 'to' square")?;
        let piece = *self.get_piece_at(from).ok_or("No piece at source")?;
        
        // Remove this check since we're checking in play_ai_game
        // if piece.color != self.current_turn {
        //     return Err("Not your turn!".to_string());
        // }
        
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
        
        let file = match s.chars().next()? {
            'a'..='h' => (s.chars().next()? as u8 - b'a') as usize,
            _ => return None
        };
        
        let rank = match s.chars().nth(1)? {
            '1'..='8' => (s.chars().nth(1)? as u8 - b'1') as usize,
            _ => return None
        };
        
        if file < 8 && rank < 8 {
            // Convert chess rank to array index
            let array_rank = 7 - rank;
            println!("DEBUG: Converting {} to coordinates ({}, {})", s, file, array_rank);
            Some((file, array_rank))
        } else {
            None
        }
    }
}
