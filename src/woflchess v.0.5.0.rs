use std::fmt;
use rand::Rng;  // Add this line

// First, let's add piece values as constants
const PAWN_VALUE: i32 = 100;
const KNIGHT_VALUE: i32 = 320;
const BISHOP_VALUE: i32 = 330;
const ROOK_VALUE: i32 = 500;
const QUEEN_VALUE: i32 = 900;
const KING_VALUE: i32 = 20000;

// Piece-square tables for positional evaluation
// These tables give bonuses/penalties based on piece positions
const PAWN_TABLE: [[i32; 8]; 8] = [
    [0,  0,  0,  0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5,  5, 10, 25, 25, 10,  5,  5],
    [0,  0,  0, 20, 20,  0,  0,  0],
    [5, -5,-10,  0,  0,-10, -5,  5],
    [5, 10, 10,-20,-20, 10, 10,  5],
    [0,  0,  0,  0,  0,  0,  0,  0]
];

const KNIGHT_TABLE: [[i32; 8]; 8] = [
    [-50,-40,-30,-30,-30,-30,-40,-50],
    [-40,-20,  0,  0,  0,  0,-20,-40],
    [-30,  0, 10, 15, 15, 10,  0,-30],
    [-30,  5, 15, 20, 20, 15,  5,-30],
    [-30,  0, 15, 20, 20, 15,  0,-30],
    [-30,  5, 10, 15, 15, 10,  5,-30],
    [-40,-20,  0,  5,  5,  0,-20,-40],
    [-50,-40,-30,-30,-30,-30,-40,-50]
];

const BISHOP_TABLE: [[i32; 8]; 8] = [
    [-20,-10,-10,-10,-10,-10,-10,-20],
    [-10,  0,  0,  0,  0,  0,  0,-10],
    [-10,  0,  5, 10, 10,  5,  0,-10],
    [-10,  5,  5, 10, 10,  5,  5,-10],
    [-10,  0, 10, 10, 10, 10,  0,-10],
    [-10, 10, 10, 10, 10, 10, 10,-10],
    [-10,  5,  0,  0,  0,  0,  5,-10],
    [-20,-10,-10,-10,-10,-10,-10,-20]
];

#[derive(Clone, Copy, PartialEq, Debug)]  // Added Debug here
enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

#[derive(Clone, Copy, PartialEq, Debug)]  // Added Debug here!
enum Color {
    White,
    Black,
}

#[derive(Clone, Copy)]
struct Piece {
    piece_type: PieceType,
    color: Color,
}

#[derive(Clone)]
struct GameState {
    board: [[Option<Piece>; 8]; 8],  // None represents empty square
    current_turn: Color,
    move_history: Vec<Move>,
    white_king_pos: (usize, usize),
    black_king_pos: (usize, usize),
    // We'll track castling rights for both players
    white_can_castle_kingside: bool,
    white_can_castle_queenside: bool,
    black_can_castle_kingside: bool,
    black_can_castle_queenside: bool,
    // For en passant
    last_pawn_double_move: Option<(usize, usize)>,
}

#[derive(Clone)]  // Added Clone
struct Move {
    from: (usize, usize),
    to: (usize, usize),
    piece_moved: Piece,
    piece_captured: Option<Piece>,
    // Special moves
    is_castling: bool,
    is_en_passant: bool,
    promotion: Option<PieceType>,
}

// First, let's add Display implementations for our piece-related types
impl fmt::Display for PieceType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Using chess notation characters
        let c = match self {
            PieceType::Pawn => 'P',
            PieceType::Knight => 'N',
            PieceType::Bishop => 'B',
            PieceType::Rook => 'R',
            PieceType::Queen => 'Q',
            PieceType::King => 'K',
        };
        write!(f, "{}", c)
    }
}

impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "\n  a b c d e f g h")?;  // Column labels
        writeln!(f, "  ---------------")?;
        
        for rank in (0..8).rev() {  // Reverse to show white at bottom
            write!(f, "{} ", rank + 1)?;  // Rank labels
            for file in 0..8 {
                match &self.board[file][rank] {
                    Some(piece) => {
                        // Use lowercase for black pieces, uppercase for white
                        let symbol = piece.piece_type.to_string();
                        if piece.color == Color::Black {
                            write!(f, "{} ", symbol.to_lowercase())?;
                        } else {
                            write!(f, "{} ", symbol)?;
                        }
                    }
                    None => write!(f, ". ")?,  // Empty square
                }
            }
            writeln!(f, "{}", rank + 1)?;  // Rank labels on right side too
        }
        writeln!(f, "  ---------------")?;
        writeln!(f, "  a b c d e f g h")?;  // Column labels at bottom
        
        // Display current turn
        writeln!(f, "\nCurrent turn: {:?}", self.current_turn)
    }
}

impl GameState {
    fn new() -> Self {
        let mut state = GameState {
            board: [[None; 8]; 8],
            current_turn: Color::White,
            move_history: Vec::new(),
            white_king_pos: (4, 0),
            black_king_pos: (4, 7),
            white_can_castle_kingside: true,
            white_can_castle_queenside: true,
            black_can_castle_kingside: true,
            black_can_castle_queenside: true,
            last_pawn_double_move: None,
        };
        state.setup_initial_position();
        state
    }

    fn setup_initial_position(&mut self) {
        let create_piece = |piece_type, color| Some(Piece { piece_type, color });
    
        // Set up white pieces (bottom of board)
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
    
        // Set up black pieces (top of board)
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

    fn is_within_bounds(pos: (usize, usize)) -> bool {
        pos.0 < 8 && pos.1 < 8
    }

    fn is_path_clear(&self, from: (usize, usize), to: (usize, usize)) -> bool {
        let (dx, dy) = (
            (to.0 as i32 - from.0 as i32).signum(),
            (to.1 as i32 - from.1 as i32).signum()
        );
        
        let mut x = from.0 as i32 + dx;
        let mut y = from.1 as i32 + dy;
        let to_x = to.0 as i32;
        let to_y = to.1 as i32;

        while (x, y) != (to_x, to_y) {
            if self.board[x as usize][y as usize].is_some() {
                return false;
            }
            x += dx;
            y += dy;
        }
        true
    }

    fn get_piece_at(&self, pos: (usize, usize)) -> Option<&Piece> {
        if Self::is_within_bounds(pos) {
            self.board[pos.0][pos.1].as_ref()
        } else {
            None
        }
    }

    fn is_valid_move(&self, move_: &Move) -> bool {
        if !self.is_valid_basic_move(move_) {
            return false;
        }
        !self.would_move_cause_check(move_)
    }

    fn is_valid_basic_move(&self, move_: &Move) -> bool {
        if !Self::is_within_bounds(move_.from) || !Self::is_within_bounds(move_.to) {
            return false;
        }

        if move_.from == move_.to {
            return false;
        }

        let piece = match self.get_piece_at(move_.from) {
            Some(p) => p,
            None => return false,
        };

        if piece.color != self.current_turn {
            return false;
        }

        if let Some(dest_piece) = self.get_piece_at(move_.to) {
            if dest_piece.color == piece.color {
                return false;
            }
        }

        match piece.piece_type {
            PieceType::Pawn => self.is_valid_pawn_move(move_, piece.color),
            PieceType::Knight => self.is_valid_knight_move(move_),
            PieceType::Bishop => self.is_valid_bishop_move(move_),
            PieceType::Rook => self.is_valid_rook_move(move_),
            PieceType::Queen => self.is_valid_queen_move(move_),
            PieceType::King => self.is_valid_king_move(move_),
        }
    }

    fn is_valid_pawn_move(&self, move_: &Move, color: Color) -> bool {
        let (from_x, from_y) = move_.from;
        let (to_x, to_y) = move_.to;
        let direction = match color {
            Color::White => 1,
            Color::Black => -1,
        };
        let start_rank = match color {
            Color::White => 1,
            Color::Black => 6,
        };
    
        // Normal moves
        let forward_one = ((from_y as i32) + direction) as usize;
        let forward_two = ((from_y as i32) + 2 * direction) as usize;
    
        // Regular forward move
        if to_x == from_x && to_y == forward_one && self.get_piece_at(move_.to).is_none() {
            return true;
        }
    
        // Initial two-square move
        if from_y == start_rank && to_x == from_x && to_y == forward_two {
            return self.get_piece_at(move_.to).is_none() && 
                   self.get_piece_at((from_x, forward_one)).is_none();
        }
    
        // Regular capture
        if (to_y as i32 - from_y as i32) == direction && 
           (to_x as i32 - from_x as i32).abs() == 1 {
            if self.get_piece_at(move_.to).is_some() {
                return true;
            }
        }
    
        // En passant
        if move_.is_en_passant {
            if let Some(last_move) = self.last_pawn_double_move {
                if (to_y as i32 - from_y as i32) == direction &&
                   (to_x as i32 - from_x as i32).abs() == 1 &&
                   to_x == last_move.0 &&
                   from_y == last_move.1 {
                    return true;
                }
            }
        }
    
        false
    }

    fn is_valid_knight_move(&self, move_: &Move) -> bool {
        let dx = (move_.to.0 as i32 - move_.from.0 as i32).abs();
        let dy = (move_.to.1 as i32 - move_.from.1 as i32).abs();
        
        (dx == 2 && dy == 1) || (dx == 1 && dy == 2)
    }

    fn is_valid_bishop_move(&self, move_: &Move) -> bool {
        let dx = (move_.to.0 as i32 - move_.from.0 as i32).abs();
        let dy = (move_.to.1 as i32 - move_.from.1 as i32).abs();
        
        if dx == dy && dx > 0 {
            self.is_path_clear(move_.from, move_.to)
        } else {
            false
        }
    }

    fn is_valid_rook_move(&self, move_: &Move) -> bool {
        let dx = move_.to.0 as i32 - move_.from.0 as i32;
        let dy = move_.to.1 as i32 - move_.from.1 as i32;
        
        if (dx == 0 && dy != 0) || (dx != 0 && dy == 0) {
            self.is_path_clear(move_.from, move_.to)
        } else {
            false
        }
    }

    fn is_valid_queen_move(&self, move_: &Move) -> bool {
        self.is_valid_bishop_move(move_) || self.is_valid_rook_move(move_)
    }

    fn is_valid_king_move(&self, move_: &Move) -> bool {
        let dx = (move_.to.0 as i32 - move_.from.0 as i32).abs();
        let dy = (move_.to.1 as i32 - move_.from.1 as i32).abs();
        
        // Normal king move
        if dx <= 1 && dy <= 1 {
            return true;
        }
    
        // Castling
        if move_.is_castling && dy == 0 && dx == 2 {
            match move_.piece_moved.color {
                Color::White => {
                    if move_.from != (4, 0) { return false; }  // King must be on starting square
                    
                    if move_.to == (6, 0) {  // Kingside
                        if !self.white_can_castle_kingside { return false; }
                        if !self.is_path_clear((4, 0), (7, 0)) { return false; }
                        
                        // Check if king passes through check
                        for x in 4..=6 {
                            if self.is_square_attacked((x, 0), Color::Black) {
                                return false;
                            }
                        }
                        return true;
                    } else if move_.to == (2, 0) {  // Queenside
                        if !self.white_can_castle_queenside { return false; }
                        if !self.is_path_clear((4, 0), (0, 0)) { return false; }
                        
                        // Check if king passes through check
                        for x in 2..=4 {
                            if self.is_square_attacked((x, 0), Color::Black) {
                                return false;
                            }
                        }
                        return true;
                    }
                },
                Color::Black => {
                    if move_.from != (4, 7) { return false; }
                    
                    if move_.to == (6, 7) {  // Kingside
                        if !self.black_can_castle_kingside { return false; }
                        if !self.is_path_clear((4, 7), (7, 7)) { return false; }
                        
                        for x in 4..=6 {
                            if self.is_square_attacked((x, 7), Color::White) {
                                return false;
                            }
                        }
                        return true;
                    } else if move_.to == (2, 7) {  // Queenside
                        if !self.black_can_castle_queenside { return false; }
                        if !self.is_path_clear((4, 7), (0, 7)) { return false; }
                        
                        for x in 2..=4 {
                            if self.is_square_attacked((x, 7), Color::White) {
                                return false;
                            }
                        }
                        return true;
                    }
                }
            }
        }
        false
    }

    fn is_promotion_move(&self, from: (usize, usize), to: (usize, usize)) -> bool {
        if let Some(piece) = self.get_piece_at(from) {
            if piece.piece_type == PieceType::Pawn {
                match piece.color {
                    Color::White => to.1 == 7,  // White pawn reaching 8th rank
                    Color::Black => to.1 == 0,  // Black pawn reaching 1st rank
                }
            } else {
                false
            }
        } else {
            false
        }
    }

    fn make_move(&mut self, move_: Move) -> Result<(), &'static str> {
        if !self.is_valid_move(&move_) {
            return Err("Invalid move");
        }

        self.make_move_without_validation(&move_);
        self.move_history.push(move_);
        
        self.current_turn = match self.current_turn {
            Color::White => Color::Black,
            Color::Black => Color::White,
        };

        Ok(())
    }
    
    // Modify make_move_without_validation to handle promotions
    fn make_move_without_validation(&mut self, move_: &Move) {
        // Handle promotion
        if let Some(promotion_type) = move_.promotion {
            let promoted_piece = Piece {
                piece_type: promotion_type,
                color: move_.piece_moved.color,
            };
            self.board[move_.to.0][move_.to.1] = Some(promoted_piece);
        } else {
            self.board[move_.to.0][move_.to.1] = Some(move_.piece_moved);
        }

        self.board[move_.from.0][move_.from.1] = None;

            // Handle castling rook movement
            if move_.is_castling {
                match (move_.from, move_.to) {
                    ((4, 0), (6, 0)) => {  // White kingside
                        self.board[5][0] = self.board[7][0];
                        self.board[7][0] = None;
                    },
                    ((4, 0), (2, 0)) => {  // White queenside
                        self.board[3][0] = self.board[0][0];
                        self.board[0][0] = None;
                    },
                    ((4, 7), (6, 7)) => {  // Black kingside
                        self.board[5][7] = self.board[7][7];
                        self.board[7][7] = None;
                    },
                    ((4, 7), (2, 7)) => {  // Black queenside
                        self.board[3][7] = self.board[0][7];
                        self.board[0][7] = None;
                    },
                    _ => {}
                }
            }

            // Update last_pawn_double_move for en passant
            if move_.piece_moved.piece_type == PieceType::Pawn {
                let dy = (move_.to.1 as i32 - move_.from.1 as i32).abs();
                if dy == 2 {
                    self.last_pawn_double_move = Some(move_.to);
                } else {
                    self.last_pawn_double_move = None;
                }

                if move_.is_en_passant {
                    let capture_y = move_.from.1;
                    self.board[move_.to.0][capture_y] = None;
                }
            } else {
                self.last_pawn_double_move = None;
            }
  
            // Update king position if king was moved
            if move_.piece_moved.piece_type == PieceType::King {
                match move_.piece_moved.color {
                    Color::White => {
                        self.white_king_pos = move_.to;
                        self.white_can_castle_kingside = false;
                        self.white_can_castle_queenside = false;
                    },
                    Color::Black => {
                        self.black_king_pos = move_.to;
                        self.black_can_castle_kingside = false;
                        self.black_can_castle_queenside = false;
                    }
                }
            }
    
            // Update castling rights if rook moves
            match (move_.from, move_.piece_moved.piece_type) {
                ((0, 0), PieceType::Rook) => self.white_can_castle_queenside = false,
                ((7, 0), PieceType::Rook) => self.white_can_castle_kingside = false,
                ((0, 7), PieceType::Rook) => self.black_can_castle_queenside = false,
                ((7, 7), PieceType::Rook) => self.black_can_castle_kingside = false,
                _ => {}
            }
        }

    fn is_valid_attack_move(&self, move_: &Move) -> bool {
        if !Self::is_within_bounds(move_.from) || !Self::is_within_bounds(move_.to) {
            return false;
        }
    
        if move_.from == move_.to {
            return false;
        }
    
        // Don't check whose turn it is for attacks
        if let Some(dest_piece) = self.get_piece_at(move_.to) {
            if dest_piece.color == move_.piece_moved.color {
                return false;
            }
        }
    
        match move_.piece_moved.piece_type {
            PieceType::Pawn => self.is_valid_pawn_move(move_, move_.piece_moved.color),
            PieceType::Knight => self.is_valid_knight_move(move_),
            PieceType::Bishop => self.is_valid_bishop_move(move_),
            PieceType::Rook => self.is_valid_rook_move(move_),
            PieceType::Queen => self.is_valid_queen_move(move_),
            PieceType::King => self.is_valid_king_move(move_),
        }
    }

    fn is_square_attacked(&self, pos: (usize, usize), by_color: Color) -> bool {
        for i in 0..8 {
            for j in 0..8 {
                if let Some(piece) = self.board[i][j] {
                    if piece.color == by_color {
                        let attack_move = Move {
                            from: (i, j),
                            to: pos,
                            piece_moved: piece,
                            piece_captured: None,
                            is_castling: false,
                            is_en_passant: false,
                            promotion: None,
                        };
                        
                        if self.is_valid_attack_move(&attack_move) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn is_in_check(&self, color: Color) -> bool {
        let king_pos = match color {
            Color::White => self.white_king_pos,
            Color::Black => self.black_king_pos,
        };
        
        let opponent_color = match color {
            Color::White => Color::Black,
            Color::Black => Color::White,
        };

        self.is_square_attacked(king_pos, opponent_color)
    }

    fn would_move_cause_check(&self, move_: &Move) -> bool {
        let mut test_state = self.clone();
        test_state.make_move_without_validation(move_);
        test_state.is_in_check(move_.piece_moved.color)
    }

    fn is_checkmate(&self) -> bool {
        let current_color = self.current_turn;
        
        if !self.is_in_check(current_color) {
            return false;
        }

        for i in 0..8 {
            for j in 0..8 {
                if let Some(piece) = self.board[i][j] {
                    if piece.color == current_color {
                        for x in 0..8 {
                            for y in 0..8 {
                                let test_move = Move {
                                    from: (i, j),
                                    to: (x, y),
                                    piece_moved: piece,
                                    piece_captured: self.board[x][y],
                                    is_castling: false,
                                    is_en_passant: false,
                                    promotion: None,
                                };

                                if self.is_valid_move(&test_move) {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        true
    }

    fn is_stalemate(&self) -> bool {
        if self.is_in_check(self.current_turn) {
            return false;
        }

        for i in 0..8 {
            for j in 0..8 {
                if let Some(piece) = self.board[i][j] {
                    if piece.color == self.current_turn {
                        for x in 0..8 {
                            for y in 0..8 {
                                let test_move = Move {
                                    from: (i, j),
                                    to: (x, y),
                                    piece_moved: piece,
                                    piece_captured: self.board[x][y],
                                    is_castling: false,
                                    is_en_passant: false,
                                    promotion: None,
                                };

                                if self.is_valid_move(&test_move) {
                                    return false;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        true
    }

    fn get_game_status(&self) -> String {
        if self.is_checkmate() {
            format!("Checkmate! {} wins!", 
                match self.current_turn {
                    Color::White => "Black",
                    Color::Black => "White",
                }
            )
        } else if self.is_in_check(self.current_turn) {
            format!("{} is in check!", 
                match self.current_turn {
                    Color::White => "White",
                    Color::Black => "Black",
                }
            )
        } else if self.is_stalemate() {
            "Stalemate! Game is a draw!".to_string()
        } else {
            format!("{}'s turn", 
                match self.current_turn {
                    Color::White => "White",
                    Color::Black => "Black",
                }
            )
        }
    }

    fn display(&self) {
        println!("{}", self);
    }

    fn clear_board(&mut self) {
        self.board = [[None; 8]; 8];
    }

    fn place_piece(&mut self, pos: (usize, usize), piece: Piece) {
        self.board[pos.0][pos.1] = Some(piece);
        if piece.piece_type == PieceType::King {
            match piece.color {
                Color::White => self.white_king_pos = pos,
                Color::Black => self.black_king_pos = pos,
            }
        }
    }

    // Add these new methods
    fn parse_square(&self, square: &str) -> Option<(usize, usize)> {
        let chars: Vec<char> = square.chars().collect();
        if chars.len() != 2 {
            return None;
        }
        
        // Convert file (a-h) to number (0-7)
        let file = match chars[0] {
            'a'..='h' => (chars[0] as u8 - b'a') as usize,
            _ => return None,
        };
        
        // Convert rank (1-8) to number (0-7)
        let rank = match chars[1] {
            '1'..='8' => (chars[1] as u8 - b'1') as usize,
            _ => return None,
        };
        
        Some((file, rank))
    }

    fn create_move(&self, from: (usize, usize), to: (usize, usize)) -> Option<Move> {
        let piece = self.get_piece_at(from)?;
        
        // Check for castling
        let is_castling = piece.piece_type == PieceType::King && 
                         (from.0 as i32 - to.0 as i32).abs() == 2;
        
        // Check for en passant
        let is_en_passant = if let Some(last_move) = self.last_pawn_double_move {
            piece.piece_type == PieceType::Pawn && 
            to.0 == last_move.0 && 
            from.1 == last_move.1
        } else {
            false
        };
        
        Some(Move {
            from,
            to,
            piece_moved: piece.clone(),
            piece_captured: self.get_piece_at(to).cloned(),
            is_castling,
            is_en_passant,
            promotion: None,  // We'll handle promotion separately
        })
    }

    fn parse_algebraic_notation(&self, move_str: &str) -> Result<((usize, usize), (usize, usize)), &'static str> {
        // Special cases first
        match move_str {
            "O-O" | "0-0" => {
                let (rank, king_file, to_file) = match self.current_turn {
                    Color::White => (0, 4, 6),
                    Color::Black => (7, 4, 6),
                };
                return Ok(((king_file, rank), (to_file, rank)));
            },
            "O-O-O" | "0-0-0" => {
                let (rank, king_file, to_file) = match self.current_turn {
                    Color::White => (0, 4, 2),
                    Color::Black => (7, 4, 2),
                };
                return Ok(((king_file, rank), (to_file, rank)));
            },
            _ => {}
        }
    
        // Regular algebraic notation (e.g., "Nf3", "e4", "Bxe5")
        let chars: Vec<char> = move_str.chars().collect();
        if chars.is_empty() {
            return Err("Empty move string");
        }
    
        let (piece_type, _start_idx) = match chars[0] {
            'N' => (PieceType::Knight, 1),
            'B' => (PieceType::Bishop, 1),
            'R' => (PieceType::Rook, 1),
            'Q' => (PieceType::Queen, 1),
            'K' => (PieceType::King, 1),
            'a'..='h' => (PieceType::Pawn, 0),  // Pawn moves don't have a prefix
            _ => return Err("Invalid move notation"),
        };
    
        // For pawn moves, we can directly parse the destination
        if piece_type == PieceType::Pawn {
            let to = self.parse_square(move_str)
                .ok_or("Invalid destination square")?;
            
            // Find the pawn that can make this move
            let from = self.find_piece_that_can_move(piece_type, to)?;
            return Ok((from, to));
        }
    
        // For other pieces, get the destination from the last two characters
        let dest_str = &move_str[move_str.len()-2..];
        let to = self.parse_square(dest_str)
            .ok_or("Invalid destination square")?;
    
        // Find the piece that can make this move
        let from = self.find_piece_that_can_move(piece_type, to)?;
    
        Ok((from, to))
    }

    
    fn get_piece_value(&self, piece: &Piece) -> i32 {
      match piece.piece_type {
          PieceType::Pawn => PAWN_VALUE,
          PieceType::Knight => KNIGHT_VALUE,
          PieceType::Bishop => BISHOP_VALUE,
          PieceType::Rook => ROOK_VALUE,
          PieceType::Queen => QUEEN_VALUE,
          PieceType::King => KING_VALUE,
      }
  }

  fn get_position_bonus(&self, piece: &Piece, pos: (usize, usize)) -> i32 {
      let (x, y) = pos;
      // For black pieces, we flip the position on the board
      let (row, col) = if piece.color == Color::Black {
          (7 - y, x)
      } else {
          (y, x)
      };

      match piece.piece_type {
          PieceType::Pawn => PAWN_TABLE[row][col],
          PieceType::Knight => KNIGHT_TABLE[row][col],
          PieceType::Bishop => BISHOP_TABLE[row][col],
          _ => 0  // No position bonus for other pieces yet
      }
  }

  fn evaluate_position(&self) -> i32 {
      let mut score = 0;
      
      // Material and basic positional evaluation
      for i in 0..8 {
          for j in 0..8 {
              if let Some(piece) = self.board[i][j] {
                  let multiplier = if piece.color == Color::White { 1 } else { -1 };
                  let piece_value = self.get_piece_value(&piece);
                  let position_bonus = self.get_position_bonus(&piece, (i, j));
                  
                  score += multiplier * (piece_value + position_bonus);
              }
          }
      }

      // Penalize doubled pawns
      for file in 0..8 {
          let mut white_pawns = 0;
          let mut black_pawns = 0;
          for rank in 0..8 {
              if let Some(piece) = self.board[file][rank] {
                  if piece.piece_type == PieceType::Pawn {
                      if piece.color == Color::White {
                          white_pawns += 1;
                      } else {
                          black_pawns += 1;
                      }
                  }
              }
          }
          if white_pawns > 1 {
              score -= 20 * (white_pawns - 1);  // Penalty for doubled pawns
          }
          if black_pawns > 1 {
              score += 20 * (black_pawns - 1);  // Penalty for opponent's doubled pawns
          }
      }

      // Additional evaluation for check and checkmate
      if self.is_checkmate() {
          if self.current_turn == Color::White {
              score = -30000;  // White is checkmated
          } else {
              score = 30000;   // Black is checkmated
          }
      } else if self.is_in_check(self.current_turn) {
          if self.current_turn == Color::White {
              score -= 50;     // White is in check
          } else {
              score += 50;     // Black is in check
          }
      }

      score
  }

    fn find_piece_that_can_move(&self, piece_type: PieceType, to: (usize, usize)) -> Result<(usize, usize), &'static str> {
        let mut valid_pieces = Vec::new();

        for i in 0..8 {
            for j in 0..8 {
                if let Some(piece) = self.board[i][j] {
                    if piece.piece_type == piece_type && piece.color == self.current_turn {
                        let test_move = Move {
                            from: (i, j),
                            to,
                            piece_moved: piece,
                            piece_captured: self.board[to.0][to.1],
                            is_castling: false,
                            is_en_passant: false,
                            promotion: None,
                        };

                        if self.is_valid_move(&test_move) {
                            valid_pieces.push((i, j));
                        }
                    }
                }
            }
        }

        match valid_pieces.len() {
            0 => Err("No piece can make that move"),
            1 => Ok(valid_pieces[0]),
            _ => Err("Ambiguous move - multiple pieces can move there"),
        }
    }

    // Add minimax search with alpha-beta pruning
    fn minimax(&mut self, depth: i32, mut alpha: i32, mut beta: i32, maximizing_player: bool) -> i32 {
        if depth == 0 {
            return self.evaluate_position();
        }

        let legal_moves = self.generate_legal_moves();
        
        // Check for terminal states
        if legal_moves.is_empty() {
            if self.is_checkmate() {
                return if maximizing_player { -30000 } else { 30000 };
            }
            return 0; // Stalemate
        }

        if maximizing_player {
            let mut max_eval = i32::MIN;
            for move_ in legal_moves {
                let mut new_state = self.clone();
                new_state.make_move_without_validation(&move_);
                let eval = new_state.minimax(depth - 1, alpha, beta, false);
                max_eval = max_eval.max(eval);
                alpha = alpha.max(eval);
                if beta <= alpha {
                    break; // Beta cutoff
                }
            }
            max_eval
        } else {
            let mut min_eval = i32::MAX;
            for move_ in legal_moves {
                let mut new_state = self.clone();
                new_state.make_move_without_validation(&move_);
                let eval = new_state.minimax(depth - 1, alpha, beta, true);
                min_eval = min_eval.min(eval);
                beta = beta.min(eval);
                if beta <= alpha {
                    break; // Alpha cutoff
                }
            }
            min_eval
        }
    }

    fn parse_move_string(&self, move_str: &str) -> Result<((usize, usize), (usize, usize)), &'static str> {
        // Parse normal moves like "e2 to e4" or "e2e4"
        let parts: Vec<&str> = move_str.split_whitespace().collect();
        let (from_str, to_str) = match parts.len() {
            2 => (parts[0], parts[1]),  // Format: "e2 e4"
            3 if parts[1] == "to" => (parts[0], parts[2]),  // Format: "e2 to e4"
            1 if move_str.len() == 4 => (&move_str[0..2], &move_str[2..4]),  // Format: "e2e4"
                _ => return Err("Invalid move format. Use 'e2e4', 'e2 e4', or 'e2 to e4'"),
        };
    
        let from = self.parse_square(from_str)
            .ok_or("Invalid starting square - use a letter (a-h) followed by a number (1-8)")?;
        let to = self.parse_square(to_str)
            .ok_or("Invalid destination square - use a letter (a-h) followed by a number (1-8)")?;
    
        Ok((from, to))
    }

   // Modify make_move_from_str to handle promotion notation
   fn make_move_from_str(&mut self, move_str: &str) -> Result<(), &'static str> {
    // Handle promotion notation like "e7e8Q" or "e7e8=Q"
    let mut promotion_type = None;
    let (basic_move, promotion_char) = if move_str.len() >= 5 {
        let (main_move, promotion) = move_str.split_at(4);
        let promotion = promotion.trim_start_matches('=');
        match promotion.chars().next() {
            Some('Q') => (main_move, Some(PieceType::Queen)),
            Some('R') => (main_move, Some(PieceType::Rook)),
            Some('B') => (main_move, Some(PieceType::Bishop)),
            Some('N') => (main_move, Some(PieceType::Knight)),
            _ => (move_str, None),
        }
    } else {
        (move_str, None)
    };

    let (from, to) = if basic_move.contains(' ') || basic_move.len() == 4 {
        self.parse_move_string(basic_move)?
    } else {
        self.parse_algebraic_notation(basic_move)?
    };

    let piece = self.get_piece_at(from)
        .ok_or("No piece at starting square")?;

    if piece.color != self.current_turn {
        return Err("It's not your turn!");
    }

    // Check if this is a promotion move
    if piece.piece_type == PieceType::Pawn && self.is_promotion_move(from, to) {
        // If no promotion piece specified, default to Queen
        promotion_type = Some(promotion_type.unwrap_or(PieceType::Queen));
    }

    let move_ = Move {
        from,
        to,
        piece_moved: piece,
        piece_captured: self.get_piece_at(to).cloned(),
        is_castling: false,
        is_en_passant: false,
        promotion: promotion_type,
    };

        if !self.is_valid_move(&move_) {
            return match piece.piece_type {
                PieceType::Pawn => Err("Invalid pawn move - pawns can only move forward one square (or two on their first move) and capture diagonally"),
                PieceType::Knight => Err("Invalid knight move - knights move in an L-shape (2 squares in one direction and 1 square perpendicular)"),
                PieceType::Bishop => Err("Invalid bishop move - bishops can only move diagonally"),
                PieceType::Rook => Err("Invalid rook move - rooks can only move horizontally or vertically"),
                PieceType::Queen => Err("Invalid queen move - queens can move horizontally, vertically, or diagonally"),
                PieceType::King => {
                    if move_.is_castling {
                        Err("Invalid castling - ensure the king hasn't moved, the path is clear, and you're not castling through check")
                    } else {
                        Err("Invalid king move - kings can only move one square in any direction")
                    }
                },
            };
        }

        self.make_move(move_)
    }
           // Make the move    
        fn handle_castling_move(&mut self, from: (usize, usize), to: (usize, usize)) -> Result<(), &'static str> {
            let piece = self.get_piece_at(from)
                .ok_or("No king at starting square")?;
            
            let move_ = Move {
                from,
                to,
                piece_moved: piece.clone(),
                piece_captured: None,
                is_castling: true,
                is_en_passant: false,
                promotion: None,
            };
    
            self.make_move(move_)
        }

  // Modify generate_legal_moves to use move ordering
  fn generate_legal_moves(&self) -> Vec<Move> {
    let mut moves = Vec::new();
    
    for i in 0..8 {
        for j in 0..8 {
            if let Some(piece) = self.board[i][j] {
                if piece.color == self.current_turn {
                    // Try all possible destination squares
                    for x in 0..8 {
                        for y in 0..8 {
                            let test_move = Move {
                                from: (i, j),
                                to: (x, y),
                                piece_moved: piece,
                                piece_captured: self.board[x][y],
                                is_castling: false,
                                is_en_passant: false,
                                promotion: None,
                            };
                            
                            if self.is_valid_move(&test_move) {
                                moves.push(test_move);
                            }
                        }
                    }
                    
                    // Check castling moves for kings
                    if piece.piece_type == PieceType::King {
                        let rank = if piece.color == Color::White { 0 } else { 7 };
                        
                        // Kingside castling
                        let kingside_move = Move {
                            from: (4, rank),
                            to: (6, rank),
                            piece_moved: piece,
                            piece_captured: None,
                            is_castling: true,
                            is_en_passant: false,
                            promotion: None,
                        };
                        if self.is_valid_move(&kingside_move) {
                            moves.push(kingside_move);
                        }
                        
                        // Queenside castling
                        let queenside_move = Move {
                            from: (4, rank),
                            to: (2, rank),
                            piece_moved: piece,
                            piece_captured: None,
                            is_castling: true,
                            is_en_passant: false,
                            promotion: None,
                        };
                        if self.is_valid_move(&queenside_move) {
                            moves.push(queenside_move);
                        }
                    }
                }
            }
        }
    }
    
    // Order moves for better alpha-beta pruning
    self.order_moves(&mut moves);
    moves
}
        
        legal_moves
    }

    fn make_computer_move(&mut self) -> Result<(), &'static str> {
        let legal_moves = self.generate_legal_moves();
        
        if legal_moves.is_empty() {
            if self.is_checkmate() {
                return Err("Checkmate!");
            } else {
                return Err("Stalemate!");
            }
        }

        let mut best_score = i32::MIN;
        let mut best_moves = Vec::new();
        let search_depth = 3; // Look ahead 3 moves (can be adjusted)
        
        // Search all possible moves
        for move_ in legal_moves {
            let mut test_state = self.clone();
            test_state.make_move_without_validation(&move_);
            
            // Evaluate this move using minimax
            let score = -test_state.minimax(search_depth - 1, i32::MIN, i32::MAX, true);
            
            if score > best_score {
                best_score = score;
                best_moves.clear();
                best_moves.push(move_);
            } else if score == best_score {
                best_moves.push(move_);
            }
        }

        // Randomly select from among the best moves
        let mut rng = rand::thread_rng();
        let selected_move = best_moves[rng.gen_range(0..best_moves.len())].clone();
        
        // Make the selected move and announce it
        let move_text = self.move_to_algebraic(&selected_move);
        println!("Computer evaluates position as: {}", best_score);
        self.make_move(selected_move)?;
        println!("Computer plays: {}", move_text);
        
        Ok(())
    }
    
    // Convert a move to algebraic notation for display
    fn move_to_algebraic(&self, move_: &Move) -> String {
        if move_.is_castling {
            if move_.to.0 == 6 { // Kingside
                return "O-O".to_string();
            } else { // Queenside
                return "O-O-O".to_string();
            }
        }
        
        let piece_char = match move_.piece_moved.piece_type {
            PieceType::Pawn => "",
            PieceType::Knight => "N",
            PieceType::Bishop => "B",
            PieceType::Rook => "R",
            PieceType::Queen => "Q",
            PieceType::King => "K",
        };
        
        let capture_char = if move_.piece_captured.is_some() || move_.is_en_passant { "x" } else { "" };
        
        let dest_file = (move_.to.0 as u8 + b'a') as char;
        let dest_rank = (move_.to.1 + 1).to_string();
        
        format!("{}{}{}{}", piece_char, capture_char, dest_file, dest_rank)
    }
}

// Modified main function to include computer play
fn main() {
    let mut game = GameState::new();
    println!("Welcome to Chess!");
    println!("You play as White against the computer.");
    println!("Enter moves in the format: 'e2e4', 'e2 e4', 'e4', or 'Nf3'");
    println!("Type 'quit' to exit\n");
    
    game.display();

    loop {
        println!("\n{}", game.get_game_status());
        
        // Player's turn (White)
        println!("Enter your move:");
        let mut input = String::new();
        
        if let Err(e) = std::io::stdin().read_line(&mut input) {
            println!("Error reading input: {}", e);
            continue;
        }

        let input = input.trim();
        
        if input == "quit" {
            break;
        }

        // Make player's move
        match game.make_move_from_str(input) {
            Ok(()) => {
                println!("Move successful!");
                game.display();
                
                // Check if game is over after player's move
                if game.is_checkmate() {
                    println!("Checkmate! You win!");
                    break;
                }
                if game.is_stalemate() {
                    println!("Stalemate! Game is a draw!");
                    break;
                }
                
                // Computer's turn
                println!("\nComputer is thinking...");
                match game.make_computer_move() {
                    Ok(()) => {
                        println!("Computer moved!");
                        game.display();
                        
                        // Check if game is over after computer's move
                        if game.is_checkmate() {
                            println!("Checkmate! Computer wins!");
                            break;
                        }
                        if game.is_stalemate() {
                            println!("Stalemate! Game is a draw!");
                            break;
                        }
                    },
                    Err(e) => {
                        println!("Computer error: {}", e);
                        break;
                    }
                }
            },
            Err(e) => println!("❌ {}", e),
        }
    }
}