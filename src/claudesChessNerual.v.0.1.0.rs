// ---------- LOAD CRATES ----------

use bitflags::bitflags;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use rand::Rng;

// Neural network activation functions
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn relu(x: f32) -> f32 {
    x.max(0.0)
}

// ---------- TYPES MODULE ----------
mod types {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum PieceType {
        Pawn = 0,
        Knight = 1, 
        Bishop = 2,
        Rook = 3,
        Queen = 4,
        King = 5,
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum Color {
        White,
        Black,
    }

    impl Color {
        #[inline]
        pub fn opposite(&self) -> Self {
            match self {
                Color::White => Color::Black,
                Color::Black => Color::White,
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct Piece {
        pub piece_type: PieceType,
        pub color: Color,
    }

    // Optimized move struct with bitflags
    #[derive(Clone, Debug)]
    pub struct Move {
        pub from: Square,
        pub to: Square,
        pub piece_moved: Piece,
        pub piece_captured: Option<Piece>,
        pub flags: MoveFlags,
        pub promotion: Option<PieceType>,
    }

    // Type aliases for clarity
    pub type Square = (u8, u8);
    pub type Board = [[Option<Piece>; 8]; 8];

    // Bitflags for move properties
    bitflags! {
        pub struct MoveFlags: u8 {
            const NORMAL     = 0b00000000;
            const CAPTURE    = 0b00000001;
            const CASTLE     = 0b00000010;
            const EN_PASSANT = 0b00000100;
            const PROMOTION  = 0b00001000;
            const CHECK      = 0b00010000;
            const CHECKMATE  = 0b00100000;
        }
    }
}

// ---------- CONSTANTS FOR PIECE MOVEMENTS ----------
const BISHOP_DIRECTIONS: [(i8, i8); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
const ROOK_DIRECTIONS: [(i8, i8); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
const KNIGHT_MOVES: [(i8, i8); 8] = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1)
];

// ---------- NEURAL MODULE ----------
mod neural {
    use super::*;
    use ndarray::{Array1, Array2};

    pub struct OptimizedNN {
        layers: Vec<Layer>,
        learning_rate: f32,
    }

    struct Layer {
        weights: Array2<f32>,
        biases: Array1<f32>,
        activation: fn(f32) -> f32,
    }

    impl OptimizedNN {
        pub fn new(layer_sizes: &[usize], learning_rate: f32) -> Self {
            let mut layers = Vec::with_capacity(layer_sizes.len() - 1);
            for i in 0..layer_sizes.len() - 1 {
                layers.push(Layer {
                    weights: Array2::zeros((layer_sizes[i+1], layer_sizes[i])),
                    biases: Array1::zeros(layer_sizes[i+1]),
                    activation: if i == layer_sizes.len() - 2 { sigmoid } else { relu },
                });
            }
            Self { layers, learning_rate }
        }

        pub fn forward(&self, input: &Array1<f32>) -> Array1<f32> {
            self.layers.iter().fold(input.clone(), |acc, layer| {
                let z = layer.weights.dot(&acc) + &layer.biases;
                z.mapv(layer.activation)
            })
        }

        pub fn train_batch(&mut self, _inputs: &[Array1<f32>], _targets: &[Array1<f32>]) {
            // TODO: Implement backpropagation logic here.
        }
    }
}

// ---------- TRANSPOSITION TABLE IMPLEMENTATION ----------
pub struct TranspositionTable {
    pub entries: Vec<Option<TableEntry>>,
}

pub struct TableEntry {
    pub zobrist_hash: u64,
    pub depth: i32,
    pub score: i32,
    pub best_move: Option<types::Move>,
}

impl TranspositionTable {
    pub fn with_capacity(size: usize) -> Self {
        Self {
            entries: vec![None; size],
        }
    }
}

// ---------- MOVE GENERATOR IMPLEMENTATION ----------
pub struct MoveGenerator {
    pub attack_tables: AttackTables,
}

pub struct AttackTables {
    pub pawn_attacks: [[u64; 64]; 2],  // One for each color
    pub knight_attacks: [u64; 64],
    pub king_attacks: [u64; 64],
}

impl MoveGenerator {
    pub fn new() -> Self {
        Self {
            attack_tables: AttackTables {
                pawn_attacks: [[0; 64]; 2],
                knight_attacks: [0; 64],
                king_attacks: [0; 64],
            },
        }
    }

    pub fn generate_moves(&self, _game: &GameState) -> Vec<types::Move> {
        // TODO: Implement move generation logic.
        Vec::new()  // Placeholder
    }
}

// ---------- GAME STATE IMPLEMENTATION ----------
use crate::types::{Board, Color, Move as ChessMove, Square, PieceType, Piece};

struct GameState {
    board: Board,
    current_turn: Color,
    move_history: Vec<types::Move>,
    king_positions: [(u8, u8); 2],  // [white_king, black_king]
    castling_rights: u8,            // Bitflags for castling
    last_move: Option<types::Move>,
    zobrist_hash: u64,              // For transposition table
}

impl GameState {
    fn new() -> Self {
        let mut state = Self {
            board: [[None; 8]; 8],
            current_turn: Color::White,
            move_history: Vec::with_capacity(40),
            king_positions: [(4, 0), (4, 7)],
            castling_rights: 0b1111,
            last_move: None,
            zobrist_hash: 0,
        };
        state.setup_initial_position();
        state.zobrist_hash = state.compute_zobrist_hash();
        state
    }

    // Stub: Set up the initial board position.
    fn setup_initial_position(&mut self) {
        // TODO: Initialize board with starting pieces.
    }

    // Stub: Compute Zobrist hash.
    fn compute_zobrist_hash(&self) -> u64 {
        // TODO: Implement proper Zobrist hash computation.
        0
    }

    // Check if a square is within bounds.
    fn is_within_bounds(&self, square: Square) -> bool {
        let (x, y) = square;
        x < 8 && y < 8
    }

    // Generate moves for the piece on the given square.
    fn generate_moves_for_piece(&self, square: Square) -> Vec<types::Move> {
        let mut moves = Vec::new();
        let piece = match self.board[square.0 as usize][square.1 as usize] {
            Some(p) => p,
            None => return moves,
        };

        match piece.piece_type {
            PieceType::Pawn => self.generate_pawn_moves(square, &mut moves),
            PieceType::Knight => self.generate_knight_moves(square, &mut moves),
            PieceType::Bishop => self.generate_sliding_moves(square, &BISHOP_DIRECTIONS, &mut moves),
            PieceType::Rook => self.generate_sliding_moves(square, &ROOK_DIRECTIONS, &mut moves),
            PieceType::Queen => {
                self.generate_sliding_moves(square, &BISHOP_DIRECTIONS, &mut moves);
                self.generate_sliding_moves(square, &ROOK_DIRECTIONS, &mut moves);
            },
            PieceType::King => self.generate_king_moves(square, &mut moves),
        }

        moves.into_iter()
            .filter(|mov| !self.would_be_in_check_after_move(mov))
            .collect()
    }

    // Stub: Generate pawn moves.
    fn generate_pawn_moves(&self, _square: Square, _moves: &mut Vec<types::Move>) -> Vec<types::Move> {
        // TODO: Implement pawn move generation.
        Vec::new()
    }

    // Stub: Generate knight moves.
    fn generate_knight_moves(&self, _square: Square, _moves: &mut Vec<types::Move>) -> Vec<types::Move> {
        // TODO: Implement knight move generation.
        Vec::new()
    }

    // Stub: Generate sliding moves (bishop, rook, queen).
    fn generate_sliding_moves(&self, _square: Square, _directions: &[(i8, i8)], _moves: &mut Vec<types::Move>) -> Vec<types::Move> {
        // TODO: Implement sliding move generation.
        Vec::new()
    }

    // Stub: Generate king moves.
    fn generate_king_moves(&self, _square: Square, _moves: &mut Vec<types::Move>) -> Vec<types::Move> {
        // TODO: Implement king move generation.
        Vec::new()
    }

    // Stub: Check if the move would leave the king in check.
    fn would_be_in_check_after_move(&self, _mov: &types::Move) -> bool {
        // TODO: Implement check detection.
        false
    }

    // Check if a move is valid.
    fn is_valid_move(&self, mov: &types::Move) -> bool {
        if !self.is_within_bounds(mov.from) || !self.is_within_bounds(mov.to) {
            return false;
        }

        let piece = match self.board[mov.from.0 as usize][mov.from.1 as usize] {
            Some(p) => p,
            None => return false,
        };

        if piece.color != self.current_turn {
            return false;
        }

        let valid_moves = self.generate_moves_for_piece(mov.from);
        valid_moves.contains(mov)
    }

    // Stub: Get piece at a square.
    fn get_piece(&self, square: Square) -> Option<Piece> {
        self.board[square.0 as usize][square.1 as usize]
    }

    // Stub: Check for checkmate.
    fn is_checkmate(&self) -> bool {
        // TODO: Implement checkmate detection.
        false
    }

    // Stub: Check for stalemate.
    fn is_stalemate(&self) -> bool {
        // TODO: Implement stalemate detection.
        false
    }

    // Dummy implementation: Convert game state to a neural network input.
    fn to_neural_input(&self) -> Array1<f32> {
        // TODO: Create a proper board encoding.
        Array1::zeros(384)
    }
}

// ---------- FREE FUNCTION: PAWN ATTACKS ----------
fn get_pawn_attacks(_square: types::Square, _color: types::Color) -> Vec<types::Square> {
    // TODO: Implement pawn attack square calculation.
    Vec::new()
}

// ---------- ENGINE MODULE ----------
mod engine {
    use super::*;
    use crate::neural::OptimizedNN;
    use crate::types::{Move, Color, PieceType};
    use ndarray::Array1;

    pub struct ChessEngine {
        pub game: GameState,
        pub neural_net: OptimizedNN,
        pub transposition_table: TranspositionTable,
        pub move_generator: MoveGenerator,
    }

    impl ChessEngine {
        pub fn new() -> Self {
            Self {
                game: GameState::new(),
                neural_net: OptimizedNN::new(&[384, 512, 256, 1], 0.001),
                transposition_table: TranspositionTable::with_capacity(1_000_000),
                move_generator: MoveGenerator::new(),
            }
        }

        // Dummy alpha-beta search stub.
        pub fn alpha_beta(&mut self, _depth: i32, _alpha: i32, _beta: i32, _maximizing_player: bool) -> i32 {
            // TODO: Implement alpha-beta search.
            0
        }

        // Dummy scoring function.
        pub fn score_move(&self, _mov: &Move) -> i32 {
            // TODO: Implement move scoring.
            0
        }

        // Stub: Display board.
        pub fn display_board(&self) {
            println!("Displaying board (stub)");
        }

        // Stub: Parse a move from a string.
        pub fn make_move_from_string(&mut self, _input: &str) -> Result<(), String> {
 
 
            / ---------- IMPROVED GAME STATE ----------
            struct GameState {
                board: Board,
                current_turn: Color,
                move_history: Vec<Move>,
                king_positions: [(u8, u8); 2],  // [white_king, black_king]
                castling_rights: u8,            // Bitflags for castling
                last_move: Option<Move>,
                zobrist_hash: u64,              // For transposition table
            }
            
            impl GameState {
                fn new() -> Self {
                    let mut state = Self {
                        board: [[None; 8]; 8],
                        current_turn: Color::White,
                        move_history: Vec::with_capacity(40),  // Pre-allocate for typical game
                        king_positions: [(4, 0), (4, 7)],
                        castling_rights: 0b1111,  // All castling initially allowed
                        last_move: None,
                        zobrist_hash: 0,
                    };
                    state.setup_initial_position();
                    state.zobrist_hash = state.compute_zobrist_hash();
                    state
                }
            
                // Optimized move validation using bitboards
                fn is_valid_move(&self, mov: &Move) -> bool {
                    // Check basic move validity
                    if !self.is_within_bounds(mov.from) || !self.is_within_bounds(mov.to) {
                        return false;
                    }
            
                    let piece = match self.board[mov.from.0 as usize][mov.from.1 as usize] {
                        Some(p) => p,
                        None => return false,
                    };
            
                    // Verify piece color matches current turn
                    if piece.color != self.current_turn {
                        return false;
                    }
            
                    // Generate valid moves for the piece
                    let valid_moves = self.generate_moves_for_piece(mov.from);
                    valid_moves.contains(mov)
                }
            
                // Optimized move generation using bitboards
                fn generate_moves_for_piece(&self, square: Square) -> Vec<Move> {
                    let mut moves = Vec::new();
                    let piece = match self.board[square.0 as usize][square.1 as usize] {
                        Some(p) => p,
                        None => return moves,
                    };
            
                    match piece.piece_type {
                        PieceType::Pawn => self.generate_pawn_moves(square, &mut moves),
                        PieceType::Knight => self.generate_knight_moves(square, &mut moves),
                        PieceType::Bishop => self.generate_sliding_moves(square, &BISHOP_DIRECTIONS, &mut moves),
                        PieceType::Rook => self.generate_sliding_moves(square, &ROOK_DIRECTIONS, &mut moves),
                        PieceType::Queen => {
                            self.generate_sliding_moves(square, &BISHOP_DIRECTIONS, &mut moves);
                            self.generate_sliding_moves(square, &ROOK_DIRECTIONS, &mut moves);
                        },
                        PieceType::King => self.generate_king_moves(square, &mut moves),
                    }
            
                    // Filter moves that would leave king in check
                    moves.into_iter()
                        .filter(|mov| !self.would_be_in_check_after_move(mov))
                        .collect()
                }
            
                // Efficient check detection using piece attack maps
                fn is_in_check(&self, color: Color) -> bool {
                    let king_pos = self.king_positions[color as usize];
                    self.is_square_attacked(king_pos, color.opposite())
                }
            
                // Optimized using pre-computed attack tables
                fn is_square_attacked(&self, square: Square, by_color: Color) -> bool {
                    // Check pawn attacks
                    let pawn_attacks = get_pawn_attacks(square, by_color.opposite());
                    for attack_square in pawn_attacks {
                        if let Some(piece) = self.get_piece(attack_square) {
                            if piece.piece_type == PieceType::Pawn && piece.color == by_color {
                                return true;
                            }
                        }
                    }
            
                    // Check knight attacks
                    const KNIGHT_MOVES: [(i8, i8); 8] = [
                        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
                        (1, -2), (1, 2), (2, -1), (2, 1)
                    ];
                    
                    for &(dx, dy) in &KNIGHT_MOVES {
                        let new_x = square.0 as i8 + dx;
                        let new_y = square.1 as i8 + dy;
                        if new_x >= 0 && new_x < 8 && new_y >= 0 && new_y < 8 {
                            if let Some(piece) = self.get_piece((new_x as u8, new_y as u8)) {
                                if piece.piece_type == PieceType::Knight && piece.color == by_color {
                                    return true;
                                }
                            }
                        }
                    }
            
                    // Check sliding pieces (bishop, rook, queen)
                    const BISHOP_DIRECTIONS: [(i8, i8); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
                    const ROOK_DIRECTIONS: [(i8, i8); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
            
                    // Check bishop and queen diagonal attacks
                    for &(dx, dy) in &BISHOP_DIRECTIONS {
                        let mut x = square.0 as i8 + dx;
                        let mut y = square.1 as i8 + dy;
                        while x >= 0 && x < 8 && y >= 0 && y < 8 {
                            if let Some(piece) = self.get_piece((x as u8, y as u8)) {
                                if piece.color == by_color && 
                                   (piece.piece_type == PieceType::Bishop || 
                                    piece.piece_type == PieceType::Queen) {
                                    return true;
                                }
                                break; // Stop checking this direction if we hit any piece
                            }
                            x += dx;
                            y += dy;
                        }
                    }
            
                    // Check rook and queen orthogonal attacks
                    for &(dx, dy) in &ROOK_DIRECTIONS {
                        let mut x = square.0 as i8 + dx;
                        let mut y = square.1 as i8 + dy;
                        while x >= 0 && x < 8 && y >= 0 && y < 8 {
                            if let Some(piece) = self.get_piece((x as u8, y as u8)) {
                                if piece.color == by_color && 
                                   (piece.piece_type == PieceType::Rook || 
                                    piece.piece_type == PieceType::Queen) {
                                    return true;
                                }
                                break; // Stop checking this direction if we hit any piece
                            }
                            x += dx;
                            y += dy;
                        }
                    }
            
                    // Check king attacks (for adjacent squares)
                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            if dx == 0 && dy == 0 { continue; }
                            let new_x = square.0 as i8 + dx;
                            let new_y = square.1 as i8 + dy;
                            if new_x >= 0 && new_x < 8 && new_y >= 0 && new_y < 8 {
                                if let Some(piece) = self.get_piece((new_x as u8, new_y as u8)) {
                                    if piece.piece_type == PieceType::King && piece.color == by_color {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
            
                    false // Square is not attacked if none of the above checks found an attacking piece
                }    // TODO: Parse input and update game state.
            Ok(())
        }

        // Stub: Execute a move.
        pub fn make_move(&mut self, _mov: &Move) -> Result<(), String> {
            // TODO: Update game state with the move.
            Ok(())
        }

        // Optimized alpha-beta search with move ordering.
        pub fn find_best_move(&mut self, depth: i32) -> Option<Move> {
            let mut alpha = i32::MIN;
            let beta = i32::MAX;

            let legal_moves = self.move_generator.generate_moves(&self.game);
            if legal_moves.is_empty() {
                return None;
            }

            let ordered_moves = self.order_moves(legal_moves);

            let mut best_move = None;
            let mut best_score = i32::MIN;

            for mov in ordered_moves {
                let score = -self.alpha_beta(depth - 1, -beta, -alpha, true);
                if score > best_score {
                    best_score = score;
                    best_move = Some(mov.clone());
                }
                alpha = alpha.max(score);
                if alpha >= beta {
                    break;
                }
            }

            best_move
        }

        // Improved move ordering.
        fn order_moves(&self, moves: Vec<Move>) -> Vec<Move> {
            let mut move_scores: Vec<(Move, i32)> = moves
                .into_iter()
                .map(|mov| {
                    let score = self.score_move(&mov);
                    (mov, score)
                })
                .collect();

            move_scores.sort_by_key(|&(_, score)| -score);
            move_scores.into_iter().map(|(mov, _)| mov).collect()
        }

        // Enhanced position evaluation using the neural network.
        #[allow(dead_code)]
        fn evaluate_position(&self) -> i32 {
            let board_tensor: Array1<f32> = Array1::zeros(384);
            let evaluation = self.neural_net.forward(&board_tensor)[0];
            (evaluation * 1000.0) as i32
        }
    }
}

// ---------- MAIN FUNCTION ----------
fn main() {
    let mut engine = engine::ChessEngine::new();

    println!("Neural Chess Engine v2.0");
    println!("Enter moves in UCI format (e.g., 'e2e4') or 'quit' to exit");

    loop {
        engine.display_board();

        if engine.game.is_checkmate() {
            println!(
                "Checkmate! {} wins!",
                match engine.game.current_turn.opposite() {
                    types::Color::White => "White",
                    types::Color::Black => "Black",
                }
            );
            break;
        }

        if engine.game.is_stalemate() {
            println!("Stalemate! Game is drawn.");
            break;
        }

        match engine.game.current_turn {
            types::Color::White => {
                // Human move
                println!("Your move: ");
                let mut input = String::new();
                std::io::stdin().read_line(&mut input).unwrap();
                let input = input.trim();
                if input == "quit" {
                    break;
                }
                match engine.make_move_from_string(input) {
                    Ok(_) => (),
                    Err(e) => {
                        println!("Invalid move: {}", e);
                        continue;
                    }
                }
            }
            types::Color::Black => {
                // Computer move
                println!("Thinking...");
                let computer_move = engine
                    .find_best_move(4)
                    .expect("No legal moves available");
                engine.make_move(&computer_move).unwrap();
                println!("Computer plays: {:?}", computer_move);
            }
        }
    }
}
