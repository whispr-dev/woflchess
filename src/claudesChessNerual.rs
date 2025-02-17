// ---------- LOAD CRATES ----------

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use rand::Rng;

#[macro_use]
extern crate bitflags;


// Neural network activation functions
fn sigmoid(x: f32) -> f32 {
  1.0 / (1.0 + (-x).exp())
}

fn relu(x: f32) -> f32 {
  x.max(0.0)
}

// ---------- OPTIMIZED MODULES ----------
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

    #[derive(Clone, Copy)]
    pub struct Piece {
        pub piece_type: PieceType,
        pub color: Color,
    }

    // Optimized move struct with bitflags
    #[derive(Clone)]
    pub struct Move {
        pub from: Square,
        pub to: Square,
        pub piece_moved: Piece,
        pub piece_captured: Option<Piece>,
        pub flags: MoveFlags,
        pub promotion: Option<PieceType>,
    }

    // Use type aliases for clarity
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

#[derive(Clone)]
pub struct Move {
    pub from: Square,
    pub to: Square,
    pub piece_moved: Piece,
    pub piece_captured: Option<Piece>,
    pub flags: MoveFlags,
    pub promotion: Option<PieceType>,
}
}

// Constants for piece movements
const BISHOP_DIRECTIONS: [(i8, i8); 4] = [(-1, -1), (-1, 1), (1, -1), (1, 1)];
const ROOK_DIRECTIONS: [(i8, i8); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
const KNIGHT_MOVES: [(i8, i8); 8] = [
(-2, -1), (-2, 1), (-1, -2), (-1, 2),
(1, -2), (1, 2), (2, -1), (2, 1)
];

od neural {
  use super::*;

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

      pub fn train_batch(&mut self, inputs: &[Array1<f32>], targets: &[Array1<f32>]) {
          // Implementation of backpropagation here
          // This is a simplified version - you'll need to implement the full training logic
      }
  }
}

mod engine {
    use super::*;
    
    pub struct ChessEngine {
        game: GameState,
        neural_net: OptimizedNN,
        transposition_table: TranspositionTable,
        move_generator: MoveGenerator,
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

        // Optimized alpha-beta search with move ordering
        pub fn find_best_move(&mut self, depth: i32) -> Option<Move> {
            let mut alpha = i32::MIN;
            let beta = i32::MAX;
            
            let legal_moves = self.move_generator.generate_moves(&self.game);
            if legal_moves.is_empty() {
                return None;
            }

            // Order moves for better pruning
            let mut ordered_moves = self.order_moves(legal_moves);
            
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

        // Improved move ordering using history heuristic and MVV-LVA
        fn order_moves(&self, moves: Vec<Move>) -> Vec<Move> {
            let mut move_scores: Vec<(Move, i32)> = moves.into_iter()
                .map(|mov| {
                    let score = self.score_move(&mov);
                    (mov, score)
                })
                .collect();

            move_scores.sort_by_key(|(_, score)| -score);
            move_scores.into_iter().map(|(mov, _)| mov).collect()
        }

        // Enhanced position evaluation using neural network
        fn evaluate_position(&self) -> i32 {
            let board_tensor = self.game.to_neural_input();
            let evaluation = self.neural_net.forward(&board_tensor)[0];
            
            (evaluation * 1000.0) as i32
        }
    }
}


// Transposition table implementation
struct TranspositionTable {
  entries: Vec<Option<TableEntry>>,
}

struct TableEntry {
  zobrist_hash: u64,
  depth: i32,
  score: i32,
  best_move: Option<types::Move>,
}

impl TranspositionTable {
  fn with_capacity(size: usize) -> Self {
      Self {
          entries: vec![None; size],
      }
  }
}

// Move generator implementation
struct MoveGenerator {
  attack_tables: AttackTables,
}

struct AttackTables {
  pawn_attacks: [[u64; 64]; 2],  // One for each color
  knight_attacks: [u64; 64],
  king_attacks: [u64; 64],
}

impl MoveGenerator {
  fn new() -> Self {
      Self {
          attack_tables: AttackTables {
              pawn_attacks: [[0; 64]; 2],
              knight_attacks: [0; 64],
              king_attacks: [0; 64],
          }
      }
  }

  fn generate_moves(&self, game: &GameState) -> Vec<types::Move> {
      // Implementation of move generation here
      Vec::new()  // Placeholder
  }
}

// ---------- IMPROVED GAME STATE ----------
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
    }
}

// ---------- MAIN GAME LOOP ----------
fn main() {
    let mut engine = ChessEngine::new();
    
    println!("Neural Chess Engine v2.0");
    println!("Enter moves in UCI format (e.g., 'e2e4') or 'quit' to exit");

    loop {
        engine.display_board();
        
        if engine.game.is_checkmate() {
            println!("Checkmate! {} wins!", engine.game.current_turn.opposite());
            break;
        }
        
        if engine.game.is_stalemate() {
            println!("Stalemate! Game is drawn.");
            break;
        }

        match engine.game.current_turn {
            Color::White => {
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
            },
            Color::Black => {
                // Computer move
                println!("Thinking...");
                let computer_move = engine.find_best_move(4)
                    .expect("No legal moves available");
                
                engine.make_move(&computer_move).unwrap();
                println!("Computer plays: {}", computer_move);
            }
        }
    }
}

// Additional optimizations and improvements could include:
// 1. Implementing opening book
// 2. Adding endgame tablebases
// 3. Improving parallel search
// 4. Adding time management
// 5. Implementing UCI protocol
// 6. Adding position evaluation caching
// 7. Implementing quiescence search
// 8. Adding null move pruning
