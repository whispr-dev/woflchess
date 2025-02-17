// ---------- LOAD CRATES ----------
use std::sync::{Arc, Mutex};
use std::time::{Instant};
use rand::Rng;
use std::fmt;
use bitflags::bitflags;

// ---------- TYPES MODULE ----------
mod types {
    use super::*;
    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum PieceType {
        Pawn,
        Knight,
        Bishop,
        Rook,
        Queen,
        King,
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum Color {
        White,
        Black,
    }

    impl Color {
        pub fn opposite(&self) -> Color {
            match self {
                Color::White => Color::Black,
                Color::Black => Color::White,
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Piece {
        pub piece_type: PieceType,
        pub color: Color,
    }

    #[derive(Clone, Debug, PartialEq)]
    pub struct Move {
        pub from: (usize, usize),
        pub to: (usize, usize),
        pub piece_moved: Piece,
        pub piece_captured: Option<Piece>,
        pub is_castling: bool,
        pub is_en_passant: bool,
        pub promotion: Option<PieceType>,
    }

    pub type Board = [[Option<Piece>; 8]; 8];

    bitflags! {
        #[derive(Copy, PartialEq)]
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

    impl Clone for MoveFlags {
        fn clone(&self) -> Self { *self }
    }

    impl fmt::Debug for MoveFlags {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_tuple("MoveFlags").field(&self.bits()).finish()
        }
    }
}

// ---------- CONSTANTS FOR PIECE MOVEMENTS ----------
const BISHOP_DIRECTIONS: [(i8, i8); 4] = [(-1,-1), (-1,1), (1,-1), (1,1)];
const ROOK_DIRECTIONS: [(i8, i8); 4] = [(-1,0), (1,0), (0,-1), (0,1)];
const KNIGHT_MOVES: [(i8, i8); 8] = [
    (-2,-1), (-2,1), (-1,-2), (-1,2),
    (1,-2), (1,2), (2,-1), (2,1)
];

// ---------- CELLULAR AUTOMATA INTERFACE ----------
struct CAInterface {
    width: usize,
    height: usize,
    cells: Vec<Vec<f32>>,
    update_rules: Vec<Box<dyn Fn(&[f32]) -> f32>>,
}

impl CAInterface {
    fn new(width: usize, height: usize) -> Self {
        let mut rng = rand::thread_rng();
        CAInterface {
            width,
            height,
            cells: (0..height).map(|_| {
                (0..width).map(|_| rng.gen::<f32>()).collect()
            }).collect(),
            update_rules: vec![
                Box::new(|neighbors: &[f32]| {
                    let sum: f32 = neighbors.iter().sum();
                    if sum > 2.0 && sum < 3.5 { 1.0 } else { 0.0 }
                }),
                Box::new(|neighbors: &[f32]| {
                    neighbors.iter().sum::<f32>() / neighbors.len() as f32
                }),
                Box::new(|neighbors: &[f32]| {
                    let center = neighbors[neighbors.len()/2];
                    let avg = neighbors.iter().sum::<f32>() / neighbors.len() as f32;
                    (center + avg) / 2.0
                }),
            ],
        }
    }

    fn update(&mut self) {
        let mut new_cells = self.cells.clone();
        let rule_idx = rand::thread_rng().gen_range(0..self.update_rules.len());
        for i in 1..self.height-1 {
            for j in 1..self.width-1 {
                let neighbors = vec![
                    self.cells[i-1][j-1], self.cells[i-1][j], self.cells[i-1][j+1],
                    self.cells[i][j-1],   self.cells[i][j],   self.cells[i][j+1],
                    self.cells[i+1][j-1], self.cells[i+1][j], self.cells[i+1][j+1],
                ];
                new_cells[i][j] = self.update_rules[rule_idx](&neighbors);
            }
        }
        self.cells = new_cells;
    }

    fn get_state(&self) -> Vec<f32> {
        self.cells.iter().flat_map(|row| row.iter().cloned()).collect()
    }
}

// ---------- NEURAL CLOCK ----------
struct NeuralClock {
    cycle_duration: f32,
    last_tick: Instant,
}

impl NeuralClock {
    fn new(cycle_duration: f32) -> Self {
        NeuralClock {
            cycle_duration,
            last_tick: Instant::now(),
        }
    }

    fn tick(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_tick).as_secs_f32();
        if elapsed >= self.cycle_duration {
            self.last_tick = now;
            true
        } else {
            false
        }
    }
}

// ---------- NEURAL NETWORK COMPONENTS ----------
fn tanh(x: f32) -> f32 { x.tanh() }
fn relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

struct RNN {
    hidden_state: Vec<f32>,
    weights: Vec<Vec<f32>>, // dimensions: hidden_size x (input_dim + hidden_size)
    discriminator_weights: Vec<Vec<f32>>,
}

impl RNN {
    fn new(input_dim: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..hidden_size).map(|_| {
            (0..(input_dim + hidden_size)).map(|_| rng.gen_range(-0.1..0.1)).collect()
        }).collect();
        let discriminator_weights = (0..hidden_size).map(|_| {
            (0..hidden_size).map(|_| rng.gen_range(-0.1..0.1)).collect()
        }).collect();
        RNN {
            hidden_state: vec![0.0; hidden_size],
            weights,
            discriminator_weights,
        }
    }

    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let hidden_size = self.hidden_state.len();
        let mut output = vec![0.0; hidden_size];
        for i in 0..hidden_size {
            let mut sum = 0.0;
            for j in 0..input.len() {
                sum += input[j] * self.weights[i][j];
            }
            for j in 0..hidden_size {
                sum += self.hidden_state[j] * self.weights[i][j + input.len()];
            }
            output[i] = tanh(sum);
        }
        self.hidden_state = output.clone();
        output
    }

    fn discriminate(&self, state: &[f32]) -> f32 {
        let hidden_size = self.hidden_state.len();
        let mut sum = 0.0;
        for i in 0..hidden_size {
            for j in 0..hidden_size {
                sum += state[j] * self.discriminator_weights[i][j];
            }
        }
        sigmoid(sum)
    }
}

// First the struct definition
struct CNN {
    filters: Vec<Vec<Vec<f32>>>, // Each filter is 3x3
    feature_maps: Vec<Vec<f32>>,
    discriminator_filters: Vec<Vec<Vec<f32>>>, // Each is 3x3
}

// Then the complete implementation
impl CNN {
  fn new(num_filters: usize, input_rows: usize, input_cols: usize) -> Self {
      let mut rng = rand::thread_rng();
      
      // Initialize 3x3 filters
      let filters = (0..num_filters).map(|_| {
          (0..3).map(|_| {
              (0..3).map(|_| rng.gen_range(-0.1..0.1)).collect()
          }).collect()
      }).collect();
      
      // Initialize feature maps with exact size needed
      let feature_maps = vec![vec![0.0; input_rows * input_cols]; num_filters];
      
      // Initialize discriminator filters
      let discriminator_filters = (0..num_filters).map(|_| {
          (0..3).map(|_| {
              (0..3).map(|_| rng.gen_range(-0.1..0.1)).collect()
          }).collect()
      }).collect();

      CNN {
          filters,
          feature_maps,
          discriminator_filters,
      }
  }

  fn forward(&mut self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
      let rows = input.len();
      let cols = if rows > 0 { input[0].len() } else { 0 };
      let num_filters = self.filters.len();
      
      // Create output maps with exact dimensions
      let mut new_feature_maps = vec![vec![0.0; rows * cols]; num_filters];
      
      // Apply convolution for each filter
      for (f, filter) in self.filters.iter().enumerate() {
          for i in 1..rows.saturating_sub(1) {
              for j in 1..cols.saturating_sub(1) {
                  let mut sum = 0.0;
                  
                  // Safe convolution
                  for di in 0..3 {
                      for dj in 0..3 {
                          if (i + di > 0) && (j + dj > 0) && 
                             (i + di - 1 < rows) && (j + dj - 1 < cols) {
                              sum += input[i+di-1][j+dj-1] * filter[di][dj];
                          }
                      }
                  }
                  
                  // Safe indexing into output
                  let out_idx = i * cols + j;
                  if out_idx < new_feature_maps[f].len() {
                      new_feature_maps[f][out_idx] = relu(sum);
                  }
              }
          }
      }
      
      self.feature_maps = new_feature_maps.clone();
      new_feature_maps
  }

  fn discriminate(&self, feature_maps: &[Vec<f32>]) -> f32 {
      let mut sum = 0.0;
      
      // Safe iteration over feature maps
      for (f, map) in feature_maps.iter().enumerate() {
          if f >= self.discriminator_filters.len() {
              break;
          }
          
          // Only process the first 3x3 area or less
          let rows = 3.min((map.len() as f32).sqrt() as usize);
          let cols = rows;
          
          for i in 0..rows {
              for j in 0..cols {
                  let idx = i * cols + j;
                  if idx < map.len() {
                      sum += map[idx] * self.discriminator_filters[f][i.min(2)][j.min(2)];
                  }
              }
          }
      }
      
      sigmoid(sum)
  }
}

struct LSTM {
    cell_state: Vec<f32>,
    hidden_state: Vec<f32>,
    weights: Vec<Vec<f32>>, // dimensions: hidden_size x (input_dim + hidden_size*2)
    discriminator_weights: Vec<Vec<f32>>,
}

impl LSTM {
    fn new(input_dim: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..hidden_size).map(|_| {
            (0..(input_dim + hidden_size * 2)).map(|_| rng.gen_range(-0.1..0.1)).collect()
        }).collect();
        let discriminator_weights = (0..hidden_size).map(|_| {
            (0..hidden_size).map(|_| rng.gen_range(-0.1..0.1)).collect()
        }).collect();
        LSTM {
            cell_state: vec![0.0; hidden_size],
            hidden_state: vec![0.0; hidden_size],
            weights,
            discriminator_weights,
        }
    }

    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let hidden_size = self.hidden_state.len();
        let mut new_hidden = vec![0.0; hidden_size];
        let mut new_cell = vec![0.0; hidden_size];
        for i in 0..hidden_size {
            let mut forget_sum = 0.0;
            for j in 0..input.len() {
                forget_sum += input[j] * self.weights[i][j];
            }
            let forget_gate = sigmoid(forget_sum);
            let mut input_sum = 0.0;
            for j in 0..input.len() {
                input_sum += input[j] * self.weights[i][j + input.len()];
            }
            let input_gate = sigmoid(input_sum);
            new_cell[i] = forget_gate * self.cell_state[i] + input_gate * tanh(input_sum);
            let mut output_sum = 0.0;
            for j in 0..input.len() {
                output_sum += input[j] * self.weights[i][j + 2 * input.len()];
            }
            let output_gate = sigmoid(output_sum);
            new_hidden[i] = output_gate * tanh(new_cell[i]);
        }
        self.hidden_state = new_hidden.clone();
        self.cell_state = new_cell;
        new_hidden
    }

    fn discriminate(&self, state: &[f32]) -> f32 {
        let hidden_size = self.hidden_state.len();
        let mut sum = 0.0;
        for i in 0..hidden_size {
            for j in 0..hidden_size {
                sum += state[j] * self.discriminator_weights[i][j];
            }
        }
        sigmoid(sum)
    }
}

// ---------- GAN TRAINER ----------
struct GANTrainer {
    learning_rate: f32,
    batch_size: usize,
    noise_dim: usize,
}

impl GANTrainer {
    fn new(learning_rate: f32, batch_size: usize, noise_dim: usize) -> Self {
        GANTrainer { learning_rate, batch_size, noise_dim }
    }

    fn generate_noise(&self) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        (0..self.noise_dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
    }

    fn train_step(
        &self,
        rnn: &mut RNN,
        cnn: &mut CNN,
        lstm: &mut LSTM,
        real_data: &[Vec<f32>],
        ca_interface: &mut CAInterface,
    ) {
        // Train discriminators
        for _ in 0..self.batch_size {
            let noise = self.generate_noise();
            ca_interface.update();
            let rnn_out = rnn.forward(&noise);
            let cnn_out = cnn.forward(&vec![rnn_out.clone()]);
            let lstm_out = lstm.forward(&rnn_out);
            for real_sample in real_data.iter() {
                let rnn_real_score = rnn.discriminate(real_sample);
                let cnn_real_score = cnn.discriminate(&vec![real_sample.to_vec()]);
                let lstm_real_score = lstm.discriminate(real_sample);
                self.update_discriminator_weights(rnn, real_sample, &rnn_out, rnn_real_score);
                self.update_discriminator_weights_cnn(cnn, real_sample, &cnn_out[0], cnn_real_score);
                self.update_discriminator_weights_lstm(lstm, real_sample, &lstm_out, lstm_real_score);
            }
        }
        // Train generators
        for _ in 0..self.batch_size {
            let noise = self.generate_noise();
            ca_interface.update();
            let rnn_out = rnn.forward(&noise);
            let cnn_out = cnn.forward(&vec![rnn_out.clone()]);
            let lstm_out = lstm.forward(&rnn_out);
            let rnn_fake_score = rnn.discriminate(&rnn_out);
            let cnn_fake_score = cnn.discriminate(&cnn_out);
            let lstm_fake_score = lstm.discriminate(&lstm_out);
            self.update_generator_weights(rnn, &noise, rnn_fake_score);
            self.update_generator_weights_cnn(cnn, &noise, cnn_fake_score);
            self.update_generator_weights_lstm(lstm, &noise, lstm_fake_score);
        }
    }

    fn update_discriminator_weights(&self, network: &mut RNN, real: &[f32], fake: &[f32], real_score: f32) {
        for i in 0..network.discriminator_weights.len() {
            for j in 0..network.discriminator_weights[0].len() {
                let real_grad = real_score * (1.0 - real_score) * real[j % real.len()];
                let fake_grad = -fake[j % fake.len()] * (1.0 - fake[j % fake.len()]);
                network.discriminator_weights[i][j] += self.learning_rate * (real_grad + fake_grad);
            }
        }
    }

    fn update_discriminator_weights_cnn(&self, network: &mut CNN, _real: &[f32], fake: &[f32], real_score: f32) {
        for f in 0..network.discriminator_filters.len() {
            for i in 0..3 {
                for j in 0..3 {
                    let real_grad = real_score * (1.0 - real_score);
                    let fake_grad = -fake[0] * (1.0 - fake[0]);
                    network.discriminator_filters[f][i][j] += self.learning_rate * (real_grad + fake_grad);
                }
            }
        }
    }

    fn update_discriminator_weights_lstm(&self, network: &mut LSTM, real: &[f32], fake: &[f32], real_score: f32) {
        for i in 0..network.discriminator_weights.len() {
            for j in 0..network.discriminator_weights[0].len() {
                let real_grad = real_score * (1.0 - real_score) * real[j % real.len()];
                let fake_grad = -fake[j % fake.len()] * (1.0 - fake[j % fake.len()]);
                network.discriminator_weights[i][j] += self.learning_rate * (real_grad + fake_grad);
            }
        }
    }

    fn update_generator_weights(&self, network: &mut RNN, noise: &[f32], fake_score: f32) {
        for i in 0..network.weights.len() {
            for j in 0..network.weights[0].len() {
                let grad = fake_score * (1.0 - fake_score) * noise[j % noise.len()];
                network.weights[i][j] += self.learning_rate * grad;
            }
        }
    }

    fn update_generator_weights_cnn(&self, network: &mut CNN, _noise: &[f32], fake_score: f32) {
        for f in 0..network.filters.len() {
            for i in 0..3 {
                for j in 0..3 {
                    let grad = fake_score * (1.0 - fake_score);
                    network.filters[f][i][j] += self.learning_rate * grad;
                }
            }
        }
    }

    fn update_generator_weights_lstm(&self, network: &mut LSTM, noise: &[f32], fake_score: f32) {
        for i in 0..network.weights.len() {
            for j in 0..network.weights[0].len() {
                let grad = fake_score * (1.0 - fake_score) * noise[j % noise.len()];
                network.weights[i][j] += self.learning_rate * grad;
            }
        }
    }
}

// ---------- CHESS NEURAL ENGINE ----------
struct ChessNeuralEngine {
    rnn: Arc<Mutex<RNN>>,
    cnn: Arc<Mutex<CNN>>,
    lstm: Arc<Mutex<LSTM>>,
    gan_trainer: GANTrainer,
    ca_interface: CAInterface,
    neural_clock: NeuralClock,
}

impl ChessNeuralEngine {
  fn new() -> Self {
      let input_dim = 64;  // 8x8 board
      let hidden_size = 16;
      ChessNeuralEngine {
          rnn: Arc::new(Mutex::new(RNN::new(input_dim, hidden_size))),
          cnn: Arc::new(Mutex::new(CNN::new(1, 8, 8))),  // One filter, 8x8 board
          lstm: Arc::new(Mutex::new(LSTM::new(input_dim, hidden_size))),
          gan_trainer: GANTrainer::new(0.01, 10, input_dim),
          ca_interface: CAInterface::new(8, 8),
          neural_clock: NeuralClock::new(0.5),
      }
  }
}

// ---------- CHESS ENGINE (GAME STATE) ----------
#[derive(Clone)]
struct GameState {
    board: types::Board,
    current_turn: types::Color,
    move_history: Vec<types::Move>,
    white_king_pos: (usize, usize),
    black_king_pos: (usize, usize),
    white_can_castle_kingside: bool,
    white_can_castle_queenside: bool,
    black_can_castle_kingside: bool,
    black_can_castle_queenside: bool,
    last_pawn_double_move: Option<(usize, usize)>,
    neural_engine: Option<Arc<Mutex<ChessNeuralEngine>>>,
}

impl GameState {
    fn new() -> Self {
        let mut state = GameState {
            board: [[None; 8]; 8],
            current_turn: types::Color::White,
            move_history: Vec::new(),
            white_king_pos: (4, 0),
            black_king_pos: (4, 7),
            white_can_castle_kingside: true,
            white_can_castle_queenside: true,
            black_can_castle_kingside: true,
            black_can_castle_queenside: true,
            last_pawn_double_move: None,
            neural_engine: Some(Arc::new(Mutex::new(ChessNeuralEngine::new()))),
        };
        state.setup_initial_position();
        state
    }

    fn setup_initial_position(&mut self) {
        let create_piece = |pt, color| Some(types::Piece { piece_type: pt, color });
        // White pieces
        self.board[0][0] = create_piece(types::PieceType::Rook, types::Color::White);
        self.board[7][0] = create_piece(types::PieceType::Rook, types::Color::White);
        self.board[1][0] = create_piece(types::PieceType::Knight, types::Color::White);
        self.board[6][0] = create_piece(types::PieceType::Knight, types::Color::White);
        self.board[2][0] = create_piece(types::PieceType::Bishop, types::Color::White);
        self.board[5][0] = create_piece(types::PieceType::Bishop, types::Color::White);
        self.board[3][0] = create_piece(types::PieceType::Queen, types::Color::White);
        self.board[4][0] = create_piece(types::PieceType::King, types::Color::White);
        for i in 0..8 { self.board[i][1] = create_piece(types::PieceType::Pawn, types::Color::White); }
        // Black pieces
        self.board[0][7] = create_piece(types::PieceType::Rook, types::Color::Black);
        self.board[7][7] = create_piece(types::PieceType::Rook, types::Color::Black);
        self.board[1][7] = create_piece(types::PieceType::Knight, types::Color::Black);
        self.board[6][7] = create_piece(types::PieceType::Knight, types::Color::Black);
        self.board[2][7] = create_piece(types::PieceType::Bishop, types::Color::Black);
        self.board[5][7] = create_piece(types::PieceType::Bishop, types::Color::Black);
        self.board[3][7] = create_piece(types::PieceType::Queen, types::Color::Black);
        self.board[4][7] = create_piece(types::PieceType::King, types::Color::Black);
        for i in 0..8 { self.board[i][6] = create_piece(types::PieceType::Pawn, types::Color::Black); }
    }

    fn board_to_neural_input(&self) -> Vec<f32> {
        let mut input = Vec::with_capacity(64); // Changed from 64 * 6
        for rank in 0..8 {
            for file in 0..8 {
                if let Some(piece) = self.board[file][rank] {
                    let value = if piece.color == types::Color::White { 1.0 } else { -1.0 };
                    input.push(value);
                } else {
                    input.push(0.0);
                }
            }
        }
        input
    }

    fn evaluate_position(&self) -> i32 {
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
                    let multiplier = if piece.color == types::Color::White { 1 } else { -1 };
                    let piece_value = match piece.piece_type {
                        types::PieceType::Pawn => pawn_value,
                        types::PieceType::Knight => knight_value,
                        types::PieceType::Bishop => bishop_value,
                        types::PieceType::Rook => rook_value,
                        types::PieceType::Queen => queen_value,
                        types::PieceType::King => king_value,
                    };
                    score += multiplier * piece_value;
                }
            }
        }
        score
    }

    impl GameState {
      fn evaluate_position_neural(&self) -> i32 {
          if let Some(engine_arc) = &self.neural_engine {
              let engine = engine_arc.lock().unwrap();
              let input = self.board_to_neural_input();
              
              let rnn_evaluation = {
                  let mut rnn = engine.rnn.lock().unwrap();
                  let output = rnn.forward(&input);
                  output.get(0).copied().unwrap_or(0.0) * 100.0
              };
              
              let cnn_evaluation = {
                  let mut cnn = engine.cnn.lock().unwrap();
                  let reshaped_input = reshape_vector_to_matrix(&input, 8, 8);
                  let output = cnn.forward(&reshaped_input);
                  output.first()
                        .and_then(|row| row.first())
                        .copied()
                        .unwrap_or(0.0) * 100.0
              };
              
              let lstm_evaluation = {
                  let mut lstm = engine.lstm.lock().unwrap();
                  let output = lstm.forward(&input);
                  output.get(0).copied().unwrap_or(0.0) * 100.0
              };
              
              let traditional_eval = self.evaluate_position() as f32;
              let combined_eval = traditional_eval * 0.4 + 
                                rnn_evaluation * 0.2 + 
                                cnn_evaluation * 0.2 + 
                                lstm_evaluation * 0.2;
              combined_eval as i32
          } else {
              self.evaluate_position()
          }
      }
  }
  
  // Helper function for reshaping vectors
  fn reshape_vector_to_matrix(vec: &[f32], rows: usize, cols: usize) -> Vec<Vec<f32>> {
      let mut matrix = vec![vec![0.0; cols]; rows];
      for (i, &value) in vec.iter().enumerate() {
          if i >= rows * cols { break; }
          let row = i / cols;
          let col = i % cols;
          matrix[row][col] = value;
      }
      matrix
  }

    fn is_valid_move(&self, mv: &types::Move) -> bool {
        self.is_valid_basic_move(mv) && !self.would_move_cause_check(mv)
    }

    fn is_valid_basic_move(&self, mv: &types::Move) -> bool {
        if !Self::is_within_bounds(mv.from) || !Self::is_within_bounds(mv.to) {
            return false;
        }
        if mv.from == mv.to { return false; }
        let piece = match self.get_piece_at(mv.from) {
            Some(p) => p,
            None => return false,
        };
        if let Some(dest) = self.get_piece_at(mv.to) {
            if dest.color == piece.color { return false; }
        }
        match piece.piece_type {
            types::PieceType::Pawn => self.is_valid_pawn_move(mv, piece.color),
            types::PieceType::Knight => self.is_valid_knight_move(mv),
            types::PieceType::Bishop => self.is_valid_bishop_move(mv),
            types::PieceType::Rook => self.is_valid_rook_move(mv),
            types::PieceType::Queen => self.is_valid_queen_move(mv),
            types::PieceType::King => self.is_valid_king_move(mv),
        }
    }

    fn is_valid_pawn_move(&self, mv: &types::Move, color: types::Color) -> bool {
        let (from_x, from_y) = mv.from;
        let (to_x, to_y) = mv.to;
        let direction = match color {
            types::Color::White => 1,
            types::Color::Black => -1,
        };
        let start_rank = match color {
            types::Color::White => 1,
            types::Color::Black => 6,
        };
        let forward_one = ((from_y as i32) + direction) as usize;
        let forward_two = ((from_y as i32) + 2 * direction) as usize;
        if to_x == from_x && to_y == forward_one && self.get_piece_at(mv.to).is_none() {
            return true;
        }
        if from_y == start_rank && to_x == from_x && to_y == forward_two {
            return self.get_piece_at(mv.to).is_none() && self.get_piece_at((from_x, forward_one)).is_none();
        }
        if (to_y as i32 - from_y as i32) == direction && (to_x as i32 - from_x as i32).abs() == 1 {
            if self.get_piece_at(mv.to).is_some() {
                return true;
            }
        }
        if mv.is_en_passant {
            if let Some(last_move) = self.last_pawn_double_move {
                if (to_y as i32 - from_y as i32) == direction &&
                   (to_x as i32 - from_x as i32).abs() == 1 &&
                   to_x == last_move.0 && from_y == last_move.1 {
                    return true;
                }
            }
        }
        false
    }

    fn is_valid_knight_move(&self, mv: &types::Move) -> bool {
        let dx = (mv.to.0 as i32 - mv.from.0 as i32).abs();
        let dy = (mv.to.1 as i32 - mv.from.1 as i32).abs();
        (dx == 2 && dy == 1) || (dx == 1 && dy == 2)
    }

    fn is_valid_bishop_move(&self, mv: &types::Move) -> bool {
        let dx = (mv.to.0 as i32 - mv.from.0 as i32).abs();
        let dy = (mv.to.1 as i32 - mv.from.1 as i32).abs();
        if dx == dy && dx > 0 {
            self.is_path_clear(mv.from, mv.to)
        } else {
            false
        }
    }

    fn is_valid_rook_move(&self, mv: &types::Move) -> bool {
        let dx = mv.to.0 as i32 - mv.from.0 as i32;
        let dy = mv.to.1 as i32 - mv.from.1 as i32;
        if (dx == 0 && dy != 0) || (dx != 0 && dy == 0) {
            self.is_path_clear(mv.from, mv.to)
        } else {
            false
        }
    }

    fn is_valid_queen_move(&self, mv: &types::Move) -> bool {
        self.is_valid_bishop_move(mv) || self.is_valid_rook_move(mv)
    }

    fn is_valid_king_move(&self, mv: &types::Move) -> bool {
        let dx = (mv.to.0 as i32 - mv.from.0 as i32).abs();
        let dy = (mv.to.1 as i32 - mv.from.1 as i32).abs();
        if dx <= 1 && dy <= 1 {
            return true;
        }
        if mv.is_castling && dy == 0 && dx == 2 {
            match mv.piece_moved.color {
                types::Color::White => {
                    if mv.from != (4, 0) { return false; }
                    if mv.to == (6, 0) {
                        if !self.white_can_castle_kingside { return false; }
                        if !self.is_path_clear((4, 0), (7, 0)) { return false; }
                        for x in 4..=6 {
                            if self.is_square_attacked((x, 0), types::Color::Black) { return false; }
                        }
                        return true;
                    } else if mv.to == (2, 0) {
                        if !self.white_can_castle_queenside { return false; }
                        if !self.is_path_clear((4, 0), (0, 0)) { return false; }
                        for x in 2..=4 {
                            if self.is_square_attacked((x, 0), types::Color::Black) { return false; }
                        }
                        return true;
                    }
                },
                types::Color::Black => {
                    if mv.from != (4, 7) { return false; }
                    if mv.to == (6, 7) {
                        if !self.black_can_castle_kingside { return false; }
                        if !self.is_path_clear((4, 7), (7, 7)) { return false; }
                        for x in 4..=6 {
                            if self.is_square_attacked((x, 7), types::Color::White) { return false; }
                        }
                        return true;
                    } else if mv.to == (2, 7) {
                        if !self.black_can_castle_queenside { return false; }
                        if !self.is_path_clear((4, 7), (0, 7)) { return false; }
                        for x in 2..=4 {
                            if self.is_square_attacked((x, 7), types::Color::White) { return false; }
                        }
                        return true;
                    }
                },
            }
        }
        false
    }

    fn is_path_clear(&self, from: (usize, usize), to: (usize, usize)) -> bool {
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

    // Only one definition of get_piece_at exists.
    fn get_piece_at(&self, pos: (usize, usize)) -> Option<&types::Piece> {
        if Self::is_within_bounds(pos) {
            self.board[pos.0][pos.1].as_ref()
        } else {
            None
        }
    }

    fn is_within_bounds(pos: (usize, usize)) -> bool {
        pos.0 < 8 && pos.1 < 8
    }

    fn would_move_cause_check(&self, mv: &types::Move) -> bool {
        let mut test_state = self.clone();
        test_state.make_move_without_validation(mv);
        test_state.is_in_check(mv.piece_moved.color)
    }

    fn is_in_check(&self, color: types::Color) -> bool {
        let king_pos = if color == types::Color::White { self.white_king_pos } else { self.black_king_pos };
        self.is_square_attacked(king_pos, color.opposite())
    }

    fn is_square_attacked(&self, pos: (usize, usize), by_color: types::Color) -> bool {
        for i in 0..8 {
            for j in 0..8 {
                if let Some(piece) = self.board[i][j] {
                    if piece.color == by_color {
                        let attack_mv = types::Move {
                            from: (i, j),
                            to: pos,
                            piece_moved: piece,
                            piece_captured: self.board[pos.0][pos.1],
                            is_castling: false,
                            is_en_passant: false,
                            promotion: None,
                        };
                        if self.is_valid_basic_move(&attack_mv) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn make_move(&mut self, mv: types::Move) -> Result<(), &'static str> {
        if !self.is_valid_move(&mv) {
            return Err("Invalid move");
        }
        self.make_move_without_validation(&mv);
        self.move_history.push(mv.clone());
        self.current_turn = match self.current_turn {
            types::Color::White => types::Color::Black,
            types::Color::Black => types::Color::White,
        };
        Ok(())
    }

    // Only one definition of make_move_without_validation exists.
    fn make_move_without_validation(&mut self, mv: &types::Move) {
        if let Some(promote) = mv.promotion {
            let promoted = types::Piece { piece_type: promote, color: mv.piece_moved.color };
            self.board[mv.to.0][mv.to.1] = Some(promoted);
        } else {
            self.board[mv.to.0][mv.to.1] = Some(mv.piece_moved);
        }
        self.board[mv.from.0][mv.from.1] = None;
        if mv.is_castling {
            match (mv.from, mv.to) {
                ((4,0), (6,0)) => { self.board[5][0] = self.board[7][0]; self.board[7][0] = None; },
                ((4,0), (2,0)) => { self.board[3][0] = self.board[0][0]; self.board[0][0] = None; },
                ((4,7), (6,7)) => { self.board[5][7] = self.board[7][7]; self.board[7][7] = None; },
                ((4,7), (2,7)) => { self.board[3][7] = self.board[0][7]; self.board[0][7] = None; },
                _ => {}
            }
        }
        if mv.piece_moved.piece_type == types::PieceType::Pawn {
            let dy = (mv.to.1 as i32 - mv.from.1 as i32).abs();
            if dy == 2 { self.last_pawn_double_move = Some(mv.to); } else { self.last_pawn_double_move = None; }
            if mv.is_en_passant {
                let capture_y = mv.from.1;
                self.board[mv.to.0][capture_y] = None;
            }
        } else {
            self.last_pawn_double_move = None;
        }
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
        match (mv.from, mv.piece_moved.piece_type) {
            ((0,0), types::PieceType::Rook) => self.white_can_castle_queenside = false,
            ((7,0), types::PieceType::Rook) => self.white_can_castle_kingside = false,
            ((0,7), types::PieceType::Rook) => self.black_can_castle_queenside = false,
            ((7,7), types::PieceType::Rook) => self.black_can_castle_kingside = false,
            _ => {}
        }
    }

    fn parse_square(&self, s: &str) -> Option<(usize, usize)> {
        let chars: Vec<char> = s.chars().collect();
        if chars.len() != 2 { return None; }
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
        match move_str {
            "O-O" | "0-0" => {
                let (rank, king, to) = if self.current_turn == types::Color::White { (0, 4, 6) } else { (7, 4, 6) };
                return Ok(((king, rank), (to, rank)));
            },
            "O-O-O" | "0-0-0" => {
                let (rank, king, to) = if self.current_turn == types::Color::White { (0, 4, 2) } else { (7, 4, 2) };
                return Ok(((king, rank), (to, rank)));
            },
            _ => {}
        }
        let chars: Vec<char> = move_str.chars().collect();
        if chars.is_empty() { return Err("Empty move string"); }
        let (piece_type, _) = match chars[0] {
            'N' => (types::PieceType::Knight, 1),
            'B' => (types::PieceType::Bishop, 1),
            'R' => (types::PieceType::Rook, 1),
            'Q' => (types::PieceType::Queen, 1),
            'K' => (types::PieceType::King, 1),
            'a'..='h' => (types::PieceType::Pawn, 0),
            _ => return Err("Invalid move notation"),
        };
        if piece_type == types::PieceType::Pawn {
            let to = self.parse_square(move_str).ok_or("Invalid destination square")?;
            let from = self.find_piece_that_can_move(piece_type, to)?;
            return Ok((from, to));
        }
        let dest_str = &move_str[move_str.len()-2..];
        let to = self.parse_square(dest_str).ok_or("Invalid destination square")?;
        let from = self.find_piece_that_can_move(piece_type, to)?;
        Ok((from, to))
    }

    fn parse_move_string(&self, move_str: &str) -> Result<((usize, usize), (usize, usize)), &'static str> {
        let parts: Vec<&str> = move_str.split_whitespace().collect();
        let (from_str, to_str) = match parts.len() {
            2 => (parts[0], parts[1]),
            3 if parts[1] == "to" => (parts[0], parts[2]),
            1 if move_str.len() == 4 => (&move_str[0..2], &move_str[2..4]),
            _ => return Err("Invalid move format. Use 'e2e4', 'e2 e4', or 'e2 to e4'"),
        };
        let from = self.parse_square(from_str).ok_or("Invalid starting square")?;
        let to = self.parse_square(to_str).ok_or("Invalid destination square")?;
        Ok((from, to))
    }

    fn find_piece_that_can_move(&self, piece_type: types::PieceType, to: (usize, usize)) -> Result<(usize, usize), &'static str> {
        let mut valid = Vec::new();
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
                        if self.is_valid_move(&test_mv) {
                            valid.push((i, j));
                        }
                    }
                }
            }
        }
        match valid.len() {
            0 => Err("No piece can make that move"),
            1 => Ok(valid[0]),
            _ => Err("Ambiguous move - multiple pieces can move there"),
        }
    }

    fn make_move_from_str(&mut self, move_str: &str) -> Result<(), &'static str> {
        let (basic_move, promotion_char) = if move_str.len() >= 5 {
            let (main_move, promotion) = move_str.split_at(4);
            (main_move, promotion.chars().next())
        } else {
            (move_str, None)
        };
        let (from, to) = if basic_move.contains(' ') || basic_move.len() == 4 {
            self.parse_move_string(basic_move)?
        } else {
            self.parse_algebraic_notation(basic_move)?
        };
        let piece = self.get_piece_at(from).ok_or("No piece at starting square")?;
        if piece.color != self.current_turn {
            return Err("It's not your turn!");
        }
        let promotion_type = if piece.piece_type == types::PieceType::Pawn && self.is_promotion_move(from, to) {
            match promotion_char {
                Some('Q') | Some('q') => Some(types::PieceType::Queen),
                Some('R') | Some('r') => Some(types::PieceType::Rook),
                Some('B') | Some('b') => Some(types::PieceType::Bishop),
                Some('N') | Some('n') => Some(types::PieceType::Knight),
                _ => Some(types::PieceType::Queen),
            }
        } else {
            None
        };
        let mv = types::Move {
            from,
            to,
            piece_moved: *piece,
            piece_captured: self.get_piece_at(to).cloned(),
            is_castling: false,
            is_en_passant: false,
            promotion: promotion_type,
        };
        if !self.is_valid_move(&mv) {
            return Err(match piece.piece_type {
                types::PieceType::Pawn => "Invalid pawn move",
                types::PieceType::Knight => "Invalid knight move",
                types::PieceType::Bishop => "Invalid bishop move",
                types::PieceType::Rook => "Invalid rook move",
                types::PieceType::Queen => "Invalid queen move",
                types::PieceType::King => {
                    if mv.is_castling { "Invalid castling" } else { "Invalid king move" }
                },
            });
        }
        self.make_move(mv)
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
        if depth == 0 { return self.evaluate_position(); }
        let legal_moves = self.generate_legal_moves();
        if legal_moves.is_empty() {
            if self.is_in_check(self.current_turn) { return if maximizing { -30000 } else { 30000 }; }
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
                if beta <= alpha { break; }
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
                if beta <= alpha { break; }
            }
            min_eval
        }
    }

    fn minimax_neural(&mut self, depth: i32, mut alpha: i32, mut beta: i32, maximizing: bool) -> i32 {
        if depth == 0 { return self.evaluate_position_neural(); }
        let moves = self.generate_moves_neural();
        if moves.is_empty() {
            if self.is_in_check(self.current_turn) { return if maximizing { -30000 } else { 30000 }; }
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
                if beta <= alpha { break; }
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
                if beta <= alpha { break; }
            }
            min_eval
        }
    }

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
                                if self.is_valid_move(&test_mv) {
                                    moves.push(test_mv);
                                }
                            }
                        }
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
                            if self.is_valid_move(&kingside) { moves.push(kingside); }
                            let queenside = types::Move {
                                from: (4, rank),
                                to: (2, rank),
                                piece_moved: piece,
                                piece_captured: None,
                                is_castling: true,
                                is_en_passant: false,
                                promotion: None,
                            };
                            if self.is_valid_move(&queenside) { moves.push(queenside); }
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
            let move_scores: Vec<(types::Move, f32)> = moves.iter().map(|mv| {
                let mut test_state = self.clone();
                test_state.make_move_without_validation(mv);
                let position_after = test_state.board_to_neural_input();
                let rnn_score = engine.rnn.lock().unwrap().discriminate(&position_after);
                let cnn_score = engine.cnn.lock().unwrap().discriminate(&vec![position_after.clone()]);
                let lstm_score = engine.lstm.lock().unwrap().discriminate(&position_after);
                let combined = (rnn_score + cnn_score + lstm_score) / 3.0;
                (mv.clone(), combined)
            }).collect();
            let mut sorted = move_scores;
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            moves = sorted.into_iter().map(|(mv, _)| mv).collect();
        }
        moves
    }

    fn make_computer_move(&mut self) -> Result<(), &'static str> {
        let moves = self.generate_moves_neural();
        if moves.is_empty() {
            if self.is_in_check(self.current_turn) { return Err("Checkmate!"); }
            else { return Err("Stalemate!"); }
        }
        let mut best_score = i32::MIN;
        let mut best_moves = Vec::new();
        let search_depth = 3;
        for mv in moves {
            let mut test_state = self.clone();
            test_state.make_move_without_validation(&mv);
            let score = -test_state.minimax_neural(search_depth - 1, i32::MIN, i32::MAX, true);
            if score > best_score {
                best_score = score;
                best_moves.clear();
                best_moves.push(mv);
            } else if score == best_score {
                best_moves.push(mv);
            }
        }
        let mut rng = rand::thread_rng();
        let selected = best_moves[rng.gen_range(0..best_moves.len())].clone();
        println!("Neural evaluation: {}", best_score);
        self.make_move(selected.clone())?;
        println!("Computer plays: {}", self.move_to_algebraic(&selected));
        Ok(())
    }

    fn move_to_algebraic(&self, mv: &types::Move) -> String {
        if mv.is_castling {
            if mv.to.0 == 6 { "O-O".to_string() } else { "O-O-O".to_string() }
        } else {
            let piece_char = match mv.piece_moved.piece_type {
                types::PieceType::Pawn => "",
                types::PieceType::Knight => "N",
                types::PieceType::Bishop => "B",
                types::PieceType::Rook => "R",
                types::PieceType::Queen => "Q",
                types::PieceType::King => "K",
            };
            let capture = if mv.piece_captured.is_some() || mv.is_en_passant { "x" } else { "" };
            let dest_file = (mv.to.0 as u8 + b'a') as char;
            let dest_rank = (mv.to.1 + 1).to_string();
            format!("{}{}{}{}", piece_char, capture, dest_file, dest_rank)
        }
    }

    fn get_game_status(&self) -> String {
        if self.is_in_check(self.current_turn) && self.generate_legal_moves().is_empty() {
            format!("Checkmate! {:?} wins!", match self.current_turn {
                types::Color::White => "Black",
                types::Color::Black => "White",
            })
        } else if self.is_in_check(self.current_turn) {
            format!("{:?} is in check!", self.current_turn)
        } else if self.generate_legal_moves().is_empty() {
            "Stalemate! Game is a draw!".to_string()
        } else {
            format!("{:?}'s turn", self.current_turn)
        }
    }
}

// Utility: reshape vector into matrix.
fn reshape_vector_to_matrix(vec: &[f32], rows: usize, cols: usize) -> Vec<Vec<f32>> {
  let mut matrix = vec![vec![0.0; cols]; rows];
  for (i, &value) in vec.iter().enumerate() {
      if i >= rows * cols { break; }
      let row = i / cols;
      let col = i % cols;
      matrix[row][col] = value;
  }
  matrix
}

// ---------- DISPLAY IMPLEMENTATION ----------
impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "\n  a b c d e f g h")?;
        writeln!(f, "  ---------------")?;
        for rank in (0..8).rev() {
            write!(f, "{} ", rank + 1)?;
            for file in 0..8 {
                match self.board[file][rank] {
                    Some(piece) => {
                        let symbol = match piece.piece_type {
                            types::PieceType::Pawn => "P",
                            types::PieceType::Knight => "N",
                            types::PieceType::Bishop => "B",
                            types::PieceType::Rook => "R",
                            types::PieceType::Queen => "Q",
                            types::PieceType::King => "K",
                        };
                        if piece.color == types::Color::Black {
                            write!(f, "{} ", symbol.to_lowercase())?;
                        } else {
                            write!(f, "{} ", symbol)?;
                        }
                    },
                    None => write!(f, ". ")?,
                }
            }
            writeln!(f, "{}", rank + 1)?;
        }
        writeln!(f, "  ---------------")?;
        writeln!(f, "  a b c d e f g h")?;
        writeln!(f, "\nCurrent turn: {:?}", self.current_turn)
    }
}

// ---------- MAIN FUNCTION ----------
fn main() {
    let mut game_state = GameState::new();
    println!("Neural Chess Engine v2.0");
    loop {
        println!("{}", game_state);
        println!("{}", game_state.get_game_status());
        if game_state.current_turn == types::Color::White {
            println!("Your move: ");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();
            if input == "quit" { break; }
            if let Err(e) = game_state.make_move_from_str(input) {
                println!("Error: {}", e);
                continue;
            }
        } else {
            println!("Computer is thinking...");
            if let Err(e) = game_state.make_computer_move() {
                println!("{}", e);
                break;
            }
        }
    }
}
