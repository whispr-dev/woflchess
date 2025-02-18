////////////////////////////////////////////////////////////
// main.rs
////////////////////////////////////////////////////////////

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use std::thread;
use std::io;
use std::fmt;

use rand::Rng;
use serde::{Serialize, Deserialize};
use ctrlc;
use colored::*;

use eframe::{egui, App, Frame, NativeOptions, run_native};
use eframe::epaint::Pos2;
use eframe::egui::Vec2;

////////////////////////////////////////////////////////////
// 1) DEFINITIONS, ERRORS, TYPES
////////////////////////////////////////////////////////////
mod types {
    use super::*;

    // ---------- Chess Move Errors ----------
    #[derive(Debug)]
    pub enum MoveError {
        OutOfBounds,
        NoPieceAtSource,
        WrongColor,
        InvalidPieceMove(String),
        BlockedPath,
        CastlingInCheck,
        CastlingThroughCheck,
        CastlingRightsLost,
        CastlingRookMissing,
        CastlingPathBlocked,
        WouldCauseCheck,
        KingNotInOriginalPosition,
    }

    impl std::fmt::Display for MoveError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                MoveError::OutOfBounds => write!(f, "Move is out of bounds"),
                MoveError::NoPieceAtSource => write!(f, "No piece at starting square"),
                MoveError::WrongColor => write!(f, "That's not your piece"),
                MoveError::InvalidPieceMove(msg) => write!(f, "{}", msg),
                MoveError::BlockedPath => write!(f, "Path is blocked by other pieces"),
                MoveError::CastlingInCheck => write!(f, "Cannot castle while in check"),
                MoveError::CastlingThroughCheck => write!(f, "Cannot castle through check"),
                MoveError::CastlingRightsLost => write!(f, "Castling rights have been lost"),
                MoveError::CastlingRookMissing => write!(f, "Rook is not in position for castling"),
                MoveError::CastlingPathBlocked => write!(f, "Cannot castle through other pieces"),
                MoveError::WouldCauseCheck => write!(f, "Move would put or leave king in check"),
                MoveError::KingNotInOriginalPosition => write!(f, "King must be in original position to castle"),
            }
        }
    }

    // ---------- Piece Types, Colors ----------
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

    impl std::fmt::Display for Color {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Color::White => write!(f, "White"),
                Color::Black => write!(f, "Black"),
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Piece {
        pub piece_type: PieceType,
        pub color: Color,
    }

    // ---------- Chess Move + Board ----------
    #[derive(Clone, Debug, PartialEq)]
    pub struct Move {
        pub from: (usize, usize),
        pub to: (usize, usize),
        pub piece_moved: Piece,
        pub piece_captured: Option<Piece>,
        // for advanced logic:
        pub is_castling: bool,
        pub is_en_passant: bool,
        pub promotion: Option<PieceType>,
    }

    pub type Board = [[Option<Piece>; 8]; 8];

    // ---------- GameResult + Stats ----------
    #[derive(Debug)]
    pub enum GameResult {
        Checkmate(Color),
        Stalemate,
        DrawByMoveLimit,
    }

    #[derive(Debug)]
    pub struct TrainingStats {
        pub games_played: usize,
        pub white_wins: usize,
        pub black_wins: usize,
        pub draws: usize,
    }

    impl TrainingStats {
        pub fn new() -> Self {
            TrainingStats {
                games_played: 0,
                white_wins: 0,
                black_wins: 0,
                draws: 0,
            }
        }
        pub fn update(&mut self, result: &GameResult) {
            self.games_played += 1;
            match result {
                GameResult::Checkmate(color) => {
                    if *color == Color::White {
                        self.white_wins += 1;
                    } else {
                        self.black_wins += 1;
                    }
                },
                GameResult::Stalemate | GameResult::DrawByMoveLimit => {
                    self.draws += 1;
                }
            }
        }
        pub fn display(&self) {
            println!("Training Statistics:");
            println!("Games Played: {}", self.games_played);
            if self.games_played > 0 {
                println!("White Wins: {} ({:.1}%)",
                    self.white_wins,
                    (self.white_wins as f32 / self.games_played as f32)*100.0
                );
                println!("Black Wins: {} ({:.1}%)",
                    self.black_wins,
                    (self.black_wins as f32 / self.games_played as f32)*100.0
                );
                println!("Draws: {} ({:.1}%)",
                    self.draws,
                    (self.draws as f32 / self.games_played as f32)*100.0
                );
            }
        }
    }

    // ---------- Neural Weights ----------
    #[derive(Serialize, Deserialize)]
    pub struct NetworkWeights {
        pub rnn_weights: Vec<Vec<f32>>,
        pub rnn_discriminator: Vec<Vec<f32>>,
        pub cnn_filters: Vec<Vec<Vec<f32>>>,
        pub cnn_discriminator: Vec<Vec<Vec<f32>>>,
        pub lstm_weights: Vec<Vec<f32>>,
        pub lstm_discriminator: Vec<Vec<f32>>,
    }

    // ---------- The main GameState ----------
    #[derive(Clone)]
    pub struct GameState {
        pub board: Board,
        pub current_turn: Color,
        // for castling
        pub white_can_castle_kingside: bool,
        pub white_can_castle_queenside: bool,
        pub black_can_castle_kingside: bool,
        pub black_can_castle_queenside: bool,
        // for en passant
        pub last_pawn_double_move: Option<(usize, usize)>,
        // neural engine
        pub neural_engine: Option<Arc<Mutex<crate::ChessNeuralEngine>>>,
        // track kings
        pub white_king_pos: (usize, usize),
        pub black_king_pos: (usize, usize),
        // track moves
        pub move_history: Vec<Move>,
    }

    // ---------- The ChessApp for eframe GUI ----------
    pub struct ChessApp {
        pub selected_square: Option<(usize, usize)>,
        // reference a shared GameState
        pub shared_gs: Arc<Mutex<GameState>>,
    }
}

use types::{
    MoveError, PieceType, Color, Piece, Board, Move, GameResult,
    TrainingStats, NetworkWeights, GameState, ChessApp,
};

////////////////////////////////////////////////////////////
// 2) NEURAL ENGINE: CA, RNN, CNN, LSTM, GAN
////////////////////////////////////////////////////////////
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
}

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

// Activation
fn tanh(x: f32) -> f32 { x.tanh() }
fn relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// RNN
struct RNN {
    hidden_state: Vec<f32>,
    weights: Vec<Vec<f32>>,
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
        let hs = self.hidden_state.len();
        let mut output = vec![0.0; hs];
        for i in 0..hs {
            let mut sum = 0.0;
            for (j, &inp) in input.iter().enumerate() {
                sum += inp * self.weights[i][j];
            }
            for j in 0..hs {
                sum += self.hidden_state[j] * self.weights[i][j + input.len()];
            }
            output[i] = tanh(sum);
        }
        self.hidden_state = output.clone();
        output
    }
    fn discriminate(&self, state: &[f32]) -> f32 {
        let hs = self.hidden_state.len();
        let mut sum = 0.0;
        for i in 0..hs {
            for j in 0..hs {
                sum += state[j] * self.discriminator_weights[i][j];
            }
        }
        sigmoid(sum)
    }
}

// CNN
struct CNN {
    filters: Vec<Vec<Vec<f32>>>,
    feature_maps: Vec<Vec<f32>>,
    discriminator_filters: Vec<Vec<Vec<f32>>>,
}
impl CNN {
    fn new(num_filters: usize, rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let filters = (0..num_filters).map(|_| {
            (0..3).map(|_| {
                (0..3).map(|_| rng.gen_range(-0.1..0.1)).collect()
            }).collect()
        }).collect();
        let feature_maps = vec![vec![0.0; rows * cols]; num_filters];
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
        let cols = if rows>0 { input[0].len() } else { 0 };
        let nf = self.filters.len();
        let mut new_feature_maps = vec![vec![0.0; rows*cols]; nf];
        for (f, filter) in self.filters.iter().enumerate() {
            for i in 1..rows.saturating_sub(1) {
                for j in 1..cols.saturating_sub(1) {
                    let mut sum = 0.0;
                    for di in 0..3 {
                        for dj in 0..3 {
                            if i+di>0 && j+dj>0 && (i+di-1<rows) && (j+dj-1<cols) {
                                sum += input[i+di-1][j+dj-1]*filter[di][dj];
                            }
                        }
                    }
                    let out_idx = i*cols + j;
                    if out_idx<new_feature_maps[f].len() {
                        new_feature_maps[f][out_idx] = relu(sum);
                    }
                }
            }
        }
        self.feature_maps = new_feature_maps.clone();
        new_feature_maps
    }
    fn discriminate(&self, fmaps: &[Vec<f32>]) -> f32 {
        let mut sum = 0.0;
        for (f, map) in fmaps.iter().enumerate() {
            if f >= self.discriminator_filters.len() { break; }
            let rows = 3.min((map.len() as f32).sqrt() as usize);
            let cols = rows;
            for i in 0..rows {
                for j in 0..cols {
                    let idx = i*cols + j;
                    if idx<map.len() {
                        sum += map[idx] * self.discriminator_filters[f][i.min(2)][j.min(2)];
                    }
                }
            }
        }
        sigmoid(sum)
    }
}

// LSTM
struct LSTM {
    cell_state: Vec<f32>,
    hidden_state: Vec<f32>,
    weights: Vec<Vec<f32>>,
    discriminator_weights: Vec<Vec<f32>>,
}
impl LSTM {
    fn new(input_dim: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..hidden_size).map(|_| {
            (0..(input_dim+hidden_size)).map(|_| rng.gen_range(-0.1..0.1)).collect()
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
        let hs = self.hidden_state.len();
        let mut new_hidden = vec![0.0; hs];
        let mut new_cell = vec![0.0; hs];
        for i in 0..hs {
            let mut forget_sum=0.0; let mut input_sum=0.0; let mut output_sum=0.0;
            for (j,&inp) in input.iter().enumerate() {
                if j<self.weights[i].len() {
                    forget_sum += inp*self.weights[i][j];
                    input_sum += inp*self.weights[i][j];
                    output_sum += inp*self.weights[i][j];
                }
            }
            for j in 0..hs {
                let widx = j+input.len();
                if widx<self.weights[i].len() {
                    forget_sum += self.hidden_state[j]*self.weights[i][widx];
                    input_sum += self.hidden_state[j]*self.weights[i][widx];
                    output_sum += self.hidden_state[j]*self.weights[i][widx];
                }
            }
            let forget_gate = sigmoid(forget_sum);
            let input_gate = sigmoid(input_sum);
            let output_gate = sigmoid(output_sum);
            new_cell[i] = forget_gate*self.cell_state[i] + input_gate*tanh(input_sum);
            new_hidden[i] = output_gate*tanh(new_cell[i]);
        }
        self.hidden_state=new_hidden.clone();
        self.cell_state=new_cell;
        new_hidden
    }
    fn discriminate(&self, state: &[f32]) -> f32 {
        let hs = self.hidden_state.len();
        let mut sum=0.0;
        for i in 0..hs {
            for j in 0..hs {
                if j<state.len() && j<self.discriminator_weights[i].len() {
                    sum += state[j]*self.discriminator_weights[i][j];
                }
            }
        }
        sigmoid(sum)
    }
}

// The GAN Trainer
struct GANTrainer {
    learning_rate: f32,
    batch_size: usize,
    noise_dim: usize,
}
impl GANTrainer {
    fn new(lr: f32, bs: usize, nd: usize) -> Self {
        GANTrainer { learning_rate: lr, batch_size: bs, noise_dim: nd }
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
        ca: &mut CAInterface,
    ) {
        // Train discriminators
        for _ in 0..self.batch_size {
            let noise = self.generate_noise();
            ca.update();
            let rnn_out = rnn.forward(&noise);
            let cnn_out = cnn.forward(&vec![rnn_out.clone()]);
            let lstm_out = lstm.forward(&rnn_out);
            for real_sample in real_data {
                let rnn_real_score = rnn.discriminate(real_sample);
                let cnn_real_score = cnn.discriminate(&vec![real_sample.clone()]);
                let lstm_real_score = lstm.discriminate(real_sample);
                self.update_discriminator_weights_rnn(rnn, real_sample, &rnn_out, rnn_real_score);
                self.update_discriminator_weights_cnn(cnn, real_sample, &cnn_out[0], cnn_real_score);
                self.update_discriminator_weights_lstm(lstm, real_sample, &lstm_out, lstm_real_score);
            }
        }
        // Train generators
        for _ in 0..self.batch_size {
            let noise = self.generate_noise();
            ca.update();
            let rnn_out = rnn.forward(&noise);
            let cnn_out = cnn.forward(&vec![rnn_out.clone()]);
            let lstm_out = lstm.forward(&rnn_out);
            let rnn_fake_score = rnn.discriminate(&rnn_out);
            let cnn_fake_score = cnn.discriminate(&cnn_out);
            let lstm_fake_score = lstm.discriminate(&lstm_out);
            self.update_generator_weights_rnn(rnn, &noise, rnn_fake_score);
            self.update_generator_weights_cnn(cnn, &noise, cnn_fake_score);
            self.update_generator_weights_lstm(lstm, &noise, lstm_fake_score);
        }
    }

    fn update_discriminator_weights_rnn(&self, rnn: &mut RNN, real: &[f32], fake: &[f32], real_score: f32) {
        for i in 0..rnn.discriminator_weights.len() {
            for j in 0..rnn.discriminator_weights[i].len() {
                let real_grad = real_score*(1.0-real_score)*real[j%real.len()];
                let fake_grad = -fake[j%fake.len()]*(1.0-fake[j%fake.len()]);
                rnn.discriminator_weights[i][j]+=self.learning_rate*(real_grad+fake_grad);
            }
        }
    }
    fn update_discriminator_weights_cnn(&self, cnn: &mut CNN, _real: &[f32], fake: &[f32], real_score: f32) {
        for f in 0..cnn.discriminator_filters.len() {
            for i in 0..3 {
                for j in 0..3 {
                    let real_grad = real_score*(1.0-real_score);
                    let fake_grad = -fake[0]*(1.0-fake[0]);
                    cnn.discriminator_filters[f][i][j]+=self.learning_rate*(real_grad+fake_grad);
                }
            }
        }
    }
    fn update_discriminator_weights_lstm(&self, lstm: &mut LSTM, real: &[f32], fake: &[f32], real_score: f32) {
        for i in 0..lstm.discriminator_weights.len() {
            for j in 0..lstm.discriminator_weights[i].len() {
                let real_grad = real_score*(1.0-real_score)*real[j%real.len()];
                let fake_grad = -fake[j%fake.len()]*(1.0-fake[j%fake.len()]);
                lstm.discriminator_weights[i][j]+=self.learning_rate*(real_grad+fake_grad);
            }
        }
    }

    fn update_generator_weights_rnn(&self, rnn: &mut RNN, noise: &[f32], fake_score: f32) {
        for i in 0..rnn.weights.len() {
            for j in 0..rnn.weights[i].len() {
                let grad = fake_score*(1.0-fake_score)*noise[j%noise.len()];
                rnn.weights[i][j]+=self.learning_rate*grad;
            }
        }
    }
    fn update_generator_weights_cnn(&self, cnn: &mut CNN, _noise: &[f32], fake_score: f32) {
        for f in 0..cnn.filters.len() {
            for i in 0..3 {
                for j in 0..3 {
                    let grad = fake_score*(1.0-fake_score);
                    cnn.filters[f][i][j]+=self.learning_rate*grad;
                }
            }
        }
    }
    fn update_generator_weights_lstm(&self, lstm: &mut LSTM, noise: &[f32], fake_score: f32) {
        for i in 0..lstm.weights.len() {
            for j in 0..lstm.weights[i].len() {
                let grad = fake_score*(1.0-fake_score)*noise[j%noise.len()];
                lstm.weights[i][j]+=self.learning_rate*grad;
            }
        }
    }
}

////////////////////////////////////////////////////////////
// 3) OUR MAIN NEURAL ENGINE
////////////////////////////////////////////////////////////
pub struct ChessNeuralEngine {
    rnn: Arc<Mutex<RNN>>,
    cnn: Arc<Mutex<CNN>>,
    lstm: Arc<Mutex<LSTM>>,
    gan_trainer: GANTrainer,
    ca_interface: CAInterface,
    neural_clock: NeuralClock,
}

impl ChessNeuralEngine {
    fn new() -> Self {
        let input_dim=64; // 8x8 board
        let hidden_size=16;
        ChessNeuralEngine {
            rnn: Arc::new(Mutex::new(RNN::new(input_dim, hidden_size))),
            cnn: Arc::new(Mutex::new(CNN::new(1, 8, 8))),
            lstm: Arc::new(Mutex::new(LSTM::new(input_dim, hidden_size))),
            gan_trainer: GANTrainer::new(0.01, 10, input_dim),
            ca_interface: CAInterface::new(8, 8),
            neural_clock: NeuralClock::new(0.5),
        }
    }

    // Save / Load
    pub fn save_weights(&self, filename: &str) -> std::io::Result<()> {
        let rnn_lock = self.rnn.lock().unwrap();
        let cnn_lock = self.cnn.lock().unwrap();
        let lstm_lock = self.lstm.lock().unwrap();

        let weights = NetworkWeights {
            rnn_weights: rnn_lock.weights.clone(),
            rnn_discriminator: rnn_lock.discriminator_weights.clone(),
            cnn_filters: cnn_lock.filters.clone(),
            cnn_discriminator: cnn_lock.discriminator_filters.clone(),
            lstm_weights: lstm_lock.weights.clone(),
            lstm_discriminator: lstm_lock.discriminator_weights.clone(),
        };
        drop(rnn_lock); drop(cnn_lock); drop(lstm_lock);

        let serialized = serde_json::to_string(&weights)?;
        std::fs::write(filename, serialized)?;
        Ok(())
    }

    pub fn load_weights(&mut self, filename: &str) -> std::io::Result<()> {
        let contents = std::fs::read_to_string(filename)?;
        let weights: NetworkWeights = serde_json::from_str(&contents)?;

        let mut rnn_lock = self.rnn.lock().unwrap();
        let mut cnn_lock = self.cnn.lock().unwrap();
        let mut lstm_lock = self.lstm.lock().unwrap();

        rnn_lock.weights = weights.rnn_weights;
        rnn_lock.discriminator_weights = weights.rnn_discriminator;
        cnn_lock.filters = weights.cnn_filters;
        cnn_lock.discriminator_filters = weights.cnn_discriminator;
        lstm_lock.weights = weights.lstm_weights;
        lstm_lock.discriminator_weights = weights.lstm_discriminator;

        Ok(())
    }

    // Train self-play
    pub fn train_self_play(
        &mut self,
        duration: Option<Duration>,
        num_games: Option<usize>,
    ) -> Result<(), String> {
        // Minimal stub:
        println!("(Stub) train_self_play called with duration={:?}, num_games={:?}",
            duration, num_games);
        Ok(())
    }
}

////////////////////////////////////////////////////////////
// 4) GAMESTATE IMPL: setup, get_piece_at, make_move, etc.
////////////////////////////////////////////////////////////

impl GameState {
    pub fn new() -> Self {
        let mut gs = GameState {
            board: [[None; 8]; 8],
            current_turn: Color::White,
            white_can_castle_kingside: true,
            white_can_castle_queenside: true,
            black_can_castle_kingside: true,
            black_can_castle_queenside: true,
            last_pawn_double_move: None,
            neural_engine: Some(Arc::new(Mutex::new(ChessNeuralEngine::new()))),
            white_king_pos: (4,0),
            black_king_pos: (4,7),
            move_history: Vec::new(),
        };
        gs.setup_initial_position();
        gs
    }

    pub fn setup_initial_position(&mut self) {
        // place pieces
        // White
        let create_piece = |pt, c| Some(Piece { piece_type: pt, color: c });
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
        // Black
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

    pub fn get_piece_at(&self, pos: (usize, usize)) -> Option<&Piece> {
        if pos.0<8 && pos.1<8 {
            self.board[pos.0][pos.1].as_ref()
        } else {
            None
        }
    }

    pub fn make_move(&mut self, mv: Move) -> Result<(), MoveError> {
        // check bounds
        if mv.from.0>=8 || mv.from.1>=8 || mv.to.0>=8 || mv.to.1>=8 {
            return Err(MoveError::OutOfBounds);
        }
        // do move
        self.board[mv.to.0][mv.to.1] = Some(mv.piece_moved);
        self.board[mv.from.0][mv.from.1] = None;
        self.current_turn = self.current_turn.opposite();
        Ok(())
    }

    pub fn get_game_status(&self) -> String {
        // minimal stub
        if self.current_turn == Color::White {
            "White's turn".to_string()
        } else {
            "Black's turn".to_string()
        }
    }

    pub fn make_computer_move(&mut self) -> Result<(), String> {
        // minimal stub
        println!("(Stub) make_computer_move: AI picks a random move or something");
        Ok(())
    }

    pub fn is_in_check(&self, color: Color) -> bool {
        // minimal stub
        false
    }

    pub fn is_square_attacked(&self, pos: (usize, usize), by_color: Color) -> bool {
        // minimal stub
        false
    }

    // parse a move from a string like "e2e4"
    pub fn make_move_from_str(&mut self, input: &str) -> Result<(), String> {
        if input.len()<4 {
            return Err("Move too short (e.g. 'e2e4')".to_string());
        }
        let from = self.parse_square(&input[0..2]).ok_or("Bad from-square")?;
        let to = self.parse_square(&input[2..4]).ok_or("Bad to-square")?;
        let piece = self.get_piece_at(from).ok_or("No piece at source")?;
        if piece.color != self.current_turn {
            return Err("Not your turn".to_string());
        }
        let mv = Move {
            from,
            to,
            piece_moved: *piece,
            piece_captured: self.get_piece_at(to).copied(),
            is_castling: false,
            is_en_passant: false,
            promotion: None,
        };
        self.make_move(mv).map_err(|e| e.to_string())
    }

    fn parse_square(&self, s: &str) -> Option<(usize, usize)> {
        if s.len()!=2 { return None; }
        let file = (s.chars().next()? as u8).wrapping_sub(b'a');
        let rank = (s.chars().nth(1)? as u8).wrapping_sub(b'1');
        if file<8 && rank<8 {
            Some((file as usize, rank as usize))
        } else {
            None
        }
    }
}

// Implement Display for printing the board
impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print file labels (a-h) at top
        write!(f, "  ")?;
        for file in 0..8 {
            write!(f, " {} ", ((file as u8 + b'a') as char).to_string().cyan())?;
        }
        writeln!(f)?;
        writeln!(f, "  {}", "─".repeat(24).bright_magenta())?;

        for rank in (0..8).rev() {
            write!(f, "{} {}",
                (rank + 1).to_string().cyan(),
                "│".bright_magenta()
            )?;
            for file in 0..8 {
                let symbol = match self.board[file][rank] {
                    Some(piece) => match piece.piece_type {
                        PieceType::Pawn => if piece.color == Color::White { 'P' } else { 'p' },
                        PieceType::Knight => if piece.color == Color::White { 'N' } else { 'n' },
                        PieceType::Bishop => if piece.color == Color::White { 'B' } else { 'b' },
                        PieceType::Rook => if piece.color == Color::White { 'R' } else { 'r' },
                        PieceType::Queen => if piece.color == Color::White { 'Q' } else { 'q' },
                        PieceType::King => if piece.color == Color::White { 'K' } else { 'k' },
                    },
                    None => '·',
                };
                let colored_symbol = match self.board[file][rank] {
                    Some(piece) => {
                        if piece.color == Color::White {
                            symbol.to_string().bright_red()
                        } else {
                            symbol.to_string().bright_blue()
                        }
                    },
                    None => symbol.to_string().bright_magenta(),
                };
                write!(f, " {} ", colored_symbol)?;
            }
            writeln!(f, "{} {}",
                "│".bright_magenta(),
                (rank + 1).to_string().cyan()
            )?;
        }

        writeln!(f, "  {}", "─".repeat(24).bright_magenta())?;
        write!(f, "  ")?;
        for file in 0..8 {
            write!(f, " {} ", ((file as u8 + b'a') as char).to_string().cyan())?;
        }
        writeln!(f)?;
        Ok(())
    }
}

////////////////////////////////////////////////////////////
// 5) eframe GUI: ChessApp Implementation
////////////////////////////////////////////////////////////
impl App for ChessApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        let mut gs = self.shared_gs.lock().unwrap();

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Concurrent Chess GUI");
            egui::Grid::new("chess_board")
                .spacing(egui::Vec2::new(0.0, 0.0))
                .show(ui, |ui| {
                    for row in (0..8).rev() {
                        for col in 0..8 {
                            let is_dark = (row + col) % 2 == 1;
                            let base_color = if is_dark {
                                egui::Color32::from_rgb(118, 150, 86)
                            } else {
                                egui::Color32::from_rgb(238, 238, 210)
                            };
                            let square_color = if let Some((sel_r, sel_c)) = self.selected_square {
                                if sel_r == row && sel_c == col {
                                    egui::Color32::YELLOW
                                } else {
                                    base_color
                                }
                            } else {
                                base_color
                            };

                            let label = match gs.board[col][row] {
                                Some(piece) => match piece.piece_type {
                                    PieceType::Pawn => "P",
                                    PieceType::Knight => "N",
                                    PieceType::Bishop => "B",
                                    PieceType::Rook => "R",
                                    PieceType::Queen => "Q",
                                    PieceType::King => "K",
                                },
                                None => " ",
                            };

                            if ui
                                .add(egui::Button::new(label)
                                    .fill(square_color)
                                    .min_size(egui::Vec2::new(50.0, 50.0)))
                                .clicked()
                            {
                                if let Some((from_r, from_c)) = self.selected_square {
                                    // do a move
                                    let piece = gs.get_piece_at((from_c, from_r));
                                    if let Some(piece) = piece {
                                        if piece.color == gs.current_turn {
                                            let mv = Move {
                                                from: (from_c, from_r),
                                                to: (col, row),
                                                piece_moved: *piece,
                                                piece_captured: gs.get_piece_at((col,row)).copied(),
                                                is_castling: false,
                                                is_en_passant: false,
                                                promotion: None,
                                            };
                                            if let Err(e) = gs.make_move(mv) {
                                                println!("[GUI] Move error: {}", e);
                                            }
                                        }
                                    }
                                    self.selected_square = None;
                                } else {
                                    self.selected_square = Some((row, col));
                                }
                            }
                        }
                        ui.end_row();
                    }
                });
        });
    }
}

////////////////////////////////////////////////////////////
// 6) MAIN: concurrency-based console loop
////////////////////////////////////////////////////////////
fn main() {
    // Create a shared GameState behind Arc<Mutex<..>>
    let shared_gs = Arc::new(Mutex::new(GameState::new()));

    // Extract the neural engine from the game state
    let engine_arc = {
        let gs = shared_gs.lock().unwrap();
        gs.neural_engine.clone().unwrap()
    };

    let running = Arc::new(AtomicBool::new(true));
    let running_c = running.clone();

    // Ctrl+C
    ctrlc::set_handler(move || {
        println!("\n[Ctrl+C] Shutting down...");
        running_c.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");

    println!("Concurrent Chess Program with GUI + Console!");
    println!("Commands:");
    println!("  play          - Play a local game vs. AI in the console");
    println!("  train time X  - Self-play for X minutes");
    println!("  train games X - Self-play for X games");
    println!("  gui           - Launch a GUI window in a separate thread");
    println!("  quit          - Exit the program");

    while running.load(Ordering::SeqCst) {
        println!("\nEnter command:");
        let mut line = String::new();
        if io::stdin().read_line(&mut line).is_err() {
            println!("[Error] reading input");
            continue;
        }
        let cmd = line.trim();
        match cmd {
            "play" => {
                // local console game
                play_game(Arc::clone(&shared_gs));
            },
            c if c.starts_with("train time ") => {
                let rest = c.trim_start_matches("train time ");
                if let Ok(minutes) = rest.parse::<u64>() {
                    println!("[Info] Training for {} minutes...", minutes);
                    let mut eng = engine_arc.lock().unwrap();
                    let dur = Duration::from_secs(minutes*60);
                    if let Err(e) = eng.train_self_play(Some(dur), None) {
                        println!("[Error] training: {}", e);
                    }
                } else {
                    println!("[Error] invalid format: train time X");
                }
            },
            c if c.starts_with("train games ") => {
                let rest = c.trim_start_matches("train games ");
                if let Ok(games) = rest.parse::<usize>() {
                    println!("[Info] Training for {} games...", games);
                    let mut eng = engine_arc.lock().unwrap();
                    if let Err(e) = eng.train_self_play(None, Some(games)) {
                        println!("[Error] training: {}", e);
                    }
                } else {
                    println!("[Error] invalid format: train games X");
                }
            },
            "gui" => {
                // spawn a new thread to run the GUI
                let gs_clone = Arc::clone(&shared_gs);
                std::thread::spawn(move || {
                    let app = ChessApp {
                        selected_square: None,
                        shared_gs: gs_clone,
                    };
                    let opts = NativeOptions {
                        initial_window_pos: Some(Pos2{x:50.0,y:50.0}),
                        initial_window_size: Some(Vec2{x:600.0,y:600.0}),
                        ..Default::default()
                    };
                    run_native("Concurrent Chess GUI", opts, Box::new(|_cc| Box::new(app)));
                    println!("[GUI Thread] closed window, returning to console");
                });
            },
            "quit" => {
                println!("[Info] Exiting program...");
                running.store(false, Ordering::SeqCst);
            },
            "" => continue,
            _ => println!("[Error] unknown command"),
        }
    }

    println!("Goodbye!");
}

// A helper for console-based game
fn play_game(shared_gs: Arc<Mutex<GameState>>) {
    println!("Playing console game vs. AI. Type 'resign' or 'quit' to exit.");
    loop {
        let mut gs = shared_gs.lock().unwrap();
        println!("{}", *gs);
        println!("{}", gs.get_game_status());

        if gs.current_turn == Color::White {
            println!("Your move (like e2e4):");
            let mut line = String::new();
            if io::stdin().read_line(&mut line).is_err() {
                println!("Error reading line");
                break;
            }
            let mv_str = line.trim();
            if mv_str=="quit" {
                break;
            }
            if mv_str=="resign" {
                println!("White resigns, black wins!");
                break;
            }
            if let Err(e) = gs.make_move_from_str(mv_str) {
                println!("Move error: {}", e);
                continue;
            }
        } else {
            println!("[AI] thinking...");
            if let Err(e) = gs.make_computer_move() {
                println!("Game Over: {}", e);
                break;
            }
        }
        // minimal check for checkmate/stalemate
        let status = gs.get_game_status();
        if status.contains("Checkmate") || status.contains("Stalemate") {
            println!("[Game Over] {}", status);
            break;
        }
    }
}
