////////////////////////////////////////////////////
// main.rs
////////////////////////////////////////////////////

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use std::io::{Write, Read};
use std::fs::File;
use std::fmt;

use rand::Rng;
use serde::{Serialize, Deserialize};
use ctrlc;
use colored::*;

use eframe::{egui, App, Frame, NativeOptions, run_native};
use eframe::epaint::Pos2;
use eframe::egui::Vec2;

////////////////////////////////////////////////////
// 1) TYPES, DATA STRUCTS, AND ERRORS
////////////////////////////////////////////////////
mod types {
    use super::*;

    // ----- MoveError with full explicit messages -----
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

    // ----- Basic piece/color definitions -----
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

    // ----- Our main chess Move struct -----
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

    // ----- The game result type -----
    #[derive(Debug)]
    pub enum GameResult {
        Checkmate(Color),
        Stalemate,
        DrawByMoveLimit,
    }

    // ----- The main 8×8 board type -----
    pub type Board = [[Option<Piece>; 8]; 8];

    // ----- TrainingStats for self-play -----
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
                         (self.white_wins as f32 / self.games_played as f32)*100.0);
                println!("Black Wins: {} ({:.1}%)", 
                         self.black_wins, 
                         (self.black_wins as f32 / self.games_played as f32)*100.0);
                println!("Draws: {} ({:.1}%)", 
                         self.draws, 
                         (self.draws as f32 / self.games_played as f32)*100.0);
            }
        }
    }

    // ----- NetworkWeights for saving/loading the neural engine -----
    #[derive(Serialize, Deserialize)]
    pub struct NetworkWeights {
        pub rnn_weights: Vec<Vec<f32>>,
        pub rnn_discriminator: Vec<Vec<f32>>,
        pub cnn_filters: Vec<Vec<Vec<f32>>>,
        pub cnn_discriminator: Vec<Vec<Vec<f32>>>,
        pub lstm_weights: Vec<Vec<f32>>,
        pub lstm_discriminator: Vec<Vec<f32>>,
    }

    // ----- The main GameState that holds the board and logic -----
    #[derive(Clone)]
    pub struct GameState {
        pub board: Board,
        pub current_turn: Color,
        pub white_can_castle_kingside: bool,
        pub white_can_castle_queenside: bool,
        pub black_can_castle_kingside: bool,
        pub black_can_castle_queenside: bool,
        pub last_pawn_double_move: Option<(usize, usize)>,
        pub neural_engine: Option<Arc<Mutex<crate::ChessNeuralEngine>>>,
        pub white_king_pos: (usize, usize),
        pub black_king_pos: (usize, usize),
        pub move_history: Vec<Move>,
    }

    // ----- The eframe ChessApp for a GUI board -----
    // We'll manually implement Default for it later.
    pub struct ChessApp { 
        pub selected_square: Option<(usize, usize)>,
        pub game_state: GameState,
    }
}

// Re‐export some stuff for convenience:
use types::{
    Color, PieceType, MoveError, Board, Move, GameResult,
    TrainingStats, NetworkWeights, ChessApp, GameState,
};

////////////////////////////////////////////////////
// 2) NEURAL NETWORK AND GAN SETUP
////////////////////////////////////////////////////
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

// Activation functions
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

// CNN
struct CNN {
    filters: Vec<Vec<Vec<f32>>>,
    feature_maps: Vec<Vec<f32>>,
    discriminator_filters: Vec<Vec<Vec<f32>>>,
}

impl CNN {
    fn new(num_filters: usize, input_rows: usize, input_cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let filters = (0..num_filters).map(|_| {
            (0..3).map(|_| {
                (0..3).map(|_| rng.gen_range(-0.1..0.1)).collect()
            }).collect()
        }).collect();
        let feature_maps = vec![vec![0.0; input_rows * input_cols]; num_filters];
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

        let mut new_feature_maps = vec![vec![0.0; rows * cols]; num_filters];
        for (f, filter) in self.filters.iter().enumerate() {
            for i in 1..rows.saturating_sub(1) {
                for j in 1..cols.saturating_sub(1) {
                    let mut sum = 0.0;
                    for di in 0..3 {
                        for dj in 0..3 {
                            if (i + di > 0) && (j + dj > 0)
                                && (i + di - 1 < rows) && (j + dj - 1 < cols)
                            {
                                sum += input[i+di-1][j+dj-1] * filter[di][dj];
                            }
                        }
                    }
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
        for (f, map) in feature_maps.iter().enumerate() {
            if f >= self.discriminator_filters.len() {
                break;
            }
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
            (0..(input_dim + hidden_size)).map(|_| rng.gen_range(-0.1..0.1)).collect()
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
            let mut input_sum = 0.0;
            let mut output_sum = 0.0;

            for (j, &input_val) in input.iter().enumerate() {
                if j < self.weights[i].len() {
                    forget_sum += input_val * self.weights[i][j];
                    input_sum += input_val * self.weights[i][j];
                    output_sum += input_val * self.weights[i][j];
                }
            }

            for j in 0..hidden_size {
                let weight_idx = j + input.len();
                if weight_idx < self.weights[i].len() {
                    forget_sum += self.hidden_state[j] * self.weights[i][weight_idx];
                    input_sum += self.hidden_state[j] * self.weights[i][weight_idx];
                    output_sum += self.hidden_state[j] * self.weights[i][weight_idx];
                }
            }
            let forget_gate = sigmoid(forget_sum);
            let input_gate = sigmoid(input_sum);
            let output_gate = sigmoid(output_sum);

            new_cell[i] = forget_gate * self.cell_state[i] + input_gate * tanh(input_sum);
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
                if j < state.len() && j < self.discriminator_weights[i].len() {
                    sum += state[j] * self.discriminator_weights[i][j];
                }
            }
        }
        sigmoid(sum)
    }
}

// The GAN trainer
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
        // This is the short version, re‐add expansions as needed
        // 1) Train discriminators
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

        // 2) Train generators
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

////////////////////////////////////////////////////
// 3) OUR NEURAL ENGINE (GAN + CA + RNN/CNN/LSTM)
////////////////////////////////////////////////////
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
        let input_dim = 64;  // 8x8 board
        let hidden_size = 16;
        ChessNeuralEngine {
            rnn: Arc::new(Mutex::new(RNN::new(input_dim, hidden_size))),
            cnn: Arc::new(Mutex::new(CNN::new(1, 8, 8))),
            lstm: Arc::new(Mutex::new(LSTM::new(input_dim, hidden_size))),
            gan_trainer: GANTrainer::new(0.01, 10, input_dim),
            ca_interface: CAInterface::new(8, 8),
            neural_clock: NeuralClock::new(0.5),
        }
    }

    // Save / Load weights
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

        drop(rnn_lock);
        drop(cnn_lock);
        drop(lstm_lock);

        let serialized = serde_json::to_string(&weights)?;
        let mut file = File::create(filename)?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    pub fn load_weights(&mut self, filename: &str) -> std::io::Result<()> {
        let mut file = File::open(filename)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let weights: NetworkWeights = serde_json::from_str(&contents)?;

        {
            let mut rnn_lock = self.rnn.lock().unwrap();
            let mut cnn_lock = self.cnn.lock().unwrap();
            let mut lstm_lock = self.lstm.lock().unwrap();

            rnn_lock.weights = weights.rnn_weights;
            rnn_lock.discriminator_weights = weights.rnn_discriminator;
            cnn_lock.filters = weights.cnn_filters;
            cnn_lock.discriminator_filters = weights.cnn_discriminator;
            lstm_lock.weights = weights.lstm_weights;
            lstm_lock.discriminator_weights = weights.lstm_discriminator;
        }

        Ok(())
    }

    fn train_on_positions(&mut self, positions: &[Vec<f32>]) {
        let mut training_data = Vec::new();
        for pos in positions {
            training_data.push(pos.clone());
            // Slightly modified versions
            for _ in 0..3 {
                let mut variation = pos.clone();
                for value in variation.iter_mut() {
                    *value += rand::thread_rng().gen_range(-0.1..0.1);
                }
                training_data.push(variation);
            }
        }

        let mut rnn = self.rnn.lock().unwrap();
        let mut cnn = self.cnn.lock().unwrap();
        let mut lstm = self.lstm.lock().unwrap();

        self.gan_trainer.train_step(
            &mut rnn,
            &mut cnn,
            &mut lstm,
            &training_data,
            &mut self.ca_interface,
        );
    }

    pub fn train_self_play(
        &mut self,
        duration: Option<Duration>,
        num_games: Option<usize>,
    ) -> Result<TrainingStats, String> {
        let running = Arc::new(AtomicBool::new(true));
        let r = running.clone();

        ctrlc::set_handler(move || {
            println!("\n[Warning] Received Ctrl+C, gracefully shutting down...");
            r.store(false, Ordering::SeqCst);
        }).map_err(|e| format!("Error setting Ctrl+C handler: {}", e))?;

        let start_time = Instant::now();
        let mut stats = TrainingStats::new();
        let mut game_count = 0;

        while running.load(Ordering::SeqCst) {
            // Termination conditions
            if let Some(dur) = duration {
                if start_time.elapsed() >= dur {
                    println!("[Info] Training time limit reached.");
                    break;
                }
            }
            if let Some(games) = num_games {
                if game_count >= games {
                    println!("[Info] Reached desired number of training games.");
                    break;
                }
            }

            match self.play_training_game(&running) {
                Ok(game_result) => {
                    stats.update(&game_result);
                    game_count += 1;
                    println!("[Info] Completed game {}", game_count);
                },
                Err(e) => {
                    if e == "Training interrupted" {
                        println!("[Warning] Training interrupted, saving progress...");
                        break;
                    }
                    println!("[Error] In game {}: {}", game_count + 1, e);
                    if e.contains("Invalid board state") {
                        println!("[Warning] Attempting to save weights before exit...");
                        if let Err(save_err) = self.save_weights("neural_weights.json") {
                            println!("[Error] Failed to save weights on error: {}", save_err);
                        }
                        return Err(e);
                    }
                }
            }

            // Save weights every 10 games
            if game_count % 10 == 0 && game_count > 0 {
                println!("[Info] Saving periodic weights...");
                if let Err(e) = self.save_weights("neural_weights.json") {
                    println!("[Warning] Failed to save weights: {}", e);
                } else {
                    println!("[Info] Saved weights after game {}", game_count);
                }
            }

            if !running.load(Ordering::SeqCst) {
                break;
            }
        }

        println!("[Info] Saving final weights...");
        if let Err(e) = self.save_weights("neural_weights.json") {
            println!("[Warning] Failed to save final weights: {}", e);
        } else {
            println!("[Info] Saved final weights successfully");
        }

        println!("\nTraining summary:");
        stats.display();
        println!("[Info] Training session complete!");
        Ok(stats)
    }

    fn play_training_game(&mut self, running: &Arc<AtomicBool>) -> Result<GameResult, String> {
        println!("\n[Self-Play] Starting new training game...");
        let mut game_state = GameState::new();
        let mut moves = 0;
        let mut positions = Vec::new();

        if let Err(e) = game_state.validate_board_state() {
            return Err(format!("Initial board state invalid: {}", e));
        }

        println!("Initial position:");
        println!("{}", game_state);

        while moves < 200 {
            if !running.load(Ordering::SeqCst) {
                return Err("Training interrupted".to_string());
            }

            let legal = game_state.generate_legal_moves();
            if legal.is_empty() {
                if game_state.is_in_check(game_state.current_turn) {
                    println!("Checkmate! {} wins!", game_state.current_turn.opposite());
                    return Ok(GameResult::Checkmate(game_state.current_turn.opposite()));
                } else {
                    println!("Stalemate!");
                    return Ok(GameResult::Stalemate);
                }
            }

            positions.push(game_state.board_to_neural_input());

            println!("\nMove {}: {} to play", moves + 1, game_state.current_turn);

            match game_state.make_computer_move() {
                Ok(_) => println!("{}", game_state),
                Err(e) => {
                    if e.contains("Checkmate") {
                        println!("Checkmate! {} wins!", game_state.current_turn.opposite());
                        return Ok(GameResult::Checkmate(game_state.current_turn.opposite()));
                    } else if e.contains("Stalemate") {
                        println!("Stalemate!");
                        return Ok(GameResult::Stalemate);
                    } else {
                        return Err(e);
                    }
                }
            }
            moves += 1;
            std::thread::sleep(Duration::from_millis(500));
        }

        // If normal exit
        self.train_on_positions(&positions);
        println!("[Info] Game reached move limit => DrawByMoveLimit");
        Ok(GameResult::DrawByMoveLimit)
    }
}

////////////////////////////////////////////////////
// 4) GAMESTATE IMPLEMENTATION
////////////////////////////////////////////////////
const BISHOP_DIRECTIONS: [(i8, i8); 4] = [(-1,-1), (-1,1), (1,-1), (1,1)];
const ROOK_DIRECTIONS: [(i8, i8); 4] = [(-1,0), (1,0), (0,-1), (0,1)];
const KNIGHT_MOVES: [(i8, i8); 8] = [
    (-2,-1), (-2,1), (-1,-2), (-1,2),
    (1,-2), (1,2), (2,-1), (2,1)
];

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
            white_king_pos: (4, 0),
            black_king_pos: (4, 7),
            move_history: Vec::new(),
            neural_engine: Some(Arc::new(Mutex::new(ChessNeuralEngine::new()))),
        };
        gs.setup_initial_position();
        gs
    }

    // The function that the GUI calls to do a move
    pub fn try_move(&mut self, from: (usize, usize), to: (usize, usize))
        -> Result<(), MoveError>
    {
        let piece = self.get_piece_at(from).ok_or(MoveError::NoPieceAtSource)?;
        if piece.color != self.current_turn {
            return Err(MoveError::WrongColor);
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
        self.make_move(mv)
    }

    pub fn setup_initial_position(&mut self) {
        let create_piece = |pt, color| Some(types::Piece { piece_type: pt, color });
        // White
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

    // For debugging or mixing with the neural net
    pub fn evaluate_position(&self) -> i32 {
        let pawn_value = 100;
        let knight_value = 320;
        let bishop_value = 330;
        let rook_value = 500;
        let queen_value = 900;
        let king_value = 20000;

        let mut score = 0;
        for rank in 0..8 {
            for file in 0..8 {
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

    pub fn evaluate_position_neural(&self) -> i32 {
        if let Some(engine_arc) = &self.neural_engine {
            let engine = engine_arc.lock().unwrap();
            let input = self.board_to_neural_input();

            let rnn_eval = {
                let mut rnn = engine.rnn.lock().unwrap();
                let out = rnn.forward(&input);
                out.get(0).copied().unwrap_or(0.0) * 100.0
            };
            let cnn_eval = {
                let mut cnn = engine.cnn.lock().unwrap();
                let reshaped = reshape_vector_to_matrix(&input, 8, 8);
                let out = cnn.forward(&reshaped);
                out.first()
                   .and_then(|row| row.first())
                   .copied()
                   .unwrap_or(0.0) * 100.0
            };
            let lstm_eval = {
                let mut lstm = engine.lstm.lock().unwrap();
                let out = lstm.forward(&input);
                out.get(0).copied().unwrap_or(0.0) * 100.0
            };
            let trad_eval = self.evaluate_position() as f32;
            let combined = trad_eval * 0.4 + rnn_eval * 0.2 + cnn_eval * 0.2 + lstm_eval * 0.2;
            combined as i32
        } else {
            self.evaluate_position()
        }
    }

    // Display or debug
    pub fn get_game_status(&self) -> String {
        let mut piece_count = 0;
        let mut has_minor_piece = false;
        for rank in 0..8 {
            for file in 0..8 {
                if let Some(piece) = self.board[file][rank] {
                    piece_count += 1;
                    if piece.piece_type != PieceType::King {
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

    // Minimally required for the training engine
    pub fn generate_legal_moves(&self) -> Vec<Move> {
        // Similar logic as before...
        let mut moves = Vec::new();
        for x in 0..8 {
            for y in 0..8 {
                if let Some(piece) = self.board[x][y] {
                    if piece.color == self.current_turn {
                        for xx in 0..8 {
                            for yy in 0..8 {
                                let test_mv = Move {
                                    from: (x, y),
                                    to: (xx, yy),
                                    piece_moved: piece,
                                    piece_captured: self.board[xx][yy],
                                    is_castling: false,
                                    is_en_passant: false,
                                    promotion: None,
                                };
                                if self.is_valid_move(&test_mv).is_ok() {
                                    moves.push(test_mv);
                                }
                            }
                        }
                    }
                }
            }
        }
        // Sort or do move ordering if you like
        moves
    }

    pub fn make_computer_move(&mut self) -> Result<(), String> {
        // Basic neural minimax approach
        if let Err(e) = self.validate_board_state() {
            return Err(format!("Invalid board state: {}", e));
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
        let depth = 3;
        for mv in moves {
            let mut test_state = self.clone();
            if test_state.make_move(mv.clone()).is_err() {
                continue;
            }
            let score = -test_state.minimax_neural(depth - 1, i32::MIN, i32::MAX, true);
            if score > best_score {
                best_score = score;
                best_moves.clear();
                best_moves.push(mv.clone());
            } else if score == best_score {
                best_moves.push(mv.clone());
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
            return Err(format!("Move resulted in invalid board state: {}", e));
        }
        println!("Computer plays: {}", self.move_to_algebraic(&selected));
        Ok(())
    }

    // Additional minimax
    pub fn minimax_neural(&mut self, depth: i32, mut alpha: i32, mut beta: i32, maximizing: bool) -> i32 {
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

    pub fn generate_moves_neural(&self) -> Vec<Move> {
        // Possibly do neural move ordering. We'll just do generate_legal_moves
        // but you could do advanced heuristics
        self.generate_legal_moves()
    }

    pub fn make_move(&mut self, mv: Move) -> Result<(), MoveError> {
        self.is_valid_move(&mv)?;
        self.make_move_without_validation(&mv);
        self.move_history.push(mv.clone());
        self.current_turn = self.current_turn.opposite();
        Ok(())
    }

    pub fn make_move_without_validation(&mut self, mv: &Move) {
        self.board[mv.to.0][mv.to.1] = None;
        if let Some(promote) = mv.promotion {
            let promoted = types::Piece {
                piece_type: promote,
                color: mv.piece_moved.color,
            };
            self.board[mv.to.0][mv.to.1] = Some(promoted);
        } else {
            self.board[mv.to.0][mv.to.1] = Some(mv.piece_moved);
        }
        self.board[mv.from.0][mv.from.1] = None;

        // castling?
        if mv.is_castling {
            // ...
        }
        // etc. (like en passant)
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

    pub fn is_valid_move(&self, mv: &Move) -> Result<(), MoveError> {
        if !Self::is_within_bounds(mv.from) || !Self::is_within_bounds(mv.to) {
            return Err(MoveError::OutOfBounds);
        }
        let piece = self.get_piece_at(mv.from).ok_or(MoveError::NoPieceAtSource)?;
        if let Some(dest) = self.get_piece_at(mv.to) {
            if dest.color == piece.color {
                return Err(MoveError::InvalidPieceMove("Cannot capture your own piece".to_string()));
            }
        }
        // You can expand to check actual piece moves, castling, etc.
        // We'll keep it short for the example
        Ok(())
    }

    pub fn is_within_bounds(pos: (usize, usize)) -> bool {
        pos.0 < 8 && pos.1 < 8
    }

    pub fn get_piece_at(&self, pos: (usize, usize)) -> Option<&types::Piece> {
        if Self::is_within_bounds(pos) {
            self.board[pos.0][pos.1].as_ref()
        } else {
            None
        }
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
        // We can do a partial check, or the real approach
        for x in 0..8 {
            for y in 0..8 {
                if let Some(piece) = self.board[x][y] {
                    if piece.color == by_color {
                        let test_mv = Move {
                            from: (x, y),
                            to: pos,
                            piece_moved: piece,
                            piece_captured: self.board[pos.0][pos.1],
                            is_castling: false,
                            is_en_passant: false,
                            promotion: None,
                        };
                        if self.is_valid_move(&test_mv).is_ok() {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }
}

// Helper for CNN input
fn reshape_vector_to_matrix(vec: &[f32], rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let mut matrix = vec![vec![0.0; cols]; rows];
    for (i, &value) in vec.iter().enumerate() {
        if i >= rows * cols {
            break;
        }
        let r = i / cols;
        let c = i % cols;
        matrix[r][c] = value;
    }
    matrix
}

////////////////////////////////////////////////////
// 5) fmt::Display for GameState
////////////////////////////////////////////////////
impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print file labels
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

////////////////////////////////////////////////////
// 6) eframe GUI: ChessApp
////////////////////////////////////////////////////

// Provide a normal Default for ChessApp (not in trait App!)
impl Default for ChessApp {
    fn default() -> Self {
        ChessApp {
            selected_square: None,
            game_state: GameState::new(),
        }
    }
}

impl App for ChessApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Chess Board");
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

                            // Show piece
                            let label = match self.game_state.board[col][row] {
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
                                .add(egui::Button::new(label).fill(square_color)
                                     .min_size(egui::Vec2::new(50.0, 50.0)))
                                .clicked()
                            {
                                if let Some((from_r, from_c)) = self.selected_square {
                                    let mv_result = self.game_state.try_move(
                                        (from_c, from_r),
                                        (col, row),
                                    );
                                    if let Err(e) = mv_result {
                                        println!("[GUI] Move error: {:?}", e);
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

////////////////////////////////////////////////////
// 7) HELPER: Move Algebraic
////////////////////////////////////////////////////
impl GameState {
    pub fn move_to_algebraic(&self, mv: &Move) -> String {
        if mv.is_castling {
            if mv.to.0 == 6 {
                "O-O".to_string()
            } else {
                "O-O-O".to_string()
            }
        } else {
            let piece_char = match mv.piece_moved.piece_type {
                PieceType::Pawn => "",
                PieceType::Knight => "N",
                PieceType::Bishop => "B",
                PieceType::Rook => "R",
                PieceType::Queen => "Q",
                PieceType::King => "K",
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
}

////////////////////////////////////////////////////
// 8) MAIN: console-based commands
////////////////////////////////////////////////////
fn main() {
    println!("Neural Chess Engine v2.0 with RNN/CNN/LSTM + CA + GAN!");
    println!("Commands:");
    println!("  'play' - Start a new game vs. the AI");
    println!("  'train time XX' - Self-play for XX minutes");
    println!("  'train games XX' - Self-play for XX games");
    println!("  'gui' - (Optional) Launch a GUI board");
    println!("  'quit' - Exit the program");

    let mut engine = ChessNeuralEngine::new();
    // Try load weights
    match engine.load_weights("neural_weights.json") {
        Ok(_) => println!("[Info] Successfully loaded existing neural weights"),
        Err(e) => println!("[Warning] No existing weights found, starting fresh: {}", e),
    }

    loop {
        println!("\nEnter command:");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        match input {
            "play" => play_game(),
            cmd if cmd.starts_with("train time ") => {
                let rest = cmd.trim_start_matches("train time ");
                if let Ok(minutes) = rest.parse::<u64>() {
                    println!("[Info] Starting training for {} minutes...", minutes);
                    match engine.train_self_play(Some(Duration::from_secs(minutes * 60)), None) {
                        Ok(_) => println!("[Info] Training completed successfully"),
                        Err(e) => println!("[Error] Training error: {}", e),
                    }
                } else {
                    println!("[Error] Invalid time format. Use 'train time XX' where XX is minutes");
                }
            },
            cmd if cmd.starts_with("train games ") => {
                let rest = cmd.trim_start_matches("train games ");
                if let Ok(games) = rest.parse::<usize>() {
                    println!("[Info] Starting training for {} games...", games);
                    match engine.train_self_play(None, Some(games)) {
                        Ok(_) => println!("[Info] Training completed successfully"),
                        Err(e) => println!("[Error] Training error: {}", e),
                    }
                } else {
                    println!("[Error] Invalid games format. Use 'train games XX' where XX is number of games");
                }
            },
            "gui" => {
                println!("[Info] Launching GUI... close it to return here.");
                run_gui();
            },
            "quit" => break,
            "" => continue,
            _ => println!("[Error] Unknown command"),
        }
    }
    println!("[Info] Goodbye!");
}

// A simple function to run the eframe GUI
fn run_gui() {
    let options = NativeOptions {
        initial_window_pos: Some(Pos2 { x: 50.0, y: 50.0 }),
        initial_window_size: Some(Vec2 { x: 600.0, y: 600.0 }),
        ..Default::default()
    };

    run_native(
        "Chess GUI",
        options,
        Box::new(|_cc| Box::new(ChessApp::default())),
    );
}

// A simple function to play one game vs. the AI
fn play_game() {
    let mut game_state = GameState::new();
    loop {
        println!("{}", game_state);
        println!("{}", game_state.get_game_status());
        if game_state.current_turn == Color::White {
            println!("Your move (e.g. 'e2e4' or 'resign'):");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();
            if input == "quit" {
                break;
            }
            if let Err(e) = game_state.make_move_from_str(input) {
                println!("[Error] {}", e);
                continue;
            }
        } else {
            println!("[AI] Thinking...");
            if let Err(e) = game_state.make_computer_move() {
                println!("[Game Over] {}", e);
                break;
            }
        }
    }
}

////////////////////////////////////////////////////
// 9) Extra: parse user moves
////////////////////////////////////////////////////
impl GameState {
    pub fn make_move_from_str(&mut self, move_str: &str) -> Result<(), String> {
        if move_str.to_lowercase() == "resign" {
            println!("{} resigns. {} wins!", self.current_turn, self.current_turn.opposite());
            return Err("Resigned".to_string());
        }
        if move_str.len() < 4 {
            return Err("Move string too short (e.g. 'e2e4')".to_string());
        }
        let from = self.parse_square(&move_str[0..2])
            .ok_or_else(|| "Invalid 'from' square".to_string())?;
        let to = self.parse_square(&move_str[2..4])
            .ok_or_else(|| "Invalid 'to' square".to_string())?;

        let piece = self.get_piece_at(from)
            .ok_or_else(|| "No piece at starting square".to_string())?;
        if piece.color != self.current_turn {
            return Err("It's not your turn!".to_string());
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
        if s.len() != 2 {
            return None;
        }
        let chars: Vec<char> = s.chars().collect();
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
}
