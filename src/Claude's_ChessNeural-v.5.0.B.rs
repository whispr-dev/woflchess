//////////////////////////
// main.rs
//////////////////////////


// welcome to Claude's ChessNeural - a Version of chess by wofl with an optional computer
// opponent employing a very special brain i've been experimenting with.
// v.2.0.0 has expanded err msgs for more useful feedback during gaming.
// v.3.0.0 has colours and nice labelling on rows/columns. enjoy!
// v.4.5.0 This implementation adds:
//
// Weight persistence:
//
// Saves neural network weights to a JSON file
// Loads weights on startup
// Periodically saves during training
//
//
// Training commands:
//
// train time XX - Train for XX minutes
// train games XX - Train for XX number of games
// Training results are displayed after completion
//
//
// Training process:
//
// Plays full games against itself
// Collects successful positions
// Creates variations for additional training data
// Updates neural networks using GAN training
// Tracks statistics (wins/losses/draws)
//
//
// Safeguards:
//
// 200 move limit per game to prevent infinite games
// Periodic weight saving every 10 games
// Error handling for file operations
//
//
//
// To use it:
//
// Start the program
// Use train time 15 to train for 15 minutes
// Use train games 100 to train for 100 games
// Use play to start a game against the AI
// The AI will get progressively better as it trains
//
// The weights are saved to "neural_weights.json" and will be loaded automatically when you restart the program.
//
// v.4.6.0 fixed bug with game not exiting gracefullly upon end self train.
//
//
// v.4.7.0 fixed bug so it handles [Ctrl]+[C] gracefully and quickly
// - Saves progress properly
// - No freeze or hang
// - Returns to the command prompt properly
// - Handles empty input gracefully
// - option for a small sleep to prevent CPU overload during training. [commented out rn]
//
//
// v.4.8.0 fixed bug with game not exiting gracefullly upon end self train.
//key changes are:
//
// Get all Mutex locks at once in save_weights and explicitly drop them
// Added more status messages to track what's happening
// Added check for running status after saving weights
// Added completion message
// Improved error handling and messaging
// Only save periodic weights after first game
//
// This should:
//
// Prevent deadlocks when saving weights
// Handle Ctrl+C more gracefully
// Exit properly after training completes
// Give better feedback about what's happening
//
//
// v.4.9.0 added one last functionality:
//
// changes include:
//
// Display the initial board position
// Show whose turn it is for each move
// Display the board after each move
// Added a 500ms pause between moves to make it easier to follow
// Better game end messages
// Move counter display
//
// Now when you run training, you'll see:
//
// The complete board after every move
// Clear indications of whose turn it is
// Move numbers
// Game results messages
//
//
// v.5.0.0 Final release version packaged, stripped and pretty.
// main changes are:
//
// Added length validation for moves (2-3 characters only)
// Added explicit validation for piece letters
// Added validation for destination squares
// Better error messages for different types of invalid input
// Separated validation for pawn moves and piece moves
// Protected against array index overflow
// woohoo! thnxu for your interest, hope you ahve fun fooling with this :)


// ---------- LOAD CRATES ----------
use std::sync::{Arc, Mutex};
use rand::Rng;
use std::fmt;
use bitflags::bitflags;
use colored::*;
use std::fs::File;
use std::io::{Write, Read};
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};
use std::sync::atomic::{AtomicBool, Ordering};
use ctrlc;


// ---------- TYPES MODULE ----------
// In the types module
mod types {
    use super::*;

    // Add this new enum:
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

    #[derive(Serialize, Deserialize)]
    pub struct NetworkWeights {
        pub rnn_weights: Vec<Vec<f32>>,
        pub rnn_discriminator: Vec<Vec<f32>>,
        pub cnn_filters: Vec<Vec<Vec<f32>>>,
        pub cnn_discriminator: Vec<Vec<Vec<f32>>>,
        pub lstm_weights: Vec<Vec<f32>>,
        pub lstm_discriminator: Vec<Vec<f32>>,
    }

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
                GameResult::Checkmate(Color::White) => self.white_wins += 1,
                GameResult::Checkmate(Color::Black) => self.black_wins += 1,
                GameResult::Stalemate | GameResult::DrawByMoveLimit => self.draws += 1,
            }
        }

        pub fn display(&self) {
            println!("Training Statistics:");
            println!("Games Played: {}", self.games_played);
            println!("White Wins: {} ({:.1}%)", 
                self.white_wins, 
                (self.white_wins as f32 / self.games_played as f32) * 100.0);
            println!("Black Wins: {} ({:.1}%)", 
                self.black_wins,
                (self.black_wins as f32 / self.games_played as f32) * 100.0);
            println!("Draws: {} ({:.1}%)",
                self.draws,
                (self.draws as f32 / self.games_played as f32) * 100.0);
        }
    }

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
}    

// Bring GameState into scope from the types module.
use crate::types::{PieceType, Color, MoveError};  // Add MoveError to your imports
use crate::types::{GameState, GameResult, TrainingStats, NetworkWeights};


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

// RNN Implementation
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

// CNN Implementation
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
                                && (i + di - 1 < rows) && (j + dj - 1 < cols) {
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

// LSTM Implementation
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
            cnn: Arc::new(Mutex::new(CNN::new(1, 8, 8))),  
            lstm: Arc::new(Mutex::new(LSTM::new(input_dim, hidden_size))),
            gan_trainer: GANTrainer::new(0.01, 10, input_dim),
            ca_interface: CAInterface::new(8, 8),
            neural_clock: NeuralClock::new(0.5),
        }
    }

    pub fn save_weights(&self, filename: &str) -> std::io::Result<()> {
        // Get all locks at once to prevent deadlock
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

        // Drop locks before file operations
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
        
        self.rnn.lock().unwrap().weights = weights.rnn_weights;
        self.rnn.lock().unwrap().discriminator_weights = weights.rnn_discriminator;
        self.cnn.lock().unwrap().filters = weights.cnn_filters;
        self.cnn.lock().unwrap().discriminator_filters = weights.cnn_discriminator;
        self.lstm.lock().unwrap().weights = weights.lstm_weights;
        self.lstm.lock().unwrap().discriminator_weights = weights.lstm_discriminator;
        
        Ok(())
    }

    fn train_on_positions(&mut self, positions: &[Vec<f32>]) {
        // Create small random variations of successful positions for training
        let mut training_data = Vec::new();
        for pos in positions {
            training_data.push(pos.clone());
            // Add slightly modified versions
            for _ in 0..3 {
                let mut variation = pos.clone();
                for value in variation.iter_mut() {
                    *value += rand::thread_rng().gen_range(-0.1..0.1);
                }
                training_data.push(variation);
            }
        }

        // Train the networks
        self.gan_trainer.train_step(
            &mut self.rnn.lock().unwrap(),
            &mut self.cnn.lock().unwrap(),
            &mut self.lstm.lock().unwrap(),
            &training_data,
            &mut self.ca_interface,
        );
    }    
    
    pub fn train_self_play(&mut self, duration: Option<Duration>, num_games: Option<usize>) -> Result<TrainingStats, String> {
        // Set up Ctrl+C handler
        let running = Arc::new(AtomicBool::new(true));
        let r = running.clone();
        
        ctrlc::set_handler(move || {
            println!("\nReceived Ctrl+C, gracefully shutting down...");
            r.store(false, Ordering::SeqCst);
        }).map_err(|e| format!("Error setting Ctrl+C handler: {}", e))?;

        let start_time = std::time::Instant::now();
        let mut stats = TrainingStats::new();
        let mut game_count = 0;

        while running.load(Ordering::SeqCst) {
            // Check termination conditions
            if let Some(dur) = duration {
                if start_time.elapsed() >= dur {
                    break;
                }
            }
            if let Some(games) = num_games {
                if game_count >= games {
                    break;
                }
            }

            // Play a complete game
            match self.play_training_game(&running) {
                Ok(game_result) => {
                    stats.update(&game_result);
                    game_count += 1;
                    println!("Completed game {}", game_count);
                },
                Err(e) => {
                    if e == "Training interrupted" {
                        println!("Training interrupted, saving progress...");
                        break;
                    }
                    println!("Error in game {}: {}", game_count + 1, e);
                    if e.contains("Invalid board state") {
                        // Save weights before exiting
                        println!("Attempting to save weights before exit...");
                        if let Err(save_err) = self.save_weights("neural_weights.json") {
                            println!("Failed to save weights on error: {}", save_err);
                        }
                        return Err(e);
                    }
                }
            }

            // Save weights periodically (every 10 games)
            if game_count % 10 == 0 && game_count > 0 {
                println!("Saving periodic weights...");
                if let Err(e) = self.save_weights("neural_weights.json") {
                    println!("Warning: Failed to save weights: {}", e);
                } else {
                    println!("Saved weights after game {}", game_count);
                }
            }

            // Check if we should exit after saving
            if !running.load(Ordering::SeqCst) {
                break;
            }
        }

        // Final weight save
        println!("Saving final weights...");
        let save_result = self.save_weights("neural_weights.json");
        
        match save_result {
            Ok(_) => println!("Saved final weights successfully"),
            Err(e) => println!("Warning: Failed to save final weights: {}", e),
        }

        println!("\nTraining summary:");
        stats.display();
        
        println!("Training session complete!");
        Ok(stats)
    }

    fn play_training_game(&mut self, running: &Arc<AtomicBool>) -> Result<GameResult, String> {
        println!("\nStarting new training game...");
        let mut game_state = GameState::new();
        let mut moves = 0;
        let mut positions = Vec::new();

        // Initial board validation
        if let Err(e) = game_state.validate_board_state() {
            return Err(format!("Initial board state invalid: {}", e));
        }

        // Show initial board
        println!("Initial position:");
        println!("{}", game_state);

        while moves < 200 { // Prevent infinite games
            // Check if we should terminate
            if !running.load(Ordering::SeqCst) {
                return Err("Training interrupted".to_string());
            }

            if game_state.generate_legal_moves().is_empty() {
                if game_state.is_in_check(game_state.current_turn) {
                    println!("Checkmate! {} wins!", game_state.current_turn.opposite());
                    return Ok(GameResult::Checkmate(game_state.current_turn.opposite()));
                } else {
                    println!("Stalemate!");
                    return Ok(GameResult::Stalemate);
                }
            }

            // Store position for training
            positions.push(game_state.board_to_neural_input());

            // Print whose turn it is
            println!("\nMove {}: {} to play", moves + 1, game_state.current_turn);

            // Make move
            match game_state.make_computer_move() {
                Ok(_) => {
                    // Display the board after each move
                    println!("{}", game_state);
                },
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
            
            // Optional: Add a small pause to make it easier to follow
            std::thread::sleep(std::time::Duration::from_millis(500));
        }

        // Train on the collected positions if we completed normally
        if running.load(Ordering::SeqCst) {
            self.train_on_positions(&positions);
        }

        println!("Game reached move limit!");
        Ok(GameResult::DrawByMoveLimit)
    }
}

// Free function for reshaping a 64-element vector into 8x8
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

// ---------- CHESS ENGINE (GAME STATE) ----------

// ---------- IMPL GameState ----------
// (All your methods remain here as you wrote them)
impl GameState {
    fn new() -> Self {
        let mut new_state = GameState {
            board: [[None; 8]; 8],
            current_turn: types::Color::White,
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
        new_state.setup_initial_position();
        new_state
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
        for i in 0..8 {
            self.board[i][1] = create_piece(types::PieceType::Pawn, types::Color::White);
        }
        // Black pieces
        self.board[0][7] = create_piece(types::PieceType::Rook, types::Color::Black);
        self.board[7][7] = create_piece(types::PieceType::Rook, types::Color::Black);
        self.board[1][7] = create_piece(types::PieceType::Knight, types::Color::Black);
        self.board[6][7] = create_piece(types::PieceType::Knight, types::Color::Black);
        self.board[2][7] = create_piece(types::PieceType::Bishop, types::Color::Black);
        self.board[5][7] = create_piece(types::PieceType::Bishop, types::Color::Black);
        self.board[3][7] = create_piece(types::PieceType::Queen, types::Color::Black);
        self.board[4][7] = create_piece(types::PieceType::King, types::Color::Black);
        for i in 0..8 {
            self.board[i][6] = create_piece(types::PieceType::Pawn, types::Color::Black);
        }
    }

    fn board_to_neural_input(&self) -> Vec<f32> {
        let mut input = Vec::with_capacity(64);
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

    fn validate_board_state(&self) -> Result<(), String> {
        let mut white_kings = 0;
        let mut black_kings = 0;
        for rank in 0..8 {
            for file in 0..8 {
                if let Some(piece) = self.board[file][rank] {
                    if piece.piece_type == types::PieceType::King {
                        match piece.color {
                            types::Color::White => white_kings += 1,
                            types::Color::Black => black_kings += 1,
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

    fn evaluate_position_neural(&self) -> i32 {
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

    fn is_valid_move(&self, mv: &types::Move) -> Result<(), MoveError> {
        self.is_valid_basic_move(mv)?;
        
        // Create a test board to check if move would cause check
        let mut test_state = self.clone();
        test_state.make_move_without_validation(mv);
        if test_state.is_in_check(mv.piece_moved.color) {
            return Err(MoveError::WouldCauseCheck);
        }
        
        Ok(())
    }

    fn is_valid_basic_move(&self, mv: &types::Move) -> Result<(), MoveError> {
        if !Self::is_within_bounds(mv.from) || !Self::is_within_bounds(mv.to) {
            return Err(MoveError::OutOfBounds);
        }

        let piece = match self.get_piece_at(mv.from) {
            Some(p) => p,
            None => return Err(MoveError::NoPieceAtSource),
        };

        if let Some(dest) = self.get_piece_at(mv.to) {
            if dest.color == piece.color {
                return Err(MoveError::InvalidPieceMove("Cannot capture your own piece".to_string()));
            }
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

    fn is_valid_pawn_move(&self, mv: &types::Move, color: types::Color) -> Result<(), MoveError> {
        let (fx, fy) = mv.from;
        let (tx, ty) = mv.to;
        let direction = match color {
            types::Color::White => 1,
            types::Color::Black => -1,
        };
        let start_rank = match color {
            types::Color::White => 1,
            types::Color::Black => 6,
        };
        let f_one = ((fy as i32) + direction) as usize;
        let f_two = ((fy as i32) + 2*direction) as usize;
    
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

    fn is_valid_knight_move(&self, mv: &types::Move) -> Result<(), MoveError> {
        let dx = (mv.to.0 as i32 - mv.from.0 as i32).abs();
        let dy = (mv.to.1 as i32 - mv.from.1 as i32).abs();
        if (dx == 2 && dy == 1) || (dx == 1 && dy == 2) {
            Ok(())
        } else {
            Err(MoveError::InvalidPieceMove("Invalid knight move".to_string()))
        }
    }
    
    fn is_valid_bishop_move(&self, mv: &types::Move) -> Result<(), MoveError> {
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
    
    fn is_valid_rook_move(&self, mv: &types::Move) -> Result<(), MoveError> {
        let dx = mv.to.0 as i32 - mv.from.0 as i32;
        let dy = mv.to.1 as i32 - mv.from.1 as i32;
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
    
    fn is_valid_queen_move(&self, mv: &types::Move) -> Result<(), MoveError> {
        if let Ok(()) = self.is_valid_bishop_move(mv) {
            Ok(())
        } else if let Ok(()) = self.is_valid_rook_move(mv) {
            Ok(())
        } else {
            Err(MoveError::InvalidPieceMove("Invalid queen move".to_string()))
        }
    }

    fn is_valid_king_move(&self, mv: &types::Move) -> Result<(), MoveError> {
        let dx = (mv.to.0 as i32 - mv.from.0 as i32).abs();
        let dy = (mv.to.1 as i32 - mv.from.1 as i32).abs();

        // Normal king move
        if dx <= 1 && dy <= 1 {
            return Ok(());
        }

        // Castling attempt
        if mv.is_castling && dy == 0 && dx == 2 {
            let (orig_pos, can_castle_kingside, can_castle_queenside) = match mv.piece_moved.color {
                types::Color::White => ((4, 0), self.white_can_castle_kingside, self.white_can_castle_queenside),
                types::Color::Black => ((4, 7), self.black_can_castle_kingside, self.black_can_castle_queenside),
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
                types::Color::White => {
                    if mv.to == (6, 0) { // Kingside
                        if !can_castle_kingside { 
                            return Err(MoveError::CastlingRightsLost);
                        }
                        if !self.is_path_clear((4,0), (7,0)) { 
                            return Err(MoveError::CastlingPathBlocked);
                        }
                        if self.board[7][0] != Some(types::Piece { 
                            piece_type: types::PieceType::Rook, 
                            color: types::Color::White 
                        }) { 
                            return Err(MoveError::CastlingRookMissing);
                        }
                        for x in 4..=6 {
                            if self.is_square_attacked((x,0), types::Color::Black) { 
                                return Err(MoveError::CastlingThroughCheck);
                            }
                        }
                    } else if mv.to == (2, 0) { // Queenside
                        if !can_castle_queenside { 
                            return Err(MoveError::CastlingRightsLost);
                        }
                        if !self.is_path_clear((4,0), (0,0)) { 
                            return Err(MoveError::CastlingPathBlocked);
                        }
                        if self.board[0][0] != Some(types::Piece { 
                            piece_type: types::PieceType::Rook, 
                            color: types::Color::White 
                        }) { 
                            return Err(MoveError::CastlingRookMissing);
                        }
                        for x in 2..=4 {
                            if self.is_square_attacked((x,0), types::Color::Black) { 
                                return Err(MoveError::CastlingThroughCheck);
                            }
                        }
                    }
                },
                types::Color::Black => {
                    if mv.to == (6, 7) { // Kingside - moves to g8
                        if !can_castle_kingside { 
                            return Err(MoveError::CastlingRightsLost);
                        }
                        if !self.is_path_clear((4,7), (7,7)) { // check path from e8 to h8
                            return Err(MoveError::CastlingPathBlocked);
                        }
                        if self.board[7][7] != Some(types::Piece { // check h8 rook
                            piece_type: types::PieceType::Rook, 
                            color: types::Color::Black 
                        }) { 
                            return Err(MoveError::CastlingRookMissing);
                        }
                        for x in 4..=6 {
                            if self.is_square_attacked((x,7), types::Color::White) { // check e8,f8,g8
                                return Err(MoveError::CastlingThroughCheck);
                            }
                        }
                    } else if mv.to == (2, 7) { // Queenside - moves to c8
                        if !can_castle_queenside { 
                            return Err(MoveError::CastlingRightsLost);
                        }
                        if !self.is_path_clear((4,7), (0,7)) { // check path from e8 to a8
                            return Err(MoveError::CastlingPathBlocked);
                        }
                        if self.board[0][7] != Some(types::Piece { // check a8 rook
                            piece_type: types::PieceType::Rook, 
                            color: types::Color::Black 
                        }) { 
                            return Err(MoveError::CastlingRookMissing);
                        }
                        for x in 2..=4 {
                            if self.is_square_attacked((x,7), types::Color::White) { // check c8,d8,e8
                                return Err(MoveError::CastlingThroughCheck);
                            }
                        }
                    }
                },
            }
            return Ok(());
        }
        Err(MoveError::InvalidPieceMove("Invalid king move".to_string()))
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
        let king_pos = if color == types::Color::White {
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


// ----------  Implement Board Display ----------
// Add these to your imports at the top
use colored::*;

// Update the Display implementation for GameState
impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print file labels (a-h) at top
        write!(f, "  ")?;
        for file in 0..8 {
            write!(f, " {} ", ((file as u8 + b'a') as char).to_string().cyan())?;
        }
        writeln!(f)?;

        // Print top border
        writeln!(f, "  {}", "─".repeat(24).bright_magenta())?;

        for rank in (0..8).rev() {
            // Print rank number on left side
            write!(f, "{} {}",
                (rank + 1).to_string().cyan(),
                "│".bright_magenta()
            )?;

            // Print board squares and pieces
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

                // Color the pieces and dots
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

            // Print rank number on right side
            writeln!(f, "{} {}",
                "│".bright_magenta(),
                (rank + 1).to_string().cyan()
            )?;
        }

        // Print bottom border
        writeln!(f, "  {}", "─".repeat(24).bright_magenta())?;

        // Print file labels (a-h) at bottom
        write!(f, "  ")?;
        for file in 0..8 {
            write!(f, " {} ", ((file as u8 + b'a') as char).to_string().cyan())?;
        }
        writeln!(f)?;

        Ok(())
    }
}


// ---------- MAIN FUNCTION ----------
fn main() {
    println!("Neural Chess Engine v2.0");
    println!("Commands:");
    println!("  'play' - Start a new game");
    println!("  'train time XX' - Train for XX minutes");
    println!("  'train games XX' - Train for XX games");
    println!("  'quit' - Exit the program");

    let mut engine = ChessNeuralEngine::new();
    
    // Try to load existing weights
    if let Err(e) = engine.load_weights("neural_weights.json") {
        println!("No existing weights found, starting fresh: {}", e);
    } else {
        println!("Successfully loaded existing neural weights");
    }

    loop {
        println!("\nEnter command:");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        match input {
            "play" => play_game(),
            cmd if cmd.starts_with("train time ") => {
                if let Ok(minutes) = cmd.trim_start_matches("train time ").parse::<u64>() {
                    println!("Starting training for {} minutes...", minutes);
                    match engine.train_self_play(Some(Duration::from_secs(minutes * 60)), None) {
                        Ok(_) => println!("Training completed successfully"),
                        Err(e) => println!("Training error: {}", e),
                    }
                } else {
                    println!("Invalid time format. Use 'train time XX' where XX is minutes");
                }
            },
            cmd if cmd.starts_with("train games ") => {
                if let Ok(games) = cmd.trim_start_matches("train games ").parse::<usize>() {
                    println!("Starting training for {} games...", games);
                    match engine.train_self_play(None, Some(games)) {
                        Ok(_) => println!("Training completed successfully"),
                        Err(e) => println!("Training error: {}", e),
                    }
                } else {
                    println!("Invalid games format. Use 'train games XX' where XX is number of games");
                }
            },
            "quit" => break,
            "" => continue, // Handle empty input gracefully
            _ => println!("Unknown command"),
        }
    }
    println!("Goodbye!");
}

fn play_game() {
    let mut game_state = GameState::new();
    loop {
        println!("{}", game_state);
        println!("{}", game_state.get_game_status());

        if game_state.current_turn == Color::White {
            println!("Your move: ");
            let mut input = String::new();
            std::io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();
            if input == "quit" {
                break;
            }
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
