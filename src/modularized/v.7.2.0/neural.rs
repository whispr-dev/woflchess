//////////////////////////
// neural.rs
//////////////////////////

#[allow(dead_code)]
use rand::Rng;
use std::time::Instant;
use std::sync::{Arc, Mutex};
use std::fs::File;
use std::io::{Write, Read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use ctrlc;

// Import GameState from crate root
use crate::GameState;

// Import other types from types module
use crate::types::{
    NetworkWeights, 
    TrainingStats, 
    Move, 
    Color,
    Piece,
    PieceType
};

// A helper to reshape a 64-element vector into 8x8
pub fn reshape_vector_to_matrix(vec: &[f32], rows: usize, cols: usize) -> Vec<Vec<f32>> {
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

// ---------- CA Interface ----------
struct CAInterface {
    width: usize,
    height: usize,
    cells: Vec<Vec<f32>>,
    update_rules: Vec<Box<dyn Fn(&[f32]) -> f32 + Send + Sync>>,  // Add Send + Sync bounds
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
                new_cells[i][j] = (self.update_rules[rule_idx])(&neighbors);
            }
        }
        self.cells = new_cells;
    }
}

// ---------- NeuralClock ----------
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

// Activation fns
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
        let nf = self.filters.len();

        let mut new_feature_maps = vec![vec![0.0; rows * cols]; nf];
        for (f, filter) in self.filters.iter().enumerate() {
            for i in 1..rows.saturating_sub(1) {
                for j in 1..cols.saturating_sub(1) {
                    let mut sum = 0.0;
                    for di in 0..3 {
                        for dj in 0..3 {
                            if i + di > 0 && j + dj > 0
                                && i + di - 1 < rows && j + dj - 1 < cols
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
        let hs = self.hidden_state.len();
        let mut new_hidden = vec![0.0; hs];
        let mut new_cell = vec![0.0; hs];

        for i in 0..hs {
            let mut forget_sum = 0.0;
            let mut input_sum = 0.0;
            let mut output_sum = 0.0;

            for (j, &val) in input.iter().enumerate() {
                if j < self.weights[i].len() {
                    forget_sum += val * self.weights[i][j];
                    input_sum += val * self.weights[i][j];
                    output_sum += val * self.weights[i][j];
                }
            }

            for j in 0..hs {
                let widx = j + input.len();
                if widx < self.weights[i].len() {
                    forget_sum += self.hidden_state[j] * self.weights[i][widx];
                    input_sum += self.hidden_state[j] * self.weights[i][widx];
                    output_sum += self.hidden_state[j] * self.weights[i][widx];
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
        let hs = self.hidden_state.len();
        let mut sum = 0.0;
        for i in 0..hs {
            for j in 0..hs {
                if j < state.len() && j < self.discriminator_weights[i].len() {
                    sum += state[j] * self.discriminator_weights[i][j];
                }
            }
        }
        sigmoid(sum)
    }
}

// GANTrainer
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
                let cnn_real_score = cnn.discriminate(&vec![real_sample.clone()]);
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
            for j in 0..network.discriminator_weights[i].len() {
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
            for j in 0..network.discriminator_weights[i].len() {
                let real_grad = real_score * (1.0 - real_score) * real[j % real.len()];
                let fake_grad = -fake[j % fake.len()] * (1.0 - fake[j % fake.len()]);
                network.discriminator_weights[i][j] += self.learning_rate * (real_grad + fake_grad);
            }
        }
    }

    fn update_generator_weights(&self, network: &mut RNN, noise: &[f32], fake_score: f32) {
        for i in 0..network.weights.len() {
            for j in 0..network.weights[i].len() {
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
            for j in 0..network.weights[i].len() {
                let grad = fake_score * (1.0 - fake_score) * noise[j % noise.len()];
                network.weights[i][j] += self.learning_rate * grad;
            }
        }
    }
}


// ---------- **ChessNeural**tm ENGINE ----------
pub struct ChessNeuralEngine {
    rnn: Arc<Mutex<RNN>>,
    cnn: Arc<Mutex<CNN>>,
    lstm: Arc<Mutex<LSTM>>,
    gan_trainer: GANTrainer,
    ca_interface: CAInterface,
    neural_clock: NeuralClock,
}

impl ChessNeuralEngine {
    pub fn new() -> Self {
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
        
    pub fn evaluate_position(&mut self, game_state: &GameState) -> f32 {
        let mut score = 0.0;
        
        // Material values
        let pawn_value = 100.0;
        let knight_value = 320.0;
        let bishop_value = 330.0;
        let rook_value = 500.0;
        let queen_value = 900.0;
        
        // Evaluate material
        for file in 0..8 {
            for rank in 0..8 {
                if let Some(piece) = game_state.board[file][rank] {
                    let value = match piece.piece_type {
                        PieceType::Pawn => pawn_value,
                        PieceType::Knight => knight_value,
                        PieceType::Bishop => bishop_value,
                        PieceType::Rook => rook_value,
                        PieceType::Queen => queen_value,
                        PieceType::King => 0.0, // King's value is handled in positional evaluation
                    };
                    score += if piece.color == Color::White { value } else { -value };
                }
            }
        }
        
        // Positional evaluation
        let safety_bonus = 50.0;
        let check_penalty = 100.0;
        
        // King safety
        let king_pos = if game_state.current_turn == Color::White {
            game_state.white_king_pos
        } else {
            game_state.black_king_pos
        };
        
        // Penalize vulnerable king positions
        if game_state.is_square_attacked(king_pos, game_state.current_turn.opposite()) {
            score -= safety_bonus;
        }
        
        // Heavy penalty for being in check
        if game_state.is_in_check(game_state.current_turn) {
            score -= check_penalty;
        }
        
        // Adjust score based on whose turn it is
        if game_state.current_turn == Color::Black {
            score = -score;
        }
        
        score
    }

    pub fn suggest_move(&mut self, game_state: &GameState) -> Option<Move> {
        println!("AI is analyzing position...");
        
        let mut legal_moves = Vec::new();
        let mut best_move = None;
        let mut best_eval = if game_state.current_turn == Color::White {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    
        // Generate and evaluate moves...
        for file_from in 0..8 {
            for rank_from in 0..8 {
                if let Some(piece) = game_state.get_piece_at((file_from, rank_from)) {
                    if piece.color == game_state.current_turn {
                        let possible_moves = self.generate_piece_moves(
                            *piece,
                            (file_from, rank_from),
                            game_state
                        );
    
                        for (file_to, rank_to) in possible_moves {
                            let test_move = Move {
                                from: (file_from, rank_from),
                                to: (file_to, rank_to),
                                piece_moved: *piece,
                                piece_captured: game_state.get_piece_at((file_to, rank_to)).cloned(),
                                is_castling: piece.piece_type == PieceType::King && (file_to as i32 - file_from as i32).abs() == 2,
                                is_en_passant: false,
                                promotion: None,
                            };
    
                            if game_state.is_valid_move(&test_move).is_ok() {
                                legal_moves.push(test_move.clone());
                                
                                let mut test_state = game_state.clone();
                                test_state.make_move_without_validation(&test_move);
                                let eval = self.evaluate_position(&test_state);
    
                                let is_better = if game_state.current_turn == Color::White {
                                    eval > best_eval
                                } else {
                                    eval < best_eval
                                };
    
                                if is_better {
                                    best_eval = eval;
                                    best_move = Some(test_move);
                                }
                            }
                        }
                    }
                }
            }
        }
    
        println!("AI found {} legal moves", legal_moves.len());
        if let Some(mv) = &best_move {
            println!("AI chose move from {:?} to {:?} with evaluation {}", 
                     mv.from, mv.to, best_eval);
        }
    
        best_move
    }

    fn generate_piece_moves(&self, piece: Piece, from: (usize, usize), game_state: &GameState) 
        -> Vec<(usize, usize)> {
        let mut moves = Vec::new();
        let (x, y) = from;

        match piece.piece_type {
            PieceType::Pawn => {
                let direction = if piece.color == Color::White { -1i32 } else { 1i32 };
                let start_rank = if piece.color == Color::White { 6 } else { 1 };

                // Forward moves
                if let Some(new_y) = (y as i32 + direction).try_into().ok() {
                    if new_y < 8 {
                        moves.push((x, new_y)); // Single step
                        
                        // Double step from start
                        if y == start_rank {
                            if let Some(double_y) = (y as i32 + 2 * direction).try_into().ok() {
                                if double_y < 8 {
                                    moves.push((x, double_y));
                                }
                            }
                        }
                    }
                }

                // Captures
                for dx in [-1, 1] {
                    if let (Some(new_x), Some(new_y)) = (
                        (x as i32 + dx).try_into().ok(),
                        (y as i32 + direction).try_into().ok()
                    ) {
                        if new_x < 8 && new_y < 8 {
                            moves.push((new_x, new_y));
                        }
                    }
                }
            },
            PieceType::Knight => {
                for (dx, dy) in [
                    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
                    (1, -2), (1, 2), (2, -1), (2, 1)
                ] {
                    if let (Some(new_x), Some(new_y)) = (
                        (x as i32 + dx).try_into().ok(),
                        (y as i32 + dy).try_into().ok()
                    ) {
                        if new_x < 8 && new_y < 8 {
                            moves.push((new_x, new_y));
                        }
                    }
                }
            },
            PieceType::Bishop => {
                for (dx, dy) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                    let mut dist = 1;
                    while let (Some(new_x), Some(new_y)) = (
                        (x as i32 + dx * dist).try_into().ok(),
                        (y as i32 + dy * dist).try_into().ok()
                    ) {
                        if new_x >= 8 || new_y >= 8 { break; }
                        moves.push((new_x, new_y));
                        if game_state.get_piece_at((new_x, new_y)).is_some() { break; }
                        dist += 1;
                    }
                }
            },
            PieceType::Rook => {
                for (dx, dy) in [(0, -1), (0, 1), (-1, 0), (1, 0)] {
                    let mut dist = 1;
                    while let (Some(new_x), Some(new_y)) = (
                        (x as i32 + dx * dist).try_into().ok(),
                        (y as i32 + dy * dist).try_into().ok()
                    ) {
                        if new_x >= 8 || new_y >= 8 { break; }
                        moves.push((new_x, new_y));
                        if game_state.get_piece_at((new_x, new_y)).is_some() { break; }
                        dist += 1;
                    }
                }
            },
            PieceType::Queen => {
                for (dx, dy) in [
                    (-1, -1), (-1, 0), (-1, 1),
                    (0, -1), (0, 1),
                    (1, -1), (1, 0), (1, 1)
                ] {
                    let mut dist = 1;
                    while let (Some(new_x), Some(new_y)) = (
                        (x as i32 + dx * dist).try_into().ok(),
                        (y as i32 + dy * dist).try_into().ok()
                    ) {
                        if new_x >= 8 || new_y >= 8 { break; }
                        moves.push((new_x, new_y));
                        if game_state.get_piece_at((new_x, new_y)).is_some() { break; }
                        dist += 1;
                    }
                }
            },
            PieceType::King => {
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        if dx == 0 && dy == 0 { continue; }
                        if let (Some(new_x), Some(new_y)) = (
                            (x as i32 + dx).try_into().ok(),
                            (y as i32 + dy).try_into().ok()
                        ) {
                            if new_x < 8 && new_y < 8 {
                                moves.push((new_x, new_y));
                            }
                        }
                    }
                }
                // Add castling moves
                if piece.color == Color::White && y == 7 {
                    moves.push((2, 7)); // Queenside
                    moves.push((6, 7)); // Kingside
                } else if piece.color == Color::Black && y == 0 {
                    moves.push((2, 0)); // Queenside
                    moves.push((6, 0)); // Kingside
                }
            },
        }

        moves
    }

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
        // Create small random variations
        let mut training_data = Vec::new();
        for pos in positions {
            training_data.push(pos.clone());
            for _ in 0..3 {
                let mut variation = pos.clone();
                for value in variation.iter_mut() {
                    *value += rand::thread_rng().gen_range(-0.1..0.1);
                }
                training_data.push(variation);
            }
        }

        // Train
        self.gan_trainer.train_step(
            &mut self.rnn.lock().unwrap(),
            &mut self.cnn.lock().unwrap(),
            &mut self.lstm.lock().unwrap(),
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
            println!("\nReceived Ctrl+C, shutting down...");
            r.store(false, Ordering::SeqCst);
        }).map_err(|e| format!("Error setting Ctrl+C handler: {}", e))?;

        let start_time = Instant::now();
        let stats = TrainingStats::new();
        let mut game_count = 0;

        while running.load(Ordering::SeqCst) {
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
            game_count += 1;
            println!("Completed game {}", game_count);
        }

        println!("Training done. Stats:");
        stats.display();
        Ok(stats)
    }
}