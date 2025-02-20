//////////////////////////
// neural.rs
//////////////////////////

use rand::Rng;
use std::time::Instant;
use std::sync::{Arc, Mutex};
use std::fs::File;
use std::io::{Write, Read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use ctrlc;

use crate::GameState;
use crate::types::{
    NetworkWeights,
    TrainingStats,
    Move,
    Color,
    Piece,
    PieceType,
};

// A helper to reshape a 64-element vector into an 8x8 matrix
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
    update_rules: Vec<Box<dyn Fn(&[f32]) -> f32 + Send + Sync>>,
}

impl CAInterface {
    fn new(width: usize, height: usize) -> Self {
        let mut rng = rand::thread_rng();
        CAInterface {
            width,
            height,
            cells: (0..height)
                .map(|_| (0..width).map(|_| rng.gen::<f32>()).collect())
                .collect(),
            update_rules: vec![
                Box::new(|neighbors: &[f32]| {
                    let sum: f32 = neighbors.iter().sum();
                    if sum > 2.0 && sum < 3.5 {
                        1.0
                    } else {
                        0.0
                    }
                }),
                Box::new(|neighbors: &[f32]| {
                    neighbors.iter().sum::<f32>() / neighbors.len() as f32
                }),
                Box::new(|neighbors: &[f32]| {
                    let center = neighbors[neighbors.len() / 2];
                    let avg = neighbors.iter().sum::<f32>() / neighbors.len() as f32;
                    (center + avg) / 2.0
                }),
            ],
        }
    }

    fn update(&mut self) {
        let mut new_cells = self.cells.clone();
        let rule_idx = rand::thread_rng().gen_range(0..self.update_rules.len());
        for i in 1..(self.height - 1) {
            for j in 1..(self.width - 1) {
                let neighbors = vec![
                    self.cells[i - 1][j - 1],
                    self.cells[i - 1][j],
                    self.cells[i - 1][j + 1],
                    self.cells[i][j - 1],
                    self.cells[i][j],
                    self.cells[i][j + 1],
                    self.cells[i + 1][j - 1],
                    self.cells[i + 1][j],
                    self.cells[i + 1][j + 1],
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

// ---------- Activation Functions ----------
fn tanh(x: f32) -> f32 {
    x.tanh()
}
fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------- RNN ----------
struct RNN {
    hidden_state: Vec<f32>,
    weights: Vec<Vec<f32>>,
    discriminator_weights: Vec<Vec<f32>>,
}

impl RNN {
    fn new(input_dim: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..hidden_size)
            .map(|_| {
                (0..(input_dim + hidden_size))
                    .map(|_| rng.gen_range(-0.1..0.1))
                    .collect()
            })
            .collect();
        let discriminator_weights = (0..hidden_size)
            .map(|_| {
                (0..hidden_size)
                    .map(|_| rng.gen_range(-0.1..0.1))
                    .collect()
            })
            .collect();

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
            // inputs
            for (j, &inp) in input.iter().enumerate() {
                sum += inp * self.weights[i][j];
            }
            // hidden state
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

// ---------- CNN ----------
struct CNN {
    filters: Vec<Vec<Vec<f32>>>,
    feature_maps: Vec<Vec<f32>>,
    discriminator_filters: Vec<Vec<Vec<f32>>>,
}

impl CNN {
    fn new(num_filters: usize, input_rows: usize, input_cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        let filters = (0..num_filters)
            .map(|_| {
                (0..3)
                    .map(|_| (0..3).map(|_| rng.gen_range(-0.1..0.1)).collect())
                    .collect()
            })
            .collect();
        let feature_maps = vec![vec![0.0; input_rows * input_cols]; num_filters];
        let discriminator_filters = (0..num_filters)
            .map(|_| {
                (0..3)
                    .map(|_| (0..3).map(|_| rng.gen_range(-0.1..0.1)).collect())
                    .collect()
            })
            .collect();

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
                            let row_idx = i + di - 1;
                            let col_idx = j + dj - 1;
                            if row_idx < rows && col_idx < cols {
                                sum += input[row_idx][col_idx] * filter[di][dj];
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
        // Summation approach: partial "convolution" with the discriminator filters
        for (f_idx, map) in feature_maps.iter().enumerate() {
            if f_idx >= self.discriminator_filters.len() {
                break;
            }
            let rows = 3.min((map.len() as f32).sqrt() as usize);
            let cols = rows;
            for i in 0..rows {
                for j in 0..cols {
                    let idx = i * cols + j;
                    if idx < map.len() {
                        let filter_val = self.discriminator_filters[f_idx][i.min(2)][j.min(2)];
                        sum += map[idx] * filter_val;
                    }
                }
            }
        }
        sigmoid(sum)
    }
}

// ---------- LSTM ----------
struct LSTM {
    cell_state: Vec<f32>,
    hidden_state: Vec<f32>,
    weights: Vec<Vec<f32>>,
    discriminator_weights: Vec<Vec<f32>>,
}

impl LSTM {
    fn new(input_dim: usize, hidden_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..hidden_size)
            .map(|_| {
                (0..(input_dim + hidden_size))
                    .map(|_| rng.gen_range(-0.1..0.1))
                    .collect()
            })
            .collect();
        let discriminator_weights = (0..hidden_size)
            .map(|_| {
                (0..hidden_size)
                    .map(|_| rng.gen_range(-0.1..0.1))
                    .collect()
            })
            .collect();

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

// ---------- GANTrainer ----------
struct GANTrainer {
    learning_rate: f32,
    batch_size: usize,
    noise_dim: usize,
}

impl GANTrainer {
    fn new(learning_rate: f32, batch_size: usize, noise_dim: usize) -> Self {
        GANTrainer {
            learning_rate,
            batch_size,
            noise_dim,
        }
    }

    fn generate_noise(&self) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        (0..self.noise_dim)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect()
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
            let cnn_out = cnn.forward(&[rnn_out.clone()]);
            let lstm_out = lstm.forward(&rnn_out);

            for real_sample in real_data.iter() {
                let rnn_real_score = rnn.discriminate(real_sample);
                let cnn_real_score = cnn.discriminate(&[real_sample.clone()]);
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
            let cnn_out = cnn.forward(&[rnn_out.clone()]);
            let lstm_out = lstm.forward(&rnn_out);

            let rnn_fake_score = rnn.discriminate(&rnn_out);
            let cnn_fake_score = cnn.discriminate(&cnn_out);
            let lstm_fake_score = lstm.discriminate(&lstm_out);

            self.update_generator_weights(rnn, &noise, rnn_fake_score);
            self.update_generator_weights_cnn(cnn, &noise, cnn_fake_score);
            self.update_generator_weights_lstm(lstm, &noise, lstm_fake_score);
        }
    }

    fn update_discriminator_weights(
        &self,
        network: &mut RNN,
        real: &[f32],
        fake: &[f32],
        real_score: f32,
    ) {
        for i in 0..network.discriminator_weights.len() {
            for j in 0..network.discriminator_weights[i].len() {
                let real_grad = real_score * (1.0 - real_score) * real[j % real.len()];
                let fake_grad = -fake[j % fake.len()] * (1.0 - fake[j % fake.len()]);
                network.discriminator_weights[i][j] += self.learning_rate * (real_grad + fake_grad);
            }
        }
    }

    fn update_discriminator_weights_cnn(
        &self,
        network: &mut CNN,
        _real: &[f32],
        fake: &[f32],
        real_score: f32,
    ) {
        for f in 0..network.discriminator_filters.len() {
            for i in 0..3 {
                for j in 0..3 {
                    let real_grad = real_score * (1.0 - real_score);
                    let fake_grad = -fake[0] * (1.0 - fake[0]);
                    network.discriminator_filters[f][i][j] +=
                        self.learning_rate * (real_grad + fake_grad);
                }
            }
        }
    }

    fn update_discriminator_weights_lstm(
        &self,
        network: &mut LSTM,
        real: &[f32],
        fake: &[f32],
        real_score: f32,
    ) {
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

// ---------- **ChessNeural** ENGINE ----------
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
        let input_dim = 64; // 8x8 board
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
                        PieceType::King => 0.0, // handle king separately
                    };
                    if piece.color == Color::White {
                        score += value;
                    } else {
                        score -= value;
                    }
                }
            }
        }

        // Positional factors
        let safety_bonus = 50.0;
        let check_penalty = 100.0;

        let king_pos = if game_state.current_turn == Color::White {
            game_state.white_king_pos
        } else {
            game_state.black_king_pos
        };

        // Penalize if king is attacked
        if game_state.is_square_attacked(king_pos, game_state.current_turn.opposite()) {
            score -= safety_bonus;
        }
        // Extra penalty if in check
        if game_state.is_in_check(game_state.current_turn) {
            score -= check_penalty;
        }
        // Flip if it's Black's turn
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

        // Generate & evaluate moves
        for file_from in 0..8 {
            for rank_from in 0..8 {
                if let Some(piece) = game_state.get_piece_at((file_from, rank_from)) {
                    if piece.color == game_state.current_turn {
                        let possible_moves =
                            self.generate_piece_moves(*piece, (file_from, rank_from), game_state);

                        for (file_to, rank_to) in possible_moves {
                            let test_move = Move {
                                from: (file_from, rank_from),
                                to: (file_to, rank_to),
                                piece_moved: *piece,
                                piece_captured: game_state.get_piece_at((file_to, rank_to)).cloned(),
                                is_castling: piece.piece_type == PieceType::King
                                    && (file_to as i32 - file_from as i32).abs() == 2,
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
            println!(
                "AI chose move from {:?} to {:?} with evaluation {}",
                mv.from, mv.to, best_eval
            );
        }

        best_move
    }

    fn generate_piece_moves(
        &self,
        piece: Piece,
        from: (usize, usize),
        game_state: &GameState,
    ) -> Vec<(usize, usize)> {
        let mut moves = Vec::new();
        let (x, y) = from;

        match piece.piece_type {
            PieceType::Pawn => {
                let direction = if piece.color == Color::White {
                    -1i32
                } else {
                    1i32
                };
                let start_rank = if piece.color == Color::White { 6 } else { 1 };

                // Single step
                if let Some(new_y) = (y as i32 + direction).try_into().ok() {
                    if new_y < 8 {
                        moves.push((x, new_y));
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
                    let nx = x as i32 + dx;
                    let ny = y as i32 + direction;
                    if nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
                        moves.push((nx as usize, ny as usize));
                    }
                }
            }
            PieceType::Knight => {
                for (dx, dy) in [
                    (-2, -1),
                    (-2, 1),
                    (-1, -2),
                    (-1, 2),
                    (1, -2),
                    (1, 2),
                    (2, -1),
                    (2, 1),
                ] {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
                        moves.push((nx as usize, ny as usize));
                    }
                }
            }
            PieceType::Bishop => {
                for (dx, dy) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                    let mut dist = 1;
                    loop {
                        let nx = x as i32 + dx * dist;
                        let ny = y as i32 + dy * dist;
                        if nx < 0 || ny < 0 || nx >= 8 || ny >= 8 {
                            break;
                        }
                        moves.push((nx as usize, ny as usize));
                        if game_state.get_piece_at((nx as usize, ny as usize)).is_some() {
                            break;
                        }
                        dist += 1;
                    }
                }
            }
            PieceType::Rook => {
                for (dx, dy) in [(0, -1), (0, 1), (-1, 0), (1, 0)] {
                    let mut dist = 1;
                    loop {
                        let nx = x as i32 + dx * dist;
                        let ny = y as i32 + dy * dist;
                        if nx < 0 || ny < 0 || nx >= 8 || ny >= 8 {
                            break;
                        }
                        moves.push((nx as usize, ny as usize));
                        if game_state.get_piece_at((nx as usize, ny as usize)).is_some() {
                            break;
                        }
                        dist += 1;
                    }
                }
            }
            PieceType::Queen => {
                // Rook-like
                for (dx, dy) in [(0, -1), (0, 1), (-1, 0), (1, 0)] {
                    let mut dist = 1;
                    loop {
                        let nx = x as i32 + dx * dist;
                        let ny = y as i32 + dy * dist;
                        if nx < 0 || ny < 0 || nx >= 8 || ny >= 8 {
                            break;
                        }
                        moves.push((nx as usize, ny as usize));
                        if game_state.get_piece_at((nx as usize, ny as usize)).is_some() {
                            break;
                        }
                        dist += 1;
                    }
                }
                // Bishop-like
                for (dx, dy) in [(-1, -1), (-1, 1), (1, -1), (1, 1)] {
                    let mut dist = 1;
                    loop {
                        let nx = x as i32 + dx * dist;
                        let ny = y as i32 + dy * dist;
                        if nx < 0 || ny < 0 || nx >= 8 || ny >= 8 {
                            break;
                        }
                        moves.push((nx as usize, ny as usize));
                        if game_state.get_piece_at((nx as usize, ny as usize)).is_some() {
                            break;
                        }
                        dist += 1;
                    }
                }
            }
            PieceType::King => {
                for (dx, dy) in [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ] {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < 8 && ny >= 0 && ny < 8 {
                        moves.push((nx as usize, ny as usize));
                    }
                }
                // Possible castling squares
                // White: e1->g1 or e1->c1
                // Black: e8->g8 or e8->c8
                if piece.color == Color::White && y == 0 && x == 4 {
                    // kingside
                    moves.push((6, 0));
                    // queenside
                    moves.push((2, 0));
                } else if piece.color == Color::Black && y == 7 && x == 4 {
                    // kingside
                    moves.push((6, 7));
                    // queenside
                    moves.push((2, 7));
                }
            }
        }
        moves
    }

    /// Train self-play stub
    pub fn train_self_play(
        &mut self,
        time_limit: Option<Duration>,
        game_limit: Option<usize>,
    ) -> Result<(), String> {
        // Basic stub -- originally you'd run training loops, self-play, gather stats, etc.
        let start = Instant::now();
        let mut games_played = 0;

        ctrlc::set_handler(|| {
            println!("Ctrl+C caught! Attempting graceful shutdown...");
        })
        .map_err(|e| e.to_string())?;

        loop {
            // We do nothing fancy here except check time or game count
            if let Some(limit) = time_limit {
                if start.elapsed() >= limit {
                    println!("Reached time limit for training.");
                    break;
                }
            }
            if let Some(limit) = game_limit {
                if games_played >= limit {
                    println!("Reached game limit for training.");
                    break;
                }
            }
            // (Pretend we do self-play training here)
            games_played += 1;
            if games_played % 10 == 0 {
                println!("Trained on {} self-play games so far...", games_played);
            }

            // Sleep a little to mimic training time & avoid hogging CPU
            // std::thread::sleep(Duration::from_millis(50));
        }
        println!("Finished training. Games: {}", games_played);
        Ok(())
    }
}
