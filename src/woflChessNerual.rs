// ---------- LOAD CRATES ----------
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use rand::Rng;
use std::fmt;

// ---------- CONSTANTS FOR BASIC CHESS ENGINE ----------
const PAWN_VALUE: i32 = 100;
const KNIGHT_VALUE: i32 = 320;
const BISHOP_VALUE: i32 = 330;
const ROOK_VALUE: i32 = 500;
const QUEEN_VALUE: i32 = 900;
const KING_VALUE: i32 = 20000;

// ---------- PIECE-SQUARE TABLES ----------
const PAWN_TABLE: [[i32; 8]; 8] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [50, 50, 50, 50, 50, 50, 50, 50],
    [10, 10, 20, 30, 30, 20, 10, 10],
    [5, 5, 10, 25, 25, 10, 5, 5],
    [0, 0, 0, 20, 20, 0, 0, 0],
    [5, -5, -10, 0, 0, -10, -5, 5],
    [5, 10, 10, -20, -20, 10, 10, 5],
    [0, 0, 0, 0, 0, 0, 0, 0]
];

const KNIGHT_TABLE: [[i32; 8]; 8] = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20, 0, 0, 0, 0, -20, -40],
    [-30, 0, 10, 15, 15, 10, 0, -30],
    [-30, 5, 15, 20, 20, 15, 5, -30],
    [-30, 0, 15, 20, 20, 15, 0, -30],
    [-30, 5, 10, 15, 15, 10, 5, -30],
    [-40, -20, 0, 5, 5, 0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]
];

const BISHOP_TABLE: [[i32; 8]; 8] = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10, 0, 0, 0, 0, 0, 0, -10],
    [-10, 0, 5, 10, 10, 5, 0, -10],
    [-10, 5, 5, 10, 10, 5, 5, -10],
    [-10, 0, 10, 10, 10, 10, 0, -10],
    [-10, 10, 10, 10, 10, 10, 10, -10],
    [-10, 5, 0, 0, 0, 0, 5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]
];

// ---------- BASIC CHESS TYPES ----------
#[derive(Clone, Copy, PartialEq, Debug)]
enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

#[derive(Clone, Copy, PartialEq, Debug)]
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
struct Move {
    from: (usize, usize),
    to: (usize, usize),
    piece_moved: Piece,
    piece_captured: Option<Piece>,
    is_castling: bool,
    is_en_passant: bool,
    promotion: Option<PieceType>,
}

// ---------- NEURAL NETWORK COMPONENTS ----------

// Activation functions
fn tanh(x: f32) -> f32 { x.tanh() }
fn relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// Cellular Automata Interface for pattern generation
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
            cells: (0..height)
                .map(|_| (0..width).map(|_| rng.gen::<f32>()).collect())
                .collect(),
            update_rules: vec![
                Box::new(|neighbors: &[f32]| {
                    let sum: f32 = neighbors.iter().sum();
                    if sum > 2.0 && sum < 3.5 { 1.0 } else { 0.0 }
                }),
                Box::new(|neighbors: &[f32]| neighbors.iter().sum::<f32>() / neighbors.len() as f32),
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
    fn get_state(&self) -> Vec<f32> {
        self.cells.iter().flat_map(|row| row.iter().cloned()).collect()
    }
}

// Neural Clock for timing operations
struct NeuralClock {
    cycle_duration: f32,
    last_tick: std::time::Instant,
}

impl NeuralClock {
    fn new(cycle_duration: f32) -> Self {
        NeuralClock { cycle_duration, last_tick: std::time::Instant::now() }
    }
    fn tick(&mut self) -> bool {
        let now = std::time::Instant::now();
        if now.duration_since(self.last_tick).as_secs_f32() >= self.cycle_duration {
            self.last_tick = now;
            true
        } else {
            false
        }
    }
}

// RNN for sequence learning
struct RNN {
    hidden_state: Vec<f32>,
    weights: Vec<Vec<f32>>,
    discriminator_weights: Vec<Vec<f32>>,
}

impl RNN {
    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; self.hidden_state.len()];
        for i in 0..self.hidden_state.len() {
            let mut sum = 0.0;
            for j in 0..input.len() { sum += input[j] * self.weights[i][j]; }
            for j in 0..self.hidden_state.len() { sum += self.hidden_state[j] * self.weights[i][j + input.len()]; }
            output[i] = tanh(sum);
        }
        self.hidden_state = output.clone();
        output
    }
    fn discriminate(&self, state: &[f32]) -> f32 {
        let mut sum = 0.0;
        for i in 0..state.len() {
            for j in 0..state.len() { sum += state[j] * self.discriminator_weights[i][j]; }
        }
        sigmoid(sum)
    }
}

// CNN for board pattern recognition
struct CNN {
    filters: Vec<Vec<Vec<f32>>>,
    feature_maps: Vec<Vec<f32>>,
    discriminator_filters: Vec<Vec<Vec<f32>>>,
}

impl CNN {
    fn forward(&mut self, input: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut new_feature_maps = vec![vec![0.0; self.feature_maps[0].len()]; self.feature_maps.len()];
        for (f, filter) in self.filters.iter().enumerate() {
            for i in 1..input.len()-1 {
                for j in 1..input[0].len()-1 {
                    let mut sum = 0.0;
                    for di in 0..3 {
                        for dj in 0..3 {
                            sum += input[i+di-1][j+dj-1] * filter[di][dj];
                        }
                    }
                    new_feature_maps[f][i*input[0].len() + j] = relu(sum);
                }
            }
        }
        self.feature_maps = new_feature_maps.clone();
        new_feature_maps
    }
    fn discriminate(&self, feature_maps: &[Vec<f32>]) -> f32 {
        let mut sum = 0.0;
        for (f, map) in feature_maps.iter().enumerate() {
            for i in 1..map.len()-1 {
                for di in 0..3 {
                    for dj in 0..3 { sum += map[i] * self.discriminator_filters[f][di][dj]; }
                }
            }
        }
        sigmoid(sum)
    }
}

// LSTM for long-term planning
struct LSTM {
    cell_state: Vec<f32>,
    hidden_state: Vec<f32>,
    weights: Vec<Vec<f32>>,
    discriminator_weights: Vec<Vec<f32>>,
}

impl LSTM {
    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let mut new_hidden = vec![0.0; self.hidden_state.len()];
        let mut new_cell = vec![0.0; self.cell_state.len()];
        for i in 0..self.hidden_state.len() {
            let mut forget_sum = 0.0;
            for j in 0..input.len() { forget_sum += input[j] * self.weights[i][j]; }
            let forget_gate = sigmoid(forget_sum);
            let mut input_sum = 0.0;
            for j in 0..input.len() { input_sum += input[j] * self.weights[i][j + input.len()]; }
            let input_gate = sigmoid(input_sum);
            new_cell[i] = forget_gate * self.cell_state[i] + input_gate * tanh(input_sum);
            let mut output_sum = 0.0;
            for j in 0..input.len() { output_sum += input[j] * self.weights[i][j + 2 * input.len()]; }
            let output_gate = sigmoid(output_sum);
            new_hidden[i] = output_gate * tanh(new_cell[i]);
        }
        self.hidden_state = new_hidden.clone();
        self.cell_state = new_cell;
        new_hidden
    }
    fn discriminate(&self, state: &[f32]) -> f32 {
        let mut sum = 0.0;
        for i in 0..state.len() {
            for j in 0..state.len() { sum += state[j] * self.discriminator_weights[i][j]; }
        }
        sigmoid(sum)
    }
}

// GAN Training Coordinator (without the unnecessary historical position code)
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
        (0..self.noise_dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect()
    }
    fn train_step(&self, rnn: &mut RNN, cnn: &mut CNN, lstm: &mut LSTM, real_data: &[Vec<f32>], ca_interface: &mut CAInterface) {
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
                self.update_discriminator_weights(lstm, real_sample, &lstm_out, lstm_real_score);
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
            self.update_generator_weights(lstm, &noise, lstm_fake_score);
        }
    }
    fn update_discriminator_weights(&self, network: &mut RNN, real: &[f32], fake: &[f32], real_score: f32) {
        for i in 0..network.discriminator_weights.len() {
            for j in 0..network.discriminator_weights[0].len() {
                let real_grad = real_score * (1.0 - real_score) * real[j];
                let fake_grad = -network.discriminate(fake) * (1.0 - network.discriminate(fake)) * fake[j];
                network.discriminator_weights[i][j] += self.learning_rate * (real_grad + fake_grad);
            }
        }
    }
    fn update_discriminator_weights_cnn(&self, network: &mut CNN, real: &[f32], fake: &[f32], real_score: f32) {
        for f in 0..network.discriminator_filters.len() {
            for i in 0..3 {
                for j in 0..3 {
                    let real_grad = real_score * (1.0 - real_score);
                    let fake_grad = -network.discriminate(&vec![fake.to_vec()]) * (1.0 - network.discriminate(&vec![fake.to_vec()]));
                    network.discriminator_filters[f][i][j] += self.learning_rate * (real_grad + fake_grad);
                }
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
    fn update_generator_weights_cnn(&self, network: &mut CNN, noise: &[f32], fake_score: f32) {
        for f in 0..network.filters.len() {
            for i in 0..3 {
                for j in 0..3 {
                    let grad = fake_score * (1.0 - fake_score);
                    network.filters[f][i][j] += self.learning_rate * grad;
                }
            }
        }
    }
}

// ---------- CHESS NEURAL ENGINE ----------
struct ChessNeuralEngine {
    rnn: Arc<Mutex<RNN>>,
    cnn: Arc<Mutex<CNN>>,
    lstm: Arc<Mutex<LSTM>>,
    ca_interface: Arc<Mutex<CAInterface>>,
    clock: Arc<Mutex<NeuralClock>>,
    gan_trainer: GANTrainer,
}

impl ChessNeuralEngine {
    fn new() -> Self {
        ChessNeuralEngine {
            rnn: Arc::new(Mutex::new(RNN {
                hidden_state: vec![0.0; 256],
                weights: vec![vec![0.0; 256]; 256],
                discriminator_weights: vec![vec![0.0; 256]; 256],
            })),
            cnn: Arc::new(Mutex::new(CNN {
                filters: vec![vec![vec![0.0; 3]; 3]; 64],
                feature_maps: vec![vec![0.0; 64]; 64],
                discriminator_filters: vec![vec![vec![0.0; 3]; 3]; 64],
            })),
            lstm: Arc::new(Mutex::new(LSTM {
                cell_state: vec![0.0; 256],
                hidden_state: vec![0.0; 256],
                weights: vec![vec![0.0; 256]; 256],
                discriminator_weights: vec![vec![0.0; 256]; 256],
            })),
            ca_interface: Arc::new(Mutex::new(CAInterface::new(8, 8))),
            clock: Arc::new(Mutex::new(NeuralClock::new(0.1))),
            gan_trainer: GANTrainer::new(0.001, 32, 256),
        }
    }
}

// ---------- CHESS GAME LOGIC ----------
#[derive(Clone)]
struct GameState {
    board: [[Option<Piece>; 8]; 8],
    current_turn: Color,
    move_history: Vec<Move>,
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
            current_turn: Color::White,
            move_history: vec![],
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
        let create_piece = |piece_type, color| Some(Piece { piece_type, color });
        // White pieces
        self.board[0][0] = create_piece(PieceType::Rook, Color::White);
        self.board[7][0] = create_piece(PieceType::Rook, Color::White);
        self.board[1][0] = create_piece(PieceType::Knight, Color::White);
        self.board[6][0] = create_piece(PieceType::Knight, Color::White);
        self.board[2][0] = create_piece(PieceType::Bishop, Color::White);
        self.board[5][0] = create_piece(PieceType::Bishop, Color::White);
        self.board[3][0] = create_piece(PieceType::Queen, Color::White);
        self.board[4][0] = create_piece(PieceType::King, Color::White);
        for i in 0..8 { self.board[i][1] = create_piece(PieceType::Pawn, Color::White); }
        // Black pieces
        self.board[0][7] = create_piece(PieceType::Rook, Color::Black);
        self.board[7][7] = create_piece(PieceType::Rook, Color::Black);
        self.board[1][7] = create_piece(PieceType::Knight, Color::Black);
        self.board[6][7] = create_piece(PieceType::Knight, Color::Black);
        self.board[2][7] = create_piece(PieceType::Bishop, Color::Black);
        self.board[5][7] = create_piece(PieceType::Bishop, Color::Black);
        self.board[3][7] = create_piece(PieceType::Queen, Color::Black);
        self.board[4][7] = create_piece(PieceType::King, Color::Black);
        for i in 0..8 { self.board[i][6] = create_piece(PieceType::Pawn, Color::Black); }
    }
    fn board_to_neural_input(&self) -> Vec<f32> {
        let mut input = Vec::with_capacity(64 * 6);
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
    fn evaluate_position_neural(&self) -> i32 {
        if let Some(engine) = &self.neural_engine {
            let mut engine = engine.lock().unwrap();
            let input = self.board_to_neural_input();
            let rnn_evaluation = { let mut rnn = engine.rnn.lock().unwrap(); rnn.forward(&input)[0] * 100.0 };
            let cnn_evaluation = { let mut cnn = engine.cnn.lock().unwrap(); cnn.forward(&vec![input.clone()])[0][0] * 100.0 };
            let lstm_evaluation = { let mut lstm = engine.lstm.lock().unwrap(); lstm.forward(&input)[0] * 100.0 };
            let traditional_eval = self.evaluate_position() as f32;
            ((traditional_eval * 0.4 + rnn_evaluation * 0.2 + cnn_evaluation * 0.2 + lstm_evaluation * 0.2) as i32)
        } else {
            self.evaluate_position()
        }
    }
    // Simplified traditional evaluation function
    fn evaluate_position(&self) -> i32 {
        let mut score = 0;
        for file in 0..8 {
            for rank in 0..8 {
                if let Some(piece) = self.board[file][rank] {
                    let multiplier = if piece.color == Color::White { 1 } else { -1 };
                    let piece_value = match piece.piece_type {
                        PieceType::Pawn => PAWN_VALUE,
                        PieceType::Knight => KNIGHT_VALUE,
                        PieceType::Bishop => BISHOP_VALUE,
                        PieceType::Rook => ROOK_VALUE,
                        PieceType::Queen => QUEEN_VALUE,
                        PieceType::King => KING_VALUE,
                    };
                    let position_bonus = match piece.piece_type {
                        PieceType::Pawn => PAWN_TABLE[rank][file],
                        PieceType::Knight => KNIGHT_TABLE[rank][file],
                        PieceType::Bishop => BISHOP_TABLE[rank][file],
                        _ => 0,
                    };
                    score += multiplier * (piece_value + position_bonus);
                }
            }
        }
        score
    }
    fn generate_moves_neural(&self) -> Vec<Move> {
        let mut moves = self.generate_legal_moves();
        if let Some(engine) = &self.neural_engine {
            let mut engine = engine.lock().unwrap();
            let input = self.board_to_neural_input();
            let mut move_scores: Vec<(Move, f32)> = moves.iter().map(|m| {
                let mut test_state = self.clone();
                test_state.make_move_without_validation(m);
                let position_after = test_state.board_to_neural_input();
                let rnn_score = engine.rnn.lock().unwrap().discriminate(&position_after);
                let cnn_score = engine.cnn.lock().unwrap().discriminate(&vec![position_after.clone()]);
                let lstm_score = engine.lstm.lock().unwrap().discriminate(&position_after);
                let combined_score = (rnn_score + cnn_score + lstm_score) / 3.0;
                (m.clone(), combined_score)
            }).collect();
            move_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            moves = move_scores.into_iter().map(|(m, _)| m).collect();
        }
        moves
    }
    fn make_computer_move(&mut self) -> Result<(), &'static str> {
        let mut moves = self.generate_moves_neural();
        if moves.is_empty() {
            if self.is_checkmate() { return Err("Checkmate!"); }
            else { return Err("Stalemate!"); }
        }
        let mut best_score = i32::MIN;
        let mut best_moves = vec![];
        let search_depth = 3;
        for m in moves {
            let mut test_state = self.clone();
            test_state.make_move_without_validation(&m);
            let score = -test_state.minimax_neural(search_depth - 1, i32::MIN, i32::MAX, true);
            if score > best_score {
                best_score = score;
                best_moves.clear();
                best_moves.push(m);
            } else if score == best_score {
                best_moves.push(m);
            }
        }
        let mut rng = rand::thread_rng();
        let selected_move = best_moves[rng.gen_range(0..best_moves.len())].clone();
        println!("Neural evaluation: {}", best_score);
        self.make_move(selected_move)?;
        println!("Computer plays: {}", self.move_to_algebraic(&selected_move));
        Ok(())
    }
    fn minimax_neural(&mut self, depth: i32, mut alpha: i32, mut beta: i32, maximizing: bool) -> i32 {
        if depth == 0 { return self.evaluate_position_neural(); }
        let moves = self.generate_moves_neural();
        if moves.is_empty() {
            if self.is_checkmate() { return if maximizing { -30000 } else { 30000 }; }
            return 0;
        }
        if maximizing {
            let mut max_eval = i32::MIN;
            for m in moves {
                let mut new_state = self.clone();
                new_state.make_move_without_validation(&m);
                let eval = new_state.minimax_neural(depth - 1, alpha, beta, false);
                max_eval = max_eval.max(eval);
                alpha = alpha.max(eval);
                if beta <= alpha { break; }
            }
            max_eval
        } else {
            let mut min_eval = i32::MAX;
            for m in moves {
                let mut new_state = self.clone();
                new_state.make_move_without_validation(&m);
                let eval = new_state.minimax_neural(depth - 1, alpha, beta, true);
                min_eval = min_eval.min(eval);
                beta = beta.min(eval);
                if beta <= alpha { break; }
            }
            min_eval
        }
    }
    // Placeholder stub – should return the legal moves available
    fn generate_legal_moves(&self) -> Vec<Move> {
        vec![]
    }
    fn is_checkmate(&self) -> bool {
        false
    }
    fn make_move(&mut self, m: Move) -> Result<(), &'static str> {
        if !self.is_valid_move(&m) { return Err("Invalid move"); }
        self.make_move_without_validation(&m);
        self.move_history.push(m);
        self.current_turn = match self.current_turn {
            Color::White => Color::Black,
            Color::Black => Color::White,
        };
        Ok(())
    }
    fn make_move_without_validation(&mut self, m: &Move) {
        if let Some(promotion) = m.promotion {
            let promoted_piece = Piece { piece_type: promotion, color: m.piece_moved.color };
            self.board[m.to.0][m.to.1] = Some(promoted_piece);
        } else {
            self.board[m.to.0][m.to.1] = Some(m.piece_moved);
        }
        self.board[m.from.0][m.from.1] = None;
        if m.is_castling {
            match (m.from, m.to) {
                ((4, 0), (6, 0)) => { self.board[5][0] = self.board[7][0]; self.board[7][0] = None; },
                ((4, 0), (2, 0)) => { self.board[3][0] = self.board[0][0]; self.board[0][0] = None; },
                ((4, 7), (6, 7)) => { self.board[5][7] = self.board[7][7]; self.board[7][7] = None; },
                ((4, 7), (2, 7)) => { self.board[3][7] = self.board[0][7]; self.board[0][7] = None; },
                _ => {}
            }
        }
        if m.piece_moved.piece_type == PieceType::Pawn {
            let dy = (m.to.1 as i32 - m.from.1 as i32).abs();
            if dy == 2 { self.last_pawn_double_move = Some(m.to); }
            else { self.last_pawn_double_move = None; }
            if m.is_en_passant {
                let capture_y = m.from.1;
                self.board[m.to.0][capture_y] = None;
            }
        } else {
            self.last_pawn_double_move = None;
        }
        if m.piece_moved.piece_type == PieceType::King {
            match m.piece_moved.color {
                Color::White => { self.white_king_pos = m.to; self.white_can_castle_kingside = false; self.white_can_castle_queenside = false; },
                Color::Black => { self.black_king_pos = m.to; self.black_can_castle_kingside = false; self.black_can_castle_queenside = false; },
            }
        }
        match (m.from, m.piece_moved.piece_type) {
            ((0, 0), PieceType::Rook) => self.white_can_castle_queenside = false,
            ((7, 0), PieceType::Rook) => self.white_can_castle_kingside = false,
            ((0, 7), PieceType::Rook) => self.black_can_castle_queenside = false,
            ((7, 7), PieceType::Rook) => self.black_can_castle_kingside = false,
            _ => {}
        }
    }
    fn move_to_algebraic(&self, m: &Move) -> String {
        if m.is_castling {
            if m.to.0 == 6 { return "O-O".to_string(); }
            else { return "O-O-O".to_string(); }
        }
        let piece_char = match m.piece_moved.piece_type {
            PieceType::Pawn => "",
            PieceType::Knight => "N",
            PieceType::Bishop => "B",
            PieceType::Rook => "R",
            PieceType::Queen => "Q",
            PieceType::King => "K",
        };
        let capture_char = if m.piece_captured.is_some() || m.is_en_passant { "x" } else { "" };
        let dest_file = (m.to.0 as u8 + b'a') as char;
        let dest_rank = (m.to.1 + 1).to_string();
        format!("{}{}{}{}", piece_char, capture_char, dest_file, dest_rank)
    }
    // Placeholder for move validation
    fn is_valid_move(&self, _m: &Move) -> bool { true }
    fn parse_square(&self, square: &str) -> Option<(usize, usize)> {
        if square.len() != 2 { return None; }
        let file = match square.chars().nth(0)? {
            'a'..='h' => (square.chars().nth(0)? as u8 - b'a') as usize,
            _ => return None,
        };
        let rank = match square.chars().nth(1)? {
            '1'..='8' => (square.chars().nth(1)? as u8 - b'1') as usize,
            _ => return None,
        };
        Some((file, rank))
    }
    fn parse_move_string(&self, move_str: &str) -> Result<((usize, usize), (usize, usize)), &'static str> {
        let parts: Vec<&str> = move_str.split_whitespace().collect();
        let (from_str, to_str) = match parts.len() {
            2 => (parts[0], parts[1]),
            3 if parts[1] == "to" => (parts[0], parts[2]),
            1 if move_str.len() == 4 => (&move_str[0..2], &move_str[2..4]),
            _ => return Err("Invalid move format"),
        };
        let from = self.parse_square(from_str).ok_or("Invalid origin square")?;
        let to = self.parse_square(to_str).ok_or("Invalid destination square")?;
        Ok((from, to))
    }
    fn make_move_from_str(&mut self, move_str: &str) -> Result<(), &'static str> {
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
        if let Some(pt) = promotion_char { promotion_type = Some(pt); }
        let (from, to) = self.parse_move_string(basic_move)?;
        let piece = self.get_piece_at(from).ok_or("No piece at starting square")?;
        if piece.color != self.current_turn { return Err("Not your turn!"); }
        if piece.piece_type == PieceType::Pawn && self.is_promotion_move(from, to) {
            promotion_type = Some(promotion_type.unwrap_or(PieceType::Queen));
        }
        let m = Move {
            from,
            to,
            piece_moved: piece,
            piece_captured: self.get_piece_at(to),
            is_castling: false,
            is_en_passant: false,
            promotion: promotion_type,
        };
        if !self.is_valid_move(&m) {
            return Err("Invalid move");
        }
        self.make_move(m)
    }
    fn get_piece_at(&self, pos: (usize, usize)) -> Option<Piece> {
        if pos.0 < 8 && pos.1 < 8 { self.board[pos.0][pos.1] } else { None }
    }
    fn is_promotion_move(&self, from: (usize, usize), to: (usize, usize)) -> bool {
        if let Some(piece) = self.get_piece_at(from) {
            if piece.piece_type == PieceType::Pawn {
                match piece.color {
                    Color::White => to.1 == 7,
                    Color::Black => to.1 == 0,
                }
            } else { false }
        } else { false }
    }
    fn get_game_status(&self) -> String {
        if self.is_checkmate() {
            format!("Checkmate! {} wins!", match self.current_turn { Color::White => "Black", Color::Black => "White" })
        } else if self.is_in_check(self.current_turn) {
            format!("{} is in check!", match self.current_turn { Color::White => "White", Color::Black => "Black" })
        } else {
            format!("{}'s turn", match self.current_turn { Color::White => "White", Color::Black => "Black" })
        }
    }
    fn is_in_check(&self, _color: Color) -> bool { false }
    fn display(&self) {
        println!("\n  a b c d e f g h");
        println!("  ---------------");
        for rank in (0..8).rev() {
            print!("{} ", rank + 1);
            for file in 0..8 {
                match self.board[file][rank] {
                    Some(piece) => {
                        let symbol = piece.piece_type.to_string();
                        if piece.color == Color::Black { print!("{} ", symbol.to_lowercase()); }
                        else { print!("{} ", symbol); }
                    },
                    None => print!(". "),
                }
            }
            println!("{}", rank + 1);
        }
        println!("  ---------------");
        println!("  a b c d e f g h");
        println!("\nCurrent turn: {:?}", self.current_turn);
    }
    fn train_network(&mut self) {
        if let Some(engine) = &self.neural_engine {
            let mut engine = engine.lock().unwrap();
            let game_history = vec![self.clone()];
            // Here you could call a training method on the engine if implemented
            // e.g., engine.train_on_game_history(game_history);
        }
    }
}

// ---------- DISPLAY IMPLEMENTATIONS ----------
impl fmt::Display for PieceType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
        writeln!(f, "\n  a b c d e f g h")?;
        writeln!(f, "  ---------------")?;
        for rank in (0..8).rev() {
            write!(f, "{} ", rank + 1)?;
            for file in 0..8 {
                match self.board[file][rank] {
                    Some(piece) => {
                        let symbol = piece.piece_type.to_string();
                        if piece.color == Color::Black { write!(f, "{} ", symbol.to_lowercase())?; }
                        else { write!(f, "{} ", symbol)?; }
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
    let mut game = GameState::new();
    println!("Welcome to Neural Chess!");
    println!("You play as White against the neural-enhanced computer.");
    println!("Enter moves in the format: 'e2e4', 'e2 e4', 'e4', or 'Nf3'");
    println!("Type 'quit' to exit\n");
    println!("Training neural network...");
    game.train_network();
    println!("Training complete!");
    game.display();
    loop {
        println!("\n{}", game.get_game_status());
        println!("Enter your move:");
        let mut input = String::new();
        if let Err(e) = std::io::stdin().read_line(&mut input) {
            println!("Error reading input: {}", e);
            continue;
        }
        let input = input.trim();
        if input == "quit" { break; }
        match game.make_move_from_str(input) {
            Ok(()) => {
                println!("Move successful!");
                game.display();
                if game.is_checkmate() {
                    println!("Checkmate! You win!");
                    break;
                }
                // Computer's turn with neural enhancement
                println!("\nNeural engine is thinking...");
                match game.make_computer_move() {
                    Ok(()) => {
                        println!("Computer moved!");
                        game.display();
                        if game.is_checkmate() {
                            println!("Checkmate! Computer wins!");
                            break;
                        }
                    },
                    Err(e) => {
                        println!("Computer error: {}", e);
                        break;
                    }
                }
            },
            Err(e) => println!("Error: {}", e),
        }
    }
}
