// neural.rs - New file for neural network code
pub mod neural {
  use super::*;
  use rand::Rng;
  use std::time::Instant;
  use std::sync::{Arc, Mutex};
  use std::fs::File;
  use std::io::{Write, Read};
  use serde::{Serialize, Deserialize};
  use std::sync::atomic::{AtomicBool, Ordering};


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
}

use rand::Rng;
use std::time::Instant;
use std::sync::{Arc, Mutex};
use std::fs::File;
use std::io::{Write, Read};
use serde::{Serialize, Deserialize};
use std::sync::atomic::{AtomicBool, Ordering};
use crate::game::GameState;
use crate::types::*;


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


// ---------- Game App Functionality Neural ----------

pub fn evaluate_position_neural(&self) -> i32 {
    if let Some(engine_arc) = &self.neural_engine {
        let engine = engine_arc.lock().unwrap();
        let input = self.board_to_neural_input();
        
        let mut total_eval = 0.0;
        let num_components = 4;

        // Traditional evaluation (40% weight)
        total_eval += self.evaluate_position() as f32 * 0.4;

        // RNN evaluation (20% weight)
        if let Ok(mut rnn) = engine.rnn.lock() {
            let out = rnn.forward(&input);
            if let Some(val) = out.get(0) {
                total_eval += val * 100.0 * 0.2;
            }
        }

        // CNN evaluation (20% weight)
        if let Ok(mut cnn) = engine.cnn.lock() {
            let reshaped = crate::reshape_vector_to_matrix(&input, 8, 8);
            if let Some(Some(val)) = cnn.forward(&reshaped).first().map(|row| row.first()) {
                total_eval += val * 100.0 * 0.2;
            }
        }

        // LSTM evaluation (20% weight)
        if let Ok(mut lstm) = engine.lstm.lock() {
            let out = lstm.forward(&input);
            if let Some(val) = out.get(0) {
                total_eval += val * 100.0 * 0.2;
            }
        }

        total_eval as i32
    } else {
        self.evaluate_position()
    }
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






