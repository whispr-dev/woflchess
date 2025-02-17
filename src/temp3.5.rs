// Neural-Enhanced Chess Integration
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use rand::Rng;

// Add to GameState struct
impl GameState {
    // Neural network components
    neural_engine: Option<Arc<Mutex<ChessNeuralEngine>>>,
    
    // Modified constructor
    fn new() -> Self {
        let mut state = GameState {
            board: [[None; 8]; 8],
            current_turn: Color::White,
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

    // Convert board state to neural network input
    fn board_to_neural_input(&self) -> Vec<f32> {
        let mut input = Vec::with_capacity(64 * 6); // 64 squares * 6 piece types
        
        for rank in 0..8 {
            for file in 0..8 {
                if let Some(piece) = self.board[file][rank] {
                    let piece_index = match piece.piece_type {
                        PieceType::Pawn => 0,
                        PieceType::Knight => 1,
                        PieceType::Bishop => 2,
                        PieceType::Rook => 3,
                        PieceType::Queen => 4,
                        PieceType::King => 5,
                    };
                    let value = if piece.color == Color::White { 1.0 } else { -1.0 };
                    input.push(value);
                } else {
                    input.push(0.0);
                }
            }
        }
        
        input
    }

    // Neural network enhanced evaluation
    fn evaluate_position_neural(&self) -> i32 {
        if let Some(engine) = &self.neural_engine {
            let mut engine = engine.lock().unwrap();
            let input = self.board_to_neural_input();
            
            // Get evaluations from different network types
            let rnn_evaluation = {
                let mut rnn = engine.rnn.lock().unwrap();
                let output = rnn.forward(&input);
                output[0] * 100.0 // Scale to centipawn value
            };
            
            let cnn_evaluation = {
                let mut cnn = engine.cnn.lock().unwrap();
                let reshaped_input = vec![input]; // Reshape for CNN
                let output = cnn.forward(&reshaped_input);
                output[0][0] * 100.0
            };
            
            let lstm_evaluation = {
                let mut lstm = engine.lstm.lock().unwrap();
                let output = lstm.forward(&input);
                output[0] * 100.0
            };
            
            // Combine evaluations with traditional evaluation
            let traditional_eval = self.evaluate_position() as f32;
            
            // Weighted average of all evaluations
            let combined_eval = (
                traditional_eval * 0.4 + // 40% weight to traditional evaluation
                rnn_evaluation * 0.2 +   // 20% weight to RNN
                cnn_evaluation * 0.2 +   // 20% weight to CNN
                lstm_evaluation * 0.2     // 20% weight to LSTM
            ) as i32;
            
            combined_eval
        } else {
            self.evaluate_position() // Fallback to traditional evaluation
        }
    }

    // Enhanced move generation using GAN
    fn generate_moves_neural(&self) -> Vec<Move> {
        let mut moves = self.generate_legal_moves();
        
        if let Some(engine) = &self.neural_engine {
            let mut engine = engine.lock().unwrap();
            let input = self.board_to_neural_input();
            
            // Use GAN to score moves
            let mut move_scores: Vec<(Move, f32)> = moves.iter().map(|move_| {
                let mut test_state = self.clone();
                test_state.make_move_without_validation(move_);
                let position_after = test_state.board_to_neural_input();
                
                // Get discriminator scores
                let rnn_score = engine.rnn.lock().unwrap().discriminate(&position_after);
                let cnn_score = engine.cnn.lock().unwrap().discriminate(&vec![position_after.clone()]);
                let lstm_score = engine.lstm.lock().unwrap().discriminate(&position_after);
                
                // Combine scores
                let combined_score = (rnn_score + cnn_score + lstm_score) / 3.0;
                (move_.clone(), combined_score)
            }).collect();
            
            // Sort moves by GAN score
            move_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            // Return sorted moves
            moves = move_scores.into_iter().map(|(move_, _)| move_).collect();
        }
        
        moves
    }

    // Enhanced computer move selection
    fn make_computer_move(&mut self) -> Result<(), &'static str> {
        let mut moves = self.generate_moves_neural(); // Use neural-enhanced move generation
        
        if moves.is_empty() {
            if self.is_checkmate() {
                return Err("Checkmate!");
            } else {
                return Err("Stalemate!");
            }
        }

        let mut best_score = i32::MIN;
        let mut best_moves = Vec::new();
        let search_depth = 3;
        
        // Use neural evaluation in minimax
        for move_ in moves {
            let mut test_state = self.clone();
            test_state.make_move_without_validation(&move_);
            
            let score = -test_state.minimax_neural(search_depth - 1, i32::MIN, i32::MAX, true);
            
            if score > best_score {
                best_score = score;
                best_moves.clear();
                best_moves.push(move_);
            } else if score == best_score {
                best_moves.push(move_);
            }
        }

        // Select and make move
        let mut rng = rand::thread_rng();
        let selected_move = best_moves[rng.gen_range(0..best_moves.len())].clone();
        
        println!("Neural evaluation: {}", best_score);
        self.make_move(selected_move)?;
        println!("Computer plays: {}", self.move_to_algebraic(&selected_move));
        
        Ok(())
    }

    // Neural-enhanced minimax
    fn minimax_neural(&mut self, depth: i32, mut alpha: i32, mut beta: i32, maximizing_player: bool) -> i32 {
        if depth == 0 {
            return self.evaluate_position_neural();
        }

        let moves = self.generate_moves_neural();
        
        if moves.is_empty() {
            if self.is_checkmate() {
                return if maximizing_player { -30000 } else { 30000 };
            }
            return 0;
        }

        if maximizing_player {
            let mut max_eval = i32::MIN;
            for move_ in moves {
                let mut new_state = self.clone();
                new_state.make_move_without_validation(&move_);
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
            for move_ in moves {
                let mut new_state = self.clone();
                new_state.make_move_without_validation(&move_);
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
}