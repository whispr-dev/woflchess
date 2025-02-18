////////////////////////////////////////////////////
// main.rs
////////////////////////////////////////////////////

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use std::thread;
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
// 1) DATA DEFINITIONS
////////////////////////////////////////////////////
mod types {
    use super::*;

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

    #[derive(Debug)]
    pub enum GameResult {
        Checkmate(Color),
        Stalemate,
        DrawByMoveLimit,
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

    #[derive(Serialize, Deserialize)]
    pub struct NetworkWeights {
        pub rnn_weights: Vec<Vec<f32>>,
        pub rnn_discriminator: Vec<Vec<f32>>,
        pub cnn_filters: Vec<Vec<Vec<f32>>>,
        pub cnn_discriminator: Vec<Vec<Vec<f32>>>,
        pub lstm_weights: Vec<Vec<f32>>,
        pub lstm_discriminator: Vec<Vec<f32>>,
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

    // The ChessApp for the GUI
    pub struct ChessApp { 
        pub selected_square: Option<(usize, usize)>,
        // Instead of storing a local GameState,
        // we store a reference to a shared GameState:
        pub shared_state: Arc<Mutex<GameState>>,
    }
}

use types::{
    PieceType, Color, MoveError, Board, Move, GameResult, TrainingStats,
    NetworkWeights, GameState, ChessApp,
};

////////////////////////////////////////////////////
// 2) NEURAL ENGINE, CA, RNN/CNN/LSTM, ETC.
////////////////////////////////////////////////////

struct CAInterface {
    // ...
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
        // ...
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

fn tanh(x: f32) -> f32 { x.tanh() }
fn relu(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }
fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

// RNN, CNN, LSTM, etc...
// [Truncated for brevity – same as before]
struct RNN { /* ... */ }
struct CNN { /* ... */ }
struct LSTM { /* ... */ }
struct GANTrainer { /* ... */ }

impl GANTrainer {
    // ...
    fn train_step(/* ... */) {
        // ...
    }
}

// Our main neural engine
pub struct ChessNeuralEngine {
    // ...
    rnn: Arc<Mutex<RNN>>,
    cnn: Arc<Mutex<CNN>>,
    lstm: Arc<Mutex<LSTM>>,
    gan_trainer: GANTrainer,
    ca_interface: CAInterface,
    neural_clock: NeuralClock,
}

impl ChessNeuralEngine {
    fn new() -> Self {
        // ...
        ChessNeuralEngine {
            rnn: Arc::new(Mutex::new(RNN{ /*...*/ })),
            cnn: Arc::new(Mutex::new(CNN{ /*...*/ })),
            lstm: Arc::new(Mutex::new(LSTM{ /*...*/ })),
            gan_trainer: GANTrainer { /*...*/ },
            ca_interface: CAInterface::new(8, 8),
            neural_clock: NeuralClock::new(0.5),
        }
    }
    // save_weights, load_weights, train_self_play, etc...
    // same as before
}

////////////////////////////////////////////////////
// 3) GAMESTATE IMPL
////////////////////////////////////////////////////
impl GameState {
    pub fn new() -> Self {
        // ...
        // same as your existing code
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

    pub fn try_move(&mut self, from: (usize, usize), to: (usize, usize))
        -> Result<(), MoveError>
    {
        // ...
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

    // plus the rest: is_valid_move, make_move, make_computer_move, etc.
}

// We also implement Display for GameState
impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // ...
        // same board printing logic
        Ok(())
    }
}

////////////////////////////////////////////////////
// 4) THE GUI APP: ChessApp
////////////////////////////////////////////////////

// We do a custom constructor that references a shared GameState
impl ChessApp {
    pub fn new(shared_state: Arc<Mutex<GameState>>) -> Self {
        ChessApp {
            selected_square: None,
            shared_state,
        }
    }
}

// We do NOT store `GameState` directly, but an Arc<Mutex<GameState>>.
// Then the eframe::App uses that shared reference to draw / update.
impl App for ChessApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut Frame) {
        let mut gs = self.shared_state.lock().unwrap();

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
                                    let result = gs.try_move((from_c, from_r), (col, row));
                                    if let Err(e) = result {
                                        println!("[GUI Error] Move failed: {}", e);
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
// 5) MAIN: RUNS BOTH CONSOLE AND GUI (ON DEMAND)
////////////////////////////////////////////////////

fn main() {
    println!("Welcome to Chess with concurrency (GUI + console)!");

    // Make a shared GameState behind Arc<Mutex<..>>
    let shared_game_state = Arc::new(Mutex::new(GameState::new()));

    // We also keep a global neural engine if you want
    // Or you can store it inside the game state
    let engine_arc = {
        let gs = shared_game_state.lock().unwrap();
        gs.neural_engine.clone().unwrap()
    };

    // We'll run the console loop in the main thread.
    // If user types "gui", we spawn a new thread that calls run_gui(...).
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    // Ctrl+C handler
    ctrlc::set_handler(move || {
        println!("\n[Ctrl+C] Shutting down...");
        running_clone.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");

    while running.load(Ordering::SeqCst) {
        println!("Enter command (play, train time X, train games X, gui, quit):");
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_err() {
            println!("[Error] Failed to read line.");
            continue;
        }
        let cmd = line.trim();

        match cmd {
            "play" => {
                // Just do a local "play" function that references the same shared state
                // But we’ll do a quick example that clones the state to avoid concurrency issues
                play_game(Arc::clone(&shared_game_state));
            },
            c if c.starts_with("train time ") => {
                let rest = c.trim_start_matches("train time ");
                if let Ok(minutes) = rest.parse::<u64>() {
                    println!("Starting training for {} minutes...", minutes);
                    let dur = Duration::from_secs(minutes * 60);
                    let mut eng = engine_arc.lock().unwrap();
                    match eng.train_self_play(Some(dur), None) {
                        Ok(_) => println!("Training done."),
                        Err(e) => println!("Training error: {}", e),
                    }
                } else {
                    println!("[Error] invalid format for train time");
                }
            },
            c if c.starts_with("train games ") => {
                let rest = c.trim_start_matches("train games ");
                if let Ok(games) = rest.parse::<usize>() {
                    println!("Starting training for {} games...", games);
                    let mut eng = engine_arc.lock().unwrap();
                    match eng.train_self_play(None, Some(games)) {
                        Ok(_) => println!("Training done."),
                        Err(e) => println!("Training error: {}", e),
                    }
                } else {
                    println!("[Error] invalid format for train games");
                }
            },
            "gui" => {
                // Spawn a new thread for the GUI
                // We can spawn a thread that calls run_native(...) so the console keeps going
                let gs_clone = Arc::clone(&shared_game_state);
                thread::spawn(move || {
                    let options = NativeOptions {
                        initial_window_pos: Some(Pos2 { x: 50.0, y: 50.0 }),
                        initial_window_size: Some(Vec2 { x: 600.0, y: 600.0 }),
                        ..Default::default()
                    };
                    run_native(
                        "Concurrent Chess GUI",
                        options,
                        Box::new(|_cc| Box::new(ChessApp::new(gs_clone))),
                    );
                    // Once the window is closed, this thread ends.
                    println!("[GUI Thread] GUI closed, returning to console only.");
                });
            },
            "quit" => {
                println!("Exiting program...");
                running.store(false, Ordering::SeqCst);
            },
            "" => continue,
            _ => println!("[Error] unknown command"),
        }
    }

    println!("Goodbye!");
}

// A little local function to do console-based moves in the same shared GameState
fn play_game(shared_gs: Arc<Mutex<GameState>>) {
    println!("Playing one game vs. AI with the shared state!");
    loop {
        let mut gs = shared_gs.lock().unwrap();
        println!("{}", *gs);
        let status = gs.get_game_status();
        println!("{}", status);

        if gs.current_turn == Color::White {
            println!("Your move (like e2e4 or resign):");
            let mut line = String::new();
            if std::io::stdin().read_line(&mut line).is_err() {
                println!("Failed to read line");
                break;
            }
            let input = line.trim();
            if input == "quit" {
                break;
            }
            if input == "resign" {
                println!("White resigns, black wins!");
                break;
            }
            if let Err(e) = gs.make_move_from_str(input) {
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

        // Check if game ended
        if status.contains("Checkmate") || status.contains("Stalemate") {
            break;
        }
    }
}

////////////////////////////////////////////////////
// Additional code: parse moves, etc
////////////////////////////////////////////////////
impl GameState {
    pub fn make_move_from_str(&mut self, input: &str) -> Result<(), String> {
        if input.len() < 4 {
            return Err("Move too short, e.g. 'e2e4'".to_string());
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
        if s.len() != 2 {
            return None;
        }
        let file = (s.chars().next().unwrap() as u8).wrapping_sub(b'a');
        let rank = (s.chars().nth(1).unwrap() as u8).wrapping_sub(b'1');
        if file < 8 && rank < 8 {
            Some((file as usize, rank as usize))
        } else {
            None
        }
    }
}
