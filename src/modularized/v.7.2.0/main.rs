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
//
//
// v6.6.0.1 this is adding human vs. human
//
// Added a GameMode enum to distinguish between human vs human and human vs computer games
// Modified the main menu to include both play options
// Updated the play_game function to handle both modes
// Added more informative game status messages
//
// v.7.0.0 a modularized code version [Claude's idea i swear!]
//
// emplying the file not folder modularization method, made the oode
// way easier to handlle and alter and upgrade.
// also, since there's a whole buncha planned things like further
// learning nerual stuff and possibles like conversion to
// compatibility with a discorb bo9t it needs easy alter.
//
// v.7.1.0 reinstigated full ai capability so that the full game is
// restored n working as v.5.0.0 but with interwebs play and
// human vs. human local.
//
// see github/whisprer/ for more fun experiments into ai and coding!


use std::io::{self, Write};
use std::time::Duration;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use claudes_chess_neural::game::GameState;
use claudes_chess_neural::neural::ChessNeuralEngine;
use claudes_chess_neural::server::start_server;
use claudes_chess_neural::types::{Color, Piece, PieceType};  // Add Piece and PieceType

#[allow(dead_code)]
enum GameMode {
    HumanVsHumanLocal,
    HumanVsComputer,
}

fn debug_print_board(board: &[[Option<Piece>; 8]; 8]) {
    println!("DEBUG: Current board state:");
    for rank in 0..8 {
        print!("  ");
        for file in 0..8 {
            if let Some(piece) = &board[file][rank] {
                print!("{:?} ", piece.piece_type);
            } else {
                print!("· ");
            }
        }
        println!();
    }
}

#[tokio::main]
async fn main() {
    println!("Welcome to the modular _**ChessNeural** tm_ engine!");
    
    // Create a flag to track if server is running
    let server_running = Arc::new(AtomicBool::new(false));

    loop {
        println!("\nCommands:");
        println!("  play local        - Play a local Human vs. Human game");
        println!("  play ai           - Play vs. AI");
        println!("  train time X      - Train self-play for X seconds (stub)");
        println!("  train games X     - Train self-play for X games (stub)");
        if server_running.load(Ordering::SeqCst) {
            println!("  server stop      - Stop the WebSocket server");
        } else {
            println!("  server start     - Start WebSocket server on ws://127.0.0.1:8000/chess");
        }
        println!("  quit              - Exit");
        print!("> ");
        let _ = io::stdout().flush();

        let mut line = String::new();
        if io::stdin().read_line(&mut line).is_err() {
            println!("Read error!");
            continue;
        }
        let cmd = line.trim();

        match cmd {
            "quit" => {
                if server_running.load(Ordering::SeqCst) {
                    println!("Stopping server first...");
                    server_running.store(false, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                println!("Goodbye!");
                break;
            }
            "play local" => {
                println!("Starting a local human vs. human game...");
                play_local_game();
            }
            "play ai" => {
                println!("Starting a local human vs. AI game (stub)...");
                play_ai_game();
            }
            "server start" if !server_running.load(Ordering::SeqCst) => {
                println!("Starting WebSocket server...");
                let sr = server_running.clone();
                tokio::spawn(async move {
                    sr.store(true, Ordering::SeqCst);
                    start_server().await;
                    sr.store(false, Ordering::SeqCst);
                });
                println!("Server started in background. Use 'server stop' to stop it.");
            }
            "server stop" if server_running.load(Ordering::SeqCst) => {
                println!("Stopping WebSocket server...");
                server_running.store(false, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(500)).await;
                println!("Server stopped.");
            }
            c if c.starts_with("train time ") => {
                let rest = c.trim_start_matches("train time ");
                if let Ok(secs) = rest.parse::<u64>() {
                    let mut engine = ChessNeuralEngine::new();
                    println!("Training for {} seconds...", secs);
                    let _ = engine.train_self_play(Some(Duration::from_secs(secs)), None);
                } else {
                    println!("Invalid format. Use: train time X");
                }
            }
            c if c.starts_with("train games ") => {
                let rest = c.trim_start_matches("train games ");
                if let Ok(games) = rest.parse::<usize>() {
                    let mut engine = ChessNeuralEngine::new();
                    println!("Training for {} games...", games);
                    let _ = engine.train_self_play(None, Some(games));
                } else {
                    println!("Invalid format. Use: train games X");
                }
            }
            _ => {
                println!("Unknown command: {}", cmd);
            }
        }
    }
}

// Example local H2H
fn play_local_game() {
    let mut game_state = GameState::new();
    loop {
        println!("{}", game_state);
        if let Err(e) = game_state.validate_board_state() {
            println!("Board invalid: {}", e);
            break;
        }

        println!("Enter move (like 'e2e4'), or 'quit':");
        let mut line = String::new();
        if io::stdin().read_line(&mut line).is_err() {
            println!("Read error!");
            break;
        }
        let mv_str = line.trim();
        if mv_str == "quit" {
            break;
        }
        match game_state.make_move_from_str(mv_str) {
            Ok(_) => {}
            Err(e) => println!("Move error: {}", e),
        }
    }
}

fn square_to_string(pos: (usize, usize)) -> String {
    let file = (b'a' + pos.0 as u8) as char;
    // Convert array rank (0-7 from top) to chess rank (1-8 from bottom)
    let rank = (8 - pos.1).to_string();
    format!("{}{}", file, rank)
}

fn play_ai_game() {
    let mut game_state = GameState::new();
    let mut engine = ChessNeuralEngine::new();
    
    println!("Playing against AI (you are White)");
    
    loop {
        println!("{}", game_state);
        debug_print_board(&game_state.board);  // Add debug print
        
        if let Err(e) = game_state.validate_board_state() {
            println!("Board invalid: {}", e);
            break;
        }

        if game_state.current_turn == Color::White {
            println!("Your turn (e.g. 'e2e4'), or 'quit':");
            let mut line = String::new();
            if io::stdin().read_line(&mut line).is_err() {
                println!("Read error!");
                break;
            }
            let mv_str = line.trim();
            if mv_str == "quit" {
                break;
            }
            match game_state.make_move_from_str(mv_str) {
                Ok(_) => {}
                Err(e) => {
                    println!("Move error: {}", e);
                    continue;
                }
            }
        } else {
            println!("AI is thinking hard. Give him a mo, this is kinda makin' his head hurt and he's stressin'...");
            if let Some(ai_move) = engine.suggest_move(&game_state) {
                let from_str = square_to_string(ai_move.from);
                let to_str = square_to_string(ai_move.to);
                match game_state.make_move(ai_move) {
                    Ok(_) => {
                        println!("AI moved from {} to {}", from_str, to_str);
                    }
                    Err(e) => {
                        println!("AI made invalid move: {}", e);
                        break;
                    }
                }
            } else {
                println!("AI couldn't find a legal move!");
                break;
            }
        }

        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}
