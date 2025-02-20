//////////////////////////
// main.rs
//////////////////////////
//
// welcome to Claude's ChessNeural - a Version of chess by wofl with an optional computer
// opponent employing a very special brain i've been experimenting with.


use std::io::{self, Write};
use std::time::Duration;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use claudes_chess_neural::game::GameState;
use claudes_chess_neural::neural::ChessNeuralEngine;
use claudes_chess_neural::server::start_server;
use claudes_chess_neural::types::{Color, Piece};

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

    let server_running = Arc::new(AtomicBool::new(false));

    loop {
        println!("\nCommands:");
        println!("  play local        - Play a local Human vs. Human game");
        println!("  play ai           - Play vs. AI");
        println!("  train time X      - Train self-play for X seconds (stub)");
        println!("  train games X     - Train self-play for X games (stub)");
        if server_running.load(Ordering::SeqCst) {
            println!("  server stop       - Stop the WebSocket server");
        } else {
            println!("  server start      - Start WebSocket server on ws://127.0.0.1:8000/chess");
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
                println!("Starting a local human vs. AI game...");
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
                    println!("Training for {} seconds (stub)...", secs);
                    let _ = engine.train_self_play(Some(Duration::from_secs(secs)), None);
                } else {
                    println!("Invalid format. Use: train time X");
                }
            }
            c if c.starts_with("train games ") => {
                let rest = c.trim_start_matches("train games ");
                if let Ok(games) = rest.parse::<usize>() {
                    let mut engine = ChessNeuralEngine::new();
                    println!("Training for {} games (stub)...", games);
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

fn play_local_game() {
    let mut game_state = GameState::new();
    loop {
        println!("{}", game_state);
        if let Err(e) = game_state.validate_board_state() {
            println!("Board invalid: {}", e);
            break;
        }

        println!("Enter move (e.g. 'e2e4'), or 'quit':");
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

fn play_ai_game() {
    let mut game_state = GameState::new();
    let mut engine = ChessNeuralEngine::new();

    println!("Playing against AI (you are White).");
    loop {
        println!("{}", game_state);
        debug_print_board(&game_state.board);

        if let Err(e) = game_state.validate_board_state() {
            println!("Board invalid: {}", e);
            break;
        }
        if game_state.current_turn == Color::White {
            println!("Your move (e.g. 'e2e4') or 'quit':");
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
        } else {
            println!("AI thinking...");
            if let Some(best_move) = engine.suggest_move(&game_state) {
                game_state.make_move_without_validation(&best_move);
            } else {
                println!("AI could not find a move. Game over?");
                break;
            }
        }
    }
}
