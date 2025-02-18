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


// ---------- LOAD CRATES ----------
// main.rs changes
use std::sync::{Arc, Mutex};
use tokio;
use warp;
use uuid::Uuid;
use colored::*;
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
use std::collections::HashMap;
use uuid::Uuid;
use tokio::sync::broadcast;

mod types;
mod game;
mod neural;
mod server;

use crate::{
    types::*, 
    game::GameState,
    neural::ChessNeuralEngine,
    server::start_server
};

#[derive(PartialEq)]
enum GameMode {
    HumanVsHuman,
    HumanVsComputer,
}


// ---------- CHESS ENGINE (GAME STATE) ----------

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
        return format!("Ha Ha! {:?} is in check!", self.current_turn);
    } else if self.generate_legal_moves().is_empty() {
        return "Stalemate! Game is a draw!".to_string();
    } else {
        return format!("{:?}'s turn", self.current_turn);
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

// Keep only the game mode enum and main function here
#[derive(PartialEq)]
enum GameMode {
    HumanVsHuman,
    HumanVsComputer,
}

#[tokio::main]
async fn main() {
    // Your existing main function...
}

// In main.rs, modify the main function to:##[tokio::main]
async fn main() {
    println!("Neural Chess Engine v2.0");
    println!("Commands:");
    println!("  'server' - Start the multiplayer server");
    println!("  'play human' - Start a local human vs human game");
    println!("  'play computer' - Start a game against the computer");
    println!("  'train time XX' - Train for XX minutes");
    println!("  'train games XX' - Train for XX games");
    println!("  'quit' - Exit the program");

    loop {
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        match input {
            "server" => {
                start_server().await;
            }
            "play human" => {
                // Your existing play_game(GameMode::HumanVsHuman) code
            }
            "play computer" => {
                // Your existing play_game(GameMode::HumanVsComputer) code
            }
            "quit" => break,
            _ => println!("Unknown command"),
        }
    }
}
