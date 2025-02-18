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

use crate::types::*;
use crate::game::GameState;
use crate::neural::ChessNeuralEngine;
use crate::server::start_server;

#[derive(PartialEq)]
enum GameMode {
    HumanVsHuman,
    HumanVsComputer,
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


// ---------- Main Game App Functionality ----------

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

##[tokio::main]
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


// game mode selection added to main()
fn main() {
    println!("Neural Chess Engine v2.0");
    println!("Commands:");
    println!("  'play human' - Start a human vs human game");
    println!("  'play computer' - Start a game against the computer");
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
            "play human" => play_game(GameMode::HumanVsHuman),
            "play computer" => play_game(GameMode::HumanVsComputer),
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
            "" => continue,
            _ => println!("Unknown command"),
        }
    }
    println!("Goodbye!");
}

// Add game mode enum
#[derive(PartialEq)]
enum GameMode {
    HumanVsHuman,
    HumanVsComputer,
}
