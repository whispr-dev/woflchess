//////////////////////////
// types.rs
//////////////////////////

use serde::{Serialize, Deserialize};
use std::fmt;

// ----- Basic Chess Types -----

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

#[derive(Debug, Clone, Copy, PartialEq)]
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

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

// Types for WebSocket communication
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GameMove {
    pub from: String,
    pub to: String,
    pub player: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ClientMessage {
    CreateGame,
    JoinGame { game_id: String },
    MakeMove { from: String, to: String },
    ChatMessage { content: String },
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ServerMessage {
    GameCreated { game_id: String },
    GameJoined { color: String },
    MoveMade { from: String, to: String },
    Error { message: String },
}

pub type Board = [[Option<Piece>; 8]; 8];

// ----- GameResult, MoveError, etc. -----
#[derive(Debug)]
pub enum GameResult {
    Checkmate(Color),
    Stalemate,
    DrawByMoveLimit,
}

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

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
            MoveError::KingNotInOriginalPosition => {
                write!(f, "King must be in original position to castle")
            }
        }
    }
}

// ----- Training Stats -----
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
            GameResult::Checkmate(Color::White) => self.white_wins += 1,
            GameResult::Checkmate(Color::Black) => self.black_wins += 1,
            GameResult::Stalemate | GameResult::DrawByMoveLimit => self.draws += 1,
        }
    }

    pub fn display(&self) {
        println!("Training Statistics:");
        println!("Games Played: {}", self.games_played);
        if self.games_played > 0 {
            println!(
                "White Wins: {} ({:.1}%)",
                self.white_wins,
                (self.white_wins as f32 / self.games_played as f32) * 100.0
            );
            println!(
                "Black Wins: {} ({:.1}%)",
                self.black_wins,
                (self.black_wins as f32 / self.games_played as f32) * 100.0
            );
            println!(
                "Draws: {} ({:.1}%)",
                self.draws,
                (self.draws as f32 / self.games_played as f32) * 100.0
            );
        }
    }
}

// ----- Network Weights -----
#[derive(Serialize, Deserialize)]
pub struct NetworkWeights {
    pub rnn_weights: Vec<Vec<f32>>,
    pub rnn_discriminator: Vec<Vec<f32>>,
    pub cnn_filters: Vec<Vec<Vec<f32>>>,
    pub cnn_discriminator: Vec<Vec<Vec<f32>>>,
    pub lstm_weights: Vec<Vec<f32>>,
    pub lstm_discriminator: Vec<Vec<f32>>,
}
