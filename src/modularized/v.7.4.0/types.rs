//////////////////////////
// types.rs
//////////////////////////

use serde::{Serialize, Deserialize};
use std::fmt;

pub type Board = [[Option<Piece>; 8]; 8];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceType {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    White,
    Black,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Piece {
    pub piece_type: PieceType,
    pub color: Color,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CastlingRights {
    pub white_kingside: bool,
    pub white_queenside: bool,
    pub black_kingside: bool,
    pub black_queenside: bool,
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
    InvalidPromotion(String),
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
            MoveError::InvalidPromotion(msg) => write!(f, "Invalid promotion: {}", msg),
        }
    }
}