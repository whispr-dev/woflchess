// lib.rs - This looks good but should also export neural
pub mod types;
pub mod server;
pub mod game;
pub mod neural;

pub use types::*;
pub use game::GameState;
pub use server::start_server;
pub use neural::ChessNeuralEngine;