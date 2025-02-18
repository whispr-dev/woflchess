use claudes_chess_neural::server::start_server;  // assuming this is your crate name
// server.rs
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use warp::ws::{WebSocket, Ws};
use warp::Filter;
use futures_util::stream::StreamExt;
use futures_util::sink::SinkExt;  // Add this for send()
use tokio_tungstenite::tungstenite::Message;
use uuid::Uuid;

use crate::types::*;
use crate::game::GameState;


use claudes_chess_neural::server::start_server;  // assuming this is your crate name


mod server;
use server::start_server;

// Game session tracking
#[derive(Clone)]
pub struct GameSession {
    pub id: String,
    pub white_player: Option<String>,
    pub black_player: Option<String>,
    pub game_state: Arc<TokioMutex<GameState>>,
    pub move_channel: broadcast::Sender<GameMove>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct GameMove {
    pub from: String,
    pub to: String,
    pub player: String,
}

#[derive(Serialize, Deserialize)]
pub enum ClientMessage {
    CreateGame,
    JoinGame { game_id: String },
    MakeMove { from: String, to: String },
    ChatMessage { content: String },
}

#[derive(Serialize, Deserialize)]
pub enum ServerMessage {
    GameCreated { game_id: String },
    GameJoined { color: String },
    MoveMade { from: String, to: String },
    GameState { fen: String },
    Error { message: String },
}

type Games = Arc<TokioMutex<HashMap<String, GameSession>>>;


async fn handle_connection(ws: WebSocket, games: Games) {
    let (mut ws_tx, mut ws_rx) = ws.split();
    let client_id = Uuid::new_v4().to_string();

    while let Some(result) = ws_rx.next().await {
        let msg = match result {
            Ok(msg) => msg,
            Err(e) => {
                eprintln!("WebSocket error: {}", e);
                break;
            }
        };

        let client_msg: ClientMessage = match serde_json::from_str(msg.to_str().unwrap_or_default()) {
            Ok(msg) => msg,
            Err(_) => continue,
        };

        match client_msg {
            ClientMessage::CreateGame => {
                let game_id = Uuid::new_v4().to_string();
                let (tx, _) = tokio::sync::broadcast::channel(100);
                
                let session = GameSession {
                    id: game_id.clone(),
                    white_player: Some(client_id.clone()),
                    black_player: None,
                    game_state: Arc::new(Mutex::new(GameState::new())),
                    move_channel: tx,
                };

                games.lock().await.insert(game_id.clone(), session);

                let response = ServerMessage::GameCreated { game_id };
                let _ = ws_tx.send(Message::Text(serde_json::to_string(&response).unwrap())).await;
            }
            // ... rest of the match cases ...
        }
    }
    // Clean up when connection closes
    cleanup_player(client_id, games).await;
}

pub async fn start_server() {
    let games: Games = Arc::new(Mutex::new(HashMap::new()));
    
    let game_sessions = games.clone();
    let ws_route = warp::path("chess")
        .and(warp::ws())
        .and(warp::any().map(move || game_sessions.clone()))
        .map(|ws: Ws, games| {
            ws.on_upgrade(move |socket| handle_connection(socket, games))
        });

    println!("Chess server starting on ws://127.0.0.1:8000");
    warp::serve(ws_route)
        .run(([127, 0, 0, 1], 8000))
        .await;
}


async fn cleanup_player(client_id: String, games: Games) {
    let mut games = games.lock().await;
    games.retain(|_, session| {
        session.white_player.as_ref() != Some(&client_id) &&
        session.black_player.as_ref() != Some(&client_id)
    });
}


  while let Some(result) = ws_rx.next().await {
      let msg = match result {
          Ok(msg) => msg,
          Err(e) => {
              eprintln!("WebSocket error: {}", e);
              break;
          }
      };

      let client_msg: ClientMessage = match serde_json::from_str(msg.to_str().unwrap_or
          Ok(msg) => msg,
          Err(_) => continue,
      };

      match client_msg {
          ClientMessage::CreateGame => {
              let game_id = Uuid::new_v4().to_string();
              let (tx, _) = broadcast::channel(100);
              
              let session = GameSession {
                  id: game_id.clone(),
                  white_player: Some(client_id.clone()),
                  black_player: None,
                  game_state: Arc::new(TokioMutex::new(GameState::new())),
                  move_channel: tx,
              };

              games.lock().await.insert(game_id.clone(), session);

              let response = ServerMessage::GameCreated { game_id };
              let _ = ws_tx.send(Message::Text(serde_json::to_string(&response).unwrap(
          }

          ClientMessage::JoinGame { game_id } => {
              let mut games = games.lock().await;
              if let Some(session) = games.get_mut(&game_id) {
                  if session.black_player.is_none() {
                      session.black_player = Some(client_id.clone());
                      
                      let response = ServerMessage::GameJoined { 
                          color: "black".to_string() 
                      };
                      let _ = ws_tx.send(Message::Text(serde_json::to_string(&response)
                  }
              }
          }

          ClientMessage::MakeMove { from, to } => {
              let games = games.lock().await;
              for session in games.values() {
                  if session.white_player.as_ref() == Some(&client_id) || 
                     session.black_player.as_ref() == Some(&client_id) {
                      let mut game_state = session.game_state.lock().await;
                      
                      if let Ok(()) = game_state.make_move_from_str(&format!("{}{}", fr
                          let move_msg = GameMove {
                              from,
                              to,
                              player: client_id.clone(),
                          };
                          
                          let _ = session.move_channel.send(move_msg);
                      }
                      break;
                  }
              }
          }

          ClientMessage::ChatMessage { content: _ } => {
              // Handle chat messages if needed
          }
      }
  }

  cleanup_player(client_id, games).await;
}

async fn cleanup_player(client_id: String, games: Games) {
  let mut games = games.lock().await;
  games.retain(|_, session| {
      session.white_player.as_ref() != Some(&client_id) &&
      session.black_player.as_ref() != Some(&client_id)
  });
}

// In main.rs, modify the main function to:
#[tokio::main]
async fn main() {
  println!("Neural Chess Engine v2.0");
  println!("Commands:");
  println!("  'server' - Start the multiplayer server");
  println!("  'play human' - Start a local human vs human game");
  println!("  'play computer' - Start a game against the computer");
  println!("  'train time XX' - Train for XX minutes");
  println!("  'train games XX' - Train for XX games");
  println!("  'quit' - Exit the program");

  let mut input = String::new();
  std::io::stdin().read_line(&mut input).unwrap();
  let input = input.trim();

  match input {
      "server" => {
          start_server().await;
      }
      // ... rest of your existing match cases ...
      _ => println!("Unknown command"),
  }
}
