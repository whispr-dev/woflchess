//////////////////////////
// server.rs
//////////////////////////

// server.rs
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use warp::ws::{WebSocket, Message};
use warp::Filter;
use futures_util::{StreamExt, SinkExt};
use uuid::Uuid;
use tokio::sync::broadcast;
use tokio::signal;


// Add these imports from your crate
use crate::types::{GameMove, ClientMessage, ServerMessage};
use crate::game::GameState;

// Define the Games type here since it uses local types
pub type Games = Arc<Mutex<HashMap<String, GameSession>>>;

#[derive(Clone)]
pub struct GameSession {
    pub id: String,
    pub white_player: Option<String>,
    pub black_player: Option<String>,
    pub game_state: Arc<Mutex<GameState>>,
    pub move_channel: broadcast::Sender<GameMove>,
}

async fn handle_connection(ws: WebSocket, games: Games) {
    let (mut ws_tx, mut ws_rx) = ws.split();
    let client_id = Uuid::new_v4().to_string();
    println!("New client connected: {}", client_id);

    while let Some(result) = ws_rx.next().await {
        let msg = match result {
            Ok(m) => m,
            Err(e) => {
                eprintln!("WebSocket error: {}", e);
                break;
            }
        };

        let text = match msg.to_str() {
            Ok(s) => s,
            Err(_) => continue,
        };

        let client_msg: ClientMessage = match serde_json::from_str(text) {
            Ok(c) => c,
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
                    game_state: Arc::new(Mutex::new(GameState::new())),
                    move_channel: tx,
                };

                // Drop the lock before the await
                {
                    let mut games_lock = games.lock().unwrap();
                    games_lock.insert(game_id.clone(), session);
                }

                let response = ServerMessage::GameCreated { game_id };
                let _ = ws_tx.send(Message::text(serde_json::to_string(&response).unwrap())).await;
            }
            ClientMessage::JoinGame { game_id } => {
                let mut response = ServerMessage::Error { 
                    message: "Game full".to_string() 
                };
                
                {
                    let mut games_lock = games.lock().unwrap();
                    if let Some(session) = games_lock.get_mut(&game_id) {
                        if session.black_player.is_none() {
                            session.black_player = Some(client_id.clone());
                            response = ServerMessage::GameJoined { 
                                color: "black".to_string() 
                            };
                        }
                    } else {
                        response = ServerMessage::Error { 
                            message: "No such game".to_string() 
                        };
                    }
                }
                
                let _ = ws_tx.send(Message::text(serde_json::to_string(&response).unwrap())).await;
            }
            ClientMessage::MakeMove { from, to } => {
                {
                    let games_lock = games.lock().unwrap();
                    for session in games_lock.values() {
                        if session.white_player.as_ref() == Some(&client_id)
                            || session.black_player.as_ref() == Some(&client_id)
                        {
                            let mut gs = session.game_state.lock().unwrap();
                            let move_str = format!("{}{}", from, to);
                            if gs.make_move_from_str(&move_str).is_ok() {
                                let move_msg = GameMove {
                                    from: from.clone(),
                                    to: to.clone(),
                                    player: client_id.clone(),
                                };
                                let _ = session.move_channel.send(move_msg);
                            }
                            break;
                        }
                    }
                }

                let response = ServerMessage::MoveMade { 
                    from: from.clone(), 
                    to: to.clone() 
                };
                
                let _ = ws_tx.send(Message::text(serde_json::to_string(&response).unwrap())).await;
            }
            ClientMessage::ChatMessage { content } => {
                println!("Chat from {}: {}", client_id, content);
            }
        }
    }

    println!("Client {} disconnected", client_id);
    cleanup_player(client_id, games).await;
}

async fn cleanup_player(client_id: String, games: Games) {
    let mut map = games.lock().unwrap();
    map.retain(|_, session| {
        session.white_player.as_ref() != Some(&client_id)
            && session.black_player.as_ref() != Some(&client_id)
    });
}

pub async fn start_server() {
    let games: Games = Arc::new(Mutex::new(HashMap::new()));

    // Create our WebSocket route
    let games = warp::any().map(move || games.clone());
    
    let routes = warp::path("chess")
        .and(warp::ws())
        .and(games)
        .map(|ws: warp::ws::Ws, games| {
            ws.on_upgrade(move |socket| handle_connection(socket, games))
        });

    println!("Server started on ws://127.0.0.1:8000/chess");

    // Run the server
    warp::serve(routes)
        .run(([127, 0, 0, 1], 8000))
        .await;
}