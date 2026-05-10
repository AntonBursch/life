//! Static + WebSocket server for the life/ viewer.
//!
//! Status: scaffolding. Serves the static `viewer/` directory and exposes a
//! `/ws` endpoint that currently echoes JSON messages back. The wire protocol
//! between server and viewer is intentionally undefined until a rung needs
//! it. We will design it then, not before.

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;
use tracing::{info, warn};
use tracing_subscriber::{prelude::*, EnvFilter};

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let static_dir = std::env::var("LIFE_STATIC_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../../viewer"));

    let port: u16 = std::env::var("LIFE_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8787);

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/health", get(|| async { "ok" }))
        .fallback_service(ServeDir::new(&static_dir))
        .layer(TraceLayer::new_for_http());

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    info!(?addr, static_dir = ?static_dir, "life/ server starting");

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("failed to bind {addr}: {e}");
            std::process::exit(1);
        }
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");
}

async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(mut socket: WebSocket) {
    info!("ws client connected");
    while let Some(msg) = socket.recv().await {
        let msg = match msg {
            Ok(m) => m,
            Err(e) => {
                warn!(error = %e, "ws recv error");
                break;
            }
        };
        match msg {
            Message::Text(t) => {
                // For now, echo. Real protocol arrives when a rung needs it.
                if socket.send(Message::Text(t)).await.is_err() {
                    break;
                }
            }
            Message::Binary(b) => {
                if socket.send(Message::Binary(b)).await.is_err() {
                    break;
                }
            }
            Message::Close(_) => break,
            Message::Ping(p) => {
                let _ = socket.send(Message::Pong(p)).await;
            }
            Message::Pong(_) => {}
        }
    }
    info!("ws client disconnected");
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("shutdown signal received");
}
