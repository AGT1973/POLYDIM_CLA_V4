// ============================================================================
// POLYDIM PMTP - Nodo Gossip Asíncrono (Tokio)
// ============================================================================
// Motor de red del enjambre PMTP. Cada nodo escucha en un puerto TCP,
// acepta conexiones entrantes, verifica tensores HMAC y los despacha
// al consumidor (JAX vía DLPack callback). El epoch_countdown rota
// la autoridad del comité M-of-N periódicamente.
// ============================================================================

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};

use crate::transport;

pub struct PmtpNode {
    pub port: u16,
    pub peers: Arc<Mutex<Vec<String>>>,
    pub epoch: Arc<AtomicU64>,
    pub seq: Arc<AtomicU64>,
    pub received_tensors: Arc<Mutex<Vec<Vec<f64>>>>,
}

impl PmtpNode {
    pub fn new(port: u16) -> Self {
        PmtpNode {
            port,
            peers: Arc::new(Mutex::new(Vec::new())),
            epoch: Arc::new(AtomicU64::new(0)),
            seq: Arc::new(AtomicU64::new(0)),
            received_tensors: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub async fn start_listening(port: u16, received: Arc<Mutex<Vec<Vec<f64>>>>) {
        let addr = format!("0.0.0.0:{}", port);
        let listener = TcpListener::bind(&addr).await.expect("Falla al bindear puerto PMTP");
        println!("[PMTP-RUST] Nodo asíncrono escuchando en {}", addr);

        loop {
            match listener.accept().await {
                Ok((socket, addr)) => {
                    println!("[PMTP-RUST] Conexión aceptada de: {}", addr);
                    let recv_clone = received.clone();
                    tokio::spawn(async move {
                        Self::handle_connection(socket, recv_clone).await;
                    });
                }
                Err(e) => {
                    eprintln!("[PMTP-RUST] Error aceptando conexión: {}", e);
                }
            }
        }
    }

    async fn handle_connection(mut socket: TcpStream, received: Arc<Mutex<Vec<Vec<f64>>>>) {
        match transport::recv_tensor(&mut socket).await {
            Ok((_header, data)) => {
                let len = data.len();
                received.lock().await.push(data);
                println!("[PMTP-RUST] Tensor almacenado en buffer. Total en cola: {} (último: {} elem)", 
                    received.lock().await.len(), len);
            }
            Err(e) => {
                eprintln!("[PMTP-RUST] Error recibiendo tensor: {}", e);
            }
        }
    }

    pub async fn send_to_peer(
        host: &str,
        port: u16,
        data: &[f64],
        shape: Vec<i64>,
        epoch: u64,
        seq: u64,
    ) -> Result<(), String> {
        transport::send_tensor(host, port, data, shape, epoch, seq).await
    }

    pub async fn epoch_countdown_loop(epoch: Arc<AtomicU64>, peers_ref: Arc<Mutex<Vec<String>>>) {
        loop {
            sleep(Duration::from_secs(10)).await;
            let new_epoch = epoch.fetch_add(1, Ordering::SeqCst) + 1;
            let peers = peers_ref.lock().await;
            println!(
                "[PMTP-GOSSIP] Epoch rotado a {}. Pares activos: {}",
                new_epoch,
                peers.len()
            );
        }
    }
}
