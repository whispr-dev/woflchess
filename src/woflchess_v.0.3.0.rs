// src/main.rs
use std::sync::Arc;
use tokio::sync::Mutex;
use log::{info, error, warn};
use pcap::{Device, Capture};
use anyhow::Result;

mod database;
mod healing;
mod logging;
mod ml;
mod network;
mod password;
mod threat;

// Define packet features structure
#[derive(Debug)]
pub struct PacketFeatures {
    length: u32,
    protocol: u8,
    src_port: u16,
    dst_port: u16,
    flags: u8,
}

#[derive(Debug)]
pub struct NetworkMonitor {
    interface: String,
    threat_detector: Arc<Mutex<threat::ThreatDetector>>,
}

impl NetworkMonitor {
    pub fn new(interface: &str, threat_detector: threat::ThreatDetector) -> Self {
        Self {
            interface: interface.to_string(),
            threat_detector: Arc::new(Mutex::new(threat_detector)),
        }
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting network monitoring on interface: {}", self.interface);
        
        let device = Device::from(self.interface.as_str())
            .map_err(|e| anyhow::anyhow!("Failed to open device: {}", e))?;
            
        let mut cap = Capture::from_device(device)?
            .promisc(true)
            .snaplen(65535)
            .open()
            .map_err(|e| anyhow::anyhow!("Failed to open capture: {}", e))?;

        info!("Capture initialized successfully");

        while let Ok(packet) = cap.next_packet() {
            let features = self.extract_features(&packet);
            
            match self.threat_detector.lock().await.detect_threat(&features).await {
                Ok(true) => {
                    info!("Threat detected in packet!");
                    if let Err(e) = self.handle_threat(&packet).await {
                        error!("Failed to handle threat: {}", e);
                    }
                },
                Ok(false) => {
                    // Normal packet, continue monitoring
                    continue;
                },
                Err(e) => {
                    warn!("Error during threat detection: {}", e);
                }
            }
        }

        Ok(())
    }

    fn extract_features(&self, packet: &pcap::Packet) -> PacketFeatures {
        // Basic feature extraction - you'll want to expand this
        PacketFeatures {
            length: packet.len() as u32,
            protocol: packet.data().get(9).copied().unwrap_or(0),
            src_port: self.extract_port(&packet.data(), 34),
            dst_port: self.extract_port(&packet.data(), 36),
            flags: packet.data().get(47).copied().unwrap_or(0),
        }
    }

    fn extract_port(&self, data: &[u8], start: usize) -> u16 {
        if data.len() > start + 1 {
            ((data[start] as u16) << 8) | (data[start + 1] as u16)
        } else {
            0
        }
    }

    async fn handle_threat(&self, packet: &pcap::Packet) -> Result<()> {
        // Implement your threat handling logic here
        // For example:
        info!("Processing threat - Packet length: {}", packet.len());
        // Add blocking rules, logging, etc.
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    info!("Starting LobsterPot Firewall...");

    // Initialize the database
    let db = database::initialize_database().await?;
    let db = Arc::new(Mutex::new(db));

    // Initialize the threat detection system
    let threat_detector = threat::ThreatDetector::new(db.clone());
    
    // Start network monitoring
    let monitor = NetworkMonitor::new("eth0", threat_detector);
    
    // Run the main monitoring loop
    monitor.start().await?;

    Ok(())
}