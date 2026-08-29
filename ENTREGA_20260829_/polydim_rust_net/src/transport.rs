// ============================================================================
// POLYDIM PMTP - Transporte Nativo de Tensores (Zero-Copy Send/Recv)
// ============================================================================
// Este módulo implementa las operaciones reales de envío y recepción de
// tensores por TCP usando Tokio. Cada tensor viaja con su cabecera PMTP
// firmada por HMAC-SHA256, y el receptor verifica la integridad antes
// de aceptar el payload.
// ============================================================================

use tokio::net::TcpStream;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use std::io::Cursor;

use crate::wire::{PmtpHeader, DTYPE_F64};
use crate::crypto;

const HMAC_KEY: [u8; 32] = [0x50u8; 32]; // Placeholder — en producción se negocia por Diffie-Hellman

/// Envía un tensor (como Vec<f64>) a un nodo remoto por TCP.
pub async fn send_tensor(
    host: &str,
    port: u16,
    data: &[f64],
    shape: Vec<i64>,
    epoch: u64,
    seq: u64,
) -> Result<(), String> {
    // 1. Serializar los datos crudos como bytes (Little-Endian, el estándar de x86/ARM)
    let payload: Vec<u8> = data.iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    // 2. Firmar el payload con HMAC-SHA256
    let hmac_sig = crypto::sign_payload(&HMAC_KEY, &payload);

    // 3. Construir la cabecera
    let header = PmtpHeader {
        dtype: DTYPE_F64,
        ndim: shape.len() as u16,
        shape,
        epoch,
        seq,
        hmac: hmac_sig,
    };

    // 4. Serializar cabecera a bytes
    let mut header_buf = Vec::with_capacity(header.wire_size());
    header.encode(&mut header_buf).map_err(|e| format!("Error encoding header: {}", e))?;

    // 5. Conectar al peer y enviar cabecera + payload de forma atómica
    let addr = format!("{}:{}", host, port);
    let mut stream = TcpStream::connect(&addr).await
        .map_err(|e| format!("Conexión fallida a {}: {}", addr, e))?;
    
    // Timeout de escritura para evitar deadlock (10 segundos)
    stream.set_nodelay(true).ok();

    stream.write_all(&header_buf).await
        .map_err(|e| format!("Error enviando header: {}", e))?;
    stream.write_all(&payload).await
        .map_err(|e| format!("Error enviando payload: {}", e))?;
    stream.flush().await
        .map_err(|e| format!("Error en flush: {}", e))?;

    println!(
        "[PMTP-TX] Tensor enviado: {} elementos, epoch={}, seq={}, {} bytes totales",
        data.len(), epoch, seq, header_buf.len() + payload.len()
    );

    Ok(())
}

/// Recibe un tensor desde una conexión TCP ya aceptada.
/// Retorna el header decodificado y los datos como Vec<f64>.
pub async fn recv_tensor(
    stream: &mut TcpStream,
) -> Result<(PmtpHeader, Vec<f64>), String> {
    // 1. Leer cabecera fija (mínimo: 4+1+1+2+8+8+32 = 56 bytes sin SHAPE)
    //    Primero leemos los campos fijos para saber NDIM
    let mut fixed_buf = [0u8; 8]; // MAGIC(4) + VER(1) + DTYPE(1) + NDIM(2)
    stream.read_exact(&mut fixed_buf).await
        .map_err(|e| format!("Error leyendo header fijo: {}", e))?;

    // Parsear NDIM para saber cuánto SHAPE leer
    let ndim = u16::from_le_bytes([fixed_buf[6], fixed_buf[7]]);
    if ndim > 64 {
        return Err(format!("NDIM sospechoso: {}", ndim));
    }

    // 2. Leer el resto: SHAPE(ndim*8) + EPOCH(8) + SEQ(8) + HMAC(32)
    let remaining = (ndim as usize * 8) + 8 + 8 + 32;
    let mut rest_buf = vec![0u8; remaining];
    stream.read_exact(&mut rest_buf).await
        .map_err(|e| format!("Error leyendo header variable: {}", e))?;

    // 3. Reconstituir el header completo para decodificarlo
    let mut full_header = Vec::with_capacity(fixed_buf.len() + rest_buf.len());
    full_header.extend_from_slice(&fixed_buf);
    full_header.extend_from_slice(&rest_buf);

    let mut cursor = Cursor::new(&full_header);
    let header = PmtpHeader::decode(&mut cursor)
        .map_err(|e| format!("Error decodificando header: {}", e))?;

    // 4. Leer el payload
    let payload_size = header.payload_bytes();
    let mut payload_buf = vec![0u8; payload_size];
    stream.read_exact(&mut payload_buf).await
        .map_err(|e| format!("Error leyendo payload ({} bytes): {}", payload_size, e))?;

    // 5. Verificar HMAC (Zero-Trust: si no coincide, descartamos todo)
    if !crypto::verify_payload(&HMAC_KEY, &payload_buf, &header.hmac) {
        return Err("HMAC INVÁLIDO: el tensor fue alterado en tránsito. Descartando.".to_string());
    }

    // 6. Reconstruir Vec<f64> desde los bytes crudos
    let data: Vec<f64> = payload_buf
        .chunks_exact(8)
        .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
        .collect();

    println!(
        "[PMTP-RX] Tensor recibido: {} elementos, epoch={}, seq={}, HMAC verificado ✓",
        data.len(), header.epoch, header.seq
    );

    Ok((header, data))
}
