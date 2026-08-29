// ============================================================================
// POLYDIM PMTP - Capa Criptográfica HMAC-SHA256 (Zero-Trust)
// ============================================================================
// Cada tensor transmitido lleva una firma HMAC-SHA256 que el receptor verifica
// antes de tocar un solo byte del payload. Si la firma no coincide, el tensor
// se descarta y el nodo emisor queda marcado como sospechoso.
// ============================================================================

use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

const HMAC_KEY_LEN: usize = 32;

/// Genera la firma HMAC-SHA256 de un bloque de bytes (el tensor serializado).
pub fn sign_payload(key: &[u8; HMAC_KEY_LEN], payload: &[u8]) -> [u8; 32] {
    let mut mac = HmacSha256::new_from_slice(key)
        .expect("HMAC acepta claves de cualquier tamaño");
    mac.update(payload);
    let result = mac.finalize();
    let mut sig = [0u8; 32];
    sig.copy_from_slice(&result.into_bytes());
    sig
}

/// Verifica la firma HMAC-SHA256. Retorna true si el payload no fue alterado.
pub fn verify_payload(key: &[u8; HMAC_KEY_LEN], payload: &[u8], signature: &[u8; 32]) -> bool {
    let mut mac = HmacSha256::new_from_slice(key)
        .expect("HMAC acepta claves de cualquier tamaño");
    mac.update(payload);
    mac.verify_slice(signature).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_verify_roundtrip() {
        let key = [0xABu8; HMAC_KEY_LEN];
        let payload = b"tensor_data_simulado_1234567890";
        let sig = sign_payload(&key, payload);
        assert!(verify_payload(&key, payload, &sig));
    }

    #[test]
    fn test_tampered_payload_fails() {
        let key = [0xCDu8; HMAC_KEY_LEN];
        let payload = b"tensor_original";
        let sig = sign_payload(&key, payload);
        let tampered = b"tensor_alterado";
        assert!(!verify_payload(&key, tampered, &sig));
    }

    #[test]
    fn test_wrong_key_fails() {
        let key1 = [0x01u8; HMAC_KEY_LEN];
        let key2 = [0x02u8; HMAC_KEY_LEN];
        let payload = b"tensor_secreto";
        let sig = sign_payload(&key1, payload);
        assert!(!verify_payload(&key2, payload, &sig));
    }
}
