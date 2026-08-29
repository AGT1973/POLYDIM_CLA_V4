// ============================================================================
// POLYDIM PMTP - Protocolo de Cable (Wire Protocol)
// ============================================================================
// Define la estructura binaria exacta de cada paquete PMTP transmitido por TCP.
//
// Layout del paquete (Little-Endian):
// ┌──────────────────────────────────────────────────────────┐
// │ MAGIC (4 bytes): "PMTP"                                 │
// │ VERSION (1 byte): 0x01                                  │
// │ DTYPE (1 byte): 0x08 = float64, 0x04 = float32          │
// │ NDIM (2 bytes): número de dimensiones del tensor         │
// │ SHAPE (NDIM * 8 bytes): cada dimensión como i64          │
// │ EPOCH (8 bytes): época del comité M-of-N                 │
// │ SEQ (8 bytes): número de secuencia anti-replay            │
// │ HMAC (32 bytes): firma SHA-256 del payload               │
// │ PAYLOAD (variable): datos crudos del tensor               │
// └──────────────────────────────────────────────────────────┘
// ============================================================================

use std::io::{self, Read, Write};

pub const MAGIC: &[u8; 4] = b"PMTP";
pub const VERSION: u8 = 0x01;
pub const DTYPE_F64: u8 = 0x08;
pub const DTYPE_F32: u8 = 0x04;

/// Cabecera del paquete PMTP (tamaño fijo excepto SHAPE).
#[derive(Debug, Clone)]
pub struct PmtpHeader {
    pub dtype: u8,
    pub ndim: u16,
    pub shape: Vec<i64>,
    pub epoch: u64,
    pub seq: u64,
    pub hmac: [u8; 32],
}

impl PmtpHeader {
    /// Tamaño en bytes de la cabecera serializada (sin contar el payload).
    pub fn wire_size(&self) -> usize {
        4 + 1 + 1 + 2 + (self.ndim as usize * 8) + 8 + 8 + 32
    }

    /// Serializa la cabecera a un buffer de bytes.
    pub fn encode<W: Write>(&self, w: &mut W) -> io::Result<()> {
        w.write_all(MAGIC)?;
        w.write_all(&[VERSION])?;
        w.write_all(&[self.dtype])?;
        w.write_all(&self.ndim.to_le_bytes())?;
        for &dim in &self.shape {
            w.write_all(&dim.to_le_bytes())?;
        }
        w.write_all(&self.epoch.to_le_bytes())?;
        w.write_all(&self.seq.to_le_bytes())?;
        w.write_all(&self.hmac)?;
        Ok(())
    }

    /// Deserializa una cabecera desde un stream de bytes.
    pub fn decode<R: Read>(r: &mut R) -> io::Result<Self> {
        // MAGIC
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Magic PMTP inválido"));
        }

        // VERSION
        let mut ver = [0u8; 1];
        r.read_exact(&mut ver)?;
        if ver[0] != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Versión PMTP no soportada: {}", ver[0]),
            ));
        }

        // DTYPE
        let mut dtype_buf = [0u8; 1];
        r.read_exact(&mut dtype_buf)?;
        let dtype = dtype_buf[0];

        // NDIM
        let mut ndim_buf = [0u8; 2];
        r.read_exact(&mut ndim_buf)?;
        let ndim = u16::from_le_bytes(ndim_buf);

        // Límite de seguridad: no aceptar más de 64 dimensiones
        if ndim > 64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("NDIM sospechosamente alto: {}", ndim),
            ));
        }

        // SHAPE
        let mut shape = Vec::with_capacity(ndim as usize);
        for _ in 0..ndim {
            let mut dim_buf = [0u8; 8];
            r.read_exact(&mut dim_buf)?;
            shape.push(i64::from_le_bytes(dim_buf));
        }

        // EPOCH
        let mut epoch_buf = [0u8; 8];
        r.read_exact(&mut epoch_buf)?;
        let epoch = u64::from_le_bytes(epoch_buf);

        // SEQ
        let mut seq_buf = [0u8; 8];
        r.read_exact(&mut seq_buf)?;
        let seq = u64::from_le_bytes(seq_buf);

        // HMAC
        let mut hmac = [0u8; 32];
        r.read_exact(&mut hmac)?;

        Ok(PmtpHeader {
            dtype,
            ndim,
            shape,
            epoch,
            seq,
            hmac,
        })
    }

    /// Calcula el número total de elementos del tensor.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Calcula el tamaño del payload en bytes.
    pub fn payload_bytes(&self) -> usize {
        self.num_elements() * (self.dtype as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_header_roundtrip() {
        let header = PmtpHeader {
            dtype: DTYPE_F64,
            ndim: 2,
            shape: vec![1000, 1000],
            epoch: 42,
            seq: 1,
            hmac: [0xAA; 32],
        };

        let mut buf = Vec::new();
        header.encode(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = PmtpHeader::decode(&mut cursor).unwrap();

        assert_eq!(decoded.dtype, DTYPE_F64);
        assert_eq!(decoded.ndim, 2);
        assert_eq!(decoded.shape, vec![1000, 1000]);
        assert_eq!(decoded.epoch, 42);
        assert_eq!(decoded.seq, 1);
        assert_eq!(decoded.hmac, [0xAA; 32]);
        assert_eq!(decoded.num_elements(), 1_000_000);
        assert_eq!(decoded.payload_bytes(), 8_000_000);
    }

    #[test]
    fn test_bad_magic_rejected() {
        let bad = b"BADMxxxxxxxx";
        let mut cursor = Cursor::new(bad.as_slice());
        assert!(PmtpHeader::decode(&mut cursor).is_err());
    }
}
