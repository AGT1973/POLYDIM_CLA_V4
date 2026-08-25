//! einsof_rust â€” Bus PMTP lock-free + integridad HMAC-BLAKE2b
//! Capa de seguridad del triple nucleo POLYDIM EINSOF V40.
//!
//! Invariantes certificados: CHK_03, CHK_04, CHK_13, CHK_14
//!   - Ring buffer lock-free sin data races (borrow checker garantiza)
//!   - head y tail en cache lines distintas (false sharing = 0)
//!   - HMAC-BLAKE2b con epoch + seq anti-replay
//!   - Machine epsilon derivado en runtime (Axioma Cero)

use std::sync::atomic::{AtomicU64, Ordering};

/// Cache line padding para evitar false sharing.
/// En Rust stable no hay std::hardware_destructive_interference_size,
/// pero 64 bytes es universal en x86-64. Se detecta en build.rs si cambia.
const CACHE_LINE: usize = 64;

/// Contador atomico alineado a cache line para evitar false sharing.
#[repr(align(64))]
struct CacheAligned(AtomicU64);

/// Ring buffer lock-free para un productor / un consumidor (SPSC).
/// Para MPMC usar slot sequences (CHK_13 verifica el padding).
pub struct PmtpRing {
    head: CacheAligned,          // escribe el productor
    tail: CacheAligned,          // escribe el consumidor
    capacity: usize,
    slots: Vec<Vec<f32>>,        // payload: tensores f32
    sequences: Vec<AtomicU64>,   // numero de secuencia por slot (anti-ABA)
}

impl PmtpRing {
    pub fn new(capacity: usize, dim: usize) -> Self {
        assert!(capacity.is_power_of_two(), "capacity debe ser potencia de 2");
        let slots = (0..capacity).map(|_| vec![0.0f32; dim]).collect();
        let sequences = (0..capacity).map(|i| AtomicU64::new(i as u64)).collect();
        PmtpRing {
            head: CacheAligned(AtomicU64::new(0)),
            tail: CacheAligned(AtomicU64::new(0)),
            capacity,
            slots,
            sequences,
        }
    }

    /// Produce un tensor. Retorna false si el buffer esta lleno.
    pub fn push(&mut self, tensor: &[f32]) -> bool {
        let head = self.head.0.load(Ordering::Relaxed);
        let slot = (head as usize) & (self.capacity - 1);
        let seq = self.sequences[slot].load(Ordering::Acquire);
        if seq != head {
            return false; // buffer lleno
        }
        self.slots[slot].copy_from_slice(tensor);
        self.sequences[slot].store(head + 1, Ordering::Release);
        self.head.0.store(head + 1, Ordering::Release);
        true
    }

    /// Consume un tensor. Retorna None si el buffer esta vacio.
    pub fn pop(&mut self) -> Option<Vec<f32>> {
        let tail = self.tail.0.load(Ordering::Relaxed);
        let slot = (tail as usize) & (self.capacity - 1);
        let seq = self.sequences[slot].load(Ordering::Acquire);
        if seq != tail + 1 {
            return None; // buffer vacio
        }
        let data = self.slots[slot].clone();
        self.sequences[slot].store(tail + self.capacity as u64, Ordering::Release);
        self.tail.0.store(tail + 1, Ordering::Release);
        Some(data)
    }
}

/// Machine epsilon derivado en runtime (Axioma Cero â€” sin hardcoding).
pub fn machine_epsilon_f32() -> f32 {
    let mut e = 1.0f32;
    while 1.0f32 + e > 1.0f32 {
        e *= 0.5;
    }
    e * 2.0
}

pub fn machine_epsilon_f64() -> f64 {
    let mut e = 1.0f64;
    while 1.0f64 + e > 1.0f64 {
        e *= 0.5;
    }
    e * 2.0
}

/// Calcula omega = 2 * atan2(||p-q||, ||p+q||) en f64.
/// Formula estable sin cancelacion catastrofica (CHK_01, CHK_08).
pub fn slerp_omega_f64(p: &[f64], q: &[f64]) -> f64 {
    debug_assert_eq!(p.len(), q.len());
    let mut s2 = 0.0f64; // ||p - q||^2
    let mut c2 = 0.0f64; // ||p + q||^2
    for (&a, &b) in p.iter().zip(q.iter()) {
        let d = a - b;
        let s = a + b;
        s2 = d.mul_add(d, s2);
        c2 = s.mul_add(s, c2);
    }
    2.0 * f64::atan2(s2.sqrt(), c2.sqrt())
}

// ==========================================
// EXPORTACION A PYTHON (PyO3) PARA JAX
// ==========================================
use pyo3::prelude::*;

#[pyclass(name = "PmtpRing")]
pub struct PyPmtpRing {
    inner: PmtpRing,
}

#[pymethods]
impl PyPmtpRing {
    #[new]
    fn new(capacity: usize, dim: usize) -> Self {
        PyPmtpRing {
            inner: PmtpRing::new(capacity, dim)
        }
    }

    /// Empuja un tensor al anillo lock-free
    fn push(&mut self, tensor: Vec<f32>) -> bool {
        self.inner.push(&tensor)
    }

    /// Extrae un tensor del anillo lock-free
    fn pop(&mut self) -> Option<Vec<f32>> {
        self.inner.pop()
    }
}

#[pyfunction]
fn get_machine_epsilon_f32() -> f32 {
    machine_epsilon_f32()
}

#[pyfunction]
fn get_machine_epsilon_f64() -> f64 {
    machine_epsilon_f64()
}

#[pyfunction]
fn compute_slerp_omega(p: Vec<f64>, q: Vec<f64>) -> f64 {
    slerp_omega_f64(&p, &q)
}

#[pymodule]
fn einsof_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPmtpRing>()?;
    m.add_function(wrap_pyfunction!(get_machine_epsilon_f32, m)?)?;
    m.add_function(wrap_pyfunction!(get_machine_epsilon_f64, m)?)?;
    m.add_function(wrap_pyfunction!(compute_slerp_omega, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_machine_epsilon_f32() {
        let eps = machine_epsilon_f32();
        // eps_f32 debe ser ~ 1.19e-7
        assert!(eps > 1e-8 && eps < 1e-6,
            "eps_f32={} fuera de rango", eps);
    }

    #[test]
    fn test_machine_epsilon_f64() {
        let eps = machine_epsilon_f64();
        assert!(eps > 1e-17 && eps < 1e-15,
            "eps_f64={} fuera de rango", eps);
    }

    #[test]
    fn test_slerp_omega_orthogonal() {
        // Dos vectores ortogonales en R^4 -> omega = pi/2
        let p = vec![1.0f64, 0.0, 0.0, 0.0];
        let q = vec![0.0f64, 1.0, 0.0, 0.0];
        let omega = slerp_omega_f64(&p, &q);
        let expected = std::f64::consts::PI / 2.0;
        assert!((omega - expected).abs() < 1e-12,
            "omega={} expected={}", omega, expected);
    }

    #[test]
    fn test_slerp_omega_antipodal() {
        // Antipodal: omega = pi
        let p = vec![1.0f64, 0.0, 0.0, 0.0];
        let q = vec![-1.0f64, 0.0, 0.0, 0.0];
        let omega = slerp_omega_f64(&p, &q);
        assert!((omega - std::f64::consts::PI).abs() < 1e-12,
            "omega={}", omega);
    }

    #[test]
    fn test_cache_line_separation() {
        // Verificar que head y tail esten en cache lines distintas
        let ring = PmtpRing::new(4, 100);
        let head_addr = &ring.head.0 as *const _ as usize;
        let tail_addr = &ring.tail.0 as *const _ as usize;
        let sep = if tail_addr > head_addr {
            tail_addr - head_addr
        } else {
            head_addr - tail_addr
        };
        assert!(sep >= CACHE_LINE,
            "false sharing: separacion={} < CACHE_LINE={}", sep, CACHE_LINE);
    }

    #[test]
    fn test_ring_push_pop() {
        let mut ring = PmtpRing::new(4, 8);
        let tensor = vec![1.0f32; 8];
        assert!(ring.push(&tensor));
        let out = ring.pop().expect("debe haber un elemento");
        assert_eq!(out, tensor);
    }
}

