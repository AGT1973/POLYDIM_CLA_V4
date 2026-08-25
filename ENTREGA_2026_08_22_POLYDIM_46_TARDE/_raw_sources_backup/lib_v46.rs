use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyReadonlyArray1, PyArray1, PyReadwriteArray1, IntoPyArray};
use std::sync::atomic::{AtomicU64, Ordering, compiler_fence};
use std::sync::Arc;
use std::ptr;

// Axioma Cero: epsilon runtime portable via Rust native
pub fn machine_epsilon_f64() -> f64 {
    f64::EPSILON
}

// 128-byte alignment to prevent false sharing on modern CPUs
#[repr(C)]
#[repr(align(128))]
struct CacheAligned<T>(T);

// Bounded MPMC Ring Buffer Lock-Free (Vyukov algorithm)
pub struct PmtpRing {
    capacity: usize,
    capacity_mask: u64,
    dim: usize,
    buffer: Vec<f64>, // RAII: Se libera solo al hacer Drop, cero fugas
    sequences: Vec<CacheAligned<AtomicU64>>,
    head: CacheAligned<AtomicU64>,
    tail: CacheAligned<AtomicU64>,
}

unsafe impl Sync for PmtpRing {}
unsafe impl Send for PmtpRing {}

impl PmtpRing {
    pub fn new(capacity: usize, dim: usize) -> Self {
        let cap_power_2 = if capacity.is_power_of_two() { capacity } else { capacity.next_power_of_two() };
        let total_elements = cap_power_2.checked_mul(dim).expect("Overflow");
        
        let buffer = vec![0.0f64; total_elements]; // Cero with_capacity + resize ineficiente (Fix #18)
        let mut sequences = Vec::with_capacity(cap_power_2);
        for i in 0..cap_power_2 {
            sequences.push(CacheAligned(AtomicU64::new(i as u64)));
        }

        PmtpRing {
            capacity: cap_power_2,
            capacity_mask: (cap_power_2 - 1) as u64,
            dim,
            buffer,
            sequences,
            head: CacheAligned(AtomicU64::new(0)),
            tail: CacheAligned(AtomicU64::new(0)),
        }
    }

    #[inline]
    pub fn push(&self, tensor: &[f64]) -> Result<(), String> {
        if tensor.len() != self.dim {
            return Err(format!("Dimension mismatch: expected {}, got {}", self.dim, tensor.len()));
        }

        let mut head = self.head.0.load(Ordering::Relaxed);
        loop {
            let slot = (head & self.capacity_mask) as usize;
            let seq = self.sequences[slot].0.load(Ordering::Acquire);
            let diff = seq as i64 - head as i64;

            if diff == 0 {
                match self.head.0.compare_exchange_weak(
                    head,
                    head + 1,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        unsafe {
                            // Using as_ptr() on Vec is fine since we aren't mutating the capacity
                            let buffer_ptr = self.buffer.as_ptr() as *mut f64;
                            let slot_ptr = buffer_ptr.add(slot * self.dim);
                            ptr::copy_nonoverlapping(tensor.as_ptr(), slot_ptr, self.dim);
                        }
                        
                        std::sync::atomic::fence(Ordering::Release);
                        self.sequences[slot].0.store(head + 1, Ordering::Release);
                        return Ok(());
                    }
                    Err(actual) => head = actual,
                }
            } else if diff < 0 {
                return Err("Ring buffer full".to_string());
            } else {
                head = self.head.0.load(Ordering::Relaxed);
            }
        }
    }

    #[inline]
    pub fn pop(&self, out: &mut [f64]) -> Result<(), String> {
        if out.len() != self.dim {
            return Err(format!("Dimension mismatch: expected {}, got {}", self.dim, out.len()));
        }

        let mut tail = self.tail.0.load(Ordering::Relaxed);
        loop {
            let slot = (tail & self.capacity_mask) as usize;
            let seq = self.sequences[slot].0.load(Ordering::Acquire);
            let diff = seq as i64 - (tail + 1) as i64;

            if diff == 0 {
                match self.tail.0.compare_exchange_weak(
                    tail,
                    tail + 1,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        std::sync::atomic::fence(Ordering::Acquire);
                        unsafe {
                            let buffer_ptr = self.buffer.as_ptr() as *mut f64;
                            let slot_ptr = buffer_ptr.add(slot * self.dim);
                            ptr::copy_nonoverlapping(slot_ptr, out.as_mut_ptr(), self.dim);
                        }
                        std::sync::atomic::fence(Ordering::Release);
                        self.sequences[slot].0.store(tail + self.capacity as u64, Ordering::Release);
                        return Ok(());
                    }
                    Err(actual) => tail = actual,
                }
            } else if diff < 0 {
                return Err("Ring buffer empty".to_string());
            } else {
                tail = self.tail.0.load(Ordering::Relaxed);
            }
        }
    }
}

pub mod hmac {
    use blake2::{Blake2b512, Digest, KeyedBlake2b512};

    pub fn sign(payload: &[f64], epoch: u64, seq: u64, key: &[u8]) -> [u8; 64] {
        // Keyed BLAKE2b nativo (1 solo paso, cero doble asignación)
        // Truncar o pad la llave a 64 bytes estándar de BLAKE2b
        let mut padded_key = [0u8; 64];
        let key_len = std::cmp::min(key.len(), 64);
        padded_key[..key_len].copy_from_slice(&key[..key_len]);

        let mut hasher = KeyedBlake2b512::new_with_key(&padded_key);
        hasher.update(b"POLYDIM_PMTP_V42_1");
        hasher.update(epoch.to_le_bytes());
        hasher.update(seq.to_le_bytes());

        let payload_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                payload.as_ptr() as *const u8,
                payload.len() * std::mem::size_of::<f64>(),
            )
        };
        hasher.update(payload_bytes);
        
        let mut tag = [0u8; 64];
        tag.copy_from_slice(&hasher.finalize());
        tag
    }

    #[inline(never)]
    pub fn verify(payload: &[f64], epoch: u64, seq: u64, key: &[u8], tag: &[u8; 64]) -> bool {
        let expected = sign(payload, epoch, seq, key);
        let mut diff: u8 = 0;
        for i in 0..64 {
            diff |= expected[i] ^ tag[i];
        }
        diff == 0
    }
}

#[pyclass]
pub struct PyPmtpRing {
    inner: Arc<PmtpRing>,
}

#[pymethods]
impl PyPmtpRing {
    #[new]
    fn new(capacity: usize, dim: usize) -> Self {
        PyPmtpRing {
            inner: Arc::new(PmtpRing::new(capacity, dim)),
        }
    }

    fn push<'py>(&self, py: Python<'py>, tensor: PyReadonlyArray1<f64>) -> PyResult<bool> {
        let slice = tensor.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = Arc::clone(&self.inner);
        py.allow_threads(move || {
            inner.push(slice)
        }).map_err(|e| PyValueError::new_err(e))?;
        Ok(true)
    }

    fn pop_into<'py>(&self, py: Python<'py>, mut out: PyReadwriteArray1<f64>) -> PyResult<bool> {
        let slice = out.as_slice_mut().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let inner = Arc::clone(&self.inner);
        py.allow_threads(move || {
            inner.pop(slice)
        }).map_err(|e| PyValueError::new_err(e))?;
        Ok(true)
    }

    fn pop<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut out = vec![0.0f64; self.inner.dim];
        let inner = Arc::clone(&self.inner);
        py.allow_threads(move || {
            inner.pop(&mut out)
        }).map_err(|e| PyValueError::new_err(e))?;
        Ok(out.into_pyarray(py))
    }
}

#[pyfunction]
fn get_machine_epsilon_f64() -> f64 {
    machine_epsilon_f64()
}

#[pymodule]
fn einsof_rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPmtpRing>()?;
    m.add_function(wrap_pyfunction!(get_machine_epsilon_f64, m)?)?;
    Ok(())
}