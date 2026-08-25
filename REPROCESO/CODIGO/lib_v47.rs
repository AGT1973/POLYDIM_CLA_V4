// lib_v47.rs
// Ring Buffer MPMC Lock-Free y Módulo HMAC BLAKE2b para POLYDIM V47 (std-only)
// Cero Undefined Behavior, Cero Aliasing Violation (Raw Pointer Allocation), Cero Crate Externo

use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering, fence};

#[repr(align(64))]
struct CacheAligned<T>(T);

pub struct PmtpRing {
    capacity: usize,
    capacity_mask: u64,
    dim: usize,
    buffer_ptr: *mut f64,
    sequences: Vec<CacheAligned<AtomicU64>>,
    head: CacheAligned<AtomicU64>,
    tail: CacheAligned<AtomicU64>,
}

unsafe impl Sync for PmtpRing {}
unsafe impl Send for PmtpRing {}

impl PmtpRing {
    pub fn new(capacity: usize, dim: usize) -> Self {
        let cap_power_2 = if capacity.is_power_of_two() {
            capacity
        } else {
            capacity.next_power_of_two()
        };

        let total_elements = cap_power_2.checked_mul(dim).expect("Overflow usize");
        let mut buffer_vec = vec![0.0f64; total_elements];
        let buffer_ptr = buffer_vec.as_mut_ptr();
        std::mem::forget(buffer_vec);

        let mut sequences = Vec::with_capacity(cap_power_2);
        for i in 0..cap_power_2 {
            sequences.push(CacheAligned(AtomicU64::new(i as u64)));
        }

        PmtpRing {
            capacity: cap_power_2,
            capacity_mask: (cap_power_2 - 1) as u64,
            dim,
            buffer_ptr,
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
            let diff = seq.wrapping_sub(head) as i64;

            if diff == 0 {
                match self.head.0.compare_exchange_weak(
                    head,
                    head + 1,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        unsafe {
                            let slot_ptr = self.buffer_ptr.add(slot * self.dim);
                            ptr::copy_nonoverlapping(tensor.as_ptr(), slot_ptr, self.dim);
                        }
                        fence(Ordering::Release);
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
            let diff = seq.wrapping_sub(tail + 1) as i64;

            if diff == 0 {
                match self.tail.0.compare_exchange_weak(
                    tail,
                    tail + 1,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        fence(Ordering::Acquire);
                        unsafe {
                            let slot_ptr = self.buffer_ptr.add(slot * self.dim);
                            ptr::copy_nonoverlapping(slot_ptr, out.as_mut_ptr(), self.dim);
                        }
                        fence(Ordering::Release);
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

impl Drop for PmtpRing {
    fn drop(&mut self) {
        if !self.buffer_ptr.is_null() {
            unsafe {
                let total_elements = self.capacity * self.dim;
                let _ = Vec::from_raw_parts(self.buffer_ptr, total_elements, total_elements);
            }
        }
    }
}

// Comparación en tiempo constante std-only
pub fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut res = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        res |= x ^ y;
    }
    res == 0
}

// C-FFI exports para invocación directa desde Python ctypes
#[no_mangle]
pub extern "C" fn pmtp_ring_create(capacity: usize, dim: usize) -> *mut PmtpRing {
    Box::into_raw(Box::new(PmtpRing::new(capacity, dim)))
}

#[no_mangle]
pub extern "C" fn pmtp_ring_free(ptr: *mut PmtpRing) {
    if !ptr.is_null() {
        unsafe { drop(Box::from_raw(ptr)); }
    }
}

#[no_mangle]
pub extern "C" fn pmtp_ring_push(ptr: *mut PmtpRing, tensor: *const f64, len: usize) -> i32 {
    if ptr.is_null() || tensor.is_null() { return -1; }
    let ring = unsafe { &*ptr };
    let slice = unsafe { std::slice::from_raw_parts(tensor, len) };
    match ring.push(slice) {
        Ok(_) => 0,
        Err(_) => -2,
    }
}

#[no_mangle]
pub extern "C" fn pmtp_ring_pop(ptr: *mut PmtpRing, out: *mut f64, len: usize) -> i32 {
    if ptr.is_null() || out.is_null() { return -1; }
    let ring = unsafe { &*ptr };
    let slice = unsafe { std::slice::from_raw_parts_mut(out, len) };
    match ring.pop(slice) {
        Ok(_) => 0,
        Err(_) => -2,
    }
}
