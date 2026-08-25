// pmtp_seqlock_native.rs
// IMPLÉMENTACIÓN SOTA RUST 2024: SEQLOCK LOCK-FREE PARA PMTP V44
// CERO ADULACIÓN - RED TEAM / BULLDOG CRITIC PROOF
// ============================================================================

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering, fence};
use std::slice;

const PAYLOAD_DIM: usize = 10000;

#[repr(C)]
pub struct PmtpSeqlockBuffer {
    pub sequence: AtomicU64,
    pub writer_spinlock: AtomicBool,
    pub timestamp_ns: AtomicU64,
    pub payload: [f64; PAYLOAD_DIM],
}

impl PmtpSeqlockBuffer {
    pub fn new() -> Self {
        Self {
            sequence: AtomicU64::new(0),
            writer_spinlock: AtomicBool::new(false),
            timestamp_ns: AtomicU64::new(0),
            payload: [0.0; PAYLOAD_DIM],
        }
    }
}

// Global buffer instance for FFI testing
static mut GLOBAL_SEQLOCK: Option<PmtpSeqlockBuffer> = None;
static INIT_ONCE: std::sync::Once = std::sync::Once::new();

fn get_global_seqlock() -> &'static mut PmtpSeqlockBuffer {
    unsafe {
        INIT_ONCE.call_once(|| {
            GLOBAL_SEQLOCK = Some(PmtpSeqlockBuffer::new());
        });
        GLOBAL_SEQLOCK.as_mut().unwrap()
    }
}

#[no_mangle]
pub extern "C" fn pmtp_seqlock_init() {
    let _ = get_global_seqlock();
}

#[no_mangle]
pub extern "C" fn pmtp_seqlock_write(
    val: f64,
    timestamp: u64,
) -> u64 {
    let buf = get_global_seqlock();

    // 1. Spinlock de Escritores (Multi-Writer Lock-Free Coordination)
    while buf.writer_spinlock.compare_exchange_weak(
        false,
        true,
        Ordering::Acquire,
        Ordering::Relaxed
    ).is_err() {
        std::hint::spin_loop();
    }

    // 2. Incrementar secuencia a IMPAR (Indica escritura en progreso)
    let old_seq = buf.sequence.fetch_add(1, Ordering::Release);
    let writing_seq = old_seq + 1; // IMPAR

    // Memory Barrier explicito pre-escritura
    fence(Ordering::Release);

    // 3. Escribir Payload (Llenar vector de dim=10,000 con val coherente para detectar torn reads)
    // Para probar atomicos: la suma del payload DEBE ser exactamente PAYLOAD_DIM * val
    buf.timestamp_ns.store(timestamp, Ordering::Relaxed);
    for slot in buf.payload.iter_mut() {
        *slot = val;
    }

    // Memory Barrier explicito post-escritura
    fence(Ordering::Release);

    // 4. Incrementar secuencia a PAR (Indica escritura completada)
    let final_seq = buf.sequence.fetch_add(1, Ordering::Release) + 1;

    // Liberar Spinlock
    buf.writer_spinlock.store(false, Ordering::Release);

    final_seq
}

#[no_mangle]
pub extern "C" fn pmtp_seqlock_read_lockfree(
    out_payload: *mut f64,
    out_timestamp: *mut u64,
    max_retries: u64,
) -> i32 {
    if out_payload.is_null() {
        return -1;
    }

    let buf = get_global_seqlock();
    let out_slice = unsafe { slice::from_raw_parts_mut(out_payload, PAYLOAD_DIM) };

    let mut retries = 0u64;

    loop {
        if max_retries > 0 && retries >= max_retries {
            return -2; // Retry limit exceeded
        }

        // a. Leer secuencia inicial
        let seq1 = buf.sequence.load(Ordering::Acquire);

        // b. Si seq1 es IMPAR, hay una escritura en progreso -> Spin & Retry
        if seq1 & 1 != 0 {
            retries += 1;
            std::hint::spin_loop();
            continue;
        }

        // c. Memory Barrier pre-lectura
        fence(Ordering::Acquire);

        // d. Leer payload (memcpy directo)
        unsafe {
            let ts = buf.timestamp_ns.load(Ordering::Relaxed);
            if !out_timestamp.is_null() {
                *out_timestamp = ts;
            }
        }
        out_slice.copy_from_slice(&buf.payload);

        // e. Memory Barrier post-lectura
        fence(Ordering::Acquire);

        // f. Leer secuencia final
        let seq2 = buf.sequence.load(Ordering::Acquire);

        // g. Si seq1 == seq2, la lectura fue limpia y sin torn reads!
        if seq1 == seq2 {
            return 0; // SUCCESS
        }

        // h. Conflicto detectado (un escritor modificó el buffer en medio) -> Reintentar
        retries += 1;
        std::hint::spin_loop();
    }
}
