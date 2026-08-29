// ============================================================================
// POLYDIM PMTP - API FFI Principal (C ABI para Python/ctypes)
// ============================================================================
// Este es el punto de entrada de la DLL. Expone funciones extern "C" que
// Python llama directamente vía ctypes. El Runtime de Tokio se inicializa
// una sola vez y todas las operaciones de red ocurren en background threads
// de Rust, liberando completamente el GIL de Python.
// ============================================================================

mod dlpack_bridge;
mod gossip;
mod crypto;
mod wire;
mod transport;

use dlpack::ManagedTensor;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::sync::{Arc, OnceLock};
use std::sync::atomic::Ordering;
use tokio::runtime::Runtime;
use gossip::PmtpNode;

static RUNTIME: OnceLock<Runtime> = OnceLock::new();
static NODE: OnceLock<Arc<PmtpNode>> = OnceLock::new();

/// Inicializa el nodo PMTP. Levanta el listener TCP y el epoch countdown.
/// Retorna 0 en éxito, 1 si ya estaba inicializado.
#[no_mangle]
pub extern "C" fn pmtp_init_node(port: u16) -> i32 {
    let rt = match Runtime::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[PMTP-ERROR] Fallo creando runtime Tokio: {}", e);
            return -1;
        }
    };

    let node = Arc::new(PmtpNode::new(port));
    let received = node.received_tensors.clone();
    let epoch = node.epoch.clone();
    let peers = node.peers.clone();

    rt.spawn(async move {
        PmtpNode::start_listening(port, received).await;
    });

    rt.spawn(async move {
        PmtpNode::epoch_countdown_loop(epoch, peers).await;
    });

    if NODE.set(node).is_err() {
        return 1;
    }
    if RUNTIME.set(rt).is_err() {
        return 1;
    }

    println!("[PMTP] Nodo inicializado en puerto {}", port);
    0
}

/// Envía un tensor DLPack a un peer remoto.
/// El tensor se lee directamente desde la memoria de JAX (Zero-Copy hasta el punto de serialización).
#[no_mangle]
pub extern "C" fn pmtp_send_tensor_dlpack(
    tensor: *mut ManagedTensor,
    host: *const c_char,
    port: u16,
) -> i32 {
    if tensor.is_null() || host.is_null() {
        return -1;
    }

    let rt = match RUNTIME.get() {
        Some(r) => r,
        None => {
            eprintln!("[PMTP-ERROR] Runtime no inicializado. Llama pmtp_init_node primero.");
            return -3;
        }
    };

    let node = match NODE.get() {
        Some(n) => n,
        None => {
            eprintln!("[PMTP-ERROR] Nodo no inicializado.");
            return -3;
        }
    };

    // 1. Extraer datos del tensor DLPack
    let data = match dlpack_bridge::read_dlpack_tensor(tensor) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[PMTP-ERROR] DLPack read: {}", e);
            return -2;
        }
    };

    // 2. Extraer shape del DLPack
    let managed = unsafe { &*tensor };
    let dltensor = &managed.dl_tensor;
    let shape: Vec<i64> = (0..dltensor.ndim)
        .map(|i| unsafe { *dltensor.shape.offset(i as isize) })
        .collect();

    // 3. Obtener host como string
    let host_str = unsafe {
        match CStr::from_ptr(host).to_str() {
            Ok(s) => s.to_owned(),
            Err(_) => {
                eprintln!("[PMTP-ERROR] Host inválido (no UTF-8)");
                return -4;
            }
        }
    };

    // 4. Obtener epoch y seq actuales
    let epoch = node.epoch.load(Ordering::SeqCst);
    let seq = node.seq.fetch_add(1, Ordering::SeqCst);

    // 5. Despachar envío asíncrono (no bloquea Python)
    rt.spawn(async move {
        match PmtpNode::send_to_peer(&host_str, port, &data, shape, epoch, seq).await {
            Ok(()) => {}
            Err(e) => eprintln!("[PMTP-TX-ERROR] {}", e),
        }
    });

    0
}

/// Envía un tensor raw (puntero a f64 + shape) a un peer remoto.
/// Útil cuando no se usa DLPack (ej. testing directo).
#[no_mangle]
pub extern "C" fn pmtp_send_raw(
    data_ptr: *const f64,
    num_elements: usize,
    shape_ptr: *const i64,
    ndim: usize,
    host: *const c_char,
    port: u16,
) -> i32 {
    if data_ptr.is_null() || shape_ptr.is_null() || host.is_null() {
        return -1;
    }

    let rt = match RUNTIME.get() {
        Some(r) => r,
        None => return -3,
    };

    let node = match NODE.get() {
        Some(n) => n,
        None => return -3,
    };

    let data = unsafe { std::slice::from_raw_parts(data_ptr, num_elements) }.to_vec();
    let shape = unsafe { std::slice::from_raw_parts(shape_ptr, ndim) }.to_vec();
    let host_str = unsafe { CStr::from_ptr(host).to_str().unwrap_or("").to_owned() };
    let epoch = node.epoch.load(Ordering::SeqCst);
    let seq = node.seq.fetch_add(1, Ordering::SeqCst);

    rt.spawn(async move {
        match PmtpNode::send_to_peer(&host_str, port, &data, shape, epoch, seq).await {
            Ok(()) => {}
            Err(e) => eprintln!("[PMTP-TX-ERROR] {}", e),
        }
    });

    0
}

/// Retorna la época actual del nodo.
#[no_mangle]
pub extern "C" fn pmtp_get_epoch() -> u64 {
    NODE.get().map_or(0, |n| n.epoch.load(Ordering::SeqCst))
}

/// Retorna la cantidad de tensores recibidos en la cola.
#[no_mangle]
pub extern "C" fn pmtp_recv_queue_len() -> i32 {
    let rt = match RUNTIME.get() {
        Some(r) => r,
        None => return -1,
    };
    let node = match NODE.get() {
        Some(n) => n,
        None => return -1,
    };
    let recv = node.received_tensors.clone();
    rt.block_on(async { recv.lock().await.len() as i32 })
}
