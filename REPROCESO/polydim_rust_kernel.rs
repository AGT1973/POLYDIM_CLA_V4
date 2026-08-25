
// POLYDIM V58 RUST FFI C-ABI KERNEL
#[repr(C)]
pub struct PMTPHeaderC {
    pub seq_word: u64,
    pub magic: u64,
    pub version: u32,
    pub dim: u32,
    pub dtype_code: u32,
    pub payload_bytes: u32,
    pub timestamp: u64,
    pub generation: u64,
    pub _reserved: [u8; 16],
}

#[no_mangle]
pub unsafe extern "C" fn polydim_rust_householder_reflect(
    x_ptr: *const f32,
    v_ptr: *const f32,
    out_ptr: *mut f32,
    dim: usize,
) -> i32 {
    if x_ptr.is_null() || v_ptr.is_null() || out_ptr.is_null() || dim == 0 {
        return -1;
    }
    let x = std::slice::from_raw_parts(x_ptr, dim);
    let v = std::slice::from_raw_parts(v_ptr, dim);
    let out = std::slice::from_raw_parts_mut(out_ptr, dim);

    let mut vv = 0.0f32;
    for i in 0..dim { vv += v[i] * v[i]; }

    if vv < 1e-15 {
        out.copy_from_slice(x);
        return 0;
    }

    let safe_norm = vv.sqrt().max(1e-15);
    let mut dot = 0.0f32;
    for i in 0..dim { dot += (v[i] / safe_norm) * x[i]; }

    let two_dot = 2.0 * dot;
    for i in 0..dim {
        out[i] = x[i] - two_dot * (v[i] / safe_norm);
    }
    0
}

use std::alloc::{alloc, dealloc, Layout};
use std::ptr;

#[repr(C)]
pub struct AlignedTensor {
    pub data: *mut f64,
    pub len: usize,
    pub capacity: usize,
}

#[no_mangle]
pub extern "C" fn polydim_alloc_aligned(len: usize) -> AlignedTensor {
    let align = 64;
    let size = len.checked_mul(8).expect("Overflow calculando size en bytes");
    let size_padded = (size + align - 1) & !(align - 1);
    let layout = Layout::from_size_align(size_padded, align).unwrap();
    
    let ptr = unsafe { alloc(layout) as *mut f64 };
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    
    unsafe { ptr::write_bytes(ptr, 0, len) };

    AlignedTensor {
        data: ptr,
        len,
        capacity: size_padded / 8,
    }
}

#[no_mangle]
pub extern "C" fn polydim_free_aligned(tensor: AlignedTensor) {
    if tensor.data.is_null() { return; }
    let align = 64;
    let size_padded = tensor.capacity * 8;
    let layout = Layout::from_size_align(size_padded, align).unwrap();
    unsafe {
        dealloc(tensor.data as *mut u8, layout);
    }
}
