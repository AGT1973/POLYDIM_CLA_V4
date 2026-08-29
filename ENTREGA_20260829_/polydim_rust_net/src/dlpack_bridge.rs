use dlpack::ManagedTensor;

pub fn read_dlpack_tensor(tensor_ptr: *mut ManagedTensor) -> Result<Vec<f64>, String> {
    if tensor_ptr.is_null() {
        return Err("Puntero nulo recibido de JAX".to_string());
    }
    
    let managed_tensor = unsafe { &*tensor_ptr };
    let dltensor = &managed_tensor.dl_tensor;
    
    if dltensor.ndim == 0 {
        return Err("Tensor de 0 dimensiones".to_string());
    }
    
    let mut num_elements: usize = 1;
    for i in 0..dltensor.ndim {
        let dim = unsafe { *dltensor.shape.offset(i as isize) };
        num_elements *= dim as usize;
    }
        
    let data_ptr = dltensor.data as *const f64;
    let slice = unsafe { std::slice::from_raw_parts(data_ptr, num_elements) };
    
    Ok(slice.to_vec())
}
