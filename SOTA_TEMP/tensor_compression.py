import numpy as np
import time
import csv
import sys

def quantize_fp32_to_int8(tensor):
    min_val = np.min(tensor)
    max_val = np.max(tensor)
    if max_val == min_val:
        scale = 1.0
    else:
        scale = 255.0 / (max_val - min_val)
    quantized = np.round((tensor - min_val) * scale).astype(np.uint8)
    return quantized, min_val, scale

def dequantize_int8_to_fp32(quantized, min_val, scale):
    return (quantized.astype(np.float32) / scale) + min_val

def benchmark_compression(d_size=10000, n_samples=50):
    np.random.seed(42)
    # Simulate slightly correlated data
    base = np.random.randn(n_samples, d_size // 10).astype(np.float32)
    proj = np.random.randn(d_size // 10, d_size).astype(np.float32)
    tensors = (base @ proj)
    
    results = []
    original_size = tensors.nbytes
    
    # 1. Quantization INT8
    start = time.time()
    quantized_batch = []
    meta = []
    for t in tensors:
        q, m, s = quantize_fp32_to_int8(t)
        quantized_batch.append(q)
        meta.append((m,s))
    
    q_time = time.time() - start
    q_size = sum(q.nbytes for q in quantized_batch) + sum(sys.getsizeof(m) * 2 for m in meta)
    
    dequantized = np.array([dequantize_int8_to_fp32(q, m, s) for q, (m, s) in zip(quantized_batch, meta)])
    mse_q = np.mean((tensors - dequantized)**2)
    
    results.append({
        'method': 'INT8_Quantization',
        'compression_ratio': original_size / q_size,
        'mse': float(mse_q),
        'time_s': q_time
    })
    
    # 2. PCA using SVD (keep top K components)
    start = time.time()
    mean = np.mean(tensors, axis=0)
    centered = tensors - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    # Keep components that explain 90% variance
    variance_ratio = (S**2) / np.sum(S**2)
    cum_var = np.cumsum(variance_ratio)
    k = np.argmax(cum_var >= 0.90) + 1
    
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    
    pca_time = time.time() - start
    
    # Compressed representation: mean + components + weights
    # But usually PCA models are shared. We just send the transformed data (U_k * S_k)
    # and assume Vt_k is already known or sent once.
    # To be conservative, let's include both in size.
    transformed = U_k * S_k
    pca_size = transformed.nbytes + Vt_k.nbytes + mean.nbytes
    
    reconstructed = (transformed @ Vt_k) + mean
    mse_pca = np.mean((tensors - reconstructed)**2)
    
    results.append({
        'method': 'Adaptive_PCA',
        'compression_ratio': original_size / pca_size,
        'mse': float(mse_pca),
        'time_s': pca_time
    })
    
    with open('E:\\POLYDIM_EINSOF\\SOTA_TEMP\\compression_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'compression_ratio', 'mse', 'time_s'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
            
    for r in results:
        print(f"Method: {r['method']}")
        print(f"  Ratio: {r['compression_ratio']:.2f}x")
        print(f"  MSE: {r['mse']:.6f}")
        print(f"  Time: {r['time_s']:.4f}s\n")

if __name__ == "__main__":
    # Test on D=10,000 to be safe
    benchmark_compression(d_size=10000, n_samples=50)
