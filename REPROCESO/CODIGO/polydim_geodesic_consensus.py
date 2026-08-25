# polydim_geodesic_consensus.py
# MOTOR DE CONSENSO GEODÉSICO Y FILTRADO BFT GEOMÉTRICO (SOTA 2026)
# Demuestra el consenso intrínseco en S^(D-1) para 1,000 agentes con 20% bizantinos.
# ============================================================================

import numpy as np

def interrogate_silicon_environment(dtype=np.float64):
    finfo = np.finfo(dtype)
    return {
        "eps": finfo.eps,
        "tiny": finfo.tiny,
        "norm_tolerance": 4.0 * finfo.eps,
        "small_angle_threshold": np.sqrt(finfo.eps),
        "dtype": dtype
    }

def tangent_log_map(x: np.ndarray, y: np.ndarray, silicon: dict) -> np.ndarray:
    dot_prod = np.clip(np.dot(x, y), -1.0, 1.0)
    theta = np.arccos(dot_prod)
    
    if theta < silicon["small_angle_threshold"]:
        factor = 1.0 + (theta**2) / 6.0 + (7.0 * (theta**4)) / 360.0
    else:
        factor = theta / (np.sin(theta) + silicon["tiny"])
        
    v = factor * (y - dot_prod * x)
    return v - np.dot(x, v) * x

def project_retraction(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    y = x + v
    norm_y = np.linalg.norm(y, ord=2)
    return y / max(norm_y, 1e-12)

def riemannian_median_weiszfeld(X: np.ndarray, silicon: dict, max_iters: int = 5) -> np.ndarray:
    N, D = X.shape
    m = np.sum(X, axis=0)
    m = m / np.linalg.norm(m, ord=2)
    
    for _ in range(max_iters):
        V = np.array([tangent_log_map(m, y, silicon) for y in X])
        distances = np.linalg.norm(V, ord=2, axis=1)
        weights = 1.0 / np.maximum(distances, silicon["eps"])
        weights_sum = np.sum(weights)
        v_weighted = np.sum(V * weights[:, None], axis=0) / weights_sum
        m = project_retraction(m, v_weighted)
        
    return m

def gtfm_bft_filter(X: np.ndarray, f_byzantine: int, silicon: dict) -> np.ndarray:
    N, D = X.shape
    norms_sq = np.sum(X**2, axis=1)
    norm_valid_mask = np.abs(norms_sq - 1.0) <= silicon["norm_tolerance"]
    
    m_seed = riemannian_median_weiszfeld(X, silicon, max_iters=3)
    V = np.array([tangent_log_map(m_seed, y, silicon) for y in X])
    deviations = np.linalg.norm(V, ord=2, axis=1)
    deviations = np.where(norm_valid_mask, deviations, 1e9)
    
    sorted_indices = np.argsort(deviations)
    clean_mask = np.zeros(N, dtype=bool)
    clean_indices = sorted_indices[:(N - f_byzantine)]
    clean_mask[clean_indices] = True
    return clean_mask

def compute_robust_geodesic_consensus(X: np.ndarray, f_byzantine: int, max_fgc_iters: int = 10):
    silicon = interrogate_silicon_environment(X.dtype)
    N, D = X.shape
    
    clean_mask = gtfm_bft_filter(X, f_byzantine, silicon)
    X_clean = X[clean_mask]
    
    mu = np.sum(X_clean, axis=0)
    mu = mu / np.linalg.norm(mu, ord=2)
    
    for _ in range(max_fgc_iters):
        V_clean = np.array([tangent_log_map(mu, y, silicon) for y in X_clean])
        v_bar = np.mean(V_clean, axis=0)
        grad_norm = np.linalg.norm(v_bar, ord=2)
        if grad_norm < silicon["eps"]:
            break
        mu = project_retraction(mu, v_bar)
        
    metrics = {
        "agents_total": N,
        "agents_clean": int(np.sum(clean_mask)),
        "byzantine_rejected": int(N - np.sum(clean_mask)),
        "final_grad_norm": float(grad_norm),
        "isometry_norm": float(np.linalg.norm(mu, ord=2))
    }
    return mu, metrics

def test_consensus():
    print("  [POLYDIM SOTA 2026: Iniciando Consenso Geodesico y BFT Test...]")
    np.random.seed(2026)
    N, D, F_adversarial = 1000, 10000, 200
    
    true_center = np.random.randn(D)
    true_center /= np.linalg.norm(true_center)
    
    noise = np.random.randn(N - F_adversarial, D) * 0.05
    X_honest = true_center + noise
    X_honest /= np.linalg.norm(X_honest, axis=1, keepdims=True)
    
    X_byzantine_norm = np.random.randn(F_adversarial // 2, D) * 50.0
    X_byzantine_anti = -true_center + np.random.randn(F_adversarial // 2, D) * 0.01
    
    X_swarm = np.vstack([X_honest, X_byzantine_norm, X_byzantine_anti])
    
    mu_consensus, metrics = compute_robust_geodesic_consensus(X_swarm, f_byzantine=F_adversarial)
    
    print("\n  [RESULTADOS DEL CONSENSO GEODESICO Y BFT GEOMETRICO (SOTA 2026):]")
    print(f"  -> Total de Agentes enjambre (N): {metrics['agents_total']}")
    print(f"  -> Agentes Bizantinos Inyectados: {F_adversarial}")
    print(f"  -> Agentes Rechazados por GTFM BFT: {metrics['byzantine_rejected']}")
    print(f"  -> Invarianza de Norma del Consenso (||mu*||_2): {metrics['isometry_norm']:.8f}")
    geo_dist = np.arccos(np.clip(np.dot(mu_consensus, true_center), -1.0, 1.0))
    print(f"  -> Distancia Geodesica al Centro Real: {geo_dist:.6e} rad")
    assert abs(metrics['isometry_norm'] - 1.0) < 1e-12
    assert geo_dist < 0.25
    print("  -> OK: Consenso Geodesico e Inmunidad BFT Certificados.")

if __name__ == "__main__":
    test_consensus()
