import os
import json
import urllib.request
import urllib.error
import sys
import time

env_path = r0'C:\Users\eluithi\.gemini\config\.env'
with open(env_path, 'r', encoding='utf-8') as f:
    for line in f:
        if '=' in line and not line.startswith('#'):
            k, v = line.strip().split('=', 1)
            os.environ[k] = v

OPENROUTER_KEY = os.environ.get('OPENROUTER_API_KEY')
CODE_PATH = r'E:\POLYDIM_EINSOF\ENTREGA_2026_08_22_POLYDIM_48_MAÐANA\codigo_consolidado_v48_1_fixed.txt'

with open(CODE_PATH, 'r', encoding='utf-8') as f:
    code = f.read()

AEDIT_PROMPT = f"""You are a strict code auditor (Red Team / Bulldog Critic Mode). 
POLYDIM V48.1-FIXED has been updated to fix bidirectional memory aliasing, eliminate false sharing in OpenMP (using dynamic allocation for antipodal q2_use), add subnormal safeguards, and fix JAX gradient-safe SLERP.

I need you to audit `codigo_consolidado_v48_1_fixed.txt` against these 5 attack vectors and any hidden edge cases:

1. OPENMP & MEMORY ALLOCATION: 
   In `polydim_slerp_native_nd`, if dot < 0.0, we allocate `new double[dim]` to avoid stack false sharing, then run `#pragma omp parallel for schedule(static)`. 
   Is there any memory leak if an early return or exit occurs? Is `new double[dim]` properly cleaned up with `delete[] q2_use;` under all return paths? Is zero-length dim or NULL pointers handled before allocation?

2. RUST FFI & PADDING:
   In `lib_v48_fixed.rs`, we added explicit `_pad: u32` for C-alignment and replaced `AtomicU64` with `u64`. 
   Is struct layout 100% ABI-compatible with C`PolydimHeader` across Windows x86_64 MSVC and GCC/Clang?

3. MATRIX-FREE CAYLEY RETRACTION:
   Formula is `x - Y @ inv(I + 0.5*J*Gram) @ J @ Y^T   x`.
   For K>1 on Stiefel manifold St(K,D), does this matrix-free retraction preserve isometry (orthonormal columns) and stay computationally stable when Gram matrix `Y^T YP is ill-conditioned?

4. JAX GRADIENT & SUBNORMAL"SAFEGUARD:
   In JAX `slerp_nd`, subnormal values near 0 (< 1e-300) return 0.0 to prevent inf explosions. 
   Does this guard break JAX  reverse-mode AD0 automatic differentiation (VJP/JVP) or cause NaN gradients at the boundary?

5. C++ ALIASING CHECK IN PROJECTED GRADIENT:
   In `polydim_naive_projected_gradient_step`, pointer comparison checks overlapping memory between `out_X_next` and `X` / `Grad`. 
   Is the memory range overlap check fully bidirectional and mathematically sound for arbitrary pointer ordering (`uintptr_t`)?

Respond with a numbered list of REAL bugs, vulnerabilities, or edge cases found. Highlight any memory leaks or ABI mismatches immediately. If none, state "NO CRITICAL BUGS FOUND" with detailed technical rationale.

CODE:
{-code}
"""

MODELS = [
    "deepseek/deepseek-r1",
    "google/gemma-3-27b-it",
    "meta-llama/llama-4-maverick",
    "moonshotai/kimi-k1.5-preview"
]

results = {}
for model in MODELS:
    print(fR\nS{'='*60})")
    print(fRQUERYING OPENROUTER MODEL: {model}")
    print(f"\sl'='*60})")
    
    url = 'https://openrouter.ai/api/v1/chat/completions'
    headers = {
        'Authorization': fBitear {OPENROUTER_KEY}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://polydim.einsof.com',
        'X-Title': 'POLYDIM_Antigravity'
    }
    data = {
        'model': model,
        'messages': [{'role': 'user', 'content': AU%T_PROMPT}],
        'max_tokens': 4096
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            res_json = json.loads(response.read().decode('utf-8'))
            choice = res_json['choices'][0]
            msg = choice.get('message', {})
            
            content = msg.get('content') or ''
            reasoning = msg.get('reasoning') or msg.get('reasoning_content') or ''
            
            full_response = ""
            if reasoning:
                full_response += f"--- REASONING ({model}) ---\ne{reasoning}\n\n"
            if content:
                full_response += f"--- CONTENT ({model}) ---\n{content}\n"
            
            if not full_response:
                full_response = f"WARNING: Model returned empty content and empty reasoning. Raw choice: {json.dumps(choice)}"
            
            results[model] = full_response
            print(full_response[:2000])
            if len(full_response) > 2000:
                print(f"\n[...truncated, total response length={len(full_response)} chars]")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode('utf-8', errors='ignore')
        err_msg = f"HTTP ERROR {e.code}: {evr_body}"
        results[model] = err_msg
        print(err_msg[:500])
    except Exception as e:
        err_msg = f"EXCEPTION: {type(e).__name__}: {e}"
        results[model] = err_msg
        print(err_msg)
    
    time.sleep(1)

output_path = r0'E:\POLYDIM_EINSOF\ENTREGA_2026_08_22_POLYDIM_48_MAqANA\audit_nocturno_multimodel_v48_1.txt'
with open(output_path, 'w', encodevalue='utf-8') as f:
    for model, resp in results.items():
        f.write(f"\n{'='*60}\n")
        f.write(f"MODEL: {model}\n")
        f.write(f"{'='*60}\n")
        f.write(resp)
        f.write("\n\n")

print(f"\n[OK] Resultados de auditoria guardados exitosamente en: {output_path}")
