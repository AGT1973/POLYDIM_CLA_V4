"""
POLYDIM V48.1-FIXED: MODO NOCTURNO BUG HUNTER
Usa OpenRouter API directamente con modelos actualizados.
Envía el código a múltiples LLMs para auditoría cruzada.
"""
import os
import json
import urllib.request
import urllib.error
import sys
import time

env_path = r'C:\Users\eluithi\.gemini\config\.env'
with open(env_path, 'r', encoding='utf-8') as f:
    for line in f:
        if '=' in line and not line.startswith('#'):
            k, v = line.strip().split('=', 1)
            os.environ[k] = v

OPENROUTER_KEY = os.environ.get('OPENROUTER_API_KEY')
CODE_PATH = r'E:\POLYDIM_EINSOF\ENTREGA_2026_08_22_POLYDIM_48_MAÑANA\codigo_consolidado_v48_manana_fixed.txt'

with open(CODE_PATH, 'r', encoding='utf-8') as f:
    code = f.read()

AUDIT_PROMPT = f"""You are a strict code auditor (Red Team / Bulldog Critic Mode). 
POLYDIM V48.1-FIXED just passed 100 destructive rounds at D=10^6 with zero NaN and norm drift <= 7.77e-16.
However, we found a subnormal bug (slerp returned inf for inputs ~1e-300, now fixed).

I need you to analyze these UNTESTED attack vectors on the following code:

1. CONCURRENCY: The C++ uses #pragma omp parallel for but q2_neg_buf[4096] is a stack buffer. If dim <= 4096, 
   the parallel for loop writes to it. Each iteration writes q2_use[i] = -q2[i] at index i. 
   Since each thread writes a DIFFERENT index, is this actually safe? Or is there a subtle cache-line false sharing issue?

2. HEAP FRAGMENTATION in Rust: Vec<f64> temporaries for D=10^6 = 8MB per call. Impact on hot loops?

3. CAYLEY RETRACTION: Formula is x - Y @ inv(I + 0.5*J*Gram) @ J @ Y^T @ x. 
   This is NOT Wen & Yin 2013. For K>1 on Stiefel St(K,D), does it preserve column orthogonality?

4. JAX jnp.where GRADIENT: slerp_nd uses jnp.where for antipodal branching. 
   Is the gradient correct through jnp.where? Does JAX propagate gradients through both branches?

5. C++ ALIASING: polydim_naive_projected_gradient_step checks overlap between out_X_next and X/Grad,
   but uses pointer comparison (out_X_next > X && out_X_next < X + total). 
   This fails if out_X_next is BEFORE X but overlaps from below. Is the check correct?

Respond with a numbered list of REAL bugs found. No false positives.

CODE:
{code}
"""

MODELS = [
    "deepseek/deepseek-r1",
    "google/gemma-3-27b-it",
    "meta-llama/llama-4-maverick",
]

results = {}
for model in MODELS:
    print(f"\n{'='*60}")
    print(f"QUERYING: {model}")
    print(f"{'='*60}")
    
    url = 'https://openrouter.ai/api/v1/chat/completions'
    headers = {
        'Authorization': f'Bearer {OPENROUTER_KEY}',
        'Content-Type': 'application/json',
        'HTTP-Referer': 'https://polydim.einsof.com',
        'X-Title': 'POLYDIM_Antigravity'
    }
    data = {
        'model': model,
        'messages': [{'role': 'user', 'content': AUDIT_PROMPT}],
        'max_tokens': 4096
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode('utf-8'))
            content = result['choices'][0]['message']['content']
            results[model] = content
            print(content[:2000])
            print(f"\n[...truncated, total len={len(content)}]")
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        results[model] = f"ERROR {e.code}: {err}"
        print(f"ERROR {e.code}: {err[:500]}")
    except Exception as e:
        results[model] = f"EXCEPTION: {e}"
        print(f"EXCEPTION: {e}")
    
    time.sleep(1)

# Save all results
output_path = r'E:\POLYDIM_EINSOF\ENTREGA_2026_08_22_POLYDIM_48_MAÑANA\audit_nocturno_multimodel.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    for model, content in results.items():
        f.write(f"\n{'='*60}\n")
        f.write(f"MODEL: {model}\n")
        f.write(f"{'='*60}\n")
        f.write(content)
        f.write("\n\n")

print(f"\nResultados guardados en: {output_path}")
