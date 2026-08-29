import re

# FIX C++ KERNEL
cpp_file = 'E:/POLYDIM_EINSOF/ENTREGA_V79_BULLDOG_/kernel_cpp_v79_fixed.cpp'
with open(cpp_file, 'r', encoding='utf-8') as f:
    cpp_code = f.read()

# Swap memset and overlap check
cpp_pattern = r"(std::memset\(out, 0, total_bytes\);)\s*(// Verificar alias global ANTES del loop.*?\n\s*if \(polydim::check_byte_overlap\(x, out, total_bytes\)\).*?\n\s*if \(polydim::check_byte_overlap\(v, out, total_bytes\)\).*?\n)"
cpp_replacement = r"\2\n    \1"
cpp_code = re.sub(cpp_pattern, cpp_replacement, cpp_code, flags=re.DOTALL)
with open(cpp_file, 'w', encoding='utf-8') as f:
    f.write(cpp_code)


# FIX PYTHON MONOLITH (PMTPNetworkLayer)
py_file = 'E:/POLYDIM_EINSOF/ENTREGA_V79_BULLDOG_/polydim_v79_monolito_fixed.py'
with open(py_file, 'r', encoding='utf-8') as f:
    py_code = f.read()

# Fix semaphore deadlock in _listen_loop:
py_code = py_code.replace(
    'self._active_connections.acquire()',
    'if not self._active_connections.acquire(timeout=5.0): continue'
)

# Fix send_tensor deadlock: inject socket.settimeout
py_code = re.sub(
    r'(s\.connect\(\(host, port\)\))',
    r'\1\n                s.settimeout(10.0)',
    py_code
)

# Fix slowloris: add global timeout in _recv_exact
py_code = re.sub(
    r'(def _recv_exact.*?deadline = time\.time\(\) \+ timeout\n.*?while len\(data\) < num_bytes:)',
    r'\1\n            if time.time() > deadline: raise TimeoutError("Slowloris timeout")\n            conn.settimeout(deadline - time.time())',
    py_code, flags=re.DOTALL
)

with open(py_file, 'w', encoding='utf-8') as f:
    f.write(py_code)

print('Sabueso patches applied to C++ and Python.')
