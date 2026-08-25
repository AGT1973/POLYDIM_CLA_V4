import os

with open('build_v63_pieces.py', 'r', encoding='utf-8') as f:
    content = f.read()

bridge = content.split("FFI_BRIDGE_CODE = '''")[1].split("'''")[0]
persist = content.split("PERSISTENCE_CODE = '''")[1].split("'''")[0]
net = content.split("NETWORK_CODE = '''")[1].split("'''")[0]

out = f"""
================================================================================
# ARCHIVO: polydim\\ffi_bridge.py
================================================================================
{bridge}

================================================================================
# ARCHIVO: polydim\\persistence.py
================================================================================
{persist}

================================================================================
# ARCHIVO: polydim\\network.py
================================================================================
{net}
"""

with open('codigo_consolidado_v63.txt', 'a', encoding='utf-8') as f:
    f.write(out)
