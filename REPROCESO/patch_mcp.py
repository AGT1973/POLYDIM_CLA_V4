import os

with open('polydim_v63_monolito.py', 'r', encoding='utf-8') as f:
    content = f.read()

MCP_CODE = '''
# ------------------------------------------------------------------------------
# PIECE 4: NATIVE MCP SERVER (MODEL CONTEXT PROTOCOL)
# ------------------------------------------------------------------------------
import json
import base64

class POLYDIM_MCP_Server:
    """
    Servidor MCP (Model Context Protocol) embebido.
    Permite que otras IAs descubran e invoquen funciones nativas ND sin salir de la arquitectura.
    """
    
    @staticmethod
    def get_capabilities():
        return {
            "tools": [
                {
                    "name": "polydim_slerp",
                    "description": "Realiza interpolación SLERP en S^{D-1}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "q1_base64": {"type": "string", "description": "Tensor q1 en base64 (float32)"},
                            "q2_base64": {"type": "string", "description": "Tensor q2 en base64 (float32)"},
                            "t": {"type": "number", "description": "Parámetro de interpolación [0,1]"}
                        },
                        "required": ["q1_base64", "q2_base64", "t"]
                    }
                }
            ]
        }
        
    @staticmethod
    def invoke_tool(name: str, args: dict):
        if name == "polydim_slerp":
            q1_bytes = base64.b64decode(args["q1_base64"])
            q2_bytes = base64.b64decode(args["q2_base64"])
            
            q1 = np.frombuffer(q1_bytes, dtype=np.float32)
            q2 = np.frombuffer(q2_bytes, dtype=np.float32)
            
            # Convierte a jax
            q1_j = jnp.array(q1)
            q2_j = jnp.array(q2)
            
            res = GeodesicKernels.slerp(q1_j, q2_j, args["t"])
            res_np = np.array(res)
            
            return {
                "result_base64": base64.b64encode(res_np.tobytes()).decode('utf-8'),
                "shape": list(res_np.shape)
            }
        raise ValueError(f"Unknown tool: {name}")

'''

if "PIECE 4: NATIVE MCP SERVER" not in content:
    idx = content.find("# SUITE DE VERIFICACIÓN AUTÓNOMA EN CALIENTE")
    new_content = content[:idx] + MCP_CODE + "\\n\\n" + content[idx:]
    with open('polydim_v63_monolito.py', 'w', encoding='utf-8') as f:
        f.write(new_content)

# Append to consolidado
with open('codigo_consolidado_v63.txt', 'a', encoding='utf-8') as f:
    f.write(f"""
================================================================================
# ARCHIVO: polydim\\mcp_server.py
================================================================================
{MCP_CODE}
""")
