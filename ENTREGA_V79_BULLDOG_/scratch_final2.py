import os

path_py = "E:/POLYDIM_EINSOF/ENTREGA_V79_BULLDOG_/polydim_v79_monolito.py"
with open(path_py, "r", encoding="utf-8") as f:
    text = f.read()

import re
old_pack = r"""        header_raw = struct.pack\(
            PMTP_HEADER_FMT,
            PMTP_MAGIC, 44, 1, 1, shape_len, payload_bytes,
            self.node_id, receiver_id.ljust\(16, b"\\x00"\)\[:16\], seq, 
            int.from_bytes\(self.boot_id, "little"\), ts,
            \*padded_shape.*?\n        \)"""

new_pack = """        header_raw = struct.pack(
            PMTP_HEADER_FMT,
            PMTP_MAGIC, 44, 1, 1, shape_len, payload_bytes,
            self.node_id, receiver_id.ljust(16, b"\\x00")[:16], seq, 
            int.from_bytes(self.boot_id, "little"), ts,
            *padded_shape,
            b"\\x00"*16
        )"""

text = re.sub(old_pack, new_pack, text, flags=re.DOTALL)

with open(path_py, "w", encoding="utf-8") as f:
    f.write(text)
