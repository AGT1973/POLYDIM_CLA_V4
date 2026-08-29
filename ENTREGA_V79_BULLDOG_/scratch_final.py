import os

path_cpp = "E:/POLYDIM_EINSOF/ENTREGA_V79_BULLDOG_/kernel_cpp_v79.cpp"
with open(path_cpp, "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("check_byte_overlap(vb, ob", "check_byte_overlap(v, out")
with open(path_cpp, "w", encoding="utf-8") as f:
    f.write(text)

with open(path_cpp + ".txt", "w", encoding="utf-8") as f:
    f.write(text)

path_py = "E:/POLYDIM_EINSOF/ENTREGA_V79_BULLDOG_/polydim_v79_monolito.py"
with open(path_py, "r", encoding="utf-8") as f:
    text = f.read()

# Fix PMTP pack 17 items issue.
import re
text = re.sub(r'\*padded_shape,\n\s*b"\\\\x00"\*16', r'*padded_shape,\n            b"\\x00"*16', text)
text = text.replace('b"\\\\x00"*16', 'b"\\x00"*16')

with open(path_py, "w", encoding="utf-8") as f:
    f.write(text)

