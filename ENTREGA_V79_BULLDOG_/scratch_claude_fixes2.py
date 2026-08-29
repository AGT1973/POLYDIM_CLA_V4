import os

with open("E:/POLYDIM_EINSOF/ENTREGA_V79_BULLDOG_/polydim_v79_monolito.py", "r", encoding="utf-8") as f:
    code = f.read()

# Make sure PMTP_BODY_FMT is defined at the module level
if "PMTP_BODY_FMT = " not in code:
    print("Adding PMTP_BODY_FMT")
    # Add it right before PMTP_MAGIC
    code = code.replace('PMTP_MAGIC = b"PMTP"', 'PMTP_BODY_FMT = "<4s B B B B Q 16s 16s Q Q d Q Q Q Q Q"\nPMTP_HEADER_FMT = PMTP_BODY_FMT + " 16s"\nPMTP_MAGIC = b"PMTP"')
else:
    print("PMTP_BODY_FMT already exists")

# Wait, if I wiped PMTP_MAGIC too?
if "PMTP_MAGIC" not in code:
    print("Wait, PMTP_MAGIC is missing!")

with open("E:/POLYDIM_EINSOF/ENTREGA_V79_BULLDOG_/polydim_v79_monolito.py", "w", encoding="utf-8") as f:
    f.write(code)
