py_file = 'E:/POLYDIM_EINSOF/ENTREGA_V79_BULLDOG_/nightly_autonomous_runner.py'
with open(py_file, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('timeout=300', 'timeout=1200')

with open(py_file, 'w', encoding='utf-8') as f:
    f.write(content)
print('Timeout increased to 1200 seconds in nightly runner.')
