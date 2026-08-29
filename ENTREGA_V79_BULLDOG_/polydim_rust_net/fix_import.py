py_file = 'src/lib.rs'
with open(py_file, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('use dlpack::DLManagedTensor;', 'use dlpack::ManagedTensor;')
content = content.replace('*mut DLManagedTensor', '*mut ManagedTensor')

with open(py_file, 'w', encoding='utf-8') as f:
    f.write(content)
print('Fixed DLPack import in Rust core.')
