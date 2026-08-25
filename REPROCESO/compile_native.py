import os
import sys
import subprocess
from distutils.ccompiler import new_compiler
import distutils.sysconfig
import distutils.msvc9compiler

try:
    cpp_compiler = new_compiler()
    objs = cpp_compiler.compile(['test_cpp.cpp'])
    cpp_compiler.link_shared_lib(objs, 'test_cpp')
    print("C++ DLL compiled successfully.")
except Exception as e:
    print(f"C++ Compilation failed: {e}")
