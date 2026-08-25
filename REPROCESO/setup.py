from setuptools import setup, Extension
setup(name="test_cpp", ext_modules=[Extension("test_cpp", ["test_cpp.cpp"])])
