from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="gru_rcn_cpp",
      ext_modules=[cpp_extension.CppExtension("ballas2016.gru_rcn_cpp", ["ballas2016/gru_rcn.cpp"])],
      cmdclass={"build_ext": cpp_extension.BuildExtension})
