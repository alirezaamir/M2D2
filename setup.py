from distutils.core import setup, Extension

extension = Extension(
    "tnml", ["tnml.cc"], extra_compile_args=["-std=c++17", "-fPIC", "-shared"],
    library_dirs=["/home/ahthomas/Research/itensor/lib",
                  "/usr/local/opt/openblas/lib"],
    libraries=["itensor", "openblas"],
    include_dirs=["/home/ahthomas/Research/itensor",
                  "/usr/local/opt/openblas/include"])
setup(name="tnml", version="1.0", ext_modules=[extension])