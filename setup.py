from distutils.core import setup, Extension

compiler_args = ["-std=c++17", 
                 "-fPIC", 
                 "-DBOOST_LOG_DYN_LINK", 
                 "-shared",
                 "-fopenmp"]
extension = Extension(
    "tnml", ["tnml.cc"], extra_compile_args=compiler_args,
    library_dirs=["/home/ahthomas/Research/itensor/lib",
                  "/usr/local/opt/openblas/lib",
                  "/usr/lib/x86_64-linux-gnu/"],
    libraries=["itensor", "openblas", "boost_log", "boost_log_setup", "gomp"],
    include_dirs=["/home/ahthomas/Research/itensor",
                  "/usr/local/opt/openblas/include",
                  "/usr/include/boost"])
setup(name="tnml", version="1.0", ext_modules=[extension])
