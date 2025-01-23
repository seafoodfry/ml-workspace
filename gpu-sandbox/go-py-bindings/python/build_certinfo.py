import cffi

ffi = cffi.FFI()

with open('cffi_certinfo.h') as f:
    ffi.cdef(f.read())

ffi.set_source("_certinfo",
    """
    #include "certinfo.h"
    """,
    libraries=['curl', 'certinfo'],
    library_dirs=['.'],
    include_dirs=['.'])

if __name__ == '__main__':
    ffi.compile()