from libc.stdint cimport *

cdef extern:
    uint16_t mul8s_1L12(uint8_t A, uint8_t B)


cpdef int mul(int a,int b):
    return mul8s_1L12(a,b)

