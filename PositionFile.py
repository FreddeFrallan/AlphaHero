from ctypes import *

class POSITION(Structure):
    _fields_ = [
        ('current_position', c_uint64),
        ('mask', c_uint64),
        ('moves', c_uint),
        ('score', c_float)
    ]
