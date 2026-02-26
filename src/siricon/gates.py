"""
Unitary gate matrices as MLX arrays (complex64).

All single-qubit gates return (2, 2) arrays.
All two-qubit gates return (4, 4) arrays.
Parameterized gates are functions: float -> mx.array.
"""

import math
import mlx.core as mx
import numpy as np


# Fixed single-qubit gates

def I() -> mx.array:
    return mx.array([[1, 0], [0, 1]], dtype=mx.complex64)

def X() -> mx.array:
    return mx.array([[0, 1], [1, 0]], dtype=mx.complex64)

def Y() -> mx.array:
    return mx.array([[0, -1j], [1j, 0]], dtype=mx.complex64)

def Z() -> mx.array:
    return mx.array([[1, 0], [0, -1]], dtype=mx.complex64)

def H() -> mx.array:
    s = 1.0 / math.sqrt(2)
    return mx.array([[s, s], [s, -s]], dtype=mx.complex64)

def S() -> mx.array:
    return mx.array([[1, 0], [0, 1j]], dtype=mx.complex64)

def T() -> mx.array:
    return mx.array([[1, 0], [0, complex(math.cos(math.pi/4), math.sin(math.pi/4))]], dtype=mx.complex64)


# Parameterized single-qubit rotations

def RX(theta: float) -> mx.array:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return mx.array([[c, -1j * s], [-1j * s, c]], dtype=mx.complex64)

def RY(theta: float) -> mx.array:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return mx.array([[c, -s], [s, c]], dtype=mx.complex64)

def RZ(theta: float) -> mx.array:
    e_neg = complex(math.cos(theta / 2), -math.sin(theta / 2))
    e_pos = complex(math.cos(theta / 2),  math.sin(theta / 2))
    return mx.array([[e_neg, 0], [0, e_pos]], dtype=mx.complex64)

def P(phi: float) -> mx.array:
    """Phase gate."""
    return mx.array([[1, 0], [0, complex(math.cos(phi), math.sin(phi))]], dtype=mx.complex64)

def U(theta: float, phi: float, lam: float) -> mx.array:
    """General single-qubit unitary (IBM U gate)."""
    ct, st = math.cos(theta / 2), math.sin(theta / 2)
    return mx.array([
        [ct, -complex(math.cos(lam), math.sin(lam)) * st],
        [complex(math.cos(phi), math.sin(phi)) * st,
         complex(math.cos(phi + lam), math.sin(phi + lam)) * ct],
    ], dtype=mx.complex64)


# Fixed two-qubit gates

def CNOT() -> mx.array:
    return mx.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=mx.complex64)

def CZ() -> mx.array:
    return mx.array([
        [1, 0, 0,  0],
        [0, 1, 0,  0],
        [0, 0, 1,  0],
        [0, 0, 0, -1],
    ], dtype=mx.complex64)

def SWAP() -> mx.array:
    return mx.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=mx.complex64)

def iSWAP() -> mx.array:
    return mx.array([
        [1,  0,  0, 0],
        [0,  0, 1j, 0],
        [0, 1j,  0, 0],
        [0,  0,  0, 1],
    ], dtype=mx.complex64)


# Parameterized two-qubit gates

def CRZ(theta: float) -> mx.array:
    e_neg = complex(math.cos(theta / 2), -math.sin(theta / 2))
    e_pos = complex(math.cos(theta / 2),  math.sin(theta / 2))
    return mx.array([
        [1, 0, 0,     0],
        [0, 1, 0,     0],
        [0, 0, e_neg, 0],
        [0, 0, 0,  e_pos],
    ], dtype=mx.complex64)

def RZZ(theta: float) -> mx.array:
    """Ising ZZ coupling gate, native to many hardware platforms."""
    e_neg = complex(math.cos(theta / 2), -math.sin(theta / 2))
    e_pos = complex(math.cos(theta / 2),  math.sin(theta / 2))
    return mx.array([
        [e_neg, 0,     0,    0],
        [0,     e_pos, 0,    0],
        [0,     0,     e_pos, 0],
        [0,     0,     0,  e_neg],
    ], dtype=mx.complex64)

def RXX(theta: float) -> mx.array:
    """Ising XX coupling gate."""
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    return mx.array([
        [c,     0,     0,  -1j*s],
        [0,     c,  -1j*s,    0],
        [0,  -1j*s,    c,     0],
        [-1j*s,  0,    0,     c],
    ], dtype=mx.complex64)
