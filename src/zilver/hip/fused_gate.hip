// One kernel. Every gate in metal.py becomes a matrix handed to it.
//
// A thread owns one COSET of the span of m masks: it gathers 2^m amplitudes,
// applies a 2^m x 2^m unitary out of registers, and scatters back in place.
// Cosets are disjoint, so there is no race and no temporary -- the whole point,
// since the torch path was measured on this device at 4.3-7.0 copies per gate
// against a floor of 1.0 (bench/copies_per_gate.py, Radeon 8060S, 207 GB/s).
//
// Three things here are quantum-specific and would be absent from a generic
// tiling kernel:
//
//   1. PAIRING BY MASK, NOT BY BIT. A CNOT is a GF(2)-linear map on the index,
//      so a whole Clifford segment composes into one invertible matrix M that
//      never has to touch memory. A later 1q gate on logical qubit q pairs
//      buffer indices i and i^mask, mask = column q of M. With no frame the
//      mask is just 1<<position and this reduces to the ordinary strided gate,
//      so one code path covers both.
//
//   2. ORIENTATION BY PARITY. Which member of a pair is the logical |0>
//      branch is NOT "pivot bit clear" -- it is parity(i & row_q), row_q from
//      M^-1. Getting this wrong applies X U X on half the pairs: max error
//      ~7e-2, norm preserved, no crash. It would ship silently. Hence
//      __popcll below, and hence the planted-answer test in bench/coset_toy.py.
//
//   3. ARBITRARY POSITIONS IN ONE PASS. The masks need not be adjacent or
//      ordered, so high- and low-stride qubits fuse into the SAME pass. That
//      is why no bit-reversal, transpose or permutation appears anywhere here,
//      unlike the FFT this otherwise resembles.
//
// Complex is carried as float2 (HIP's hipFloatComplex is the same layout);
// ROCm supports complex64 natively, unlike DirectML, which aborts the process
// on it.

#include <hip/hip_runtime.h>

#define MAXM 5                       // 2^5 = 32 amplitudes = 64 VGPRs

extern "C" __global__ void fused_apply(
        float2* __restrict__ state,
        const unsigned long long* __restrict__ masks,   // [m] XOR partner per fused qubit
        const int*                __restrict__ pivots,  // [m] distinct pivot bit of each mask
        const unsigned long long* __restrict__ rows,    // [m] parity row of M^-1, 0 = no frame
        const float2*             __restrict__ U,       // [2^m][2^m] row-major
        int m,
        unsigned long long n_cosets)
{
    unsigned long long t = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (t >= n_cosets) return;

    // Scatter the thread id across the NON-pivot bits. Pivots are distinct, so
    // inserting a zero at each (ascending) reconstructs the coset representative.
    // This is the address arithmetic that costs a full index array in torch and
    // nothing at all here -- the reason the frame is only truly free in HIP.
    unsigned long long base = t;
    for (int b = 0; b < m; ++b) {
        unsigned long long low = base & ((1ULL << pivots[b]) - 1ULL);
        base = ((base ^ low) << 1) | low;
    }

    const int D = 1 << m;

    // Gather. Local index k selects a subset of the masks to XOR in.
    unsigned long long idx[1 << MAXM];
    float2 amp[1 << MAXM];
    for (int k = 0; k < D; ++k) {
        unsigned long long i = base;
        for (int b = 0; b < m; ++b)
            if ((k >> b) & 1) i ^= masks[b];
        idx[k] = i;
        amp[k] = state[i];
    }

    // Orientation. Under a frame, the logical bit b of this amplitude is
    // parity(i & rows[b]), which need not equal the local index bit. Permute
    // the gathered amplitudes into logical order before applying U.
    bool framed = false;
    for (int b = 0; b < m; ++b) if (rows[b]) framed = true;
    if (framed) {
        float2 tmp[1 << MAXM];
        for (int k = 0; k < D; ++k) {
            int lg = 0;
            for (int b = 0; b < m; ++b)
                lg |= (__popcll(idx[k] & rows[b]) & 1) << b;
            tmp[lg] = amp[k];
        }
        for (int k = 0; k < D; ++k) amp[k] = tmp[k];
    }

    // Apply, in registers. This is the arithmetic that is free on a
    // bandwidth-bound problem: m stages for the price of one pass.
    float2 out[1 << MAXM];
    for (int r = 0; r < D; ++r) {
        float re = 0.0f, im = 0.0f;
        for (int c = 0; c < D; ++c) {
            float2 u = U[r * D + c], a = amp[c];
            re += u.x * a.x - u.y * a.y;
            im += u.x * a.y + u.y * a.x;
        }
        out[r] = make_float2(re, im);
    }

    // Scatter back through the same orientation.
    if (framed) {
        for (int k = 0; k < D; ++k) {
            int lg = 0;
            for (int b = 0; b < m; ++b)
                lg |= (__popcll(idx[k] & rows[b]) & 1) << b;
            state[idx[k]] = out[lg];
        }
    } else {
        for (int k = 0; k < D; ++k) state[idx[k]] = out[k];
    }
}
