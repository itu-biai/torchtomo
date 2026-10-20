// Cone beam kernels for ConeBeam, compiled at run time by NVRTC (see _nvrtc.py).
//
// A ray runs from its view's source to one detector element. It is clipped to the
// volume's box, the outer faces of its outermost voxels, and sampled at the
// midpoints of n equal steps; the projection is the step length times the sum of
// the trilinear samples, with zero outside the volume. That is the PyTorch path's
// arithmetic, in float32.
//
// Layout. Volumes are stored with a border of one zero voxel on every side, in
// bricks of 4 x 4 x 4 voxels. Every sample lies inside the volume's box, so its
// eight voxels lie inside the border and read zero where the volume ends: zero
// padding with no bounds checks. The bricks keep the eight voxels in one or two
// 256 byte blocks per channel however the ray crosses the volume, where rows of a
// [Z, Y, X] array would spread them over four lines.
//
// The forward and the geometry gradient read batches packed, up to four volumes
// interleaved as channels, so one vector load serves the whole group and every ray
// is set up once for it; packed projections are [rays, C]. The forward writes
// unpacked projections, [B, ...] with `plane` elements per item.
//
// `box` holds the volume centre (x, y, z), the half extent, the inverse voxel
// spacing, and (size + 1) / 2, the stored coordinate of the centre: twelve floats.
//
// Exactness. The adjoint scatters every sample's trilinear weights from the same
// code that the forward gathers them with, so the two are each other's transpose
// up to the order of float32 additions.

template <int C>
__device__ __forceinline__ void fetch(const float* __restrict__ base, unsigned index, float (&out)[C])
{
    if constexpr (C == 1) {
        out[0] = __ldg(base + index);
    } else if constexpr (C == 2) {
        const float2 t = __ldg(reinterpret_cast<const float2*>(base) + index);
        out[0] = t.x;
        out[1] = t.y;
    } else {
        const float4 t = __ldg(reinterpret_cast<const float4*>(base) + index);
        out[0] = t.x;
        out[1] = t.y;
        out[2] = t.z;
        out[3] = t.w;
    }
}

struct ConeRay {
    // Sample 0 and the step between samples, in stored voxel coordinates.
    float px, py, pz, sx, sy, sz;
    // Step length; the integral is its product with the sum of the samples.
    float weight;
    // Entry and exit along the ray from the source, in the ray's length unit, the
    // axes whose faces set them (-1 for the source or the detector), and the
    // source-to-detector vector and length, for the geometry gradient.
    float entry, exit, length, ex, ey, ez;
    int entry_axis, exit_axis;
};

__device__ __forceinline__ void clip_axis(float origin, float direction, float half, int axis, float& entry,
                                          float& exit, int& entry_axis, int& exit_axis, bool& missed)
{
    if (fabsf(direction) > 1.1920929e-7f) {
        const float t0 = (-half - origin) / direction, t1 = (half - origin) / direction;
        const float lower = fminf(t0, t1), upper = fmaxf(t0, t1);
        if (lower > entry) {
            entry = lower;
            entry_axis = axis;
        }
        if (upper < exit) {
            exit = upper;
            exit_axis = axis;
        }
    } else if (fabsf(origin) > half) {
        missed = true;
    }
}

__device__ __forceinline__ bool cone_ray(const float* __restrict__ source, const float* __restrict__ detector,
                                         const float* __restrict__ box, int n, ConeRay& ray)
{
    const float sx = source[0], sy = source[1], sz = source[2];
    ray.ex = detector[0] - sx;
    ray.ey = detector[1] - sy;
    ray.ez = detector[2] - sz;
    ray.length = sqrtf(ray.ex * ray.ex + ray.ey * ray.ey + ray.ez * ray.ez);
    if (!(ray.length > 0.f)) return false;
    const float ux = ray.ex / ray.length, uy = ray.ey / ray.length, uz = ray.ez / ray.length;
    const float ox = sx - box[0], oy = sy - box[1], oz = sz - box[2];
    ray.entry = 0.f;
    ray.exit = ray.length;
    ray.entry_axis = ray.exit_axis = -1;
    bool missed = false;
    clip_axis(ox, ux, box[3], 0, ray.entry, ray.exit, ray.entry_axis, ray.exit_axis, missed);
    clip_axis(oy, uy, box[4], 1, ray.entry, ray.exit, ray.entry_axis, ray.exit_axis, missed);
    clip_axis(oz, uz, box[5], 2, ray.entry, ray.exit, ray.entry_axis, ray.exit_axis, missed);
    const float span = ray.exit - ray.entry;
    if (missed || !(span > 0.f)) return false;
    ray.weight = span / (float)n;
    const float first = ray.entry + 0.5f * ray.weight;
    ray.px = __fmaf_rn(first, ux, ox) * box[6] + box[9];
    ray.py = __fmaf_rn(first, uy, oy) * box[7] + box[10];
    ray.pz = __fmaf_rn(first, uz, oz) * box[8] + box[11];
    ray.sx = ray.weight * ux * box[6];
    ray.sy = ray.weight * uy * box[7];
    ray.sz = ray.weight * uz * box[8];
    return true;
}

// The stored volume: X, Y, Z voxels plus the border, in bricks.
struct Layout {
    int X, Y, Z;
    unsigned bricks_x, row, plane;
};

__device__ __forceinline__ Layout layout(int X, int Y, int Z)
{
    Layout l;
    l.X = X;
    l.Y = Y;
    l.Z = Z;
    l.bricks_x = (unsigned)(X + 5) >> 2;
    l.row = l.bricks_x << 6;
    l.plane = l.row * ((unsigned)(Y + 5) >> 2);
    return l;
}

// The eight voxels around a sample: the stored index of the lowest, the steps to
// its neighbours along x, y and z, which cross into the next brick from the last
// voxel of one, and the trilinear weights.
struct Stencil {
    int x0, y0, z0;
    unsigned index, ox, oy, oz;
    float weight[8];
    float dx, dy, dz;
};

__device__ __forceinline__ unsigned corner(const Stencil& s, int q)
{
    return s.index + ((q & 1) ? s.ox : 0u) + ((q & 2) ? s.oy : 0u) + ((q & 4) ? s.oz : 0u);
}

// The stored index of voxel (x, y, z) and the steps to its neighbours along x, y
// and z, which cross into the next brick from the last voxel of one. A sample inside
// the box has 0 <= floor <= size; the clamp only guards memory.
__device__ __forceinline__ void cell(int x0, int y0, int z0, const Layout& l, unsigned& index, unsigned& ox,
                                     unsigned& oy, unsigned& oz)
{
    const unsigned x = min(max(x0, 0), l.X), y = min(max(y0, 0), l.Y), z = min(max(z0, 0), l.Z);
    index = ((z >> 2) * (l.plane >> 6) + (y >> 2) * l.bricks_x + (x >> 2)) << 6 | (z & 3) << 4 | (y & 3) << 2 |
            (x & 3);
    ox = (x & 3) == 3 ? 61u : 1u;
    oy = (y & 3) == 3 ? l.row - 12u : 4u;
    oz = (z & 3) == 3 ? l.plane - 48u : 16u;
}

__device__ __forceinline__ void stencil(float px, float py, float pz, const Layout& l, Stencil& s)
{
    s.x0 = __float2int_rd(px);
    s.y0 = __float2int_rd(py);
    s.z0 = __float2int_rd(pz);
    s.dx = px - (float)s.x0;
    s.dy = py - (float)s.y0;
    s.dz = pz - (float)s.z0;
    cell(s.x0, s.y0, s.z0, l, s.index, s.ox, s.oy, s.oz);
    const float wx[2] = {1.f - s.dx, s.dx}, wy[2] = {1.f - s.dy, s.dy}, wz[2] = {1.f - s.dz, s.dz};
#pragma unroll
    for (int q = 0; q < 8; ++q) s.weight[q] = (wz[q >> 2] * wy[(q >> 1) & 1]) * wx[q & 1];
}

__device__ __forceinline__ bool ray_coordinates(int R, int W, size_t& ray, int& view)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    view = blockIdx.z;
    ray = ((size_t)view * R + row) * W + column;
    return column < W && row < R;
}

__device__ __forceinline__ void sample(const ConeRay& ray, int k, const Layout& l, Stencil& s)
{
    const float kf = (float)k;
    stencil(__fmaf_rn(kf, ray.sx, ray.px), __fmaf_rn(kf, ray.sy, ray.py), __fmaf_rn(kf, ray.sz, ray.pz), l, s);
}

// One thread per ray, an 8 x 16 patch of detector elements per block, so a warp's
// loads at each sample land in a compact patch of the volume. The loop is bound by
// its instructions, not by memory: consecutive samples a fraction of a voxel apart
// often share all eight voxels, and then neither their indices nor their values are
// computed or loaded again.
template <int C>
__device__ void cone_forward(const float* __restrict__ volume, const float* __restrict__ sources,
                             const float* __restrict__ detectors, const float* __restrict__ box,
                             float* __restrict__ out, int X, int Y, int Z, int R, int W, int n, int nb,
                             size_t plane)
{
    size_t r;
    int view;
    if (!ray_coordinates(R, W, r, view)) return;
    const Layout l = layout(X, Y, Z);
    float acc[C];
#pragma unroll
    for (int c = 0; c < C; ++c) acc[c] = 0.f;
    float weight = 0.f;
    ConeRay ray;
    if (cone_ray(sources + 3 * view, detectors + 3 * r, box, n, ray)) {
        weight = ray.weight;
        float v[8][C];
        int x0 = -8, y0 = -8, z0 = -8;
        for (int k = 0; k < n; ++k) {
            const float kf = (float)k;
            const float px = __fmaf_rn(kf, ray.sx, ray.px), py = __fmaf_rn(kf, ray.sy, ray.py),
                        pz = __fmaf_rn(kf, ray.sz, ray.pz);
            const int ix = __float2int_rd(px), iy = __float2int_rd(py), iz = __float2int_rd(pz);
            if (ix != x0 || iy != y0 || iz != z0) {
                x0 = ix;
                y0 = iy;
                z0 = iz;
                unsigned index, ox, oy, oz;
                cell(ix, iy, iz, l, index, ox, oy, oz);
#pragma unroll
                for (int q = 0; q < 8; ++q)
                    fetch<C>(volume, index + ((q & 1) ? ox : 0u) + ((q & 2) ? oy : 0u) + ((q & 4) ? oz : 0u), v[q]);
            }
            const float dx = px - (float)ix, dy = py - (float)iy, dz = pz - (float)iz;
            if constexpr (C == 1) {
                const float a = __fmaf_rn(dx, v[1][0] - v[0][0], v[0][0]);
                const float b = __fmaf_rn(dx, v[3][0] - v[2][0], v[2][0]);
                const float c = __fmaf_rn(dx, v[5][0] - v[4][0], v[4][0]);
                const float d = __fmaf_rn(dx, v[7][0] - v[6][0], v[6][0]);
                const float lower = __fmaf_rn(dy, b - a, a), upper = __fmaf_rn(dy, d - c, c);
                acc[0] += __fmaf_rn(dz, upper - lower, lower);
            } else {
                const float wx[2] = {1.f - dx, dx}, wy[2] = {1.f - dy, dy}, wz[2] = {1.f - dz, dz};
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    const float w = (wz[q >> 2] * wy[(q >> 1) & 1]) * wx[q & 1];
#pragma unroll
                    for (int c = 0; c < C; ++c) acc[c] = __fmaf_rn(w, v[q][c], acc[c]);
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < C; ++c)
        if (c < nb) out[(size_t)c * plane + r] = acc[c] * weight;
}

// The transpose, for one volume: each ray scatters its value, times the step
// length, into the voxels its samples read. Atomics, not the samples, bound it, so
// a ray sums its consecutive samples' weights for each voxel in registers and
// scatters a voxel once its stencil has moved past it. A stencil that moves at most
// one voxel along each axis keeps the voxels it shares with the last one. Packing
// volumes as channels would not help: the atomics are per channel.
__device__ void cone_adjoint(const float* __restrict__ projections, const float* __restrict__ sources,
                             const float* __restrict__ detectors, const float* __restrict__ box,
                             float* __restrict__ out, int X, int Y, int Z, int R, int W, int n)
{
    size_t r;
    int view;
    if (!ray_coordinates(R, W, r, view)) return;
    float value = projections[r];
    ConeRay ray;
    if (value == 0.f || !cone_ray(sources + 3 * view, detectors + 3 * r, box, n, ray)) return;
    value *= ray.weight;
    const Layout l = layout(X, Y, Z);
    float pending[8];
    unsigned pending_index[8];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        pending[q] = 0.f;
        pending_index[q] = 0;
    }
    int x0 = -8, y0 = -8, z0 = -8;
    for (int k = 0; k <= n; ++k) {
        Stencil s;
        if (k < n) {
            sample(ray, k, l, s);
        } else {
            s.x0 = s.y0 = s.z0 = -16;
        }
        const int dx = x0 - s.x0, dy = y0 - s.y0, dz = z0 - s.z0;
        if (dx != 0 || dy != 0 || dz != 0) {
            const bool near = abs(dx) <= 1 && abs(dy) <= 1 && abs(dz) <= 1;
            float kept[8];
#pragma unroll
            for (int q = 0; q < 8; ++q) kept[q] = 0.f;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                const int i = (q & 1) + dx, j = ((q >> 1) & 1) + dy, h = (q >> 2) + dz;
                if (near && i >= 0 && i <= 1 && j >= 0 && j <= 1 && h >= 0 && h <= 1) {
                    const int target = i + 2 * j + 4 * h;
#pragma unroll
                    for (int u = 0; u < 8; ++u)
                        if (u == target) kept[u] += pending[q];
                } else if (pending[q] != 0.f) {
                    atomicAdd(out + pending_index[q], pending[q]);
                }
            }
            x0 = s.x0;
            y0 = s.y0;
            z0 = s.z0;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                pending[q] = kept[q];
                pending_index[q] = corner(s, q);
            }
        }
        if (k < n) {
#pragma unroll
            for (int q = 0; q < 8; ++q) pending[q] = __fmaf_rn(s.weight[q], value, pending[q]);
        }
    }
}

// d<g, A x> / d(source) and / d(detector element), summed over the batch. With
// P_k = S + tau_k E, E = D - S, l = |E|, and tau_k running from the entry to the exit
// fraction in n equal steps, the projection is l (tau_x - tau_e) / n sum_k f(P_k).
// Its derivative needs, per ray, F = sum_k f, G = sum_k grad f, and
// M = sum_k (k + 1/2) / n grad f; an entry or exit on a face of axis a, at fraction
// tau, moves with dtau = -((1 - tau) dS_a + tau dD_a) / E_a. f is g . volume, the
// batch weighted by the incoming gradient of this ray.
template <int C>
__device__ void cone_geometry_grad(const float* __restrict__ volume, const float* __restrict__ grad,
                                   const float* __restrict__ sources, const float* __restrict__ detectors,
                                   const float* __restrict__ box, float* __restrict__ grad_sources,
                                   float* __restrict__ grad_detectors, int X, int Y, int Z, int R, int W, int n)
{
    size_t r;
    int view;
    const bool active = ray_coordinates(R, W, r, view);
    float gs[3] = {0.f, 0.f, 0.f}, gd[3] = {0.f, 0.f, 0.f};
    float g[C];
    bool any = false;
    if (active) {
        fetch<C>(grad, (unsigned)r, g);
#pragma unroll
        for (int c = 0; c < C; ++c) any = any || g[c] != 0.f;
    }
    ConeRay ray;
    if (any && cone_ray(sources + 3 * view, detectors + 3 * r, box, n, ray)) {
        const Layout l = layout(X, Y, Z);
        float F = 0.f, G[3] = {0.f, 0.f, 0.f}, M[3] = {0.f, 0.f, 0.f};
        float v[8];
        unsigned loaded = 0xffffffffu;
        for (int k = 0; k < n; ++k) {
            Stencil s;
            sample(ray, k, l, s);
            if (s.index != loaded) {
#pragma unroll
                for (int q = 0; q < 8; ++q) {
                    float value[C];
                    fetch<C>(volume, corner(s, q), value);
                    float weighted = 0.f;
#pragma unroll
                    for (int c = 0; c < C; ++c) weighted = __fmaf_rn(g[c], value[c], weighted);
                    v[q] = weighted;
                }
                loaded = s.index;
            }
            const float ax = 1.f - s.dx, ay = 1.f - s.dy, az = 1.f - s.dz;
            const float f = az * (ay * (ax * v[0] + s.dx * v[1]) + s.dy * (ax * v[2] + s.dx * v[3])) +
                            s.dz * (ay * (ax * v[4] + s.dx * v[5]) + s.dy * (ax * v[6] + s.dx * v[7]));
            const float fx = az * (ay * (v[1] - v[0]) + s.dy * (v[3] - v[2])) +
                             s.dz * (ay * (v[5] - v[4]) + s.dy * (v[7] - v[6]));
            const float fy = az * (ax * (v[2] - v[0]) + s.dx * (v[3] - v[1])) +
                             s.dz * (ax * (v[6] - v[4]) + s.dx * (v[7] - v[5]));
            const float fz = ay * (ax * (v[4] - v[0]) + s.dx * (v[5] - v[1])) +
                             s.dy * (ax * (v[6] - v[2]) + s.dx * (v[7] - v[3]));
            const float w = ((float)k + 0.5f) / (float)n;
            F += f;
            G[0] += fx;
            G[1] += fy;
            G[2] += fz;
            M[0] = __fmaf_rn(w, fx, M[0]);
            M[1] = __fmaf_rn(w, fy, M[1]);
            M[2] = __fmaf_rn(w, fz, M[2]);
        }
        const float E[3] = {ray.ex, ray.ey, ray.ez};
        const float length = ray.length;
        const float tau_e = ray.entry / length, tau_x = ray.exit / length, delta = tau_x - tau_e;
        const float step = ray.weight;  // length * delta / n
        float GE = 0.f, ME = 0.f;
#pragma unroll
        for (int a = 0; a < 3; ++a) {
            G[a] *= box[6 + a];
            M[a] *= box[6 + a];
            GE = __fmaf_rn(G[a], E[a], GE);
            ME = __fmaf_rn(M[a], E[a], ME);
        }
        const float along = delta * F / (float)n;
        const float ends = length * F / (float)n;
#pragma unroll
        for (int a = 0; a < 3; ++a) {
            const float H = tau_e * G[a] + delta * M[a];
            gd[a] = along * E[a] / length + step * H;
            gs[a] = step * G[a] - gd[a];
        }
#pragma unroll
        for (int a = 0; a < 3; ++a) {
            if (a == ray.exit_axis) {
                const float c = (ends + step * ME) / E[a];
                gs[a] -= c * (1.f - tau_x);
                gd[a] -= c * tau_x;
            }
            if (a == ray.entry_axis) {
                const float c = (-ends + step * (GE - ME)) / E[a];
                gs[a] -= c * (1.f - tau_e);
                gd[a] -= c * tau_e;
            }
        }
    }
    if (active) {
#pragma unroll
        for (int a = 0; a < 3; ++a) grad_detectors[3 * r + a] = gd[a];
    }
    // Every ray of a block belongs to one view.
#pragma unroll
    for (int a = 0; a < 3; ++a) {
        float total = gs[a];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) total += __shfl_down_sync(0xffffffffu, total, offset);
        if ((threadIdx.x + blockDim.x * threadIdx.y) % 32 == 0 && total != 0.f)
            atomicAdd(grad_sources + 3 * view + a, total);
    }
}

#define TT_CONE(C)                                                                                            \
    extern "C" __global__ void __launch_bounds__(128)                                                         \
        cone_forward_c##C(const float* volume, const float* sources, const float* detectors, const float* box, \
                          float* out, int X, int Y, int Z, int R, int W, int n, int nb, size_t plane)          \
    {                                                                                                         \
        cone_forward<C>(volume, sources, detectors, box, out, X, Y, Z, R, W, n, nb, plane);                   \
    }                                                                                                         \
    extern "C" __global__ void __launch_bounds__(128) cone_geometry_grad_c##C(                                \
        const float* volume, const float* grad, const float* sources, const float* detectors, const float* box, \
        float* grad_sources, float* grad_detectors, int X, int Y, int Z, int R, int W, int n)                  \
    {                                                                                                         \
        cone_geometry_grad<C>(volume, grad, sources, detectors, box, grad_sources, grad_detectors, X, Y, Z, R, \
                              W, n);                                                                          \
    }

TT_CONE(1)
TT_CONE(2)
TT_CONE(4)

extern "C" __global__ void __launch_bounds__(128)
    cone_adjoint_c1(const float* projections, const float* sources, const float* detectors, const float* box,
                    float* out, int X, int Y, int Z, int R, int W, int n)
{
    cone_adjoint(projections, sources, detectors, box, out, X, Y, Z, R, W, n);
}
