// Projector kernels for torchtomo, compiled at run time by NVRTC (see _nvrtc.py).
//
// Images and sinograms reach these kernels packed: up to eight images of the batch
// interleaved as channels, so one vector load serves every image and the geometry
// (coordinates, bilinear weights) is computed once per sample, not once per image.
// A packed image is [S, S, C] and a packed sinogram [A, n_det, C], C in {1, 2, 4, 8}.
// Outputs are written unpacked, [B, ...] with `plane` elements per image.
//
// Pixel coordinates follow grid_sample with align_corners=True: pixel j sits at
// normalised x = -1 + 2 j / (S - 1), so the centre is c = (S - 1) / 2.
//
// Exactness. The adjoint kernels are the transpose of the forward kernels, not of
// the eager path: both sides compute each sample point with the same fused
// multiply-adds and derive the bilinear weights with the same floor, so every
// weight is bit-identical and <A x, y> = <x, A^T y> holds to float32 roundoff.

#define TT_SQRT2_MARGIN 1.4152f

template <int C>
__device__ __forceinline__ void fetch(const float* __restrict__ base, int index, float (&out)[C])
{
    if constexpr (C == 1) {
        out[0] = __ldg(base + index);
    } else if constexpr (C == 2) {
        const float2 t = __ldg(reinterpret_cast<const float2*>(base) + index);
        out[0] = t.x;
        out[1] = t.y;
    } else {
        const float4* p = reinterpret_cast<const float4*>(base) + (size_t)index * (C / 4);
#pragma unroll
        for (int q = 0; q < C / 4; ++q) {
            const float4 t = __ldg(p + q);
            out[4 * q + 0] = t.x;
            out[4 * q + 1] = t.y;
            out[4 * q + 2] = t.z;
            out[4 * q + 3] = t.w;
        }
    }
}

// Bilinear gather at pixel coordinates (px, py) with zero padding, accumulated into acc.
template <int C>
__device__ __forceinline__ void bilinear_add(const float* __restrict__ img, int S, float px, float py, float (&acc)[C])
{
    const float xf = floorf(px), yf = floorf(py);
    const int x0 = (int)xf, y0 = (int)yf;
    const float dx = px - xf, dy = py - yf;
    const float wx0 = (x0 >= 0 && x0 < S) ? 1.f - dx : 0.f;
    const float wx1 = (x0 + 1 >= 0 && x0 + 1 < S) ? dx : 0.f;
    const float wy0 = (y0 >= 0 && y0 < S) ? 1.f - dy : 0.f;
    const float wy1 = (y0 + 1 >= 0 && y0 + 1 < S) ? dy : 0.f;
    if (wx0 == 0.f && wx1 == 0.f) return;
    if (wy0 == 0.f && wy1 == 0.f) return;
    const int cx0 = min(max(x0, 0), S - 1), cx1 = min(max(x0 + 1, 0), S - 1);
    const int cy0 = min(max(y0, 0), S - 1), cy1 = min(max(y0 + 1, 0), S - 1);
    float p00[C], p10[C], p01[C], p11[C];
    fetch<C>(img, cy0 * S + cx0, p00);
    fetch<C>(img, cy0 * S + cx1, p10);
    fetch<C>(img, cy1 * S + cx0, p01);
    fetch<C>(img, cy1 * S + cx1, p11);
    const float w00 = wx0 * wy0, w10 = wx1 * wy0, w01 = wx0 * wy1, w11 = wx1 * wy1;
#pragma unroll
    for (int k = 0; k < C; ++k) acc[k] += w00 * p00[k] + w10 * p10[k] + w01 * p01[k] + w11 * p11[k];
}

// The forward's weight for pixel index `pix` from sample coordinate `p`, bit for bit.
__device__ __forceinline__ float tent_as_forward(float p, float pix)
{
    const float f = floorf(p);
    const float d = p - f;
    return (f == pix) ? 1.f - d : ((f + 1.f == pix) ? d : 0.f);
}

// ---------------------------------------------------------------------------------
// Parallel beam. Lattice point (h, w) of angle a, with u = w - c, v = h - c, lands at
//   px = c + cos(a) u + sin(a) v,  py = c + cos(a) v - sin(a) u
// and the projection sums the samples over h, times the pixel size.
// ---------------------------------------------------------------------------------

__device__ __forceinline__ void parallel_point(float cs, float sn, float c, float u, float v, float& px, float& py)
{
    px = __fmaf_rn(sn, v, __fmaf_rn(cs, u, c));
    py = __fmaf_rn(-sn, u, __fmaf_rn(cs, v, c));
}

// One warp owns TW detector bins of one angle; its lanes cover a TW x (32 / TW) patch
// of the lattice and step down the rays together, so a warp's loads land in a
// compact rotated patch of the image instead of a 32 pixel line.
template <int C, int TW>
__device__ void parallel_forward(const float* __restrict__ img, const float2* __restrict__ trig,
                                 float* __restrict__ out, int S, int A, int nb, int plane, float scale, float r2)
{
    constexpr int TH = 32 / TW;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int lw = lane % TW, lh = lane / TW;
    const int a = blockIdx.y;
    const int w_first = (blockIdx.x * (blockDim.x >> 5) + warp) * TW;
    if (w_first >= S) return;
    const int w = w_first + lw;
    const float c = 0.5f * (float)(S - 1);
    const float2 t = trig[a];
    const float cs = t.x, sn = t.y;
    const float u = (float)w - c;

    // Rays are bounded by the disc that can still touch the masked image; the bound
    // is the warp's widest ray so all lanes take the same number of steps.
    const float u_lo = (float)w_first - c, u_hi = (float)min(w_first + TW - 1, S - 1) - c;
    const float u_min = (u_lo <= 0.f && u_hi >= 0.f) ? 0.f : fminf(fabsf(u_lo), fabsf(u_hi));
    int h0 = 0, h1 = S - 1;
    const float room = r2 - u_min * u_min;
    if (room < 0.f) {
        h1 = -1;
    } else if (room < c * c * 4.f) {
        const float e = sqrtf(room);
        h0 = max(0, (int)floorf(c - e));
        h1 = min(S - 1, (int)ceilf(c + e));
    }

    float acc[C];
#pragma unroll
    for (int k = 0; k < C; ++k) acc[k] = 0.f;
    if (w < S) {
        for (int h = h0 + lh; h <= h1; h += TH) {
            float px, py;
            parallel_point(cs, sn, c, u, (float)h - c, px, py);
            bilinear_add<C>(img, S, px, py, acc);
        }
    }
#pragma unroll
    for (int offset = TW; offset < 32; offset <<= 1) {
#pragma unroll
        for (int k = 0; k < C; ++k) acc[k] += __shfl_xor_sync(0xffffffffu, acc[k], offset);
    }
    if (lh == 0 && w < S) {
        float* o = out + (size_t)a * S + w;
#pragma unroll
        for (int k = 0; k < C; ++k)
            if (k < nb) o[(size_t)k * plane] = acc[k] * scale;
    }
}

// Gather form of the transpose: each pixel collects from the lattice points whose
// bilinear footprint covers it. Those lie within sqrt(2) of the pixel's rotated
// position in both lattice directions, so three columns and three rows suffice.
template <int C>
__device__ void parallel_adjoint(const float* __restrict__ sino, const float2* __restrict__ trig,
                                 const float* __restrict__ mask, float* __restrict__ out, int S, int A, int nb,
                                 int plane, float scale)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= S || j >= S) return;
    const float m = mask ? mask[i * S + j] : 1.f;
    float acc[C];
#pragma unroll
    for (int k = 0; k < C; ++k) acc[k] = 0.f;
    if (m != 0.f) {
        const float c = 0.5f * (float)(S - 1);
        const float X = (float)j - c, Y = (float)i - c;
        const float jf = (float)j, iff = (float)i;
        for (int a = 0; a < A; ++a) {
            const float2 t = trig[a];
            const float cs = t.x, sn = t.y;
            const float us = cs * X - sn * Y + c;
            const float vs = sn * X + cs * Y + c;
            const int wb = (int)floorf(us - TT_SQRT2_MARGIN) + 1;
            const int hb = (int)floorf(vs - TT_SQRT2_MARGIN) + 1;
            const float* row = sino + (size_t)a * S * C;
#pragma unroll
            for (int dw = 0; dw < 3; ++dw) {
                const int w = wb + dw;
                if (w < 0 || w >= S) continue;
                const float u = (float)w - c;
                float kw = 0.f;
#pragma unroll
                for (int dh = 0; dh < 3; ++dh) {
                    const int h = hb + dh;
                    if (h < 0 || h >= S) continue;
                    float px, py;
                    parallel_point(cs, sn, c, u, (float)h - c, px, py);
                    kw += tent_as_forward(px, jf) * tent_as_forward(py, iff);
                }
                if (kw != 0.f) {
                    float y[C];
                    fetch<C>(row, w, y);
#pragma unroll
                    for (int k = 0; k < C; ++k) acc[k] += kw * y[k];
                }
            }
        }
    }
    float* o = out + (size_t)i * S + j;
#pragma unroll
    for (int k = 0; k < C; ++k)
        if (k < nb) o[(size_t)k * plane] = acc[k] * scale * m;
}

// FBP backprojection: pixel-driven, linear interpolation on the detector.
template <int C>
__device__ void parallel_backproject(const float* __restrict__ sino, const float2* __restrict__ trig,
                                     const float* __restrict__ coords, const float* __restrict__ mask,
                                     float* __restrict__ out, int S, int A, int n_det, int nb, int plane, float scale)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= S || j >= S) return;
    const float m = mask ? mask[i * S + j] : 1.f;
    float acc[C];
#pragma unroll
    for (int k = 0; k < C; ++k) acc[k] = 0.f;
    if (m != 0.f) {
        const float nx = coords[j], ny = coords[i];
        const float half = 0.5f * (float)(n_det - 1);
        for (int a = 0; a < A; ++a) {
            const float2 t = trig[a];
            const float g = __fmaf_rn(ny, -t.y, nx * t.x);
            const float d = (g + 1.f) * half;
            const float f = floorf(d);
            const int d0 = (int)f;
            const float frac = d - f;
            const float* row = sino + (size_t)a * n_det * C;
            if (d0 >= 0 && d0 < n_det) {
                float y[C];
                fetch<C>(row, d0, y);
#pragma unroll
                for (int k = 0; k < C; ++k) acc[k] += (1.f - frac) * y[k];
            }
            if (d0 + 1 >= 0 && d0 + 1 < n_det) {
                float y[C];
                fetch<C>(row, d0 + 1, y);
#pragma unroll
                for (int k = 0; k < C; ++k) acc[k] += frac * y[k];
            }
        }
    }
    float* o = out + (size_t)i * S + j;
#pragma unroll
    for (int k = 0; k < C; ++k)
        if (k < nb) o[(size_t)k * plane] = acc[k] * scale * m;
}

#define TT_PARALLEL(C)                                                                                        \
    extern "C" __global__ void __launch_bounds__(128) parallel_forward_c##C(                                  \
        const float* img, const float2* trig, float* out, int S, int A, int nb, int plane, float scale,       \
        float r2)                                                                                             \
    {                                                                                                         \
        parallel_forward<C, TT_FORWARD_TW>(img, trig, out, S, A, nb, plane, scale, r2);                       \
    }                                                                                                         \
    extern "C" __global__ void parallel_adjoint_c##C(const float* sino, const float2* trig, const float* mask, \
                                                     float* out, int S, int A, int nb, int plane, float scale) \
    {                                                                                                         \
        parallel_adjoint<C>(sino, trig, mask, out, S, A, nb, plane, scale);                                   \
    }                                                                                                         \
    extern "C" __global__ void parallel_backproject_c##C(const float* sino, const float2* trig,               \
                                                         const float* coords, const float* mask, float* out,  \
                                                         int S, int A, int n_det, int nb, int plane,          \
                                                         float scale)                                         \
    {                                                                                                         \
        parallel_backproject<C>(sino, trig, coords, mask, out, S, A, n_det, nb, plane, scale);                \
    }

#ifndef TT_FORWARD_TW
#define TT_FORWARD_TW 4
#endif
#ifndef TT_FAN_ROWS
#define TT_FAN_ROWS 2
#endif

TT_PARALLEL(1)
TT_PARALLEL(2)
TT_PARALLEL(4)
TT_PARALLEL(8)

// ---------------------------------------------------------------------------------
// Fan beam, flat detector. Every ray (a, d) is a table row in pixel coordinates:
// start (x0, y0) where it enters the unit disc, step (sx, sy) between its n samples,
// and its weight, the chord length over n. Sample k sits at
//   px = x0 + k sx,  py = y0 + k sy
// and the projection is the weight times the sum of the samples.
// ---------------------------------------------------------------------------------

__device__ __forceinline__ void fan_point(const float4 ray, float k, float& px, float& py)
{
    px = __fmaf_rn(k, ray.z, ray.x);
    py = __fmaf_rn(k, ray.w, ray.y);
}

template <int C, int TW>
__device__ void fan_forward(const float* __restrict__ img, const float4* __restrict__ rays,
                            const float* __restrict__ ray_weight, float* __restrict__ out, int S, int A, int n_det,
                            int n_samples, int nb, int plane)
{
    constexpr int TH = 32 / TW;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int lw = lane % TW, lh = lane / TW;
    const int a = blockIdx.y;
    const int d_first = (blockIdx.x * (blockDim.x >> 5) + warp) * TW;
    if (d_first >= n_det) return;
    const int d = d_first + lw;
    float acc[C];
#pragma unroll
    for (int k = 0; k < C; ++k) acc[k] = 0.f;
    float weight = 0.f;
    if (d < n_det) {
        const size_t r = (size_t)a * n_det + d;
        weight = ray_weight[r];
        if (weight != 0.f) {
            const float4 ray = rays[r];
            for (int k = lh; k < n_samples; k += TH) {
                float px, py;
                fan_point(ray, (float)k, px, py);
                bilinear_add<C>(img, S, px, py, acc);
            }
        }
    }
#pragma unroll
    for (int offset = TW; offset < 32; offset <<= 1) {
#pragma unroll
        for (int k = 0; k < C; ++k) acc[k] += __shfl_xor_sync(0xffffffffu, acc[k], offset);
    }
    if (lh == 0 && d < n_det) {
        float* o = out + (size_t)a * n_det + d;
#pragma unroll
        for (int k = 0; k < C; ++k)
            if (k < nb) o[(size_t)k * plane] = acc[k] * weight;
    }
}

// Samples k of a ray whose coordinate p0 + k s lies within one pixel of q, widened by
// TT_WINDOW_SLACK pixels. `inv` is 1 / s, or 0 for a ray that does not move along
// this axis. The window only has to be conservative: the exact forward weights are
// evaluated inside it, and the slack exceeds the rounding of p0 + k s (half an ulp
// of the coordinate, 3e-5 px at 512 px) by a wide margin.
#define TT_WINDOW_SLACK 1e-3f
__device__ __forceinline__ void sample_window(float p0, float inv, float q, float& lo, float& hi)
{
    const float reach = 1.f + TT_WINDOW_SLACK;
    if (inv == 0.f) {
        const bool inside = fabsf(p0 - q) < reach;
        lo = inside ? -1e30f : 1e30f;
        hi = inside ? 1e30f : -1e30f;
        return;
    }
    const float t0 = (q - reach - p0) * inv, t1 = (q + reach - p0) * inv;
    lo = fminf(t0, t1);
    hi = fmaxf(t0, t1);
}

// The same for coordinates within `half` + slack of q, covering several pixels at once.
__device__ __forceinline__ void sample_window_reach(float p0, float inv, float q, float half, float& lo, float& hi)
{
    const float reach = half + TT_WINDOW_SLACK;
    if (inv == 0.f) {
        const bool inside = fabsf(p0 - q) < reach;
        lo = inside ? -1e30f : 1e30f;
        hi = inside ? 1e30f : -1e30f;
        return;
    }
    const float t0 = (q - reach - p0) * inv, t1 = (q + reach - p0) * inv;
    lo = fminf(t0, t1);
    hi = fmaxf(t0, t1);
}

// Gather form of the transpose. For each angle the pixel's tent support, the open
// square of half-width one around it, is projected from the source onto the
// detector; only rays in that bin range can have samples inside it, and along each
// such ray only the samples in the matching window. The sinogram arrives already
// multiplied by the ray weights.
//
// Each thread owns PY pixels of one column. They share the candidate rays, the
// sample positions, and the column's tent, and evaluate only the row tent apiece.
template <int C, int PY>
__device__ void fan_adjoint(const float* __restrict__ sino, const float4* __restrict__ rays,
                            const float2* __restrict__ inv_steps, const float4* __restrict__ views,
                            const float* __restrict__ mask, float* __restrict__ out, int S, int A, int n_det,
                            int n_samples, float alpha, float beta, int nb, int plane)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i0 = (blockIdx.y * blockDim.y + threadIdx.y) * PY;
    if (i0 >= S || j >= S) return;
    float m[PY];
    bool any = false;
#pragma unroll
    for (int p = 0; p < PY; ++p) {
        m[p] = (i0 + p < S) ? (mask ? mask[(i0 + p) * S + j] : 1.f) : 0.f;
        any = any || m[p] != 0.f;
    }
    float acc[PY][C];
#pragma unroll
    for (int p = 0; p < PY; ++p)
#pragma unroll
        for (int k = 0; k < C; ++k) acc[p][k] = 0.f;
    if (any) {
        const float jf = (float)j, if0 = (float)i0;
        const float mid = if0 + 0.5f * (float)(PY - 1);
        const float half_y = 1.f + 0.5f * (float)(PY - 1);
        const float last = (float)(n_samples - 1);
        for (int a = 0; a < A; ++a) {
            // v0: source (x, y), detector direction (x, y); v1: source-to-detector direction
            const float4 v0 = views[2 * a], v1 = views[2 * a + 1];
            const float rx = jf - v0.x, ry = mid - v0.y;
            const float lat = rx * v0.z + ry * v0.w, dep = rx * v1.x + ry * v1.y;
            // Bin of each corner of the pixels' joint support, alpha l / e + beta. Far
            // from the source (every default geometry) 1 / (dep + de) is expanded to
            // second order about 1 / dep: de is under two pixels, so beyond 64 px the
            // third-order remainder is far inside the 0.01 bin slack below. Nearer,
            // each corner is divided exactly; a corner at or behind the source plane
            // widens the range to the whole detector, and a support entirely behind
            // it holds no samples at all, since every sample lies in front.
            float dmin = 1e30f, dmax = -1e30f;
            if (dep > 64.f) {
                const float rcp = __frcp_rn(dep);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const float ox = (q & 1) ? 1.f : -1.f, oy = (q & 2) ? half_y : -half_y;
                    const float l = lat + ox * v0.z + oy * v0.w;
                    const float x = (ox * v1.x + oy * v1.y) * rcp;
                    const float dd = l * rcp * (1.f - x + x * x) * alpha + beta;
                    dmin = fminf(dmin, dd);
                    dmax = fmaxf(dmax, dd);
                }
            } else {
                int behind = 0;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const float ox = (q & 1) ? 1.f : -1.f, oy = (q & 2) ? half_y : -half_y;
                    const float l = lat + ox * v0.z + oy * v0.w;
                    const float e = dep + ox * v1.x + oy * v1.y;
                    if (e <= 0.f) {
                        ++behind;
                        continue;
                    }
                    const float dd = l / e * alpha + beta;
                    dmin = fminf(dmin, dd);
                    dmax = fmaxf(dmax, dd);
                }
                if (behind == 4) continue;
                if (behind > 0) {
                    dmin = -1e30f;
                    dmax = 1e30f;
                }
            }
            const int d_lo = (int)fmaxf(0.f, ceilf(dmin - 0.01f));
            const int d_hi = (int)fminf((float)(n_det - 1), floorf(dmax + 0.01f));
            const float* row = sino + (size_t)a * n_det * C;
            for (int d = d_lo; d <= d_hi; ++d) {
                const size_t r = (size_t)a * n_det + d;
                const float4 ray = rays[r];
                const float2 inv = inv_steps[r];
                float xlo, xhi, ylo, yhi;
                sample_window(ray.x, inv.x, jf, xlo, xhi);
                sample_window_reach(ray.y, inv.y, mid, half_y, ylo, yhi);
                const float lo = fmaxf(fmaxf(xlo, ylo), 0.f);
                const float hi = fminf(fminf(xhi, yhi), last);
                if (lo > hi) continue;
                const int k_lo = (int)ceilf(lo);
                const int k_hi = (int)floorf(hi);
                float kw[PY];
#pragma unroll
                for (int p = 0; p < PY; ++p) kw[p] = 0.f;
                for (int k = k_lo; k <= k_hi; ++k) {
                    float px, py;
                    fan_point(ray, (float)k, px, py);
                    const float wx = tent_as_forward(px, jf);
                    const float fy = floorf(py);
                    const float dy = py - fy;
#pragma unroll
                    for (int p = 0; p < PY; ++p) {
                        const float pix = if0 + (float)p;
                        const float wy = (fy == pix) ? 1.f - dy : ((fy + 1.f == pix) ? dy : 0.f);
                        kw[p] += wx * wy;
                    }
                }
                bool hit = false;
#pragma unroll
                for (int p = 0; p < PY; ++p) hit = hit || kw[p] != 0.f;
                if (hit) {
                    float y[C];
                    fetch<C>(row, d, y);
#pragma unroll
                    for (int p = 0; p < PY; ++p)
#pragma unroll
                        for (int k = 0; k < C; ++k) acc[p][k] += kw[p] * y[k];
                }
            }
        }
    }
#pragma unroll
    for (int p = 0; p < PY; ++p) {
        if (i0 + p >= S) break;
        float* o = out + (size_t)(i0 + p) * S + j;
#pragma unroll
        for (int k = 0; k < C; ++k)
            if (k < nb) o[(size_t)k * plane] = acc[p][k] * m[p];
    }
}

// FBP backprojection with the 1/U^2 distance weight, in the normalised coordinates
// of the eager path: source at src (-sin, cos), detector centre at det (sin, -cos).
template <int C>
__device__ void fan_backproject(const float* __restrict__ sino, const float2* __restrict__ trig,
                                const float* __restrict__ coords, const float* __restrict__ mask,
                                float* __restrict__ out, int S, int A, int n_det, float src, float det,
                                float half_width, int nb, int plane, float scale)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= S || j >= S) return;
    const float m = mask ? mask[i * S + j] : 1.f;
    float acc[C];
#pragma unroll
    for (int k = 0; k < C; ++k) acc[k] = 0.f;
    if (m != 0.f) {
        const float gx = coords[j], gy = coords[i];
        const float half = 0.5f * (float)(n_det - 1);
        const float span = src + det;
        for (int a = 0; a < A; ++a) {
            const float2 t = trig[a];
            const float cs = t.x, sn = t.y;
            const float src_x = -src * sn, src_y = src * cs;
            const float det_x = det * sn, det_y = -det * cs;
            const float px = gx - src_x, py = gy - src_y;
            const float proj = px * sn - py * cs;
            const float tt = span / (proj + 1e-8f);
            const float ix = src_x + tt * px, iy = src_y + tt * py;
            const float offset = (ix - det_x) * cs + (iy - det_y) * sn;
            const float dpos = (offset / half_width + 1.f) * half;
            const float U = (src + gx * sn - gy * cs) / src;
            const float Uc = fmaxf(U, 1e-6f);
            const float wgt = 1.f / (Uc * Uc);
            const float f = floorf(dpos);
            const int d0 = (int)f;
            const float frac = dpos - f;
            const float* row = sino + (size_t)a * n_det * C;
            if (d0 >= 0 && d0 < n_det) {
                float y[C];
                fetch<C>(row, d0, y);
#pragma unroll
                for (int k = 0; k < C; ++k) acc[k] += wgt * (1.f - frac) * y[k];
            }
            if (d0 + 1 >= 0 && d0 + 1 < n_det) {
                float y[C];
                fetch<C>(row, d0 + 1, y);
#pragma unroll
                for (int k = 0; k < C; ++k) acc[k] += wgt * frac * y[k];
            }
        }
    }
    float* o = out + (size_t)i * S + j;
#pragma unroll
    for (int k = 0; k < C; ++k)
        if (k < nb) o[(size_t)k * plane] = acc[k] * scale * m;
}

#define TT_FAN(C)                                                                                             \
    extern "C" __global__ void __launch_bounds__(128) fan_forward_c##C(                                       \
        const float* img, const float4* rays, const float* ray_weight, float* out, int S, int A, int n_det,  \
        int n_samples, int nb, int plane)                                                                     \
    {                                                                                                         \
        fan_forward<C, TT_FORWARD_TW>(img, rays, ray_weight, out, S, A, n_det, n_samples, nb, plane);         \
    }                                                                                                         \
    extern "C" __global__ void fan_adjoint_c##C(const float* sino, const float4* rays, const float2* inv,     \
                                                const float4* views, const float* mask, float* out, int S,    \
                                                int A, int n_det, int n_samples, float alpha, float beta,     \
                                                int nb, int plane)                                            \
    {                                                                                                         \
        fan_adjoint<C, TT_FAN_ROWS>(sino, rays, inv, views, mask, out, S, A, n_det, n_samples, alpha, beta, nb,  \
                                    plane);                                                                   \
    }                                                                                                         \
    extern "C" __global__ void fan_backproject_c##C(const float* sino, const float2* trig,                    \
                                                    const float* coords, const float* mask, float* out,       \
                                                    int S, int A, int n_det, float src, float det,            \
                                                    float half_width, int nb, int plane, float scale)         \
    {                                                                                                         \
        fan_backproject<C>(sino, trig, coords, mask, out, S, A, n_det, src, det, half_width, nb, plane,       \
                           scale);                                                                            \
    }

TT_FAN(1)
TT_FAN(2)
TT_FAN(4)
TT_FAN(8)
