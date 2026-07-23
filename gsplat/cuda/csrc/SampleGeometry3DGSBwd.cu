/*
 * SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// ---------------------------------------------------------------------------
// Backward of sample_geometry. One thread per query point: PASS 1 recomputes
// the forward blend (front-to-back) to recover T_final and the accumulated
// depth/normal; PASS 2 walks the same Gaussians back-to-front with the standard
// suffix-buffer recurrence to accumulate gradients. All per-Gaussian gradient
// formulas match RasterizeToPixels3DGSBwd so they compose with gsplat's
// projection VJP. Gaussian grads use atomicAdd (points share tiles); the
// per-point grad on the query pixel is written once.
// ---------------------------------------------------------------------------

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Sample.h"

namespace gsplat {

namespace cg = cooperative_groups;

// MEDIAN: the sampled depth was the opacity-volume level-set median (not the
// expected mean), so its gradient is the implicit-function-theorem one (needs
// the forward median via out_depth). MEDIAN implies the depth channel is median.
template <bool SAMPLE_NORMALS, bool MEDIAN, typename scalar_t>
__global__ void sample_geometry_3dgs_bwd_kernel(
    const uint32_t P,
    const uint32_t N,
    const vec2 *__restrict__ points2d,   // [P, 2]
    const vec2 *__restrict__ means2d,    // [N, 2]
    const vec3 *__restrict__ conics,     // [N, 3]
    const scalar_t *__restrict__ opacities, // [N]
    const vec4 *__restrict__ ray_planes, // [N, 4]
    const vec3 *__restrict__ normals,    // [N, 3] read only when SAMPLE_NORMALS
    const scalar_t *__restrict__ Ks,     // [9]
    const int geometry_mode,             // 0=RD, 1=MD, 2=PD
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    const uint32_t n_isects,
    const scalar_t *__restrict__ out_depth, // [P] forward median z-depth (MEDIAN)
    // upstream gradients
    const scalar_t *__restrict__ v_depth,  // [P]
    const scalar_t *__restrict__ v_alpha,  // [P]
    const scalar_t *__restrict__ v_normal, // [P, 3] when SAMPLE_NORMALS
    // outputs (Gaussian grads via atomicAdd, point grad written once)
    vec2 *__restrict__ v_means2d,    // [N, 2]
    vec3 *__restrict__ v_conics,     // [N, 3]
    scalar_t *__restrict__ v_opacities, // [N]
    vec4 *__restrict__ v_ray_planes, // [N, 4]
    vec3 *__restrict__ v_normals,    // [N, 3] when SAMPLE_NORMALS
    vec2 *__restrict__ v_points2d    // [P, 2]
) {
    uint32_t p = cg::this_grid().thread_rank();
    if (p >= P) {
        return;
    }
    v_points2d[p] = {0.f, 0.f};

    const float px = points2d[p].x;
    const float py = points2d[p].y;
    if (px < 0.f || px >= (float)image_width || py < 0.f || py >= (float)image_height) {
        return;
    }

    const uint32_t tx = min((uint32_t)(px / tile_size), tile_width - 1);
    const uint32_t ty = min((uint32_t)(py / tile_size), tile_height - 1);
    const uint32_t tile_id = ty * tile_width + tx;
    const int32_t start = tile_offsets[tile_id];
    const int32_t end = (tile_id + 1 < tile_width * tile_height)
                            ? tile_offsets[tile_id + 1]
                            : (int32_t)n_isects;

    // ---- PASS 1: recompute forward (front-to-back) ----
    float T = 1.f;
    float D = 0.f;
    vec3 Nrm = {0.f, 0.f, 0.f};
    int32_t k_stop = end; // first index that broke the blend (exclusive)
    for (int32_t k = start; k < end; ++k) {
        const int32_t g = flatten_ids[k];
        const vec2 delta = {means2d[g].x - px, means2d[g].y - py};
        const vec3 con = conics[g];
        const float sigma = 0.5f * (con.x * delta.x * delta.x +
                                    con.z * delta.y * delta.y) +
                            con.y * delta.x * delta.y;
        const float alpha = min(0.999f, opacities[g] * __expf(-sigma));
        if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
            continue;
        }
        const float next_T = T * (1.0f - alpha);
        if (next_T <= 1e-4f) {
            k_stop = k;
            break;
        }
        const float w = alpha * T;
        const vec4 rp = ray_planes[g];
        D += (rp.x * delta.x + rp.y * delta.y + rp.z) * w;
        if constexpr (SAMPLE_NORMALS) {
            const vec3 nm = normals[g];
            Nrm.x += nm.x * w;
            Nrm.y += nm.y * w;
            Nrm.z += nm.z * w;
        }
        T = next_T;
    }

    const float T_final = T;
    const float A = 1.f - T_final;
    if (A <= 1e-7f) {
        return; // no coverage: gradients are zero
    }

    const float fx = Ks[0], fy = Ks[4], cx = Ks[2], cy = Ks[5];
    const float ndx = (px - cx) / fx;
    const float ndy = (py - cy) / fy;
    const float rln = rsqrtf(ndx * ndx + ndy * ndy + 1.f);
    const float d_out = D * rln / A;

    // ---- upstream grads -> internal accumulators ----
    const float gd = v_depth[p];
    const float ga = v_alpha[p];
    // Expected-depth channel = D * rln / A. Only active for the mean reduction;
    // for MEDIAN the depth output is the level-set median, handled separately.
    float dL_dpixel_t = 0.f;    // grad on D (expected)
    float v_depth_finalT = 0.f; // expected depth routed through T_final
    float grln = 0.f;           // grad on rln (-> query pixel)
    float mDepth = 0.f;         // MEDIAN: forward median in ray-distance
    float dL_dmt_dT_dtm = 0.f;  // MEDIAN: dL/d(median) / (-dT/d ts)
    // PGSR unbiased plane depth: out_depth = D / -(Nrm.ray + eps). alpha and rln
    // cancel; the normal seed gets the denominator's Nrm dependence. Query-pixel
    // (points2d) grad from the PD ray is not propagated (grln=0): the multi-view
    // neighbour sample is used detached w.r.t. its query pixels.
    bool pd_done = false;
    if (geometry_mode == 2) {
        const float tmp = Nrm.x * ndx + Nrm.y * ndy + Nrm.z + 1e-8f;
        dL_dpixel_t = -gd / tmp;             // dL/d(depth_acc D)
        v_depth_finalT = 0.f;
        grln = 0.f;
        pd_done = true;
    }
    if (pd_done) {
        // skip the RD/MD depth-seed branch below
    } else if constexpr (MEDIAN) {
        const float inv_rln = rln > 1e-8f ? 1.f / rln : 0.f;
        mDepth = out_depth[p] * inv_rln; // z -> ray-distance
        grln = gd * mDepth;              // out_depth = mDepth * rln
        if (mDepth > 0.f) {
            // dT/dts at the median: per-point, over this pixel's contributors
            float dT_dtm = 0.f;
            for (int32_t k = start; k < k_stop; ++k) {
                const int32_t g = flatten_ids[k];
                const vec2 delta = {means2d[g].x - px, means2d[g].y - py};
                const vec3 con = conics[g];
                const float sigma = 0.5f * (con.x * delta.x * delta.x +
                                            con.z * delta.y * delta.y) +
                                    con.y * delta.x * delta.y;
                if (sigma < 0.f) {
                    continue;
                }
                const float alpha = min(0.999f, opacities[g] * __expf(-sigma));
                if (alpha < ALPHA_THRESHOLD) {
                    continue;
                }
                const vec4 rp = ray_planes[g];
                const float t_peak = rp.x * delta.x + rp.y * delta.y + rp.z;
                const float rsigma = rp.w;
                const float t_delta = (mDepth - t_peak) * rsigma;
                const float G_exp = __expf(-0.5f * t_delta * t_delta);
                const float Gt = alpha * G_exp;
                dT_dtm += -0.25f * Gt / (1.f - Gt) * fabsf(t_delta) * rsigma;
            }
            dL_dmt_dT_dtm = gd * rln / fmaxf(-dT_dtm, 1e-7f); // dL_dpixel_mt / (-dT_dtm)
        }
    } else {
        dL_dpixel_t = gd * rln / A;
        v_depth_finalT = gd * d_out / A;
        grln = gd * D / A;
    }

    // normal normalization VJP (only when sampling normals)
    float dL_dpixel_normal[3] = {0.f, 0.f, 0.f};
    if constexpr (SAMPLE_NORMALS) {
        const vec3 vn = {
            v_normal[p * 3 + 0], v_normal[p * 3 + 1], v_normal[p * 3 + 2]
        };
        if (geometry_mode == 2) {
            // PD: rendered normal is RAW (Nrm), so vn maps directly; plus the
            // plane-depth denominator's dependence on Nrm.
            const float tmp = Nrm.x * ndx + Nrm.y * ndy + Nrm.z + 1e-8f;
            const float coef = gd * D / (tmp * tmp); // gd*depth_acc/tmp^2
            dL_dpixel_normal[0] = vn.x + coef * ndx;
            dL_dpixel_normal[1] = vn.y + coef * ndy;
            dL_dpixel_normal[2] = vn.z + coef;
        } else {
            const float nlen = sqrtf(Nrm.x * Nrm.x + Nrm.y * Nrm.y + Nrm.z * Nrm.z);
            const float denom = fmaxf(nlen, 1e-6f);
            const float large = nlen < 1e-6f ? 0.f : 1.f;
            const vec3 nout = {Nrm.x / denom, Nrm.y / denom, Nrm.z / denom};
            const float dp = (vn.x * nout.x + vn.y * nout.y + vn.z * nout.z) * large;
            dL_dpixel_normal[0] = (vn.x - dp * nout.x) / denom;
            dL_dpixel_normal[1] = (vn.y - dp * nout.y) / denom;
            dL_dpixel_normal[2] = (vn.z - dp * nout.z) / denom;
        }
    }

    // ---- PASS 2: gradients (back-to-front, suffix buffers) ----
    float Tb = T_final;
    float buffer_t = 0.f;
    vec3 buffer_normal = {0.f, 0.f, 0.f};
    vec2 v_point = {0.f, 0.f}; // grad on the query pixel from delta terms

    for (int32_t k = k_stop - 1; k >= start; --k) {
        const int32_t g = flatten_ids[k];
        const vec2 delta = {means2d[g].x - px, means2d[g].y - py};
        const vec3 con = conics[g];
        const float sigma = 0.5f * (con.x * delta.x * delta.x +
                                    con.z * delta.y * delta.y) +
                            con.y * delta.x * delta.y;
        const float G = __expf(-sigma);
        const float opac = opacities[g];
        const float alpha = min(0.999f, opac * G);
        if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
            continue;
        }
        const float ra = 1.0f / (1.0f - alpha);
        Tb *= ra;               // recover transmittance before this Gaussian
        const float w = alpha * Tb;

        const vec4 rp = ray_planes[g];
        const float tt = rp.x * delta.x + rp.y * delta.y + rp.z;

        // dL/dalpha from depth channel + T_final routing + alpha output.
        // For MEDIAN the expected terms vanish (dL_dpixel_t = v_depth_finalT = 0).
        float v_a = (tt * Tb - buffer_t * ra) * dL_dpixel_t;
        v_a += -T_final * ra * v_depth_finalT;
        v_a += T_final * ra * ga;

        // plane-depth grad (ray_plane + moves the 2D mean)
        float dL_dt = w * dL_dpixel_t; // expected (0 when MEDIAN)
        float v_rsigma = 0.f;
        if constexpr (MEDIAN) {
            // Opacity-volume median: implicit-function-theorem grad at mDepth.
            const float rsigma = rp.w;
            const float t_delta = (mDepth - tt) * rsigma;
            const float G_exp = __expf(-0.5f * t_delta * t_delta);
            const float Gt = alpha * G_exp;
            float dL_dGt = dL_dmt_dT_dtm * 0.25f / (1.f - Gt);
            dL_dGt = (mDepth > tt) ? dL_dGt : -dL_dGt;
            dL_dGt = (rsigma > 0.f) ? dL_dGt : 0.f;
            const float dL_dopa_sigma =
                dL_dGt * G_exp -
                dL_dmt_dT_dtm * (t_delta > 0.f ? 0.5f / (1.f - alpha) : 0.f);
            const float dL_ddelta = -dL_dGt * Gt * t_delta;
            v_rsigma = dL_ddelta * (mDepth - tt);
            dL_dt += -dL_ddelta * rsigma; // dL/dt_peak
            v_a += dL_dopa_sigma;
        }
        vec4 v_rp_local = {dL_dt * delta.x, dL_dt * delta.y, dL_dt, v_rsigma};
        vec2 v_xy = {dL_dt * rp.x, dL_dt * rp.y};

        vec3 v_n_local = {0.f, 0.f, 0.f};
        if constexpr (SAMPLE_NORMALS) {
            const vec3 nm = normals[g];
            v_a += (nm.x * Tb - buffer_normal.x * ra) * dL_dpixel_normal[0];
            v_a += (nm.y * Tb - buffer_normal.y * ra) * dL_dpixel_normal[1];
            v_a += (nm.z * Tb - buffer_normal.z * ra) * dL_dpixel_normal[2];
            v_n_local = {w * dL_dpixel_normal[0],
                         w * dL_dpixel_normal[1],
                         w * dL_dpixel_normal[2]};
        }

        // alpha -> sigma -> {conic, mean, opacity}, only when not clamped
        vec3 v_con_local = {0.f, 0.f, 0.f};
        float v_opac_local = 0.f;
        if (opac * G <= 0.999f) {
            const float v_sigma = -opac * G * v_a;
            v_con_local = {0.5f * v_sigma * delta.x * delta.x,
                           v_sigma * delta.x * delta.y,
                           0.5f * v_sigma * delta.y * delta.y};
            v_xy.x += v_sigma * (con.x * delta.x + con.y * delta.y);
            v_xy.y += v_sigma * (con.y * delta.x + con.z * delta.y);
            v_opac_local = G * v_a;
        }

        // delta = mean2d - point2d: the mean and the query pixel get opposite grads
        atomicAdd(&v_means2d[g].x, v_xy.x);
        atomicAdd(&v_means2d[g].y, v_xy.y);
        v_point.x -= v_xy.x;
        v_point.y -= v_xy.y;
        atomicAdd(&v_conics[g].x, v_con_local.x);
        atomicAdd(&v_conics[g].y, v_con_local.y);
        atomicAdd(&v_conics[g].z, v_con_local.z);
        atomicAdd(&v_opacities[g], v_opac_local);
        atomicAdd(&v_ray_planes[g].x, v_rp_local.x);
        atomicAdd(&v_ray_planes[g].y, v_rp_local.y);
        atomicAdd(&v_ray_planes[g].z, v_rp_local.z);
        if constexpr (MEDIAN) {
            atomicAdd(&v_ray_planes[g].w, v_rp_local.w);
        }
        if constexpr (SAMPLE_NORMALS) {
            atomicAdd(&v_normals[g].x, v_n_local.x);
            atomicAdd(&v_normals[g].y, v_n_local.y);
            atomicAdd(&v_normals[g].z, v_n_local.z);
        }

        // update suffix buffers (used by the next, more-front Gaussian)
        buffer_t += tt * w;
        if constexpr (SAMPLE_NORMALS) {
            const vec3 nm = normals[g];
            buffer_normal.x += nm.x * w;
            buffer_normal.y += nm.y * w;
            buffer_normal.z += nm.z * w;
        }
    }

    // grad on the query pixel from the ray-length -> z-depth conversion (rln)
    const float rln3 = rln * rln * rln;
    v_point.x += grln * (-ndx * rln3 / fx);
    v_point.y += grln * (-ndy * rln3 / fy);
    v_points2d[p] = v_point;
}

void launch_sample_geometry_3dgs_bwd_kernel(
    const at::Tensor points2d,
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor opacities,
    const at::Tensor ray_planes,
    const at::optional<at::Tensor> normals,
    const at::Tensor Ks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    const bool sample_normals,
    const int geometry_mode,
    const at::optional<at::Tensor> out_depth,
    const at::Tensor v_depth,
    const at::Tensor v_alpha,
    const at::optional<at::Tensor> v_normal,
    at::Tensor v_means2d,
    at::Tensor v_conics,
    at::Tensor v_opacities,
    at::Tensor v_ray_planes,
    at::optional<at::Tensor> v_normals,
    at::Tensor v_points2d
) {
    const bool median = (geometry_mode == 1);
    const uint32_t P = points2d.size(0);
    const uint32_t N = means2d.size(0);
    const uint32_t tile_height = tile_offsets.size(-2);
    const uint32_t tile_width = tile_offsets.size(-1);
    const uint32_t n_isects = flatten_ids.size(0);
    if (P == 0) {
        return;
    }

    const uint32_t threads = 256;
    const uint32_t blocks = (P + threads - 1) / threads;
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        means2d.scalar_type(), "sample_geometry_3dgs_bwd", [&]() {
            auto normals_ptr =
                sample_normals
                    ? reinterpret_cast<vec3 *>(normals.value().data_ptr<scalar_t>())
                    : nullptr;
            auto v_normal_ptr =
                sample_normals ? v_normal.value().data_ptr<scalar_t>() : nullptr;
            auto v_normals_ptr =
                sample_normals
                    ? reinterpret_cast<vec3 *>(v_normals.value().data_ptr<scalar_t>())
                    : nullptr;
            const scalar_t *out_depth_ptr =
                median ? out_depth.value().data_ptr<scalar_t>() : nullptr;
            auto fn =
                median
                    ? (sample_normals
                           ? sample_geometry_3dgs_bwd_kernel<true, true, scalar_t>
                           : sample_geometry_3dgs_bwd_kernel<false, true, scalar_t>)
                    : (sample_normals
                           ? sample_geometry_3dgs_bwd_kernel<true, false, scalar_t>
                           : sample_geometry_3dgs_bwd_kernel<false, false, scalar_t>);
            fn<<<blocks, threads, 0, stream>>>(
                P,
                N,
                reinterpret_cast<vec2 *>(points2d.data_ptr<scalar_t>()),
                reinterpret_cast<vec2 *>(means2d.data_ptr<scalar_t>()),
                reinterpret_cast<vec3 *>(conics.data_ptr<scalar_t>()),
                opacities.data_ptr<scalar_t>(),
                reinterpret_cast<vec4 *>(ray_planes.data_ptr<scalar_t>()),
                normals_ptr,
                Ks.data_ptr<scalar_t>(),
                geometry_mode,
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                n_isects,
                out_depth_ptr,
                v_depth.data_ptr<scalar_t>(),
                v_alpha.data_ptr<scalar_t>(),
                v_normal_ptr,
                reinterpret_cast<vec2 *>(v_means2d.data_ptr<scalar_t>()),
                reinterpret_cast<vec3 *>(v_conics.data_ptr<scalar_t>()),
                v_opacities.data_ptr<scalar_t>(),
                reinterpret_cast<vec4 *>(v_ray_planes.data_ptr<scalar_t>()),
                v_normals_ptr,
                reinterpret_cast<vec2 *>(v_points2d.data_ptr<scalar_t>())
            );
        }
    );
}

} // namespace gsplat
