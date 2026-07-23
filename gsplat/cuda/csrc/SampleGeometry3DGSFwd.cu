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
// Sample the alpha-weighted ("expected") surface depth (+ optional camera-space
// normal) at arbitrary query pixels, without rendering a full image. Building
// block for multi-view surface-consistency losses.
//
// One thread per query point: look up the point's tile in the precomputed
// per-tile Gaussian intersection (tile_offsets + flatten_ids) and composite that
// tile's depth-sorted Gaussians front-to-back. Each Gaussian's tangent-plane
// depth comes from its ray_plane; the accumulated depth is normalized by the
// accumulated opacity (expected reduction, not median) and converted from
// ray-distance to z-depth via the pixel ray. Blend conventions match
// RasterizeToPixels3DGSFwd, so a sample at a pixel centre reproduces that
// render's expected-depth channel. Single camera; non-packed.
// ---------------------------------------------------------------------------

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Sample.h"

namespace gsplat {

namespace cg = cooperative_groups;

// MEDIAN selects the sampled depth: false = alpha-weighted mean (expected);
// true = opacity-volume level-set median (per-point binary search over the query
// pixel's tile Gaussians, using the per-Gaussian ray spread rsigma).
template <bool SAMPLE_NORMALS, bool MEDIAN, typename scalar_t>
__global__ void sample_geometry_3dgs_fwd_kernel(
    const uint32_t P,               // number of query points
    const uint32_t N,               // number of Gaussians
    const vec2 *__restrict__ points2d,   // [P, 2] query pixel coords (this camera)
    const vec2 *__restrict__ means2d,    // [N, 2]
    const vec3 *__restrict__ conics,     // [N, 3]
    const scalar_t *__restrict__ opacities, // [N]
    const vec4 *__restrict__ ray_planes, // [N, 4] {gx, gy, tc, rsigma}
    const vec3 *__restrict__ normals,    // [N, 3] read only when SAMPLE_NORMALS
    const scalar_t *__restrict__ Ks,     // [9] pinhole intrinsics
    const int geometry_mode,             // 0=RD, 1=MD, 2=PD
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    const uint32_t n_isects,
    // outputs
    scalar_t *__restrict__ out_depth,  // [P]
    scalar_t *__restrict__ out_alpha,  // [P]
    scalar_t *__restrict__ out_normal  // [P, 3] when SAMPLE_NORMALS
) {
    uint32_t p = cg::this_grid().thread_rank();
    if (p >= P) {
        return;
    }

    const float px = points2d[p].x;
    const float py = points2d[p].y;

    // Defaults for out-of-image / no-coverage points
    out_depth[p] = 0.f;
    out_alpha[p] = 0.f;
    if constexpr (SAMPLE_NORMALS) {
        out_normal[p * 3 + 0] = 0.f;
        out_normal[p * 3 + 1] = 0.f;
        out_normal[p * 3 + 2] = 0.f;
    }

    if (px < 0.f || px >= (float)image_width || py < 0.f || py >= (float)image_height) {
        return;
    }

    // Tile containing this query pixel; clamp guards the image-edge boundary.
    const uint32_t tx = min((uint32_t)(px / tile_size), tile_width - 1);
    const uint32_t ty = min((uint32_t)(py / tile_size), tile_height - 1);
    const uint32_t tile_id = ty * tile_width + tx;
    const int32_t start = tile_offsets[tile_id];
    const int32_t end = (tile_id + 1 < tile_width * tile_height)
                            ? tile_offsets[tile_id + 1]
                            : (int32_t)n_isects;

    float T = 1.f;
    float depth_acc = 0.f;
    vec3 normal_acc = {0.f, 0.f, 0.f};
    float mseed = 0.f;      // MEDIAN: ray-distance of the T>0.5 crossing splat
    int32_t k_stop = end;   // MEDIAN: exclusive bound of contributing splats

    for (int32_t k = start; k < end; ++k) {
        const int32_t g = flatten_ids[k];
        const vec2 xy = means2d[g];
        const vec2 delta = {xy.x - px, xy.y - py};
        const vec3 conic = conics[g];
        const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                    conic.z * delta.y * delta.y) +
                            conic.y * delta.x * delta.y;
        const float alpha = min(0.999f, opacities[g] * __expf(-sigma));
        if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
            continue;
        }
        const float next_T = T * (1.0f - alpha);
        if (next_T <= 1e-4f) { // exclusive early-out, matches the full render
            k_stop = k;
            break;
        }
        const float vis = alpha * T;
        const vec4 rp = ray_planes[g];
        const float depth_t = rp.x * delta.x + rp.y * delta.y + rp.z;
        depth_acc += depth_t * vis;
        if constexpr (MEDIAN) {
            if (T > 0.5f) {
                mseed = depth_t;
            }
        }
        if constexpr (SAMPLE_NORMALS) {
            const vec3 nm = normals[g];
            normal_acc.x += nm.x * vis;
            normal_acc.y += nm.y * vis;
            normal_acc.z += nm.z * vis;
        }
        T = next_T;
    }

    const float alpha_pix = 1.f - T;
    const float fx = Ks[0], fy = Ks[4], cx = Ks[2], cy = Ks[5];
    const float ndx = (px - cx) / fx;
    const float ndy = (py - cy) / fy;
    const float rln = rsqrtf(ndx * ndx + ndy * ndy + 1.f);

    out_alpha[p] = alpha_pix;
    if (geometry_mode == 2) {
        // PGSR unbiased plane depth: depth_acc = sum(vis*dist), normal_acc =
        // sum(vis*n); alpha cancels in the ratio, no rln. Mirrors the rasterizer.
        const float tmp = normal_acc.x * ndx + normal_acc.y * ndy + normal_acc.z + 1e-8f;
        out_depth[p] = depth_acc / -tmp;
    } else if constexpr (MEDIAN) {
        // Opacity-volume level-set median over this pixel's contributing splats
        // (per-point binary search). Works in ray-distance; -> z-depth via rln.
        constexpr int OAV_SPLIT = 8;
        constexpr int OAV_ITERS = 5;
        constexpr float OAV_RANGE = 0.4f;
        constexpr float OAV_MIN_T = 0.45f;
        float oav_median = 0.f;
        if (alpha_pix >= (1.f - OAV_MIN_T)) { // T <= OAV_MIN_T: opaque enough
            float depth_min = fmaxf(mseed - OAV_RANGE, 0.f);
            float depth_max = fmaxf(mseed + OAV_RANGE, 0.f);
            bool in_range = true;
            float T_p[OAV_SPLIT + 1];
            for (int it = 0; it < OAV_ITERS; ++it) {
                const bool first = (it == 0);
                const int s_lo = first ? 0 : 1;
                const int s_hi = first ? OAV_SPLIT + 1 : OAV_SPLIT;
                for (int s = s_lo; s < s_hi; ++s) {
                    T_p[s] = 1.f;
                }
                const float interval = (depth_max - depth_min) / (float)OAV_SPLIT;
                for (int32_t k = start; k < k_stop; ++k) {
                    const int32_t g = flatten_ids[k];
                    const vec2 xy = means2d[g];
                    const vec2 delta = {xy.x - px, xy.y - py};
                    const vec3 conic = conics[g];
                    const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                                conic.z * delta.y * delta.y) +
                                        conic.y * delta.x * delta.y;
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
                    const bool ball = rsigma > 0.f;
                    for (int s = s_lo; s < s_hi; ++s) {
                        const float ts = depth_min + interval * s;
                        const float dl = (ts - t_peak) * rsigma;
                        const float gg = ball ? __expf(-0.5f * dl * dl) : 0.f;
                        const float omg = 1.f - alpha * gg;
                        const float rvac = rsqrtf(omg);
                        T_p[s] *= (ts > t_peak ? (1.f - alpha) : omg) * rvac;
                    }
                }
                if (first) {
                    in_range = (T_p[0] >= 0.5f) && (T_p[OAV_SPLIT] <= 0.5f);
                }
                int sid = 0;
                for (int s = 1; s < OAV_SPLIT; ++s) {
                    sid = (T_p[s] >= 0.5f) ? s : sid;
                }
                depth_max = depth_min + (sid + 1) * interval;
                depth_min = depth_min + (sid + 0) * interval;
                T_p[0] = T_p[sid];
                T_p[OAV_SPLIT] = T_p[sid + 1];
            }
            const float w_max =
                __saturatef((T_p[0] - 0.5f) / (T_p[0] - T_p[OAV_SPLIT]));
            oav_median = in_range ? (w_max * depth_max + (1.f - w_max) * depth_min)
                                  : 0.f;
        }
        out_depth[p] = oav_median * rln;
    } else {
        out_depth[p] = alpha_pix > 1e-7f ? depth_acc * rln / alpha_pix : 0.f;
    }
    if constexpr (SAMPLE_NORMALS) {
        // PD emits the RAW accumulated normal (PGSR-faithful); RD/MD normalize.
        const float inv_nlen =
            geometry_mode == 2
                ? 1.0f
                : 1.0f / fmaxf(sqrtf(normal_acc.x * normal_acc.x +
                                     normal_acc.y * normal_acc.y +
                                     normal_acc.z * normal_acc.z),
                               1e-6f);
        out_normal[p * 3 + 0] = normal_acc.x * inv_nlen;
        out_normal[p * 3 + 1] = normal_acc.y * inv_nlen;
        out_normal[p * 3 + 2] = normal_acc.z * inv_nlen;
    }
}

void launch_sample_geometry_3dgs_fwd_kernel(
    const at::Tensor points2d,     // [P, 2]
    const at::Tensor means2d,      // [N, 2]
    const at::Tensor conics,       // [N, 3]
    const at::Tensor opacities,    // [N]
    const at::Tensor ray_planes,   // [N, 4]
    const at::optional<at::Tensor> normals, // [N, 3]
    const at::Tensor Ks,           // [3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets, // [tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    const bool sample_normals,
    const int geometry_mode,   // 0=RD, 1=MD, 2=PD
    // outputs
    at::Tensor out_depth,  // [P]
    at::Tensor out_alpha,  // [P]
    at::Tensor out_normal  // [P, 3] (0-size when !sample_normals)
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
        means2d.scalar_type(), "sample_geometry_3dgs_fwd", [&]() {
            auto normals_ptr =
                sample_normals
                    ? reinterpret_cast<vec3 *>(normals.value().data_ptr<scalar_t>())
                    : nullptr;
            auto out_normal_ptr =
                sample_normals ? out_normal.data_ptr<scalar_t>() : nullptr;
            auto fn =
                median
                    ? (sample_normals
                           ? sample_geometry_3dgs_fwd_kernel<true, true, scalar_t>
                           : sample_geometry_3dgs_fwd_kernel<false, true, scalar_t>)
                    : (sample_normals
                           ? sample_geometry_3dgs_fwd_kernel<true, false, scalar_t>
                           : sample_geometry_3dgs_fwd_kernel<false, false, scalar_t>);
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
                out_depth.data_ptr<scalar_t>(),
                out_alpha.data_ptr<scalar_t>(),
                out_normal_ptr
            );
        }
    );
}

} // namespace gsplat
