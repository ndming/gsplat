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
// Accumulate the opacity-volume transmittance at arbitrary 3D query points
// under one camera. Building block for tetrahedra-based surface extraction: the
// signed field is `transmittance - 0.5`, so its 0.5 level set is the surface.
//
// One thread per query point: project (done on the host) gives the point's pixel
// and its ray-distance depth `point_t`; the thread looks up the point's tile in
// the precomputed per-tile Gaussian intersection (tile_offsets + flatten_ids) and
// composites that tile's depth-sorted Gaussians front-to-back. The transmittance
// recurrence is the same opacity-volume gate as RasterizeToPixels3DGSFwd's median
// pass, but evaluated once at the query point's own depth (no binary search):
//   delta  = (t_peak - point_t) * rsigma
//   g      = rsigma > 0 ? exp(-0.5 delta^2) : 0
//   T     *= (point_t > t_peak ? (1 - alpha) : (1 - alpha g)) * rsqrt(1 - alpha g)
// A separate ordinary alpha-composite transmittance drives the front-to-back
// early-out. Forward-only (extraction runs under no_grad). Single camera; non-packed.
// ---------------------------------------------------------------------------

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Integrate.h"

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void integrate_transmittance_3dgs_fwd_kernel(
    const uint32_t P,               // number of query points
    const uint32_t N,               // number of Gaussians
    const vec2 *__restrict__ points2d,      // [P, 2] query pixel coords (this camera)
    const scalar_t *__restrict__ point_t,   // [P] query ray-distance ||p_cam||
    const vec2 *__restrict__ means2d,       // [N, 2]
    const vec3 *__restrict__ conics,        // [N, 3]
    const scalar_t *__restrict__ opacities, // [N]
    const vec4 *__restrict__ ray_planes,    // [N, 4] {gx, gy, tc, rsigma}
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    const uint32_t n_isects,
    // outputs
    scalar_t *__restrict__ out_transmittance  // [P]
) {
    uint32_t p = cg::this_grid().thread_rank();
    if (p >= P) {
        return;
    }

    const float px = points2d[p].x;
    const float py = points2d[p].y;

    // Default for out-of-image / behind-near query points (marked out-of-image
    // by the host projection): fully occluded so `sdf = T - 0.5 < 0`.
    out_transmittance[p] = 0.f;
    if (px < 0.f || px >= (float)image_width || py < 0.f || py >= (float)image_height) {
        return;
    }
    const float pt = (float)point_t[p];

    // Tile containing this query pixel; clamp guards the image-edge boundary.
    const uint32_t tx = min((uint32_t)(px / tile_size), tile_width - 1);
    const uint32_t ty = min((uint32_t)(py / tile_size), tile_height - 1);
    const uint32_t tile_id = ty * tile_width + tx;
    const int32_t start = tile_offsets[tile_id];
    const int32_t end = (tile_id + 1 < tile_width * tile_height)
                            ? tile_offsets[tile_id + 1]
                            : (int32_t)n_isects;

    float T_point = 1.f; // opacity-volume transmittance (output)
    float T_gauss = 1.f; // ordinary alpha-composite, drives the early-out only

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
        const float test_T = T_gauss * (1.f - alpha);
        if (test_T <= 1e-4f) { // depth-sorted early-out, matches the full render
            break;
        }
        const vec4 rp = ray_planes[g];
        const float t_peak = rp.x * delta.x + rp.y * delta.y + rp.z;
        const float rsigma = rp.w;
        const float dl = (t_peak - pt) * rsigma;
        const float gg = rsigma > 0.f ? __expf(-0.5f * dl * dl) : 0.f;
        const float omg = 1.f - alpha * gg;
        const float rvac = rsqrtf(omg);
        T_point *= (pt > t_peak ? (1.f - alpha) : omg) * rvac;
        T_gauss = test_T;
    }

    out_transmittance[p] = T_point;
}

void launch_integrate_transmittance_3dgs_fwd_kernel(
    const at::Tensor points2d,     // [P, 2]
    const at::Tensor point_t,      // [P]
    const at::Tensor means2d,      // [N, 2]
    const at::Tensor conics,       // [N, 3]
    const at::Tensor opacities,    // [N]
    const at::Tensor ray_planes,   // [N, 4]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets, // [tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // outputs
    at::Tensor out_transmittance   // [P]
) {
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
        means2d.scalar_type(), "integrate_transmittance_3dgs_fwd", [&]() {
            integrate_transmittance_3dgs_fwd_kernel<scalar_t>
                <<<blocks, threads, 0, stream>>>(
                    P,
                    N,
                    reinterpret_cast<vec2 *>(points2d.data_ptr<scalar_t>()),
                    point_t.data_ptr<scalar_t>(),
                    reinterpret_cast<vec2 *>(means2d.data_ptr<scalar_t>()),
                    reinterpret_cast<vec3 *>(conics.data_ptr<scalar_t>()),
                    opacities.data_ptr<scalar_t>(),
                    reinterpret_cast<vec4 *>(ray_planes.data_ptr<scalar_t>()),
                    image_width,
                    image_height,
                    tile_size,
                    tile_width,
                    tile_height,
                    tile_offsets.data_ptr<int32_t>(),
                    flatten_ids.data_ptr<int32_t>(),
                    n_isects,
                    out_transmittance.data_ptr<scalar_t>()
                );
        }
    );
}

} // namespace gsplat
