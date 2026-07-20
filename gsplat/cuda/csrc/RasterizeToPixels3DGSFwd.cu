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

#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"

namespace gsplat {

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////

// MEDIAN selects the median-depth reduction written to render_medians:
//   false -> depth of the T>0.5 transmittance crossing (cheap; the default)
//   true  -> opacity-volume level-set median (block-cooperative binary search
//            over the per-Gaussian ray spread rsigma = ray_plane.w)
// MEDIAN implies GEOMETRY. The expected depth and normal are always produced.
template <uint32_t CDIM, bool GEOMETRY, bool MEDIAN, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_fwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec2 *__restrict__ means2d,         // [I, N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [I, N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [I, N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [I, N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [I, CDIM]
    const bool *__restrict__ masks,           // [I, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [I, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // geometry inputs (read only when GEOMETRY)
    const scalar_t *__restrict__ ray_planes, // [I, N, 4] or [nnz, 4]
    const scalar_t *__restrict__ normals,    // [I, N, 3] or [nnz, 3]
    const scalar_t *__restrict__ Ks,         // [I, 3, 3]
    scalar_t
        *__restrict__ render_colors, // [I, image_height, image_width, CDIM]
    scalar_t *__restrict__ render_alphas, // [I, image_height, image_width, 1]
    int32_t *__restrict__ last_ids,       // [I, image_height, image_width]
    // geometry outputs (written only when GEOMETRY)
    scalar_t *__restrict__ render_normals, // [I, H, W, 3] camera-space
    scalar_t *__restrict__ render_depths,  // [I, H, W, 1] expected z-depth
    scalar_t *__restrict__ render_medians, // [I, H, W, 1] median z-depth
    scalar_t *__restrict__ normal_length,  // [I, H, W, 1]
    int32_t *__restrict__ median_ids       // [I, H, W]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t image_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += image_id * tile_height * tile_width;
    render_colors += image_id * image_height * image_width * CDIM;
    render_alphas += image_id * image_height * image_width;
    last_ids += image_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }
    if constexpr (GEOMETRY) {
        render_normals += image_id * image_height * image_width * 3;
        render_depths += image_id * image_height * image_width;
        render_medians += image_id * image_height * image_width;
        normal_length += image_id * image_height * image_width;
        median_ids += image_id * image_height * image_width;
        Ks += image_id * 9;
    }

    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // Ray direction cosine that converts ray-distance depth to z-depth
    float rln = 0.f;
    if constexpr (GEOMETRY) {
        const float fx = Ks[0], fy = Ks[4], cx = Ks[2], cy = Ks[5];
        const float ndx = (px - cx) / fx;
        const float ndy = (py - cy) / fy;
        rln = rsqrtf(ndx * ndx + ndy * ndy + 1.f);
    }

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && inside && !masks[tile_id]) {
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
        if constexpr (GEOMETRY) {
            render_depths[pix_id] = 0.f;
            render_medians[pix_id] = 0.f;
            normal_length[pix_id] = 0.f;
            render_normals[pix_id * 3 + 0] = 0.f;
            render_normals[pix_id * 3 + 1] = 0.f;
            render_normals[pix_id * 3 + 2] = 0.f;
            median_ids[pix_id] = 0;
        }
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]
    // Geometry batches live past the classic ones; only allocated (by the
    // launcher) and touched when GEOMETRY is true.
    vec4 *ray_plane_batch =
        reinterpret_cast<vec4 *>(&conic_batch[block_size]); // [block_size]
    vec3 *normal_batch =
        reinterpret_cast<vec3 *>(&ray_plane_batch[block_size]); // [block_size]

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x
    // slower so we stick with float for now.
    float T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    float pix_out[CDIM] = {0.f};
    // geometry accumulators (used only when GEOMETRY)
    float depth_acc = 0.f;      // sum_i (t_i * alpha_i * T_i), ray-distance
    float median_depth = 0.f;   // ray-distance depth of the T>0.5 crossing splat
    float normal_acc[3] = {0.f, 0.f, 0.f};
    int32_t median_idx = 0;
    // 1-based ordinal (within this tile's range) of the last contributing splat;
    // only tracked/used by the opacity-volume median reduction.
    uint32_t last_contributor = 0;
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            if constexpr (GEOMETRY) {
                ray_plane_batch[tr] = {
                    ray_planes[g * 4],
                    ray_planes[g * 4 + 1],
                    ray_planes[g * 4 + 2],
                    ray_planes[g * 4 + 3]
                };
                normal_batch[tr] = {
                    normals[g * 3], normals[g * 3 + 1], normals[g * 3 + 2]
                };
            }
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3 conic = conic_batch[t];
            const vec3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;
            const vec2 delta = {xy_opac.x - px, xy_opac.y - py};
            const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                        conic.z * delta.y * delta.y) +
                                conic.y * delta.x * delta.y;
            float alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4f) { // this pixel is done: exclusive
                done = true;
                break;
            }

            int32_t g = id_batch[t];
            const float vis = alpha * T;
            const float *c_ptr = colors + g * CDIM;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }
            if constexpr (GEOMETRY) {
                const vec4 rp = ray_plane_batch[t];
                const vec3 nm = normal_batch[t];
                // Plane depth of this splat along the pixel ray (ray-distance).
                // delta = mean2d - pixel, matching the projection-side convention.
                const float depth_t = rp.x * delta.x + rp.y * delta.y + rp.z;
                depth_acc += depth_t * vis;
                normal_acc[0] += nm.x * vis;
                normal_acc[1] += nm.y * vis;
                normal_acc[2] += nm.z * vis;
                // T is the transmittance in front of this splat (pre-update),
                // so the last splat with T>0.5 is the median crossing.
                if (T > 0.5f) {
                    median_depth = depth_t;
                    median_idx = batch_start + t;
                }
            }
            cur_idx = batch_start + t;
            if constexpr (MEDIAN) {
                last_contributor = cur_idx - range_start + 1;
            }

            T = next_T;
        }
    }

    // Opacity-volume level-set median (block-cooperative). Runs for every thread
    // so the shared-memory reloads and block syncs stay collective. Works in
    // ray-distance; converted to z-depth (via rln) at write time.
    float oav_median = 0.f;
    if constexpr (GEOMETRY && MEDIAN) {
        constexpr int OAV_SPLIT = 8;         // sub-intervals per refinement
        constexpr int OAV_ITERS = 5;         // refinement iterations
        constexpr float OAV_RANGE = 0.4f;    // initial +/- search window (ray-dist)
        constexpr float OAV_MIN_T = 0.45f;   // pixel must be opaque enough

        // Max contributing count across the tile bounds the re-iteration.
        __shared__ uint32_t s_block_max;
        if (tr == 0) {
            s_block_max = 0u;
        }
        block.sync();
        atomicMax(&s_block_max, last_contributor);
        block.sync();
        const uint32_t max_contributor = s_block_max;
        const uint32_t oav_rounds =
            (max_contributor + block_size - 1) / block_size;

        // Seed the window at the T>0.5 crossing depth (ray-distance).
        float depth_min = fmaxf(median_depth - OAV_RANGE, 0.f);
        float depth_max = fmaxf(median_depth + OAV_RANGE, 0.f);
        bool in_range = T <= OAV_MIN_T;
        float T_p[OAV_SPLIT + 1];

        for (int it = 0; it < OAV_ITERS; ++it) {
            const bool first = (it == 0);
            const int s_lo = first ? 0 : 1;           // inclusive
            const int s_hi = first ? OAV_SPLIT + 1 : OAV_SPLIT; // exclusive
            for (int s = s_lo; s < s_hi; ++s) {
                T_p[s] = 1.f;
            }
            const float interval = (depth_max - depth_min) / (float)OAV_SPLIT;
            bool rdone = !in_range;
            uint32_t contributor = 0;
            int toDo = (int)max_contributor;
            for (uint32_t r = 0; r < oav_rounds; ++r, toDo -= block_size) {
                block.sync();
                const uint32_t progress = r * block_size + tr;
                if (progress < max_contributor) {
                    int32_t g = flatten_ids[range_start + progress];
                    const vec2 xy = means2d[g];
                    xy_opacity_batch[tr] = {xy.x, xy.y, opacities[g]};
                    conic_batch[tr] = conics[g];
                    ray_plane_batch[tr] = {
                        ray_planes[g * 4], ray_planes[g * 4 + 1],
                        ray_planes[g * 4 + 2], ray_planes[g * 4 + 3]
                    };
                }
                block.sync();
                const int bsz = min((int)block_size, toDo);
                for (int t = 0; !rdone && t < bsz; ++t) {
                    contributor++;
                    rdone = contributor >= last_contributor;
                    const vec3 xy_opac = xy_opacity_batch[t];
                    const vec2 delta = {xy_opac.x - px, xy_opac.y - py};
                    const vec3 conic = conic_batch[t];
                    const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                                conic.z * delta.y * delta.y) +
                                        conic.y * delta.x * delta.y;
                    if (sigma < 0.f) {
                        continue;
                    }
                    const float alpha = min(0.999f, xy_opac.z * __expf(-sigma));
                    if (alpha < ALPHA_THRESHOLD) {
                        continue;
                    }
                    const vec4 rp = ray_plane_batch[t];
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
            }
            if (first) {
                in_range = (T_p[0] >= 0.5f) && (T_p[OAV_SPLIT] <= 0.5f) && in_range;
            }
            int sid = 0;
            for (int p = 1; p < OAV_SPLIT; ++p) {
                sid = (T_p[p] >= 0.5f) ? p : sid;
            }
            depth_max = depth_min + (sid + 1) * interval;
            depth_min = depth_min + (sid + 0) * interval;
            T_p[0] = T_p[sid];
            T_p[OAV_SPLIT] = T_p[sid + 1];
        }
        const float w_max =
            __saturatef((T_p[0] - 0.5f) / (T_p[0] - T_p[OAV_SPLIT]));
        const float w_min = 1.f - w_max;
        oav_median = in_range ? (w_max * depth_max + w_min * depth_min) : 0.f;
    }

    if (inside) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        // We (should) store double precision as T would be used in backward
        // pass and it can be very small and causing large diff in gradients
        // with float32. However, double precision makes the backward pass 1.5x
        // slower so we stick with float for now.
        render_alphas[pix_id] = 1.0f - T;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr ? pix_out[k]
                                       : (pix_out[k] + T * backgrounds[k]);
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);

        if constexpr (GEOMETRY) {
            const float alpha_pix = 1.0f - T;
            // Expected z-depth: alpha-weighted mean plane depth, then
            // ray-distance -> z via rln.
            render_depths[pix_id] =
                alpha_pix > 1e-6f ? (depth_acc * rln) / alpha_pix : 0.f;
            // Median z-depth: opacity-volume level-set when MEDIAN, else the
            // depth of the T>0.5 crossing splat.
            if constexpr (MEDIAN) {
                render_medians[pix_id] = oav_median * rln;
            } else {
                render_medians[pix_id] = median_depth * rln;
            }
            const float nlen = sqrtf(
                normal_acc[0] * normal_acc[0] + normal_acc[1] * normal_acc[1] +
                normal_acc[2] * normal_acc[2]
            );
            normal_length[pix_id] = nlen;
            const float inv_nlen = 1.0f / fmaxf(nlen, 1e-6f);
            render_normals[pix_id * 3 + 0] = normal_acc[0] * inv_nlen;
            render_normals[pix_id * 3 + 1] = normal_acc[1] * inv_nlen;
            render_normals[pix_id * 3 + 2] = normal_acc[2] * inv_nlen;
            median_ids[pix_id] = median_idx;
        }
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,    // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities, // [..., N]  or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // outputs
    at::Tensor renders, // [..., image_height, image_width, channels]
    at::Tensor alphas,  // [..., image_height, image_width]
    at::Tensor last_ids, // [..., image_height, image_width]
    // geometry outputs; ignored unless render_geometry
    const bool render_geometry,
    const uint32_t reduction,                      // median flavor: 0=crossing, 1=opacity-volume
    const at::optional<at::Tensor> ray_planes,     // [..., N, 4]
    const at::optional<at::Tensor> normals_in,     // [..., N, 3]
    const at::optional<at::Tensor> Ks,             // [..., 3, 3]
    at::optional<at::Tensor> render_normals,       // [..., H, W, 3]
    at::optional<at::Tensor> render_depths,        // [..., H, W, 1]
    at::optional<at::Tensor> render_medians,       // [..., H, W, 1]
    at::optional<at::Tensor> normal_length,        // [..., H, W, 1]
    at::optional<at::Tensor> median_ids            // [..., H, W]
) {
    bool packed = means2d.dim() == 2;

    uint32_t N = packed ? 0 : means2d.size(-2); // number of gaussians
    uint32_t I = alphas.numel() / (image_height * image_width); // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // I * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {I, tile_height, tile_width};

    int64_t shmem_size =
        tile_size * tile_size * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3));
    if (render_geometry) {
        // extra shared batches for ray_planes (vec4) and normals (vec3)
        shmem_size += tile_size * tile_size * (sizeof(vec4) + sizeof(vec3));
    }

    // Geometry buffers resolved to raw pointers (nullptr when disabled).
    const float *ray_planes_ptr =
        render_geometry ? ray_planes.value().data_ptr<float>() : nullptr;
    const float *normals_ptr =
        render_geometry ? normals_in.value().data_ptr<float>() : nullptr;
    const float *Ks_ptr = render_geometry ? Ks.value().data_ptr<float>() : nullptr;
    float *render_normals_ptr =
        render_geometry ? render_normals.value().data_ptr<float>() : nullptr;
    float *render_depths_ptr =
        render_geometry ? render_depths.value().data_ptr<float>() : nullptr;
    float *render_medians_ptr =
        render_geometry ? render_medians.value().data_ptr<float>() : nullptr;
    float *normal_length_ptr =
        render_geometry ? normal_length.value().data_ptr<float>() : nullptr;
    int32_t *median_ids_ptr =
        render_geometry ? median_ids.value().data_ptr<int32_t>() : nullptr;

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
#define __RD_LAUNCH__(GEOM, MED)                                               \
    do {                                                                       \
        if (cudaFuncSetAttribute(                                              \
                rasterize_to_pixels_3dgs_fwd_kernel<CDIM, GEOM, MED, float>,   \
                cudaFuncAttributeMaxDynamicSharedMemorySize,                   \
                shmem_size) != cudaSuccess) {                                  \
            AT_ERROR(                                                          \
                "Failed to set maximum shared memory size (requested ",        \
                shmem_size,                                                    \
                " bytes), try lowering tile_size."                             \
            );                                                                 \
        }                                                                      \
        rasterize_to_pixels_3dgs_fwd_kernel<CDIM, GEOM, MED, float>            \
            <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>( \
                I,                                                             \
                N,                                                            \
                n_isects,                                                      \
                packed,                                                        \
                reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),           \
                reinterpret_cast<vec3 *>(conics.data_ptr<float>()),            \
                colors.data_ptr<float>(),                                      \
                opacities.data_ptr<float>(),                                   \
                backgrounds.has_value()                                        \
                    ? backgrounds.value().data_ptr<float>()                    \
                    : nullptr,                                                 \
                masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,  \
                image_width,                                                   \
                image_height,                                                  \
                tile_size,                                                     \
                tile_width,                                                    \
                tile_height,                                                   \
                tile_offsets.data_ptr<int32_t>(),                              \
                flatten_ids.data_ptr<int32_t>(),                               \
                ray_planes_ptr,                                                \
                normals_ptr,                                                   \
                Ks_ptr,                                                        \
                renders.data_ptr<float>(),                                     \
                alphas.data_ptr<float>(),                                      \
                last_ids.data_ptr<int32_t>(),                                  \
                render_normals_ptr,                                            \
                render_depths_ptr,                                             \
                render_medians_ptr,                                            \
                normal_length_ptr,                                             \
                median_ids_ptr                                                 \
            );                                                                 \
    } while (0)

    if (render_geometry) {
        // Geometry rendering is only compiled/allowed for RGB (3 channels).
        if constexpr (CDIM == 3) {
            if (reduction == 1) {
                __RD_LAUNCH__(true, true);
            } else {
                __RD_LAUNCH__(true, false);
            }
        } else {
            AT_ERROR(
                "render_geometry requires 3 color channels, got ",
                CDIM
            );
        }
    } else {
        __RD_LAUNCH__(false, false);
    }
#undef __RD_LAUNCH__
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_3dgs_fwd_kernel<CDIM>(            \
        const at::Tensor means2d,                                              \
        const at::Tensor conics,                                               \
        const at::Tensor colors,                                               \
        const at::Tensor opacities,                                            \
        const at::optional<at::Tensor> backgrounds,                            \
        const at::optional<at::Tensor> masks,                                  \
        uint32_t image_width,                                                  \
        uint32_t image_height,                                                 \
        uint32_t tile_size,                                                    \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        at::Tensor renders,                                                    \
        at::Tensor alphas,                                                     \
        at::Tensor last_ids,                                                   \
        const bool render_geometry,                                            \
        const uint32_t reduction,                                              \
        const at::optional<at::Tensor> ray_planes,                             \
        const at::optional<at::Tensor> normals_in,                             \
        const at::optional<at::Tensor> Ks,                                     \
        at::optional<at::Tensor> render_normals,                               \
        at::optional<at::Tensor> render_depths,                                \
        at::optional<at::Tensor> render_medians,                               \
        at::optional<at::Tensor> normal_length,                                \
        at::optional<at::Tensor> median_ids                                    \
    );

__INS__(1)
__INS__(2)
__INS__(3)
__INS__(4)
__INS__(5)
__INS__(8)
__INS__(9)
__INS__(16)
__INS__(17)
__INS__(32)
__INS__(33)
__INS__(64)
__INS__(65)
__INS__(128)
__INS__(129)
__INS__(256)
__INS__(257)
__INS__(512)
__INS__(513)
#undef __INS__

} // namespace gsplat
