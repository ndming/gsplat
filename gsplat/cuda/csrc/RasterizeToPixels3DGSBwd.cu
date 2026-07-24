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
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

// MEDIAN selects the median-depth gradient: false routes v_render_medians to the
// T>0.5 crossing splat (matches the crossing reduction); true is the
// opacity-volume level-set gradient (implicit-function-theorem via rsigma).
// MEDIAN implies GEOMETRY.
template <uint32_t CDIM, bool GEOMETRY, bool MEDIAN, typename scalar_t>
__global__ void rasterize_to_pixels_3dgs_bwd_kernel(
    const uint32_t I,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    // fwd inputs
    const vec2 *__restrict__ means2d,         // [..., N, 2] or [nnz, 2]
    const vec3 *__restrict__ conics,          // [..., N, 3] or [nnz, 3]
    const scalar_t *__restrict__ colors,      // [..., N, CDIM] or [nnz, CDIM]
    const scalar_t *__restrict__ opacities,   // [..., N] or [nnz]
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] or [nnz, CDIM]
    const bool *__restrict__ masks,           // [..., tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // geometry fwd inputs (read only when GEOMETRY)
    const scalar_t *__restrict__ ray_planes, // [..., N, 4] or [nnz, 4]
    const scalar_t *__restrict__ normals_in, // [..., N, 3] or [nnz, 3]
    const scalar_t *__restrict__ Ks,         // [..., 3, 3]
    const int geometry_mode,                 // 0=RD, 1=MD, 2=PD
    // fwd outputs
    const scalar_t
        *__restrict__ render_alphas,      // [..., image_height, image_width, 1]
    const int32_t *__restrict__ last_ids, // [..., image_height, image_width]
    // geometry fwd outputs (read only when GEOMETRY)
    const scalar_t *__restrict__ render_normals,  // [..., H, W, 3] normalized
    const scalar_t *__restrict__ render_depths,   // [..., H, W, 1] expected z
    const scalar_t *__restrict__ render_medians,  // [..., H, W, 1] median z (MEDIAN)
    const scalar_t *__restrict__ normal_length,   // [..., H, W, 1]
    const int32_t *__restrict__ median_ids,       // [..., H, W]
    // grad outputs
    const scalar_t *__restrict__ v_render_colors, // [..., image_height,
                                                  // image_width, CDIM]
    const scalar_t
        *__restrict__ v_render_alphas, // [..., image_height, image_width, 1]
    const scalar_t *__restrict__ v_render_normals, // [..., H, W, 3] or null
    const scalar_t *__restrict__ v_render_depths,  // [..., H, W, 1] or null
    const scalar_t *__restrict__ v_render_medians, // [..., H, W, 1] or null
    // grad inputs
    vec2 *__restrict__ v_means2d_abs,  // [..., N, 2] or [nnz, 2]
    vec2 *__restrict__ v_means2d,      // [..., N, 2] or [nnz, 2]
    vec3 *__restrict__ v_conics,       // [..., N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_colors,   // [..., N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities, // [..., N] or [nnz]
    // geometry grad inputs (written only when GEOMETRY)
    scalar_t *__restrict__ v_ray_planes, // [..., N, 4] or [nnz, 4]
    scalar_t *__restrict__ v_normals     // [..., N, 3] or [nnz, 3]
) {
    auto block = cg::this_thread_block();
    uint32_t image_id = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += image_id * tile_height * tile_width;
    render_alphas += image_id * image_height * image_width;
    last_ids += image_id * image_height * image_width;
    v_render_colors += image_id * image_height * image_width * CDIM;
    v_render_alphas += image_id * image_height * image_width;
    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }
    if constexpr (GEOMETRY) {
        render_normals += image_id * image_height * image_width * 3;
        render_depths += image_id * image_height * image_width;
        if constexpr (MEDIAN) {
            render_medians += image_id * image_height * image_width;
        }
        normal_length += image_id * image_height * image_width;
        median_ids += image_id * image_height * image_width;
        v_render_normals += image_id * image_height * image_width * 3;
        v_render_depths += image_id * image_height * image_width;
        if (v_render_medians != nullptr) {
            v_render_medians += image_id * image_height * image_width;
        }
        Ks += image_id * 9;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

    // Ray-distance -> z-depth cosine (matches the forward)
    float rln = 0.f;
    if constexpr (GEOMETRY) {
        const float fx = Ks[0], fy = Ks[4], cx = Ks[2], cy = Ks[5];
        const float ndx = (px - cx) / fx;
        const float ndy = (py - cy) / fy;
        rln = rsqrtf(ndx * ndx + ndy * ndy + 1.f);
    }

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]
    vec3 *conic_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]
    float *rgbs_batch =
        (float *)&conic_batch[block_size]; // [block_size * CDIM]
    // Geometry batches live past the classic ones; only allocated (by the
    // launcher) and touched when GEOMETRY is true.
    vec4 *ray_plane_batch =
        reinterpret_cast<vec4 *>(&rgbs_batch[block_size * CDIM]); // [block_size]
    vec3 *normal_batch =
        reinterpret_cast<vec3 *>(&ray_plane_batch[block_size]); // [block_size]

    // this is the T AFTER the last gaussian in this pixel
    float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;
    // the contribution from gaussians behind the current one
    float buffer[CDIM] = {0.f};
    // behind-contribution accumulators for the geometry channels (GEOMETRY)
    float buffer_normal[3] = {0.f, 0.f, 0.f};
    float buffer_t = 0.f;
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // df/d_out for this pixel
    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }
    const float v_render_a = v_render_alphas[pix_id];

    // per-pixel geometry gradients
    float dL_dpixel_t = 0.f;       // d(loss)/d(this splat's plane depth), per unit weight
    float dL_dpixel_mt = 0.f;      // d(loss)/d(median plane depth)
    float v_depth_finalT = 0.f;    // depth's contribution routed through T_final
    float dL_dpixel_normal[3] = {0.f, 0.f, 0.f};
    if constexpr (GEOMETRY) {
        const vec3 vn = {
            v_render_normals[pix_id * 3 + 0],
            v_render_normals[pix_id * 3 + 1],
            v_render_normals[pix_id * 3 + 2]
        };
        if (geometry_mode == 2) {
            // PGSR unbiased plane depth. plane_depth = depth_acc / -(n_acc.ray + eps).
            // The rendered normal is RAW (n_acc), so v_render_normals maps directly to
            // dL/d(n_acc) with NO unit-normalize Jacobian; plane_depth adds a further
            // n_acc dependence via the denominator, folded into the normal seed. No
            // /alpha and no rln (both cancel), so v_depth_finalT stays 0.
            const float fx = Ks[0], fy = Ks[4], cx = Ks[2], cy = Ks[5];
            const float ndx = (px - cx) / fx;
            const float ndy = (py - cy) / fy;
            const vec3 nacc = {
                render_normals[pix_id * 3 + 0],
                render_normals[pix_id * 3 + 1],
                render_normals[pix_id * 3 + 2]
            };
            const float tmp = nacc.x * ndx + nacc.y * ndy + nacc.z + 1e-8f;
            const float vd = v_render_depths[pix_id];
            const float depth_acc = -render_depths[pix_id] * tmp; // recover numerator
            dL_dpixel_t = -vd / tmp;                              // dL/d(depth_acc)
            const float coef = vd * depth_acc / (tmp * tmp);      // dL/d(n_acc) via denom
            dL_dpixel_normal[0] = vn.x + coef * ndx;
            dL_dpixel_normal[1] = vn.y + coef * ndy;
            dL_dpixel_normal[2] = vn.z + coef;
        } else {
            const float w_final = render_alphas[pix_id];
            const float inv_w = w_final > 1e-6f ? 1.f / w_final : 0.f;
            const float vd = v_render_depths[pix_id];
            // out_depth = accum_depth / w_final = expected z-depth
            dL_dpixel_t = vd * inv_w * rln;
            v_depth_finalT = vd * render_depths[pix_id] * inv_w;
            dL_dpixel_mt =
                (v_render_medians != nullptr ? v_render_medians[pix_id] : 0.f) * rln;
            // Jacobian of L2-normalization of the accumulated normal
            const float nlen = normal_length[pix_id];
            const float denom = fmaxf(nlen, 1e-6f);
            const float large = nlen < 1e-6f ? 0.f : 1.f;
            const vec3 nmap = {
                render_normals[pix_id * 3 + 0],
                render_normals[pix_id * 3 + 1],
                render_normals[pix_id * 3 + 2]
            };
            const float dp = (vn.x * nmap.x + vn.y * nmap.y + vn.z * nmap.z) * large;
            dL_dpixel_normal[0] = (vn.x - dp * nmap.x) / denom;
            dL_dpixel_normal[1] = (vn.y - dp * nmap.y) / denom;
            dL_dpixel_normal[2] = (vn.z - dp * nmap.z) / denom;
        }
    }

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());

    // Opacity-volume median: implicit-function-theorem setup. Before the main
    // back-to-front pass, compute per-pixel dT/dts at the median depth so the
    // per-Gaussian pass can scale its gradients. Block-cooperative (front-to-back
    // shared reloads); mDepth is the forward median in ray-distance.
    // Reciprocal (one-sided) variant of the level-set gradient: the per-sub-step
    // 1/sqrt normalization is dropped, so only primitives whose peak sits at or
    // behind the level set vary the transmittance through their soft factor; those
    // the level set has already passed contribute only via their (1 - alpha) factor.
    const bool reciprocal = (geometry_mode == 3);
    float mDepth = 0.f;
    float dL_dmt_dT_dtm = 0.f;
    if constexpr (MEDIAN) {
        const float inv_rln = rln > 1e-8f ? 1.f / rln : 0.f;
        mDepth = (inside && render_medians[pix_id] > 0.f)
                     ? render_medians[pix_id] * inv_rln
                     : 0.f;
        // 1-based count (within range) of this pixel's last contributor
        const int lc =
            (inside && bin_final >= range_start) ? (bin_final - range_start + 1) : 0;
        __shared__ uint32_t s_bwd_max;
        if (tr == 0) {
            s_bwd_max = 0u;
        }
        block.sync();
        atomicMax(&s_bwd_max, (uint32_t)max(lc, 0));
        block.sync();
        const uint32_t max_contributor = s_bwd_max;
        const uint32_t oav_rounds =
            (max_contributor + block_size - 1) / block_size;

        float dT_dtm = 0.f;
        bool done = (mDepth == 0.f) || (lc == 0) || !inside;
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
            for (int t = 0; !done && t < bsz; ++t) {
                contributor++;
                done = contributor >= (uint32_t)lc;
                const vec3 xy_opac = xy_opacity_batch[t];
                const vec2 d = {xy_opac.x - px, xy_opac.y - py};
                const vec3 conic = conic_batch[t];
                const float sigma = 0.5f * (conic.x * d.x * d.x +
                                            conic.z * d.y * d.y) +
                                    conic.y * d.x * d.y;
                if (sigma < 0.f) {
                    continue;
                }
                const float alpha = min(0.999f, xy_opac.z * __expf(-sigma));
                if (alpha < ALPHA_THRESHOLD) {
                    continue;
                }
                const vec4 rp = ray_plane_batch[t];
                const float t_peak = rp.x * d.x + rp.y * d.y + rp.z;
                const float rsigma = rp.w;
                const float t_delta = (mDepth - t_peak) * rsigma;
                const float G_exp = __expf(-0.5f * t_delta * t_delta);
                const float Gt = alpha * G_exp;
                if (reciprocal) {
                    // Only the front side (peak at/behind the level set) varies
                    // the transmittance w.r.t. the level-set depth; coefficient
                    // 0.5 vs the symmetric variant's 0.25.
                    if (t_peak >= mDepth && rsigma > 0.f) {
                        dT_dtm += -0.5f * Gt / (1.f - Gt) * fabsf(t_delta) * rsigma;
                    }
                } else {
                    dT_dtm += -0.25f * Gt / (1.f - Gt) * fabsf(t_delta) * rsigma;
                }
            }
        }
        dL_dmt_dT_dtm = dL_dpixel_mt / fmaxf(-dT_dtm, 1e-7f);
    }

    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        // These values can be negative so must be int32 instead of uint32
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
            if constexpr (GEOMETRY) {
                ray_plane_batch[tr] = {
                    ray_planes[g * 4],
                    ray_planes[g * 4 + 1],
                    ray_planes[g * 4 + 2],
                    ray_planes[g * 4 + 3]
                };
                normal_batch[tr] = {
                    normals_in[g * 3], normals_in[g * 3 + 1], normals_in[g * 3 + 2]
                };
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;
             ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            float alpha;
            float opac;
            vec2 delta;
            vec3 conic;
            float vis;

            if (valid) {
                conic = conic_batch[t];
                vec3 xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                      conic.z * delta.y * delta.y) +
                              conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            float v_rgb_local[CDIM] = {0.f};
            vec3 v_conic_local = {0.f, 0.f, 0.f};
            vec2 v_xy_local = {0.f, 0.f};
            vec2 v_xy_abs_local = {0.f, 0.f};
            float v_opacity_local = 0.f;
            vec3 v_ray_plane_local = {0.f, 0.f, 0.f}; // GEOMETRY {gx,gy,tc}
            float v_rsigma_local = 0.f;               // MEDIAN: ray_plane.w grad
            vec3 v_normal_local = {0.f, 0.f, 0.f};    // GEOMETRY
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian
                float ra = 1.0f / (1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const float fac = alpha * T;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                float v_alpha = 0.f;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) *
                               v_render_c[k];
                }

                // geometry: normals composite like colors, plane depth
                // is an extra channel, plus the depth's T_final contribution.
                float dL_dt = 0.f, rp_x = 0.f, rp_y = 0.f;
                if constexpr (GEOMETRY) {
                    const vec3 nm = normal_batch[t];
                    v_alpha += (nm.x * T - buffer_normal[0] * ra) * dL_dpixel_normal[0];
                    v_alpha += (nm.y * T - buffer_normal[1] * ra) * dL_dpixel_normal[1];
                    v_alpha += (nm.z * T - buffer_normal[2] * ra) * dL_dpixel_normal[2];
                    v_normal_local = {
                        fac * dL_dpixel_normal[0],
                        fac * dL_dpixel_normal[1],
                        fac * dL_dpixel_normal[2]
                    };
                    const vec4 rp = ray_plane_batch[t];
                    const float tt = rp.x * delta.x + rp.y * delta.y + rp.z;
                    v_alpha += (tt * T - buffer_t * ra) * dL_dpixel_t;
                    dL_dt = fac * dL_dpixel_t;
                    if constexpr (MEDIAN) {
                        // Implicit-function-theorem gradient of the level-set depth
                        // at the fixed per-pixel crossing (tt == t_peak).
                        const float rsigma = rp.w;
                        const float t_delta = (mDepth - tt) * rsigma;
                        const float G_exp = __expf(-0.5f * t_delta * t_delta);
                        const float Gt = alpha * G_exp;
                        if (reciprocal) {
                            // One-sided variant. Front primitives (peak at/behind the
                            // crossing, mDepth <= tt) vary through omg = 1 - alpha*g;
                            // those the crossing has passed contribute only via their
                            // (1 - alpha) factor (alpha gradient, no t_peak/rsigma).
                            float dL_dopa_sigma;
                            if (mDepth > tt) {
                                dL_dopa_sigma = -0.5f * dL_dmt_dT_dtm / (1.f - alpha);
                            } else if (rsigma > 0.f) {
                                const float inv_omg = 1.f / (1.f - Gt);
                                dL_dopa_sigma = -0.5f * dL_dmt_dT_dtm * G_exp * inv_omg;
                                const float c = 0.5f * dL_dmt_dT_dtm * Gt * t_delta * inv_omg;
                                dL_dt += -c * rsigma;                // dL/dt_peak
                                v_rsigma_local = c * (mDepth - tt);  // dL/drsigma
                            } else {
                                dL_dopa_sigma = 0.f;
                            }
                            v_alpha += dL_dopa_sigma;
                        } else {
                            float dL_dGt = dL_dmt_dT_dtm * 0.25f / (1.f - Gt);
                            dL_dGt = (mDepth > tt) ? dL_dGt : -dL_dGt;
                            dL_dGt = (rsigma > 0.f) ? dL_dGt : 0.f;
                            const float dL_dopa_sigma =
                                dL_dGt * G_exp -
                                dL_dmt_dT_dtm *
                                    (t_delta > 0.f ? 0.5f / (1.f - alpha) : 0.f);
                            const float dL_ddelta = -dL_dGt * Gt * t_delta;
                            v_rsigma_local = dL_ddelta * (mDepth - tt);
                            dL_dt += -dL_ddelta * rsigma; // dL/dt_peak
                            v_alpha += dL_dopa_sigma;
                        }
                    } else {
                        // Crossing median: route the median grad to the T>0.5 splat.
                        if ((batch_end - (int32_t)t) == median_ids[pix_id]) {
                            dL_dt += dL_dpixel_mt;
                        }
                    }
                    v_ray_plane_local = {dL_dt * delta.x, dL_dt * delta.y, dL_dt};
                    rp_x = rp.x;
                    rp_y = rp.y;
                    v_alpha += -T_final * ra * v_depth_finalT;
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if (backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if (opac * vis <= 0.999f) {
                    const float v_sigma = -opac * vis * v_alpha;
                    v_conic_local = {
                        0.5f * v_sigma * delta.x * delta.x,
                        v_sigma * delta.x * delta.y,
                        0.5f * v_sigma * delta.y * delta.y
                    };
                    v_xy_local = {
                        v_sigma * (conic.x * delta.x + conic.y * delta.y),
                        v_sigma * (conic.y * delta.x + conic.z * delta.y)
                    };
                    v_opacity_local = vis * v_alpha;
                }

                // Plane depth moves with the 2D mean (delta = mean - pixel).
                if constexpr (GEOMETRY) {
                    v_xy_local.x += dL_dt * rp_x;
                    v_xy_local.y += dL_dt * rp_y;
                    const vec3 nm2 = normal_batch[t];
                    buffer_normal[0] += nm2.x * fac;
                    buffer_normal[1] += nm2.y * fac;
                    buffer_normal[2] += nm2.z * fac;
                    const vec4 rp2 = ray_plane_batch[t];
                    buffer_t += (rp2.x * delta.x + rp2.y * delta.y + rp2.z) * fac;
                }
                if (v_means2d_abs != nullptr) {
                    v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                }

#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }
            }
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum(v_conic_local, warp);
            warpSum(v_xy_local, warp);
            if (v_means2d_abs != nullptr) {
                warpSum(v_xy_abs_local, warp);
            }
            warpSum(v_opacity_local, warp);
            if constexpr (GEOMETRY) {
                warpSum(v_ray_plane_local, warp);
                warpSum(v_normal_local, warp);
                if constexpr (MEDIAN) {
                    warpSum(v_rsigma_local, warp);
                }
            }
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t]; // flatten index in [I * N] or [nnz]
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_conic_ptr = (float *)(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if (v_means2d_abs != nullptr) {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                gpuAtomicAdd(v_opacities + g, v_opacity_local);

                if constexpr (GEOMETRY) {
                    float *v_ray_ptr = (float *)(v_ray_planes) + 4 * g;
                    gpuAtomicAdd(v_ray_ptr, v_ray_plane_local.x);
                    gpuAtomicAdd(v_ray_ptr + 1, v_ray_plane_local.y);
                    gpuAtomicAdd(v_ray_ptr + 2, v_ray_plane_local.z);
                    if constexpr (MEDIAN) {
                        // rsigma (ray_plane.w) gradient from the opacity-volume median
                        gpuAtomicAdd(v_ray_ptr + 3, v_rsigma_local);
                    }
                    float *v_nrm_ptr = (float *)(v_normals) + 3 * g;
                    gpuAtomicAdd(v_nrm_ptr, v_normal_local.x);
                    gpuAtomicAdd(v_nrm_ptr + 1, v_normal_local.y);
                    gpuAtomicAdd(v_nrm_ptr + 2, v_normal_local.z);
                }
            }
        }
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_3dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor conics,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::optional<at::Tensor> backgrounds, // [..., 3]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    // gradients of outputs
    const at::Tensor v_render_colors, // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas, // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_conics,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities,                 // [..., N] or [nnz]
    // geometry outputs; ignored unless render_geometry
    const bool render_geometry,
    const int geometry_mode,
    const at::optional<at::Tensor> ray_planes,
    const at::optional<at::Tensor> normals_in,
    const at::optional<at::Tensor> Ks,
    const at::optional<at::Tensor> render_normals,
    const at::optional<at::Tensor> render_depths,
    const at::optional<at::Tensor> render_medians,
    const at::optional<at::Tensor> normal_length,
    const at::optional<at::Tensor> median_ids,
    const at::optional<at::Tensor> v_render_normals,
    const at::optional<at::Tensor> v_render_depths,
    const at::optional<at::Tensor> v_render_medians,
    at::optional<at::Tensor> v_ray_planes,
    at::optional<at::Tensor> v_normals
) {
    bool packed = means2d.dim() == 2;

    uint32_t N = packed ? 0 : means2d.size(-2); // number of gaussians
    uint32_t I = render_alphas.numel() / (image_height * image_width); // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // I * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {I, tile_height, tile_width};

    int64_t shmem_size =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(float) * CDIM);
    if (render_geometry) {
        shmem_size += tile_size * tile_size * (sizeof(vec4) + sizeof(vec3));
    }

    if (n_isects == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    // geometry pointers (nullptr when disabled)
    const float *ray_planes_ptr =
        render_geometry ? ray_planes.value().data_ptr<float>() : nullptr;
    const float *normals_ptr =
        render_geometry ? normals_in.value().data_ptr<float>() : nullptr;
    const float *Ks_ptr = render_geometry ? Ks.value().data_ptr<float>() : nullptr;
    const float *render_normals_ptr =
        render_geometry ? render_normals.value().data_ptr<float>() : nullptr;
    const float *render_depths_ptr =
        render_geometry ? render_depths.value().data_ptr<float>() : nullptr;
    const float *render_medians_ptr =
        (render_geometry && render_medians.has_value())
            ? render_medians.value().data_ptr<float>()
            : nullptr;
    const float *normal_length_ptr =
        render_geometry ? normal_length.value().data_ptr<float>() : nullptr;
    const int32_t *median_ids_ptr =
        render_geometry ? median_ids.value().data_ptr<int32_t>() : nullptr;
    const float *v_render_normals_ptr =
        render_geometry ? v_render_normals.value().data_ptr<float>() : nullptr;
    const float *v_render_depths_ptr =
        render_geometry ? v_render_depths.value().data_ptr<float>() : nullptr;
    const float *v_render_medians_ptr =
        (render_geometry && v_render_medians.has_value())
            ? v_render_medians.value().data_ptr<float>()
            : nullptr;
    float *v_ray_planes_ptr =
        render_geometry ? v_ray_planes.value().data_ptr<float>() : nullptr;
    float *v_normals_ptr =
        render_geometry ? v_normals.value().data_ptr<float>() : nullptr;

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
#define __RD_BWD_LAUNCH__(GEOM, MED)                                           \
    do {                                                                       \
        if (cudaFuncSetAttribute(                                             \
                rasterize_to_pixels_3dgs_bwd_kernel<CDIM, GEOM, MED, float>,   \
                cudaFuncAttributeMaxDynamicSharedMemorySize,                   \
                shmem_size) != cudaSuccess) {                                  \
            AT_ERROR(                                                          \
                "Failed to set maximum shared memory size (requested ",        \
                shmem_size,                                                    \
                " bytes), try lowering tile_size."                             \
            );                                                                 \
        }                                                                      \
        rasterize_to_pixels_3dgs_bwd_kernel<CDIM, GEOM, MED, float>            \
            <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>( \
                I, N, n_isects, packed,                                        \
                reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),           \
                reinterpret_cast<vec3 *>(conics.data_ptr<float>()),            \
                colors.data_ptr<float>(), opacities.data_ptr<float>(),         \
                backgrounds.has_value()                                        \
                    ? backgrounds.value().data_ptr<float>()                    \
                    : nullptr,                                                 \
                masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,  \
                image_width, image_height, tile_size, tile_width, tile_height, \
                tile_offsets.data_ptr<int32_t>(),                              \
                flatten_ids.data_ptr<int32_t>(),                               \
                ray_planes_ptr, normals_ptr, Ks_ptr, geometry_mode,            \
                render_alphas.data_ptr<float>(),                               \
                last_ids.data_ptr<int32_t>(),                                  \
                render_normals_ptr, render_depths_ptr, render_medians_ptr,     \
                normal_length_ptr, median_ids_ptr,                             \
                v_render_colors.data_ptr<float>(),                             \
                v_render_alphas.data_ptr<float>(),                             \
                v_render_normals_ptr, v_render_depths_ptr,                     \
                v_render_medians_ptr,                                          \
                v_means2d_abs.has_value()                                      \
                    ? reinterpret_cast<vec2 *>(                                \
                          v_means2d_abs.value().data_ptr<float>())             \
                    : nullptr,                                                 \
                reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),         \
                reinterpret_cast<vec3 *>(v_conics.data_ptr<float>()),          \
                v_colors.data_ptr<float>(), v_opacities.data_ptr<float>(),     \
                v_ray_planes_ptr, v_normals_ptr                                \
            );                                                                 \
    } while (0)

    if (render_geometry) {
        if constexpr (CDIM == 3) {
            // Modes 1 and 3 both use the opacity-volume level-set backward; mode 3
            // (reciprocal) drops the per-sub-step normalization at runtime.
            if (geometry_mode == 1 || geometry_mode == 3) {
                __RD_BWD_LAUNCH__(true, true);
            } else {
                __RD_BWD_LAUNCH__(true, false);
            }
        } else {
            AT_ERROR(
                "render_geometry requires 3 color channels, got ",
                CDIM
            );
        }
    } else {
        __RD_BWD_LAUNCH__(false, false);
    }
#undef __RD_BWD_LAUNCH__
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_3dgs_bwd_kernel<CDIM>(            \
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
        const at::Tensor render_alphas,                                        \
        const at::Tensor last_ids,                                             \
        const at::Tensor v_render_colors,                                      \
        const at::Tensor v_render_alphas,                                      \
        at::optional<at::Tensor> v_means2d_abs,                                \
        at::Tensor v_means2d,                                                  \
        at::Tensor v_conics,                                                   \
        at::Tensor v_colors,                                                   \
        at::Tensor v_opacities,                                                \
        const bool render_geometry,                                            \
        const int geometry_mode,                                               \
        const at::optional<at::Tensor> ray_planes,                             \
        const at::optional<at::Tensor> normals_in,                             \
        const at::optional<at::Tensor> Ks,                                     \
        const at::optional<at::Tensor> render_normals,                         \
        const at::optional<at::Tensor> render_depths,                          \
        const at::optional<at::Tensor> render_medians,                         \
        const at::optional<at::Tensor> normal_length,                          \
        const at::optional<at::Tensor> median_ids,                             \
        const at::optional<at::Tensor> v_render_normals,                       \
        const at::optional<at::Tensor> v_render_depths,                        \
        const at::optional<at::Tensor> v_render_medians,                       \
        at::optional<at::Tensor> v_ray_planes,                                 \
        at::optional<at::Tensor> v_normals                                     \
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
