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

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"
#include "Ops.h"
#include "Sample.h"

namespace gsplat {

std::tuple<at::Tensor, at::Tensor, at::Tensor> sample_geometry_3dgs_fwd(
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
    const bool sample_normals
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(points2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(opacities);
    CHECK_INPUT(ray_planes);
    CHECK_INPUT(Ks);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    if (sample_normals) {
        TORCH_CHECK(normals.has_value(), "sample_normals requires normals");
        CHECK_INPUT(normals.value());
    }

    const uint32_t P = points2d.size(0);
    auto opt = means2d.options();
    at::Tensor out_depth = at::empty({P}, opt);
    at::Tensor out_alpha = at::empty({P}, opt);
    at::Tensor out_normal =
        sample_normals ? at::empty({P, 3}, opt) : at::empty({0}, opt);

    launch_sample_geometry_3dgs_fwd_kernel(
        points2d,
        means2d,
        conics,
        opacities,
        ray_planes,
        normals,
        Ks,
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        flatten_ids,
        sample_normals,
        out_depth,
        out_alpha,
        out_normal
    );

    return std::make_tuple(out_depth, out_alpha, out_normal);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
sample_geometry_3dgs_bwd(
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
    const at::Tensor v_depth,      // [P]
    const at::Tensor v_alpha,      // [P]
    const at::optional<at::Tensor> v_normal // [P, 3]
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(points2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(opacities);
    CHECK_INPUT(ray_planes);
    CHECK_INPUT(Ks);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(v_depth);
    CHECK_INPUT(v_alpha);
    if (sample_normals) {
        TORCH_CHECK(normals.has_value() && v_normal.has_value(),
                    "sample_normals requires normals and v_normal");
        CHECK_INPUT(normals.value());
        CHECK_INPUT(v_normal.value());
    }

    const uint32_t N = means2d.size(0);
    const uint32_t P = points2d.size(0);
    auto opt = means2d.options();
    // Gaussian grads are accumulated via atomicAdd -> zero-initialize.
    at::Tensor v_means2d = at::zeros({N, 2}, opt);
    at::Tensor v_conics = at::zeros({N, 3}, opt);
    at::Tensor v_opacities = at::zeros({N}, opt);
    at::Tensor v_ray_planes = at::zeros({N, 4}, opt);
    at::Tensor v_normals =
        sample_normals ? at::zeros({N, 3}, opt) : at::empty({0}, opt);
    at::Tensor v_points2d = at::empty({P, 2}, opt);

    launch_sample_geometry_3dgs_bwd_kernel(
        points2d, means2d, conics, opacities, ray_planes, normals, Ks,
        image_width, image_height, tile_size, tile_offsets, flatten_ids,
        sample_normals, v_depth, v_alpha, v_normal,
        v_means2d, v_conics, v_opacities, v_ray_planes, v_normals, v_points2d
    );

    return std::make_tuple(
        v_means2d, v_conics, v_opacities, v_ray_planes, v_normals, v_points2d
    );
}

} // namespace gsplat
