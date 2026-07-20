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

#pragma once

#include <ATen/core/Tensor.h>

namespace gsplat {

// Launcher for the geometry-sampling forward kernel (see SampleGeometry3DGSFwd.cu).
void launch_sample_geometry_3dgs_fwd_kernel(
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
    at::Tensor out_depth,
    at::Tensor out_alpha,
    at::Tensor out_normal
);

// Launcher for the geometry-sampling backward kernel (see SampleGeometry3DGSBwd.cu).
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
    const at::Tensor v_depth,
    const at::Tensor v_alpha,
    const at::optional<at::Tensor> v_normal,
    at::Tensor v_means2d,
    at::Tensor v_conics,
    at::Tensor v_opacities,
    at::Tensor v_ray_planes,
    at::optional<at::Tensor> v_normals,
    at::Tensor v_points2d
);

} // namespace gsplat
