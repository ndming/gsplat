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

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"
#include "Integrate.h"
#include "Ops.h"

namespace gsplat {

at::Tensor integrate_transmittance_3dgs_fwd(
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
    const at::Tensor flatten_ids   // [n_isects]
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(points2d);
    CHECK_INPUT(point_t);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(opacities);
    CHECK_INPUT(ray_planes);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);

    const uint32_t P = points2d.size(0);
    at::Tensor out_transmittance = at::empty({P}, means2d.options());

    launch_integrate_transmittance_3dgs_fwd_kernel(
        points2d,
        point_t,
        means2d,
        conics,
        opacities,
        ray_planes,
        image_width,
        image_height,
        tile_size,
        tile_offsets,
        flatten_ids,
        out_transmittance
    );

    return out_transmittance;
}

} // namespace gsplat
