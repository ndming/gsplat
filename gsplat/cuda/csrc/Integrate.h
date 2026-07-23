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

// Launcher for the point-transmittance forward kernel (see
// IntegrateTransmittance3DGSFwd.cu). Forward-only.
void launch_integrate_transmittance_3dgs_fwd_kernel(
    const at::Tensor points2d,
    const at::Tensor point_t,
    const at::Tensor means2d,
    const at::Tensor conics,
    const at::Tensor opacities,
    const at::Tensor ray_planes,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const at::Tensor tile_offsets,
    const at::Tensor flatten_ids,
    at::Tensor out_transmittance
);

} // namespace gsplat
