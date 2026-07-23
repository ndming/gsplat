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
#include "Projection.h"
#include "Utils.cuh"

namespace gsplat {

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// Per-Gaussian depth-plane coefficients and camera-space normal. The depth of
// the Gaussian's tangent plane along the ray through pixel offset
// d = (mean2d - pix) is  t = rp.x*d.x + rp.y*d.y + rp.z, with rp.z = tc (ray
// distance to the center) and rp.w = rsigma (per-Gaussian spread along the ray,
// consumed only by the opacity-volume median reduction).
// ---------------------------------------------------------------------------
inline __device__ void compute_ray_plane_normal(
    const vec3 mean_c,
    const mat3 W,     // TRANSPOSED world->camera rotation (glm column-major
                      // = glm::transpose(gsplat W2C rotation)). See call sites.
    const vec4 quat,  // [w, x, y, z], unnormalized
    const vec3 scale, // per-axis scale (already activated)
    const float fx,
    const float fy,
    const int geometry_mode, // 0=RD, 1=MD (both -> ray-plane); 2=PD (plane)
    vec4 &ray_plane,  // out: {gx, gy, tc, rsigma}  (PD: {0,0,dist,0})
    vec3 &normal_out  // out: camera-space unit normal
) {
    const vec3 t = mean_c;
    const float tc = sqrtf(t.x * t.x + t.y * t.y + t.z * t.z);
    const float u = t.x / t.z;
    const float v = t.y / t.z;

    // inverse scaling matrix
    const float sx = scale[0], sy = scale[1], sz = scale[2];
    mat3 S_inv = mat3(1.f / sx, 0.f, 0.f, 0.f, 1.f / sy, 0.f, 0.f, 0.f, 1.f / sz);

    // rotation from the normalized quaternion (column-major convention;
    // NOT gsplat's quat_to_rotmat, which is the transpose of this)
    const float qn =
        rsqrtf(quat[0] * quat[0] + quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);
    const float r = quat[0] * qn, x = quat[1] * qn, y = quat[2] * qn, z = quat[3] * qn;
    mat3 R = mat3(
        1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
        2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
        2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
    );

    // inverse camera-space covariance, with a rank-1 fallback for flat Gaussians
    int min_id = (sy < sx) ? 1 : 0;
    if (sz < scale[min_id]) {
        min_id = 2;
    }

    // PGSR unbiased plane depth: the per-Gaussian plane is the disk through the
    // center with the smallest-scale axis as normal. rmin = min_id column of the
    // standard body->world rotation (= min_id row of glm R, which is R_std^T);
    // n_cam = R_cw * rmin (glm row-vec mult v*M = M^T*v, with W = R_cw^T). Oriented
    // toward the camera; distance = |n_cam . mean_c|. Normal stays UNNORMALIZED
    // (already unit) to match PGSR (no glm::normalize here).
    if (geometry_mode == 2) {
        vec3 rmin = {R[0][min_id], R[1][min_id], R[2][min_id]};
        vec3 n_cam = rmin * W;
        if (glm::dot(n_cam, mean_c) > 0.f) {
            n_cam = -n_cam;
        }
        const float dist = fabsf(glm::dot(n_cam, mean_c));
        ray_plane = {0.f, 0.f, dist, 0.f};
        normal_out = n_cam;
        return;
    }

    const bool well_conditioned = scale[min_id] > 1e-7f;

    mat3 cov_cam_inv;
    if (well_conditioned) {
        mat3 M_inv = S_inv * R * W;
        cov_cam_inv = glm::transpose(M_inv) * M_inv;
    } else {
        vec3 rmin = {R[0][min_id], R[1][min_id], R[2][min_id]};
        vec3 M_inv = rmin * W;
        cov_cam_inv = glm::outerProduct(M_inv, M_inv);
    }

    const vec3 uvh = {u, v, 1.f};
    const vec3 uvh_m = cov_cam_inv * uvh;
    const vec3 uvh_mn = glm::normalize(uvh_m);

    if (isnan(uvh_mn.x)) {
        ray_plane = {0.f, 0.f, 0.f, 0.f};
        normal_out = {0.f, 0.f, -1.f};
        return;
    }

    const float u2 = u * u, v2 = v * v, uv = u * v;
    const float l = tc;
    mat3 nJ_inv = mat3(v2 + 1.f, -uv, 0.f, -uv, u2 + 1.f, 0.f, -u, -v, 0.f);

    const float vbn = glm::dot(uvh_mn, uvh);
    const float vb = glm::dot(uvh_m, uvh);
    const float ray_len2 = u2 + v2 + 1.f;
    const float factor_normal = l / ray_len2;
    vec3 plane = nJ_inv * (uvh_mn / fmaxf(vbn, 1e-7f));
    const float rsigma = well_conditioned ? sqrtf(fmaxf(vb / ray_len2, 0.f)) : 0.f;

    ray_plane = {plane[0] * factor_normal / fx, plane[1] * factor_normal / fy, tc, rsigma};

    vec3 ray_normal_vector = {-plane[0] * factor_normal, -plane[1] * factor_normal, -1.f};
    mat3 nJ = mat3(
        1.f / t.z, 0.f, -t.x / (t.z * t.z),
        0.f, 1.f / t.z, -t.y / (t.z * t.z),
        t.x / l, t.y / l, t.z / l
    );
    normal_out = glm::normalize(nJ * ray_normal_vector);
}

// ---------------------------------------------------------------------------
// VJP of compute_ray_plane_normal (geometry-only: no conic/coef path, no
// frustum clamp, to match the forward here). Produces the geometry
// contributions to {mean_c, scale, quat_raw}. `W` is the transposed
// world->camera rotation (see the forward).
// ---------------------------------------------------------------------------
inline __device__ void compute_ray_plane_normal_vjp(
    const vec3 mean_c,
    const mat3 W,
    const vec4 quat_raw,
    const vec3 scale,
    const float fx,
    const float fy,
    const int geometry_mode, // 0=RD, 1=MD, 2=PD
    const vec4 v_ray_plane, // {dL/dgx, dL/dgy, dL/dtc, -}
    const vec3 v_normal,    // dL/dnormal (camera space, unnormalized grad)
    vec3 &v_mean_c,         // out (added to)
    vec3 &v_scale_out,      // out
    vec4 &v_quat_out        // out (w.r.t. raw quaternion)
) {
    vec3 t = mean_c;

    // gradient of tc = |mean_c| (ray_plane.z)
    const float rtc = rsqrtf(t.x * t.x + t.y * t.y + t.z * t.z);
    const float dL_dtc = v_ray_plane[2];
    const vec3 dL_dt_tc = {t.x * rtc * dL_dtc, t.y * rtc * dL_dtc, t.z * rtc * dL_dtc};

    // undo the /focal baked into ray_plane.xy in the forward
    const float dL_drp_x = v_ray_plane[0] / fx;
    const float dL_drp_y = v_ray_plane[1] / fy;

    const float u = t.x / t.z;
    const float v = t.y / t.z;

    // scaling and rotation (column-major convention, normalized quaternion)
    const float sx = scale[0], sy = scale[1], sz = scale[2];
    mat3 S_inv = mat3(1.f / sx, 0.f, 0.f, 0.f, 1.f / sy, 0.f, 0.f, 0.f, 1.f / sz);
    const float qn = rsqrtf(
        quat_raw[0] * quat_raw[0] + quat_raw[1] * quat_raw[1] +
        quat_raw[2] * quat_raw[2] + quat_raw[3] * quat_raw[3]
    );
    const float qr = quat_raw[0] * qn, qx = quat_raw[1] * qn, qy = quat_raw[2] * qn,
                qz = quat_raw[3] * qn;
    mat3 R = mat3(
        1.f - 2.f * (qy * qy + qz * qz), 2.f * (qx * qy - qr * qz), 2.f * (qx * qz + qr * qy),
        2.f * (qx * qy + qr * qz), 1.f - 2.f * (qx * qx + qz * qz), 2.f * (qy * qz - qr * qx),
        2.f * (qx * qz - qr * qy), 2.f * (qy * qz + qr * qx), 1.f - 2.f * (qx * qx + qy * qy)
    );

    int min_id = (sy < sx) ? 1 : 0;
    if (sz < scale[min_id]) {
        min_id = 2;
    }

    // ---- PGSR plane-depth VJP -------------------------------------------------
    // fwd: rmin = R_std[:,min_id] (= glm R's min_id row); n0 = rmin*W (= R_cw*rmin);
    //      n_cam = flip*n0 (orient to cam); dist = -dot(n_cam, mean_c) (>=0).
    // Scale gets NO gradient (argmin index detached). rmin(quat) is differentiated
    // analytically per min_id, then chained through quaternion normalization.
    if (geometry_mode == 2) {
        vec3 rmin = {R[0][min_id], R[1][min_id], R[2][min_id]};
        vec3 n0 = rmin * W;
        const float flip = (glm::dot(n0, mean_c) > 0.f) ? -1.f : 1.f;
        vec3 n_cam = n0 * flip;
        const float dL_ddist = v_ray_plane[2];
        // dist = -dot(n_cam, mean_c)
        vec3 dL_dn_cam = v_normal - dL_ddist * mean_c;
        v_mean_c += (-dL_ddist) * n_cam;
        vec3 dL_dn0 = dL_dn_cam * flip;
        vec3 dL_drmin = W * dL_dn0; // n0 = W^T*rmin => dL/drmin = W*dL/dn0

        // d(rmin)/d(normalized quat), per min_id column of R_std(qr,qx,qy,qz).
        const float g0 = dL_drmin.x, g1 = dL_drmin.y, g2 = dL_drmin.z;
        vec4 dL_dqn = {0.f, 0.f, 0.f, 0.f};
        if (min_id == 0) {
            // rmin = (1-2y^2-2z^2, 2xy+2rz, 2xz-2ry)
            dL_dqn[0] = g1 * (2.f * qz) + g2 * (-2.f * qy);
            dL_dqn[1] = g1 * (2.f * qy) + g2 * (2.f * qz);
            dL_dqn[2] = g0 * (-4.f * qy) + g1 * (2.f * qx) + g2 * (-2.f * qr);
            dL_dqn[3] = g0 * (-4.f * qz) + g1 * (2.f * qr) + g2 * (2.f * qx);
        } else if (min_id == 1) {
            // rmin = (2xy-2rz, 1-2x^2-2z^2, 2yz+2rx)
            dL_dqn[0] = g0 * (-2.f * qz) + g2 * (2.f * qx);
            dL_dqn[1] = g0 * (2.f * qy) + g1 * (-4.f * qx) + g2 * (2.f * qr);
            dL_dqn[2] = g0 * (2.f * qx) + g2 * (2.f * qz);
            dL_dqn[3] = g0 * (-2.f * qr) + g1 * (-4.f * qz) + g2 * (2.f * qy);
        } else {
            // rmin = (2xz+2ry, 2yz-2rx, 1-2x^2-2y^2)
            dL_dqn[0] = g0 * (2.f * qy) + g1 * (-2.f * qx);
            dL_dqn[1] = g0 * (2.f * qz) + g1 * (-2.f * qr) + g2 * (-4.f * qx);
            dL_dqn[2] = g0 * (2.f * qr) + g1 * (2.f * qz) + g2 * (-4.f * qy);
            dL_dqn[3] = g0 * (2.f * qx) + g1 * (2.f * qy);
        }
        // backprop through quaternion normalization: dL/dq_raw
        const vec4 q_n = {qr, qx, qy, qz};
        const float qdot =
            q_n[0] * dL_dqn[0] + q_n[1] * dL_dqn[1] + q_n[2] * dL_dqn[2] + q_n[3] * dL_dqn[3];
        v_quat_out[0] = (dL_dqn[0] - q_n[0] * qdot) * qn;
        v_quat_out[1] = (dL_dqn[1] - q_n[1] * qdot) * qn;
        v_quat_out[2] = (dL_dqn[2] - q_n[2] * qdot) * qn;
        v_quat_out[3] = (dL_dqn[3] - q_n[3] * qdot) * qn;
        v_scale_out = {0.f, 0.f, 0.f};
        return;
    }

    const bool well_conditioned = scale[min_id] > 1e-7f;

    mat3 cov_cam_inv;
    mat3 Vrk_inv;
    if (well_conditioned) {
        mat3 M_inv = S_inv * R * W;
        cov_cam_inv = glm::transpose(M_inv) * M_inv;
        mat3 M_inv2 = S_inv * R;
        Vrk_inv = glm::transpose(M_inv2) * M_inv2;
    } else {
        vec3 rmin = {R[0][min_id], R[1][min_id], R[2][min_id]};
        vec3 M_inv = rmin * W;
        cov_cam_inv = glm::outerProduct(M_inv, M_inv);
        Vrk_inv = glm::outerProduct(rmin, rmin);
    }

    const vec3 uvh = {u, v, 1.f};
    const vec3 uvh_m = cov_cam_inv * uvh;
    const vec3 uvh_mn = glm::normalize(uvh_m);

    const float u2 = u * u, v2 = v * v, uv = u * v;

    mat3 dL_dVrk;
    vec3 dL_dr;
    float dL_du, dL_dv, dL_dz;
    {
        const float vb = glm::dot(uvh_m, uvh);
        const float l = 1.f / rtc; // |t|
        mat3 nJ = mat3(
            1.f / t.z, 0.f, -(t.x) / (t.z * t.z),
            0.f, 1.f / t.z, -(t.y) / (t.z * t.z),
            t.x / l, t.y / l, t.z / l
        );
        mat3 nJ_inv = mat3(v2 + 1.f, -uv, 0.f, -uv, u2 + 1.f, 0.f, -u, -v, 0.f);

        const float clamp_vb = fmaxf(vb, 1e-7f);
        const float vbn = glm::dot(uvh_mn, uvh);
        const float clamp_vbn = fmaxf(vbn, 1e-7f);
        const float ray_len2 = u2 + v2 + 1.f;
        const float ray_len_inv = rsqrtf(ray_len2);
        const float factor_normal = l / ray_len2;
        const vec3 uvh_m_vb = uvh_mn / clamp_vbn;
        vec3 plane = nJ_inv * uvh_m_vb;
        vec3 ray_normal_vector = {-plane.x * factor_normal, -plane.y * factor_normal, -1.f};

        vec3 cam_normal_vector = nJ * ray_normal_vector;
        vec3 normal_vector = glm::normalize(cam_normal_vector);
        // NOTE (deliberate): rlv is taken from the ALREADY-normalized normal_vector
        // (so rlv ~= 1), NOT from the raw cam_normal_vector. The mathematically correct
        // normalize-VJP factor is 1/|cam_normal_vector| (rsqrtf of the un-normalized
        // vector); using ~1 drops that 1/|x|, over-weighting the normal gradient for
        // tilted disks and flattening them harder. This matches the reference training
        // dynamics on purpose; see the port notes before "fixing" it.
        const float rlv = rsqrtf(
            normal_vector.x * normal_vector.x +
            normal_vector.y * normal_vector.y +
            normal_vector.z * normal_vector.z
        );
        vec3 dL_dcam_normal_vector =
            (v_normal - normal_vector * glm::dot(normal_vector, v_normal)) * rlv;
        vec3 dL_dray_normal_vector = glm::transpose(nJ) * dL_dcam_normal_vector;
        mat3 dL_dnJ = glm::outerProduct(dL_dcam_normal_vector, ray_normal_vector);
        const float dL_dfactor_normal =
            plane.x * (-dL_dray_normal_vector.x + dL_drp_x) +
            plane.y * (-dL_dray_normal_vector.y + dL_drp_y);

        vec2 dL_dplane = {
            (-dL_dray_normal_vector.x + dL_drp_x) * factor_normal,
            (-dL_dray_normal_vector.y + dL_drp_y) * factor_normal
        };
        vec3 dL_dplane_append = {dL_dplane.x, dL_dplane.y, 0.f};

        const float aux = dL_dplane.x * plane.x + dL_dplane.y * plane.y;
        vec3 W_uvh = W * uvh;
        vec3 dL_duvh_plane =
            2.f * (-aux) * uvh_m_vb +
            (cov_cam_inv / clamp_vb) * glm::transpose(nJ_inv) * dL_dplane_append;

        const float aux_nJ =
            (-dL_dnJ[2][0] * u - dL_dnJ[2][1] * v - dL_dnJ[2][2]) / ray_len2 * ray_len_inv;
        const float dL_du_nJ = -dL_dnJ[0][2] / t.z + dL_dnJ[2][0] * ray_len_inv + aux_nJ * u;
        const float dL_dv_nJ = -dL_dnJ[1][2] / t.z + dL_dnJ[2][1] * ray_len_inv + aux_nJ * v;
        const float dL_dz_nJ =
            (dL_dnJ[0][0] + dL_dnJ[1][1] - dL_dnJ[0][2] * u - dL_dnJ[1][2] * v) /
            (-t.z * t.z);

        mat3 dL_dnJ_inv = glm::outerProduct(dL_dplane_append, uvh_m_vb);
        const float dL_du_plane = dL_duvh_plane.x +
                                  (dL_dnJ_inv[0][1] + dL_dnJ_inv[1][0]) * (-v) +
                                  2.f * dL_dnJ_inv[1][1] * u - dL_dnJ_inv[2][0];
        const float dL_dv_plane = dL_duvh_plane.y +
                                  (dL_dnJ_inv[0][1] + dL_dnJ_inv[1][0]) * (-u) +
                                  2.f * dL_dnJ_inv[0][0] * v - dL_dnJ_inv[2][1];

        const float aux_factor = dL_dfactor_normal * (-t.z / ray_len2 * ray_len_inv);
        const float dL_du_factor = aux_factor * u;
        const float dL_dv_factor = aux_factor * v;
        const float dL_dz_factor = dL_dfactor_normal * ray_len_inv;

        dL_du = dL_du_nJ + dL_du_plane + dL_du_factor;
        dL_dv = dL_dv_nJ + dL_dv_plane + dL_dv_factor;
        dL_dz = dL_dz_nJ + dL_dz_factor;

        const float dL_dvb_xvb = -aux;
        if (well_conditioned) {
            dL_dVrk = -glm::outerProduct(
                Vrk_inv * W_uvh,
                Vrk_inv *
                    (W * glm::transpose(nJ_inv) * dL_dplane_append + W_uvh * dL_dvb_xvb)
            );
            dL_dVrk = dL_dVrk / vb;
            dL_dr = {0.f, 0.f, 0.f};
        } else {
            dL_dVrk = mat3(0.f);
            vec3 nJ_inv_dL_dplane_xvb =
                glm::transpose(nJ_inv) * vec3(dL_dplane.x, dL_dplane.y, 0.f);
            mat3 dL_dVrk_inv =
                glm::outerProduct(W_uvh, W_uvh * dL_dvb_xvb + W * nJ_inv_dL_dplane_xvb) / vb;
            vec3 eigenvector_min = {R[0][min_id], R[1][min_id], R[2][min_id]};
            dL_dr = (dL_dVrk_inv + glm::transpose(dL_dVrk_inv)) * eigenvector_min;
        }
    }

    // dL/d mean_c from the (u, v, z) chain plus the tc term
    const float tz = 1.f / t.z;
    const float tz2 = tz * tz;
    v_mean_c.x += dL_du * tz + dL_dt_tc.x;
    v_mean_c.y += dL_dv * tz + dL_dt_tc.y;
    v_mean_c.z += -(dL_du * t.x + dL_dv * t.y) * tz2 + dL_dz + dL_dt_tc.z;

    // dL/d cov3D (world), symmetric packing
    float dL_dcov3D[6];
    dL_dcov3D[0] = dL_dVrk[0][0];
    dL_dcov3D[1] = dL_dVrk[0][1] + dL_dVrk[1][0];
    dL_dcov3D[2] = dL_dVrk[0][2] + dL_dVrk[2][0];
    dL_dcov3D[3] = dL_dVrk[1][1];
    dL_dcov3D[4] = dL_dVrk[1][2] + dL_dVrk[2][1];
    dL_dcov3D[5] = dL_dVrk[2][2];

    // computeCov3D VJP (Vrk = M^T M with M = S*R), column-major convention
    mat3 S = mat3(sx, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, sz);
    mat3 M = S * R;
    mat3 dL_dSigma = mat3(
        dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
        0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
        0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
    );
    mat3 dL_dM = 2.f * M * dL_dSigma;
    mat3 Rt = glm::transpose(R);
    mat3 dL_dMt = glm::transpose(dL_dM);
    v_scale_out[0] = glm::dot(Rt[0], dL_dMt[0]);
    v_scale_out[1] = glm::dot(Rt[1], dL_dMt[1]);
    v_scale_out[2] = glm::dot(Rt[2], dL_dMt[2]);
    dL_dMt[0] *= sx;
    dL_dMt[1] *= sy;
    dL_dMt[2] *= sz;
    dL_dMt[min_id] += dL_dr;

    // dL/d normalized quaternion
    vec4 dL_dqn;
    dL_dqn[0] = 2.f * qz * (dL_dMt[0][1] - dL_dMt[1][0]) +
                2.f * qy * (dL_dMt[2][0] - dL_dMt[0][2]) +
                2.f * qx * (dL_dMt[1][2] - dL_dMt[2][1]);
    dL_dqn[1] = 2.f * qy * (dL_dMt[1][0] + dL_dMt[0][1]) +
                2.f * qz * (dL_dMt[2][0] + dL_dMt[0][2]) +
                2.f * qr * (dL_dMt[1][2] - dL_dMt[2][1]) -
                4.f * qx * (dL_dMt[2][2] + dL_dMt[1][1]);
    dL_dqn[2] = 2.f * qx * (dL_dMt[1][0] + dL_dMt[0][1]) +
                2.f * qr * (dL_dMt[2][0] - dL_dMt[0][2]) +
                2.f * qz * (dL_dMt[1][2] + dL_dMt[2][1]) -
                4.f * qy * (dL_dMt[2][2] + dL_dMt[0][0]);
    dL_dqn[3] = 2.f * qr * (dL_dMt[0][1] - dL_dMt[1][0]) +
                2.f * qx * (dL_dMt[2][0] + dL_dMt[0][2]) +
                2.f * qy * (dL_dMt[1][2] + dL_dMt[2][1]) -
                4.f * qz * (dL_dMt[1][1] + dL_dMt[0][0]);

    // backprop through quaternion normalization: dL/dq_raw
    const vec4 q_n = {quat_raw[0] * qn, quat_raw[1] * qn, quat_raw[2] * qn, quat_raw[3] * qn};
    const float qdot =
        q_n[0] * dL_dqn[0] + q_n[1] * dL_dqn[1] + q_n[2] * dL_dqn[2] + q_n[3] * dL_dqn[3];
    v_quat_out[0] = (dL_dqn[0] - q_n[0] * qdot) * qn;
    v_quat_out[1] = (dL_dqn[1] - q_n[1] * qdot) * qn;
    v_quat_out[2] = (dL_dqn[2] - q_n[2] * qdot) * qn;
    v_quat_out[3] = (dL_dqn[3] - q_n[3] * qdot) * qn;
}

template <typename scalar_t>
__global__ void projection_ewa_3dgs_fused_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [B, N, 3]
    const scalar_t *__restrict__ covars,   // [B, N, 6] optional
    const scalar_t *__restrict__ quats,    // [B, N, 4] optional
    const scalar_t *__restrict__ scales,   // [B, N, 3] optional
    const scalar_t *__restrict__ opacities, // [B, N] optional
    const scalar_t *__restrict__ viewmats, // [B, C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    int32_t *__restrict__ radii,         // [B, C, N, 2]
    scalar_t *__restrict__ means2d,      // [B, C, N, 2]
    scalar_t *__restrict__ depths,       // [B, C, N]
    scalar_t *__restrict__ conics,       // [B, C, N, 3]
    scalar_t *__restrict__ compensations,// [B, C, N] optional
    // geometry outputs, optional (written only when render_geometry)
    const bool render_geometry,
    const int geometry_mode,             // 0=RD, 1=MD, 2=PD
    scalar_t *__restrict__ ray_planes,   // [B, C, N, 4] optional {gx,gy,tc,rsigma}
    scalar_t *__restrict__ normals       // [B, C, N, 3] optional (camera space)
) {
    // parallelize over B * C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * C * N) {
        return;
    }
    const uint32_t bid = idx / (C * N); // batch id
    const uint32_t cid = (idx / N) % C; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += bid * N * 3 + gid * 3;
    viewmats += bid * C * 16 + cid * 16;
    Ks += bid * C * 9 + cid * 9;

    // glm is column-major but input is row-major
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to camera space
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    if (mean_c.z < near_plane || mean_c.z > far_plane) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // transform Gaussian covariance to camera space
    mat3 covar;
    if (covars != nullptr) {
        covars += bid * N * 6 + gid * 6;
        covar = mat3(
            covars[0],
            covars[1],
            covars[2], // 1st column
            covars[1],
            covars[3],
            covars[4], // 2nd column
            covars[2],
            covars[4],
            covars[5] // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quats += bid * N * 4 + gid * 4;
        scales += bid * N * 3 + gid * 3;
        quat_scale_to_covar_preci(
            glm::make_vec4(quats), glm::make_vec3(scales), &covar, nullptr
        );
    }
    mat3 covar_c;
    covarW2C(R, covar, covar_c);

    // perspective projection
    mat2 covar2d;
    vec2 mean2d;

    switch (camera_model) {
    case CameraModelType::PINHOLE: // perspective projection
        persp_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    case CameraModelType::ORTHO: // orthographic projection
        ortho_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    case CameraModelType::FISHEYE: // fisheye projection
        fisheye_proj(
            mean_c,
            covar_c,
            Ks[0],
            Ks[4],
            Ks[2],
            Ks[5],
            image_width,
            image_height,
            covar2d,
            mean2d
        );
        break;
    }

    float compensation;
    float det = add_blur(eps2d, covar2d, compensation);
    if (det <= 0.f) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // compute the inverse of the 2d covariance
    mat2 covar2d_inv = glm::inverse(covar2d);

    float extend = 3.33f;
    if (opacities != nullptr) {
        float opacity = opacities[bid * N + gid];
        if (compensations != nullptr) {
            // we assume compensation term will be applied later on.
            opacity *= compensation;
        }
        if (opacity < ALPHA_THRESHOLD) {
            radii[idx * 2] = 0;
            radii[idx * 2 + 1] = 0;
            return;
        }
        // Compute opacity-aware bounding box.
        // https://arxiv.org/pdf/2402.00525 Section B.2
        extend = min(extend, sqrt(2.0f * __logf(opacity / ALPHA_THRESHOLD)));
    }

    // compute tight rectangular bounding box (non differentiable)
    // https://arxiv.org/pdf/2402.00525
    float radius_x = ceilf(extend * sqrtf(covar2d[0][0]));
    float radius_y = ceilf(extend * sqrtf(covar2d[1][1]));

    if (radius_x <= radius_clip && radius_y <= radius_clip) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // mask out gaussians outside the image region
    if (mean2d.x + radius_x <= 0 || mean2d.x - radius_x >= image_width ||
        mean2d.y + radius_y <= 0 || mean2d.y - radius_y >= image_height) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // write to outputs
    radii[idx * 2] = (int32_t)radius_x;
    radii[idx * 2 + 1] = (int32_t)radius_y;
    means2d[idx * 2] = mean2d.x;
    means2d[idx * 2 + 1] = mean2d.y;
    depths[idx] = mean_c.z;
    conics[idx * 3] = covar2d_inv[0][0];
    conics[idx * 3 + 1] = covar2d_inv[0][1];
    conics[idx * 3 + 2] = covar2d_inv[1][1];
    if (compensations != nullptr) {
        compensations[idx] = compensation;
    }

    // Per-Gaussian depth-plane + normal. Only reached when render_geometry is set,
    // which the caller only allows with {quats, scales} (not precomputed covars),
    // so the advanced quats/scales pointers point to this Gaussian here.
    if (render_geometry) {
        vec4 ray_plane;
        vec3 normal;
        compute_ray_plane_normal(
            mean_c,
            // M_inv = S_inv*R_gaussian*W expects W as the world->camera rotation
            // in glm column-major form, which equals the TRANSPOSE of gsplat's R
            // (=W2C rotation). Passing R untransposed leaves depth ~ok (tc-dominated)
            // but rotates the per-Gaussian normal on general orientations; the
            // transpose is required. Verified against a full render.
            glm::transpose(R),
            glm::make_vec4(quats),
            glm::make_vec3(scales),
            Ks[0],
            Ks[4],
            geometry_mode,
            ray_plane,
            normal
        );
        ray_planes[idx * 4] = ray_plane[0];
        ray_planes[idx * 4 + 1] = ray_plane[1];
        ray_planes[idx * 4 + 2] = ray_plane[2];
        ray_planes[idx * 4 + 3] = ray_plane[3];
        normals[idx * 3] = normal[0];
        normals[idx * 3 + 1] = normal[1];
        normals[idx * 3 + 2] = normal[2];
    }
}

void launch_projection_ewa_3dgs_fused_fwd_kernel(
    // inputs
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6] optional
    const at::optional<at::Tensor> quats,  // [..., N, 4] optional
    const at::optional<at::Tensor> scales, // [..., N, 3] optional
    const at::optional<at::Tensor> opacities, // [..., N] optional
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const CameraModelType camera_model,
    // outputs
    at::Tensor radii,                       // [..., C, N, 2]
    at::Tensor means2d,                     // [..., C, N, 2]
    at::Tensor depths,                      // [..., C, N]
    at::Tensor conics,                      // [..., C, N, 3]
    at::optional<at::Tensor> compensations, // [..., C, N] optional
    const bool render_geometry,
    const int geometry_mode,                // 0=RD, 1=MD, 2=PD
    at::optional<at::Tensor> ray_planes,    // [..., C, N, 4] optional
    at::optional<at::Tensor> normals        // [..., C, N, 3] optional
) {
    uint32_t N = means.size(-2);    // number of gaussians
    uint32_t C = viewmats.size(-3); // number of cameras
    uint32_t B = means.numel() / (N * 3);    // number of batches

    int64_t n_elements = B * C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_ewa_3dgs_fused_fwd_kernel",
        [&]() {
            projection_ewa_3dgs_fused_fwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    B,
                    C,
                    N,
                    means.data_ptr<scalar_t>(),
                    covars.has_value() ? covars.value().data_ptr<scalar_t>()
                                       : nullptr,
                    quats.has_value() ? quats.value().data_ptr<scalar_t>()
                                      : nullptr,
                    scales.has_value() ? scales.value().data_ptr<scalar_t>()
                                       : nullptr,
                    opacities.has_value() ? opacities.value().data_ptr<scalar_t>()
                                         : nullptr,
                    viewmats.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    image_width,
                    image_height,
                    eps2d,
                    near_plane,
                    far_plane,
                    radius_clip,
                    camera_model,
                    radii.data_ptr<int32_t>(),
                    means2d.data_ptr<scalar_t>(),
                    depths.data_ptr<scalar_t>(),
                    conics.data_ptr<scalar_t>(),
                    compensations.has_value()
                        ? compensations.value().data_ptr<scalar_t>()
                        : nullptr,
                    render_geometry,
                    geometry_mode,
                    render_geometry ? ray_planes.value().data_ptr<scalar_t>()
                                    : nullptr,
                    render_geometry ? normals.value().data_ptr<scalar_t>()
                                    : nullptr
                );
        }
    );
}

template <typename scalar_t>
__global__ void projection_ewa_3dgs_fused_bwd_kernel(
    // fwd inputs
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [B, N, 3]
    const scalar_t *__restrict__ covars,   // [B, N, 6] optional
    const scalar_t *__restrict__ quats,    // [B, N, 4] optional
    const scalar_t *__restrict__ scales,   // [B, N, 3] optional
    const scalar_t *__restrict__ viewmats, // [B, C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const int32_t *__restrict__ radii,          // [B, C, N, 2]
    const scalar_t *__restrict__ conics,        // [B, C, N, 3]
    const scalar_t *__restrict__ compensations, // [B, C, N] optional
    // grad outputs
    const scalar_t *__restrict__ v_means2d,       // [B, C, N, 2]
    const scalar_t *__restrict__ v_depths,        // [B, C, N]
    const scalar_t *__restrict__ v_conics,        // [B, C, N, 3]
    const scalar_t *__restrict__ v_compensations, // [B, C, N] optional
    // geometry grad outputs (read only when non-null)
    const int geometry_mode,                   // 0=RD, 1=MD, 2=PD
    const scalar_t *__restrict__ v_ray_planes, // [B, C, N, 4] optional
    const scalar_t *__restrict__ v_normals,    // [B, C, N, 3] optional
    // grad inputs
    scalar_t *__restrict__ v_means,   // [B, N, 3]
    scalar_t *__restrict__ v_covars,  // [B, N, 6] optional
    scalar_t *__restrict__ v_quats,   // [B, N, 4] optional
    scalar_t *__restrict__ v_scales,  // [B, N, 3] optional
    scalar_t *__restrict__ v_viewmats // [B, C, 4, 4] optional
) {
    // parallelize over B * C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * C * N || radii[idx * 2] <= 0 || radii[idx * 2 + 1] <= 0) {
        return;
    }
    const uint32_t bid = idx / (C * N); // batch id
    const uint32_t cid = (idx / N) % C; // camera id
    const uint32_t gid = idx % N; // gaussian id

    // shift pointers to the current camera and gaussian
    means += bid * N * 3 + gid * 3;
    viewmats += bid * C * 16 + cid * 16;
    Ks += bid * C * 9 + cid * 9;

    conics += idx * 3;

    v_means2d += idx * 2;
    v_depths += idx;
    v_conics += idx * 3;

    // vjp: compute the inverse of the 2d covariance
    mat2 covar2d_inv = mat2(conics[0], conics[1], conics[1], conics[2]);
    mat2 v_covar2d_inv =
        mat2(v_conics[0], v_conics[1] * .5f, v_conics[1] * .5f, v_conics[2]);
    mat2 v_covar2d(0.f);
    inverse_vjp(covar2d_inv, v_covar2d_inv, v_covar2d);

    if (v_compensations != nullptr) {
        // vjp: compensation term
        const float compensation = compensations[idx];
        const float v_compensation = v_compensations[idx];
        add_blur_vjp(
            eps2d, covar2d_inv, compensation, v_compensation, v_covar2d
        );
    }

    // transform Gaussian to camera space
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);

    mat3 covar;
    vec4 quat;
    vec3 scale;
    if (covars != nullptr) {
        covars += bid * N * 6 + gid * 6;
        covar = mat3(
            covars[0],
            covars[1],
            covars[2], // 1st column
            covars[1],
            covars[3],
            covars[4], // 2nd column
            covars[2],
            covars[4],
            covars[5] // 3rd column
        );
    } else {
        // compute from quaternions and scales
        quat = glm::make_vec4(quats + bid * N * 4 + gid * 4);
        scale = glm::make_vec3(scales + bid * N * 3 + gid * 3);
        quat_scale_to_covar_preci(quat, scale, &covar, nullptr);
    }
    vec3 mean_c;
    posW2C(R, t, glm::make_vec3(means), mean_c);
    mat3 covar_c;
    covarW2C(R, covar, covar_c);

    // vjp: perspective projection
    float fx = Ks[0], cx = Ks[2], fy = Ks[4], cy = Ks[5];
    mat3 v_covar_c(0.f);
    vec3 v_mean_c(0.f);

    switch (camera_model) {
    case CameraModelType::PINHOLE: // perspective projection
        persp_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            glm::make_vec2(v_means2d),
            v_mean_c,
            v_covar_c
        );
        break;
    case CameraModelType::ORTHO: // orthographic projection
        ortho_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            glm::make_vec2(v_means2d),
            v_mean_c,
            v_covar_c
        );
        break;
    case CameraModelType::FISHEYE: // fisheye projection
        fisheye_proj_vjp(
            mean_c,
            covar_c,
            fx,
            fy,
            cx,
            cy,
            image_width,
            image_height,
            v_covar2d,
            glm::make_vec2(v_means2d),
            v_mean_c,
            v_covar_c
        );
        break;
    }

    // add contribution from v_depths
    v_mean_c.z += v_depths[0];

    // Geometry VJP: fold depth-plane + normal gradients into the
    // camera-space mean here (transformed to world by posW2C_VJP below) and
    // into scale/quat in the {quats, scales} branch.
    vec3 v_scale_geo(0.f);
    vec4 v_quat_geo(0.f);
    if (v_ray_planes != nullptr) {
        const vec4 v_rp = {
            v_ray_planes[idx * 4],
            v_ray_planes[idx * 4 + 1],
            v_ray_planes[idx * 4 + 2],
            v_ray_planes[idx * 4 + 3]
        };
        const vec3 v_nrm = {
            v_normals[idx * 3], v_normals[idx * 3 + 1], v_normals[idx * 3 + 2]
        };
        compute_ray_plane_normal_vjp(
            // Must match the forward: pass the transposed W2C rotation (see the
            // compute_ray_plane_normal call in the forward kernel).
            mean_c, glm::transpose(R), quat, scale, Ks[0], Ks[4], geometry_mode, v_rp,
            v_nrm, v_mean_c, v_scale_geo, v_quat_geo
        );
    }

    // vjp: transform Gaussian covariance to camera space
    vec3 v_mean(0.f);
    mat3 v_covar(0.f);
    mat3 v_R(0.f);
    vec3 v_t(0.f);
    posW2C_VJP(R, t, glm::make_vec3(means), v_mean_c, v_R, v_t, v_mean);
    covarW2C_VJP(R, covar, v_covar_c, v_R, v_covar);

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means += bid * N * 3 + gid * 3;
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) {
                gpuAtomicAdd(v_means + i, v_mean[i]);
            }
        }
    }
    if (v_covars != nullptr) {
        // Output gradients w.r.t. the covariance matrix
        warpSum(v_covar, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_covars += bid * N * 6 + gid * 6;
            gpuAtomicAdd(v_covars, v_covar[0][0]);
            gpuAtomicAdd(v_covars + 1, v_covar[0][1] + v_covar[1][0]);
            gpuAtomicAdd(v_covars + 2, v_covar[0][2] + v_covar[2][0]);
            gpuAtomicAdd(v_covars + 3, v_covar[1][1]);
            gpuAtomicAdd(v_covars + 4, v_covar[1][2] + v_covar[2][1]);
            gpuAtomicAdd(v_covars + 5, v_covar[2][2]);
        }
    } else {
        // Directly output gradients w.r.t. the quaternion and scale
        mat3 rotmat = quat_to_rotmat(quat);
        vec4 v_quat(0.f);
        vec3 v_scale(0.f);
        quat_scale_to_covar_vjp(quat, scale, rotmat, v_covar, v_quat, v_scale);
        // add geometry (depth-plane + normal) contributions
        v_quat += v_quat_geo;
        v_scale += v_scale_geo;
        warpSum(v_quat, warp_group_g);
        warpSum(v_scale, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_quats += bid * N * 4 + gid * 4;
            v_scales += bid * N * 3 + gid * 3;
            gpuAtomicAdd(v_quats, v_quat[0]);
            gpuAtomicAdd(v_quats + 1, v_quat[1]);
            gpuAtomicAdd(v_quats + 2, v_quat[2]);
            gpuAtomicAdd(v_quats + 3, v_quat[3]);
            gpuAtomicAdd(v_scales, v_scale[0]);
            gpuAtomicAdd(v_scales + 1, v_scale[1]);
            gpuAtomicAdd(v_scales + 2, v_scale[2]);
        }
    }
    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += bid * C * 16 + cid * 16;
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) { // rows
#pragma unroll
                for (uint32_t j = 0; j < 3; j++) { // cols
                    gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}

void launch_projection_ewa_3dgs_fused_bwd_kernel(
    // inputs
    // fwd inputs
    const at::Tensor means,                // [..., N, 3]
    const at::optional<at::Tensor> covars, // [..., N, 6] optional
    const at::optional<at::Tensor> quats,  // [..., N, 4] optional
    const at::optional<at::Tensor> scales, // [..., N, 3] optional
    const at::Tensor viewmats,             // [..., C, 4, 4]
    const at::Tensor Ks,                   // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float eps2d,
    const CameraModelType camera_model,
    // fwd outputs
    const at::Tensor radii,                       // [..., C, N, 2]
    const at::Tensor conics,                      // [..., C, N, 3]
    const at::optional<at::Tensor> compensations, // [..., C, N] optional
    // grad outputs
    const at::Tensor v_means2d,                     // [..., C, N, 2]
    const at::Tensor v_depths,                      // [..., C, N]
    const at::Tensor v_conics,                      // [..., C, N, 3]
    const at::optional<at::Tensor> v_compensations, // [..., C, N] optional
    const int geometry_mode,                        // 0=RD, 1=MD, 2=PD
    const at::optional<at::Tensor> v_ray_planes,    // [..., C, N, 4] optional
    const at::optional<at::Tensor> v_normals,       // [..., C, N, 3] optional
    const bool viewmats_requires_grad,
    // outputs
    at::Tensor v_means,   // [..., N, 3]
    at::Tensor v_covars,  // [..., N, 3, 3]
    at::Tensor v_quats,   // [..., N, 4]
    at::Tensor v_scales,  // [..., N, 3]
    at::Tensor v_viewmats // [..., C, 4, 4]
) {
    uint32_t N = means.size(-2);    // number of gaussians
    uint32_t C = viewmats.size(-3); // number of cameras
    uint32_t B = means.numel() / (N * 3); // number of batches

    int64_t n_elements = B * C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "projection_ewa_3dgs_fused_bwd_kernel",
        [&]() {
            projection_ewa_3dgs_fused_bwd_kernel<scalar_t>
                <<<grid,
                   threads,
                   shmem_size,
                   at::cuda::getCurrentCUDAStream()>>>(
                    B, 
                    C,
                    N,
                    means.data_ptr<scalar_t>(),
                    covars.has_value() ? covars.value().data_ptr<scalar_t>()
                                       : nullptr,
                    covars.has_value() ? nullptr
                                       : quats.value().data_ptr<scalar_t>(),
                    covars.has_value() ? nullptr
                                       : scales.value().data_ptr<scalar_t>(),
                    viewmats.data_ptr<scalar_t>(),
                    Ks.data_ptr<scalar_t>(),
                    image_width,
                    image_height,
                    eps2d,
                    camera_model,
                    radii.data_ptr<int32_t>(),
                    conics.data_ptr<scalar_t>(),
                    compensations.has_value()
                        ? compensations.value().data_ptr<scalar_t>()
                        : nullptr,
                    v_means2d.data_ptr<scalar_t>(),
                    v_depths.data_ptr<scalar_t>(),
                    v_conics.data_ptr<scalar_t>(),
                    v_compensations.has_value()
                        ? v_compensations.value().data_ptr<scalar_t>()
                        : nullptr,
                    geometry_mode,
                    v_ray_planes.has_value()
                        ? v_ray_planes.value().data_ptr<scalar_t>()
                        : nullptr,
                    v_normals.has_value()
                        ? v_normals.value().data_ptr<scalar_t>()
                        : nullptr,
                    v_means.data_ptr<scalar_t>(),
                    covars.has_value() ? v_covars.data_ptr<scalar_t>()
                                       : nullptr,
                    covars.has_value() ? nullptr : v_quats.data_ptr<scalar_t>(),
                    covars.has_value() ? nullptr
                                       : v_scales.data_ptr<scalar_t>(),
                    viewmats_requires_grad ? v_viewmats.data_ptr<scalar_t>()
                                           : nullptr
                );
        }
    );
}

} // namespace gsplat
