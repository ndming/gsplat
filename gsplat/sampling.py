import math
from typing import Tuple
from torch import Tensor

from .cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    sample_geometry as _sample_geometry_op,
)


def sample_geometry(
    means: Tensor,        # [N, 3]
    quats: Tensor,        # [N, 4]
    scales: Tensor,       # [N, 3]
    opacities: Tensor,    # [N]
    viewmats: Tensor,     # [1, 4, 4] or [4, 4]
    Ks: Tensor,           # [1, 3, 3] or [3, 3]
    width: int,
    height: int,
    points2d: Tensor,     # [P, 2]
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    eps2d: float = 0.3,
    tile_size: int = 16,
    sample_normals: bool = False,
    reduction: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Sample the primary surface depth (and optionally camera-space normal)
    at arbitrary query pixels in a single camera.

    The Gaussians are projected and tile-binned exactly as in a full render. At a
    pixel centre this reproduces that render's depth and/or normal channel. Query
    pixels may be continuous (sub-pixel); the surface field is evaluated directly
    at each one, which is both cheaper and sharper at depth discontinuities than
    bilinearly sampling a rendered depth map.

    The result is differentiable w.r.t. ``means``, ``quats``, ``scales``,
    ``opacities`` and ``points2d`` (so gradients flow both to the Gaussians and,
    through the query pixels, to whatever produced them). Any world-space (3D Mip)
    filter is *not* applied here; dilate ``scales``/``opacities`` before calling
    if the render being matched uses it.

    Scope: a single camera and non-packed Gaussians (matching the multi-view use
    case). ``viewmats``/``Ks`` may carry a leading singleton camera dim for
    consistency with :func:`gsplat.rendering.rasterization`; it is squeezed out.

    Args:
        means: Gaussian centres in world space. ``[N, 3]``.
        quats: Gaussian rotation quaternions ``[w, x, y, z]`` (need not be
            normalized). ``[N, 4]``.
        scales: Per-axis Gaussian scales (already activated, i.e. not log).
            ``[N, 3]``.
        opacities: Per-Gaussian opacities in ``[0, 1]`` (already activated).
            ``[N]``.
        viewmats: World-to-camera transform. ``[1, 4, 4]`` or ``[4, 4]``.
        Ks: Pinhole intrinsics. ``[1, 3, 3]`` or ``[3, 3]``.
        width: Image width in pixels (defines the tile grid / image bounds).
        height: Image height in pixels.
        points2d: Query pixel coordinates in this camera, in pixel units with the
            top-left pixel centre at ``(0.5, 0.5)``. Points outside the image
            yield zero depth/alpha/normal. ``[P, 2]``.
        near_plane: Near clip used during projection.
        far_plane: Far clip used during projection.
        eps2d: 2D screen-space covariance dilation used during projection. Match
            the render being sampled (e.g. ``0.0`` when no 2D low-pass is used).
        tile_size: Tile edge length used for the intersection grid. Must match
            the value the projection is binned against; ``16`` is gsplat's
            default.
        sample_normals: If ``True``, also composite and return the camera-space
            surface normal. Skipped entirely otherwise (no extra cost).

    Returns:
        A tuple ``(depth, alpha, normal)``:

        - ``depth``: ``[P]`` sampled expected z-depth. ``0`` where the query pixel
          is out of image or hits no surface, which callers can use as a validity
          test.
        - ``alpha``: ``[P]`` accumulated opacity ``1 - T`` at the query pixel,
          usable as a soft coverage / inside mask.
        - ``normal``: ``[P, 3]`` sampled unit camera-space normal when
          ``sample_normals`` is set, else an empty ``[0]`` tensor.
    """
    if viewmats.dim() == 3:
        viewmats = viewmats[0]
    if Ks.dim() == 3:
        Ks = Ks[0]

    # Project + tile-bin the Gaussians into this camera
    radii, means2d, depths, conics, _, ray_planes, normals = fully_fused_projection(
        means,
        None,  # covars: use quats/scales instead
        quats,
        scales,
        viewmats[None],
        Ks[None],
        width,
        height,
        eps2d=eps2d,
        near_plane=near_plane,
        far_plane=far_plane,
        opacities=opacities,
        render_geometry=True,
    )
    tile_w = math.ceil(width / tile_size)
    tile_h = math.ceil(height / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, tile_size, tile_w, tile_h)
    isect_offsets = isect_offset_encode(isect_ids, 1, tile_w, tile_h)  # [1, tile_h, tile_w]

    return _sample_geometry_op(
        points2d,
        means2d[0],
        conics[0],
        opacities,
        ray_planes[0],
        Ks,
        width,
        height,
        tile_size,
        isect_offsets[0],
        flatten_ids,
        normals=normals[0] if sample_normals else None,
        median=(reduction == 1),
    )
