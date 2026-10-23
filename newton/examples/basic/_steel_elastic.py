# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Linear tetrahedral FE helpers for building reduced elastic links in the Steel examples."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import warp as wp

import newton

# Kuhn split of a hex cell (corner index = 4 * x + 2 * y + z) into 6 tetrahedra sharing the 0-7 diagonal.
_KUHN_TETS = ((0, 4, 6, 7), (0, 4, 5, 7), (0, 1, 5, 7), (0, 1, 3, 7), (0, 2, 3, 7), (0, 2, 6, 7))


class SolidMesh:
    """A tetrahedral solid with linear-elastic nodal M and K and its boundary surface."""

    def __init__(self, nodes: np.ndarray, tets: np.ndarray, young: float, poisson: float, density: float):
        self.nodes = np.asarray(nodes, dtype=np.float64)
        self.tets = np.asarray(tets, dtype=np.int64)
        self.mass_matrix, self.stiffness_matrix, self.node_mass = _assemble(
            self.nodes, self.tets, young, poisson, density
        )
        self.surface_vertices, self.surface_indices, self.surface_nodes = _boundary(self.nodes, self.tets)

    @property
    def mass(self) -> float:
        return float(self.node_mass.sum())

    @property
    def com(self) -> np.ndarray:
        return (self.node_mass[:, None] * self.nodes).sum(0) / self.mass

    def inertia(self) -> np.ndarray:
        """Inertia tensor about the centre of mass [kg m^2] from the lumped nodal masses."""
        d = self.nodes - self.com
        r2 = (d * d).sum(1)
        return np.einsum("n,nij->ij", self.node_mass, r2[:, None, None] * np.eye(3) - d[:, :, None] * d[:, None, :])

    def modal_basis(
        self,
        mode_count: int,
        fixed_nodes: np.ndarray | None = None,
        discard_mode_count: int = 0,
        damping_ratio: float = 0.02,
        label: str | None = None,
    ) -> newton.ModalBasis:
        """Linear modes sampled exactly at every node, with node masses for the frame coupling integrals.

        A positive ``discard_mode_count`` marks a free body: its rigid modes are removed by eigenvalue instead.
        """
        free = discard_mode_count > 0
        basis = newton.ModalGeneratorFEM(
            node_positions=self.nodes,
            mass_matrix=self.mass_matrix,
            stiffness_matrix=self.stiffness_matrix,
            sample_node_indices=np.arange(len(self.nodes)),
            mode_count=mode_count,
            fixed_node_indices=fixed_nodes,
            eigenvalue_tolerance=1.0 if free else 1.0e-8,
            damping_ratio=damping_ratio,
            derive_psi=False,
            label=label,
        ).build()
        return newton.ModalBasis(
            sample_points=self.nodes.astype(np.float32),
            sample_phi=basis.sample_phi,
            sample_mass=self.node_mass.astype(np.float32),
            mode_mass=basis.mode_mass,
            mode_stiffness=basis.mode_stiffness,
            mode_damping=basis.mode_damping,
            label=label,
        )

    def add_render_mesh(self, builder: newton.ModelBuilder, body: int, cfg: newton.ModelBuilder.ShapeConfig) -> int:
        mesh = newton.Mesh(self.surface_vertices.astype(np.float32), self.surface_indices.astype(np.int32).ravel())
        return builder.add_shape_mesh(body, mesh=mesh, cfg=cfg)


def box_solid(
    half_extents: tuple[float, float, float],
    cells: tuple[int, int, int],
    young: float,
    poisson: float,
    density: float,
    warp_nodes: Callable[[np.ndarray], np.ndarray] | None = None,
) -> SolidMesh:
    """Box-shaped (optionally re-mapped) tetrahedral solid centred at the origin."""
    axes = [np.linspace(-h, h, c + 1) for h, c in zip(half_extents, cells, strict=True)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    nodes = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    ny, nz = cells[1] + 1, cells[2] + 1
    tets = []
    for i in range(cells[0]):
        for j in range(cells[1]):
            for k in range(cells[2]):
                corner = [((i + (c >> 2)) * ny + (j + ((c >> 1) & 1))) * nz + (k + (c & 1)) for c in range(8)]
                tets.extend([[corner[a] for a in tet] for tet in _KUHN_TETS])
    if warp_nodes is not None:
        nodes = warp_nodes(nodes)
    return SolidMesh(nodes, np.asarray(tets), young, poisson, density)


def square_to_disk(axis: int, radius: float, half_side: float) -> Callable[[np.ndarray], np.ndarray]:
    """Map a square cross-section (perpendicular to ``axis``) onto a disk, for cylinders and fingers."""

    def mapping(nodes: np.ndarray) -> np.ndarray:
        out = nodes.copy()
        a, b = [i for i in range(3) if i != axis]
        u, v = nodes[:, a] / half_side, nodes[:, b] / half_side
        out[:, a] = radius * u * np.sqrt(np.maximum(1.0 - 0.5 * v * v, 0.0))
        out[:, b] = radius * v * np.sqrt(np.maximum(1.0 - 0.5 * u * u, 0.0))
        return out

    return mapping


def cube_to_ball(radius: float, half_side: float) -> Callable[[np.ndarray], np.ndarray]:
    """Map a cube onto a ball with the smooth cube-to-sphere mapping."""

    def mapping(nodes: np.ndarray) -> np.ndarray:
        x, y, z = (nodes / half_side).T
        out = np.stack(
            [
                x * np.sqrt(np.maximum(1 - y * y / 2 - z * z / 2 + y * y * z * z / 3, 0.0)),
                y * np.sqrt(np.maximum(1 - z * z / 2 - x * x / 2 + z * z * x * x / 3, 0.0)),
                z * np.sqrt(np.maximum(1 - x * x / 2 - y * y / 2 + x * x * y * y / 3, 0.0)),
            ],
            axis=1,
        )
        return radius * out

    return mapping


def elastic_link_kwargs(solid: SolidMesh, basis: newton.ModalBasis, frame_origin: np.ndarray | None = None) -> dict:
    """Mass, COM and inertia arguments for ``add_link_elastic`` consistent with the solid's lumped masses."""
    origin = np.zeros(3) if frame_origin is None else np.asarray(frame_origin, dtype=np.float64)
    return {
        "mass": solid.mass,
        "com": wp.vec3(*(solid.com - origin)),
        "inertia": wp.mat33(*solid.inertia().ravel()),
        "modal_basis": basis,
    }


def _assemble(nodes, tets, young, poisson, density):
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    mu = young / (2 * (1 + poisson))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.arange(3), np.arange(3)] += 2 * mu
    D[np.arange(3, 6), np.arange(3, 6)] = mu
    n = len(nodes)
    K = np.zeros((3 * n, 3 * n))
    node_mass = np.zeros(n)
    for tet in tets:
        x = nodes[tet]
        A = np.hstack([np.ones((4, 1)), x])
        volume = abs(np.linalg.det(A)) / 6.0
        grad = np.linalg.inv(A)[1:, :]  # shape-function gradients, 3 x 4
        B = np.zeros((6, 12))
        for a in range(4):
            gx, gy, gz = grad[:, a]
            B[:, 3 * a : 3 * a + 3] = [[gx, 0, 0], [0, gy, 0], [0, 0, gz], [gy, gx, 0], [0, gz, gy], [gz, 0, gx]]
        ke = volume * B.T @ D @ B
        dofs = (3 * tet[:, None] + np.arange(3)).ravel()
        K[np.ix_(dofs, dofs)] += ke
        node_mass[tet] += density * volume / 4.0
    M = np.diag(np.repeat(node_mass, 3))
    return M, K, node_mass


def _boundary(nodes, tets):
    faces = {}
    for tet in tets:
        for opposite in range(4):
            face = tuple(int(v) for v in np.delete(tet, opposite))
            key = tuple(sorted(face))
            if key in faces:
                del faces[key]
            else:
                faces[key] = (face, int(tet[opposite]))
    tris = []
    for face, opposite in faces.values():
        a, b, c = nodes[list(face)]
        inward = np.dot(np.cross(b - a, c - a), nodes[opposite] - a) > 0.0
        tris.append((face[0], face[2], face[1]) if inward else face)
    tris = np.asarray(tris)
    used = np.unique(tris)
    remap = -np.ones(len(nodes), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return nodes[used], remap[tris], used
