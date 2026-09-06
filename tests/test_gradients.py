from pathlib import Path
import sys, json
import numpy as np
import warp as wp
from curved_lbm.two_d import curved_boundary_warp as c2
from curved_lbm.three_d import curved_boundary_warp3d as c3


@wp.kernel
def eval2(
    u: wp.array4d(dtype=wp.float64),
    mask: wp.array3d(dtype=wp.int32),
    out: wp.array4d(dtype=wp.float64),
    n: int,
    h: wp.float64,
):
    i, j = wp.tid()
    for c in range(2):
        out[2 * c, i, j, 0] = c2._local_dudx(u, mask, c, i, j, n, h)
        out[2 * c + 1, i, j, 0] = c2._local_dudy(u, mask, c, i, j, n, h)


@wp.kernel
def eval3(
    u: wp.array2d(dtype=wp.float64),
    nbr: wp.array2d(dtype=wp.int32),
    out: wp.array3d(dtype=wp.float64),
    h: wp.float64,
):
    p, A, c = wp.tid()
    out[c, A, p] = c3._sparse_local_du_axis(u, nbr, c, p, A, A + 3, h)


def test_quadratic_gradients_and_rest_state():
    n = 24
    h = 1 / n
    g = c2.CurvedBoundaryGeometry.circle(nx=n, ny=n, radius=0.35)
    m = c2.WarpHyperelasticMaterial.from_poisson("neo_hooke", 0.2, mu=1.0)
    s = c2.WarpCurvedBoundaryLBM2D(
        g,
        material=m,
        boundary=c2.CurvedBoundarySpec(kind="neumann"),
        boundary_reconstruction="local_f",
        lattice_speed=10.0,
        collision_omega=1.8,
        local_displacement_interval=1,
        local_compatibility_interval=1,
        local_compatibility_blend=0.5,
    )
    x, y = np.meshgrid((np.arange(n) + 0.5) * h, (np.arange(n) + 0.5) * h, indexing="ij")
    u = np.stack((x * x + x * y + y * y, 0.3 * x * x + 2 * x * y - y * y))[..., None]
    arr = wp.array(u, dtype=wp.float64, device=s.device)
    out = wp.zeros((4, n, n, 1), dtype=wp.float64, device=s.device)
    wp.launch(eval2, dim=(n, n), inputs=[arr, s.active, out, n, h], device=s.device)
    got = out.numpy()[..., 0]
    exact = np.array((2 * x + y, x + 2 * y, 0.6 * x + 2 * y, 2 * x - 2 * y))
    count2 = 0
    max2 = 0.0
    active = s.active_np.astype(bool)
    for i, j in zip(*np.where(active)):
        for A in range(2):

            def ok(k):
                ij = [i, j]
                ij[A] += k
                return all(0 <= v < n for v in ij) and active[tuple(ij)]

            if (ok(-1) and ok(1)) or (ok(1) and ok(2)) or (ok(-1) and ok(-2)):
                max2 = max(
                    max2, float(np.max(np.abs(got[[A, A + 2], i, j] - exact[[A, A + 2], i, j])))
                )
                count2 += 1
    U = np.zeros((6, n, n))
    U[2] = 1
    U[5] = 1
    s.initialize_from_numpy(U=U, displacement=np.zeros((2, n, n)), apply_initial_boundaries=False)
    for _ in range(5):
        s.step()
    s.refresh_current()
    equilibrium_error = float(np.max(np.abs(s.system_state()[:, active] - U[:, active])))
    assert max2 < 1e-12 and equilibrium_error < 1e-11, (max2, equilibrium_error)

    g3 = c3.CurvedBoundaryGeometry3D.sphere(nx=n, ny=n, nz=n, radius=0.35)
    m3 = c3.WarpHyperelasticMaterial3D.from_poisson("neo_hooke", 0.2, mu=1.0)
    s3 = c3.WarpCurvedBoundaryLBM3D(
        g3, material=m3, boundary=c3.CurvedBoundarySpec3D(kind="dirichlet"), lattice_speed=10.0
    )
    x, y, z = s3.active_coordinates()
    u3 = np.array((x * x + y * z, y * y + x * z, z * z + x * y))
    exact3 = np.array(((2 * x, z, y), (z, 2 * y, x), (y, x, 2 * z)))
    out3 = wp.zeros((3, 3, s3.n_active), dtype=wp.float64, device=s3.device)
    wp.launch(
        eval3,
        dim=(s3.n_active, 3, 3),
        inputs=[wp.array(u3, dtype=wp.float64, device=s3.device), s3.neighbor, out3, h],
        device=s3.device,
    )
    got3 = out3.numpy()
    nbr = s3.neighbor.numpy()
    max3 = 0.0
    count3 = 0
    for p in range(s3.n_active):
        for A in range(3):
            pp = nbr[A, p]
            pm = nbr[A + 3, p]
            if (
                (pp >= 0 and pm >= 0)
                or (pp >= 0 and nbr[A, pp] >= 0)
                or (pm >= 0 and nbr[A + 3, pm] >= 0)
            ):
                max3 = max(max3, float(np.max(np.abs(got3[:, A, p] - exact3[:, A, p]))))
                count3 += 1
    assert max3 < 1e-12, max3
    result = {
        "quadratic_2d_max_error": max2,
        "quadratic_2d_verified_directions": count2,
        "quadratic_3d_max_error": max3,
        "quadratic_3d_verified_directions": count3,
        "neumann_2d_identity_after_5_steps": equilibrium_error,
    }
    print(json.dumps(result, indent=2))
