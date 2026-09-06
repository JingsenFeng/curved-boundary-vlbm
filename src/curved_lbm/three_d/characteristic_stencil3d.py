"""Fixed-radius geometry weights for the D3Q6 local characteristic closure.

The construction is the three-dimensional version of the D2Q4 normal probes
and quadratic state-gradient fit.  Only geometry is processed on the host;
all time-dependent state samples remain on the device.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace
import numpy as np

MAX_GRADIENT_SAMPLES = 48
MAX_PROBE_SAMPLES = 16
MAX_CHARACTERISTIC_SAMPLES = MAX_GRADIENT_SAMPLES + MAX_PROBE_SAMPLES


def build_characteristic_weights(geometry, tables, *, batch_size=1024, link_mask=None):
    if link_mask is not None:
        mask = np.asarray(link_mask, dtype=bool)
        if mask.shape != (geometry.n_links,):
            raise ValueError("link_mask must select one entry per cut link")
        ids = np.flatnonzero(mask)
        subset = SimpleNamespace(
            active=geometry.active, dx=geometry.dx, dy=geometry.dy, dz=geometry.dz, n_links=len(ids)
        )
        for name in ("i", "j", "k", "xb", "yb", "zb", "nx", "ny", "nz", "q", "eta"):
            setattr(subset, "link_" + name, np.asarray(getattr(geometry, "link_" + name))[ids])
        fit = build_characteristic_weights(subset, tables, batch_size=batch_size)
        indices = np.full((MAX_CHARACTERISTIC_SAMPLES, geometry.n_links), -1, np.int32)
        weights = np.zeros((4, MAX_CHARACTERISTIC_SAMPLES, geometry.n_links), np.float64)
        modes = np.zeros(geometry.n_links, np.int32)
        indices[:, ids] = fit["indices"]
        weights[:, :, ids] = fit["weights"]
        modes[ids] = fit["probe_modes"]
        fit.update(indices=indices, weights=weights, probe_modes=modes, link_mask=mask)
        return fit
    nlinks = geometry.n_links
    centers = np.column_stack((geometry.link_i, geometry.link_j, geometry.link_k))
    cuts = np.column_stack((geometry.link_xb, geometry.link_yb, geometry.link_zb))
    normals = np.column_stack((geometry.link_nx, geometry.link_ny, geometry.link_nz))
    spacing = np.array((geometry.dx, geometry.dy, geometry.dz))
    if not np.allclose(spacing, spacing[0], rtol=1e-12, atol=0.0):
        raise ValueError("local_f requires an isotropic reference lattice")
    dx = spacing[0]
    shape = np.array(geometry.active.shape)
    active_id = tables["active_id"]
    indices = np.full((MAX_CHARACTERISTIC_SAMPLES, nlinks), -1, np.int32)
    weights = np.zeros((4, MAX_CHARACTERISTIC_SAMPLES, nlinks), np.float64)
    modes = np.zeros(nlinks, np.int32)
    offsets = np.array(list(itertools.product(range(-3, 4), repeat=3)), np.int32)
    corners = np.array(list(itertools.product((0, 1), repeat=3)), np.int32)
    identity = np.eye(10)
    dirs = np.array(((1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)))

    for start in range(0, nlinks, batch_size):
        stop = min(start + batch_size, nlinks)
        ctr = centers[start:stop]
        xb = cuts[start:stop] / dx - 0.5
        normal = normals[start:stop]
        count = stop - start
        nodes = ctr[:, None, :] + offsets[None, :, :]
        inside = np.all((nodes >= 0) & (nodes < shape), axis=2)
        safe = np.clip(nodes, 0, shape - 1)
        ids = active_id[safe[:, :, 0], safe[:, :, 1], safe[:, :, 2]]
        xi = nodes - xb[:, None, :]
        r2 = np.sum(xi * xi, axis=2)
        valid = inside & (ids >= 0) & (r2 <= 9.5)
        distance = np.where(valid, r2, np.inf)
        order = np.argsort(distance, axis=1, kind="stable")[:, :MAX_GRADIENT_SAMPLES]
        rows = np.arange(count)[:, None]
        r2 = distance[rows, order]
        xi = xi[rows, order]
        selected = np.isfinite(r2)
        x, y, z = np.moveaxis(xi, 2, 0)
        A = np.stack(
            (np.ones_like(x), x, y, z, 0.5 * x * x, x * y, x * z, 0.5 * y * y, y * z, 0.5 * z * z),
            axis=2,
        )
        W = np.where(selected, np.exp(-r2 / (2 * 1.5**2)) + 1e-4, 0.0)
        M = np.einsum("bmi,bm,bmj->bij", A, W, A)
        condition = np.linalg.cond(M)
        good = (np.sum(selected, axis=1) >= 10) & np.isfinite(condition) & (condition <= 1e8)
        if not np.all(good):
            bad = (np.flatnonzero(~good) + start)[:8]
            raise ValueError(
                f"local_f needs a full-rank quadratic stencil within three cells; cut links {bad.tolist()}"
            )
        C = np.linalg.solve(M + 1e-12 * identity, np.swapaxes(A * W[:, :, None], 1, 2))
        gradients = C[:, 1:4, :] / dx
        good = np.all(np.isfinite(gradients), axis=(1, 2))
        good &= np.all(np.abs(np.sum(gradients, axis=2)) * dx < 1e-6, axis=1)
        good &= np.all(np.sum(np.abs(gradients), axis=2) * dx <= 12.0, axis=1)
        if not np.all(good):
            raise ValueError(
                f"invalid local_f gradient weights near link {start + np.flatnonzero(~good)[0]}"
            )
        indices[:MAX_GRADIENT_SAMPLES, start:stop] = np.where(selected, ids[rows, order], -1).T
        weights[1:, :MAX_GRADIENT_SAMPLES, start:stop] = np.transpose(gradients, (1, 2, 0))

        def probe(h):
            point = xb - h * normal
            lower = np.floor(point).astype(np.int32)
            frac = point - lower
            grid = lower[:, None, :] + corners[None, :, :]
            in_grid = np.all((grid >= 0) & (grid < shape), axis=2)
            in_radius = np.all(np.abs(grid - ctr[:, None, :]) <= 3, axis=2)
            clipped = np.clip(grid, 0, shape - 1)
            pp = active_id[clipped[:, :, 0], clipped[:, :, 1], clipped[:, :, 2]]
            ww = np.prod(
                np.where(corners[None, :, :] == 1, frac[:, None, :], 1 - frac[:, None, :]), axis=2
            )
            ok = np.all(in_grid & in_radius & (pp >= 0), axis=1)
            return pp, ww, ok

        done = np.zeros(count, bool)
        pred_p = np.full((count, MAX_PROBE_SAMPLES), -1, np.int32)
        pred_w = np.zeros((count, MAX_PROBE_SAMPLES))
        mode = np.zeros(count, np.int32)
        for h1, h2 in ((0.75, 1.75), (1.0, 2.0), (1.25, 2.5), (1.5, 3.0)):
            p1, w1, ok1 = probe(h1)
            p2, w2, ok2 = probe(h2)
            take = ~done & ok1 & ok2
            pred_p[take, :8] = p1[take]
            pred_p[take, 8:] = p2[take]
            pred_w[take, :8] = h2 / (h2 - h1) * w1[take]
            pred_w[take, 8:] = -h1 / (h2 - h1) * w2[take]
            mode[take] = 2
            done |= take
        for h in (0.75, 1.0, 1.25, 1.5, 2.0):
            pp, ww, ok = probe(h)
            take = ~done & ok
            pred_p[take, :8] = pp[take]
            pred_w[take, :8] = ww[take]
            mode[take] = 1
            done |= take
        for row in np.flatnonzero(~done):
            ell = start + row
            eta = geometry.link_eta[ell]
            direction = dirs[geometry.link_q[ell]]
            pp = []
            for k in range(3):
                node = ctr[row] - k * direction
                if np.any(node < 0) or np.any(node >= shape):
                    break
                p = active_id[tuple(node)]
                if p < 0:
                    break
                pp.append(p)
            if len(pp) == 3:
                ww = (0.5 * (eta + 1) * (eta + 2), -eta * (eta + 2), 0.5 * eta * (eta + 1))
            elif len(pp) == 2:
                ww = (1 + eta, -eta)
            else:
                ww = (1.0,)
            pred_p[row, : len(pp)] = pp
            pred_w[row, : len(pp)] = ww
            mode[row] = -len(pp)
        indices[MAX_GRADIENT_SAMPLES:, start:stop] = pred_p.T
        weights[0, MAX_GRADIENT_SAMPLES:, start:stop] = pred_w.T
        modes[start:stop] = mode

    return {
        "indices": np.ascontiguousarray(indices),
        "weights": np.ascontiguousarray(weights),
        "probe_modes": modes,
        "radius": 3,
        "gradient_samples": MAX_GRADIENT_SAMPLES,
        "max_samples": MAX_CHARACTERISTIC_SAMPLES,
    }
