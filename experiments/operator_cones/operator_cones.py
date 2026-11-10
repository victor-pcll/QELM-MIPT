"""Operator light cones of monitored Haar U(4) circuits: OTOC and Krylov (#315).

Four parts, each one subcommand, each writing one JSON file to ``--out``
(the method of each is in :mod:`QELM.evaluation.operator_cones`):

- ``chain``: outcome-averaged Pauli-weight density ``rho(x, t)`` of
  ``Z_{x0}(t)``, Haar-averaged Markov chain, ``N = 41``, ``T = 25``
  (``chain.json``);
- ``traj``: exact trajectories of the Choi state, Pauli weight of the
  connected operator image of a reference qubit and its entropy ``S_R``,
  ``N = 8`` and ``10`` (``traj_N<N>.json``);
- ``krylov``: Arnoldi chain of the outcome-averaged Floquet channel,
  ``h_{n+1,n}``, ``|phi_n(t)|^2`` and ``K(t)``, ``N = 6`` and ``8``
  (``krylov_N<N>.json``);
- ``parker``: Lanczos ``b_n`` of the Liouvillian of the chaotic and of the
  integrable Ising chain, the reference of Parker et al. (``parker.json``).

    uv run python experiments/operator_cones/operator_cones.py chain \
        --out outputs/operator_cones
    uv run python experiments/operator_cones/operator_cones.py traj \
        --N 10 --traj 80 --chunk 5 --jobs 4 --out outputs/operator_cones

``operator_cones.sh`` runs all four with the defaults of the README, and
``plot_operator_cones.py`` draws the figures from the JSON files.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from QELM.evaluation.chaos_locality import ising_chain_hamiltonian
from QELM.evaluation.operator_cones import (
    arnoldi,
    floquet_heisenberg_channel,
    free_fermion_lanczos,
    krylov_wavefunction,
    liouvillian_lanczos,
    pauli_weight_chain,
    single_branch,
    single_operator_span,
    site_operator,
    slowest_mode_modulus,
    trajectory_cones,
)

# The five values of p of the figures and the grid of the summary panels.
P_MAIN = [0.0, 0.05, 0.17, 0.35, 0.65]
P_GRID = [0.0, 0.025, 0.05, 0.1, 0.13, 0.17, 0.21, 0.25, 0.35, 0.5, 0.65]


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v]


def _write(out: Path, name: str, data: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(data, indent=1))
    print(f"wrote {out / name}")


def run_chain(args: argparse.Namespace) -> None:
    """Haar-averaged Pauli-weight chain on the grid of p."""
    rng = np.random.default_rng(args.seed)
    runs = {}
    for p in sorted(set(args.p)):
        t0 = time.time()
        r = pauli_weight_chain(args.N, args.T, p, rng, population=args.population)
        runs[f"{p:g}"] = {
            "p": p,
            "density": r.density.tolist(),
            "survival": r.survival.tolist(),
            "size": r.size.tolist(),
            "ess_min": float(r.ess.min()),
        }
        print(f"chain p={p:g}: {time.time() - t0:.1f} s, size(T)={r.size[-1]:.2f}")
    _write(
        args.out,
        "chain.json",
        {
            "N": args.N,
            "T": args.T,
            "x0": args.N // 2,
            "population": args.population,
            "seed": args.seed,
            "runs": runs,
        },
    )


def _traj_chunk(job: tuple[int, int, float, int, int]) -> dict:
    N, T, p, n, seed = job
    r = trajectory_cones(N, T, p, n, np.random.default_rng(seed))
    return {
        "p": p,
        "n": n,
        "wrho": (r.weighted_density * n).tolist(),
        "w": (r.weight * n).tolist(),
        "s_R": r.per_traj_entropy.tolist(),
        "s_sys": r.per_traj_system_entropy.tolist(),
    }


def run_traj(args: argparse.Namespace) -> None:
    """Exact trajectories of the Choi state, chunks in parallel."""
    T = args.T if args.T is not None else int(1.5 * args.N)
    chunk = max(1, args.chunk)
    jobs = []
    seq = np.random.SeedSequence([args.seed, args.N])
    for p in sorted(set(args.p)):
        n_left = args.traj
        while n_left > 0:
            n = min(chunk, n_left)
            seed = int(seq.spawn(1)[0].generate_state(1)[0])
            jobs.append((args.N, T, p, n, seed))
            n_left -= n
    t0 = time.time()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            parts = list(pool.map(_traj_chunk, jobs))
    else:
        parts = [_traj_chunk(job) for job in jobs]
    runs: dict[str, dict] = {}
    for part in parts:
        key = f"{part['p']:g}"
        run = runs.setdefault(
            key,
            {
                "p": part["p"],
                "n_traj": 0,
                "wrho": np.zeros((T + 1, args.N)),
                "w": np.zeros(T + 1),
                "s_R": [],
                "s_sys": [],
            },
        )
        run["n_traj"] += part["n"]
        run["wrho"] += np.asarray(part["wrho"])
        run["w"] += np.asarray(part["w"])
        run["s_R"] += part["s_R"]
        run["s_sys"] += part["s_sys"]
    for run in runs.values():
        n = run["n_traj"]
        s = np.asarray(run.pop("s_R"))
        run["weighted_density"] = (run.pop("wrho") / n).tolist()
        run["weight"] = (run.pop("w") / n).tolist()
        run["entropy_R"] = s.mean(axis=0).tolist()
        run["entropy_R_err"] = (s.std(axis=0, ddof=1) / np.sqrt(n)).tolist()
        s_sys = np.asarray(run.pop("s_sys"))
        run["entropy_sys"] = s_sys.mean(axis=0).tolist()
        run["entropy_sys_err"] = (s_sys.std(axis=0, ddof=1) / np.sqrt(n)).tolist()
    print(f"traj N={args.N}: {len(jobs)} chunks in {time.time() - t0:.0f} s")
    _write(
        args.out,
        f"traj_N{args.N}.json",
        {"N": args.N, "T": T, "x0": args.N // 2, "seed": args.seed, "runs": runs},
    )


def run_krylov(args: argparse.Namespace) -> None:
    """Arnoldi chain of the Floquet channel, one circuit per seed."""
    N = args.N
    O0 = site_operator(N, N // 2, 3)
    runs = {}
    for p in sorted(set(args.p)):
        t0 = time.time()
        sub, prob, K, norm, lam = [], [], [], [], []
        for s in range(args.seeds):
            rng = np.random.default_rng([args.seed, s])
            channel = floquet_heisenberg_channel(N, p, rng)
            a = arnoldi(channel, O0, args.n_max)
            n = a.basis.shape[0]
            wf = krylov_wavefunction(channel, O0, a.basis, args.n_max - 1)
            pad_sub = np.full(args.n_max - 1, np.nan)
            pad_sub[: n - 1] = a.subdiagonal
            pad_prob = np.zeros((args.n_max, args.n_max))
            pad_prob[:, :n] = wf.prob
            sub.append(pad_sub)
            prob.append(pad_prob)
            K.append(wf.complexity)
            norm.append(wf.norm)
            lam.append(slowest_mode_modulus(a.hessenberg))
        K_arr = np.asarray(K)
        runs[f"{p:g}"] = {
            "p": p,
            "subdiagonal": np.nanmean(sub, axis=0).tolist(),
            "subdiagonal_err": (
                np.nanstd(sub, axis=0, ddof=1) / np.sqrt(args.seeds)
            ).tolist(),
            "prob": np.mean(prob, axis=0).tolist(),
            "complexity": K_arr.mean(axis=0).tolist(),
            "complexity_err": (
                K_arr.std(axis=0, ddof=1) / np.sqrt(args.seeds)
            ).tolist(),
            "log_norm": np.mean(np.log(np.maximum(norm, 1e-300)), axis=0).tolist(),
            "slowest_modulus": lam,
        }
        print(
            f"krylov N={N} p={p:g}: {time.time() - t0:.0f} s, "
            f"K(T)={K_arr.mean(0)[-1]:.2f}"
        )
    _write(
        args.out,
        f"krylov_N{N}.json",
        {
            "N": N,
            "n_max": args.n_max,
            "seeds": args.seeds,
            "seed": args.seed,
            "runs": runs,
        },
    )


def run_single(args: argparse.Namespace) -> None:
    """One operator in the circuits and masks of the effect rank (PR #314).

    Same draws as ``experiments/effect_rank/effect_rank.py``: for each seed,
    ``u = default_rng((seed, N, 1000)).random((T, N))``, mask ``u < p``,
    Haar circuit from ``default_rng((seed, N, 0))``, depth ``T = N``. With
    ``--rank``, the dense effect rank of #314 on the same circuit and mask.
    """
    from QELM.evaluation.effect_rank import draw_circuit, effect_rank

    N, T = args.N, args.N
    runs = {}
    for p in sorted(set(args.p)):
        t0 = time.time()
        dims, dens, adens, anorm, ranks = [], [], [], [], []
        for seed in range(args.seeds):
            u = np.random.default_rng((seed, N, 1000)).random((T, N))
            circuit = draw_circuit("haar", N, T, np.random.default_rng((seed, N, 0)))
            r = single_operator_span(circuit, u < p, N // 2)
            dims.append(r.dims)
            dens.append(r.density)
            adens.append(r.averaged_density)
            anorm.append(r.averaged_norm)
            if args.rank:
                ranks.append(effect_rank(circuit, u < p).dense)
        d = np.asarray(dims, dtype=float)
        runs[f"{p:g}"] = {
            "p": p,
            "dims": d.tolist(),
            "log2_dim_mean": np.log2(d).mean(axis=0).tolist(),
            "density": np.nanmean(dens, axis=0).tolist(),
            "averaged_density": np.nanmean(adens, axis=0).tolist(),
            "averaged_norm": np.mean(anorm, axis=0).tolist(),
            "effect_rank": ranks,
        }
        print(
            f"single N={N} p={p:g}: {time.time() - t0:.0f} s, dim={d[:, -1].mean():.0f}"
        )
    _write(
        args.out,
        f"single_N{N}.json",
        {"N": N, "T": T, "x0": N // 2, "seeds": args.seeds, "runs": runs},
    )


def run_branch(args: argparse.Namespace) -> None:
    """One sampled branch of ``Z_x0`` per draw, circuits and masks of #314.

    Same draws as :func:`run_single` (seed ``s`` gives the circuit and mask
    of seed ``s`` of ``experiments/effect_rank``), ``--branches`` sampled
    records per circuit. Cheap at ``N = 10`` for every ``p``.
    """
    from QELM.evaluation.effect_rank import draw_circuit

    N, T = args.N, args.N
    runs = {}
    for p in sorted(set(args.p)):
        t0 = time.time()
        dens = []
        for seed in range(args.seeds):
            u = np.random.default_rng((seed, N, 1000)).random((T, N))
            circuit = draw_circuit("haar", N, T, np.random.default_rng((seed, N, 0)))
            rng = np.random.default_rng((seed, N, 2000))
            for _ in range(args.branches):
                dens.append(single_branch(circuit, u < p, N // 2, rng))
        D = np.asarray(dens)
        runs[f"{p:g}"] = {
            "p": p,
            "density": D.mean(axis=0).tolist(),
            "size": D.sum(axis=2).mean(axis=0).tolist(),
            "size_err": (D.sum(axis=2).std(axis=0, ddof=1) / np.sqrt(len(D))).tolist(),
        }
        print(f"branch N={N} p={p:g}: {time.time() - t0:.0f} s")
    _write(
        args.out,
        f"branch_N{N}.json",
        {
            "N": N,
            "T": T,
            "x0": N // 2,
            "seeds": args.seeds,
            "branches": args.branches,
            "runs": runs,
        },
    )


def run_parker(args: argparse.Namespace) -> None:
    """Lanczos ``b_n`` of ``X_{N/2}`` in the Ising chain, chaotic and integrable.

    Chaotic (``h_z = 0.5``): dense ``4**N`` operator space, sizes
    ``--N`` and ``--N - 2``. Integrable (``h_z = 0``): free fermions, exact
    for the same sizes and for ``N = 200`` (no finite-size plateau).
    """
    out: dict[str, dict] = {}
    for N in (args.N - 2, args.N):
        H = ising_chain_hamiltonian(np.ones(N - 1), h_x=1.05, h_z=0.5)
        b = liouvillian_lanczos(H, site_operator(N, N // 2, 1), args.n_max)
        out[f"chaotic_N{N}"] = {"N": N, "h_x": 1.05, "h_z": 0.5, "b": b.tolist()}
        print(f"parker chaotic N={N}: {len(b)} coefficients")
    for N in (args.N - 2, args.N, 200):
        b = free_fermion_lanczos(N, 1.0, 1.05, N // 2, args.n_max)
        out[f"integrable_N{N}"] = {"N": N, "h_x": 1.05, "h_z": 0.0, "b": b.tolist()}
    _write(args.out, "parker.json", {"operator": "X_{N//2}", "J": 1.0, "models": out})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="part", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", type=Path, default=Path("outputs/operator_cones"))
    common.add_argument("--seed", type=int, default=42)
    p_help = "comma-separated values of p"
    grid = ",".join(f"{p:g}" for p in sorted(set(P_MAIN + P_GRID)))

    c = sub.add_parser("chain", parents=[common])
    c.add_argument("--N", type=int, default=41)
    c.add_argument("--T", type=int, default=25)
    c.add_argument("--population", type=int, default=50000)
    c.add_argument("--p", type=_floats, default=_floats(grid), help=p_help)
    c.set_defaults(func=run_chain)

    t = sub.add_parser("traj", parents=[common])
    t.add_argument("--N", type=int, default=10)
    t.add_argument("--T", type=int, default=None, help="half-layers (1.5 N)")
    t.add_argument("--traj", type=int, default=100, help="trajectories per p")
    t.add_argument("--chunk", type=int, default=10)
    t.add_argument("--jobs", type=int, default=4)
    t.add_argument("--p", type=_floats, default=_floats(grid), help=p_help)
    t.set_defaults(func=run_traj)

    k = sub.add_parser("krylov", parents=[common])
    k.add_argument("--N", type=int, default=8)
    k.add_argument("--n-max", dest="n_max", type=int, default=60)
    k.add_argument("--seeds", type=int, default=10)
    k.add_argument("--p", type=_floats, default=_floats(grid), help=p_help)
    k.set_defaults(func=run_krylov)

    g = sub.add_parser("single", parents=[common])
    g.add_argument("--N", type=int, default=7)
    g.add_argument("--seeds", type=int, default=10)
    g.add_argument("--rank", action="store_true", help="also the #314 rank")
    g.add_argument("--p", type=_floats, default=_floats(grid), help=p_help)
    g.set_defaults(func=run_single)

    b = sub.add_parser("branch", parents=[common])
    b.add_argument("--N", type=int, default=10)
    b.add_argument("--seeds", type=int, default=10)
    b.add_argument("--branches", type=int, default=5, help="records per circuit")
    b.add_argument("--p", type=_floats, default=_floats(grid), help=p_help)
    b.set_defaults(func=run_branch)

    q = sub.add_parser("parker", parents=[common])
    q.add_argument("--N", type=int, default=10)
    q.add_argument("--n-max", dest="n_max", type=int, default=40)
    q.set_defaults(func=run_parker)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
