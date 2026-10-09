"""Phase 12 training loop: drive the Rust environment from Python, train a
small policy, export ONNX, and check the Rust evaluator against a reference
runtime on identical observations.

What this script is for, in the plan's terms:

* "Persistent, synchronous headless stepping interface from Python to Rust;
  avoid starting a process for every action." — `F1Env` loads the crate's
  cdylib with ctypes and holds one live handle for the whole run. The Rust
  side never forks, and the environment state never crosses a socket.

* "Define reset, action frequency, observation units/normalization, rewards,
  and episode termination." — none of that is decided here. It is read from
  the Rust `EnvSpec` via `f1_env_spec_default_json`, so the trainer and the
  engine cannot disagree about the environment.

* "Record observations with the actions that produced them; split training and
  evaluation episodes." — transitions come out of the Rust episode log, which
  `step()` appends to in the same call that produced them. Seeds come from the
  Rust split ledger, which refuses to hand a seed to both splits.

* "Train a small policy, export ONNX, and verify Rust-side results on
  identical observations." — behaviour cloning against the deterministic
  Phase 4/5 driver, then `parity()`, which feeds the same observation vectors
  to the Rust evaluator and to onnxruntime and reports the worst difference.

* "First compare a learned driver with the Phase 1/4 baseline; do not assume
  self-play produces a good teacher." — the teacher here is the existing
  deterministic driver, deliberately not a learned one. `evaluate()` runs both
  controllers over the evaluation split and reports the same metrics for each,
  and the caller decides whether the learned driver is worth enabling.

Usage:
    python tools/gym_train.py --episodes 40 --out artifacts/phase12
    python tools/gym_train.py --parity-only --out artifacts/phase12
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import pathlib
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Loading the Rust library
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parent
# The build directory deliberately lives outside OneDrive (see
# apexsim/.cargo/config.toml), so the default in-tree location is checked too but
# is rarely the one that answers.
_CANDIDATES = [
    _ROOT / "apexsim" / "target" / "release" / "apexsim.dll",
    _ROOT / "apexsim" / "target" / "debug" / "apexsim.dll",
    _ROOT / "target" / "release" / "apexsim.dll",
    pathlib.Path(os.environ.get("CARGO_TARGET_DIR", "")) / "release" / "apexsim.dll",
    pathlib.Path.home() / ".local" / "share" / "apexsim-target" / "release" / "apexsim.dll",
]


def find_library() -> pathlib.Path:
    for p in _CANDIDATES:
        if p.is_file():
            _assert_fresh(p)
            return p
    tried = "\n  ".join(str(p) for p in _CANDIDATES)
    raise SystemExit(
        "could not find apexsim.dll. Build it with:\n"
        "  cd apexsim && cargo build --release\n"
        f"looked in:\n  {tried}"
    )


def _assert_fresh(dll: pathlib.Path) -> None:
    """Refuse to train against a library older than the engine that built it.

    Silently loading a stale DLL is the worst failure this script can have,
    because nothing downstream reports it: the run completes, prints plausible
    metrics, and exports a model, all of it fitted to behaviour the engine no
    longer has. It happened here. A release DLL predating the Phase 12 driver
    fix collected the entire expert dataset, so every pedal label came from the
    buggy teacher — the run reported "not enabled" and the real cause was a
    file timestamp.

    Comparing against the newest source file is the honest check. Cargo hashes
    content, so a touched-but-unchanged file costs only a rebuild, while a
    genuinely newer engine can never be silently bypassed.
    """
    src = _ROOT / "apexsim" / "src"
    if not src.is_dir():
        return
    newest = max((p.stat().st_mtime for p in src.rglob("*.rs")), default=0.0)
    if dll.stat().st_mtime + 1.0 < newest:
        age_min = (newest - dll.stat().st_mtime) / 60.0
        raise SystemExit(
            f"{dll} is older than apexsim/src by {age_min:.0f} min.\n"
            "Training against it would fit a model to engine behaviour that no\n"
            "longer exists. Rebuild first:\n"
            "  cd apexsim && cargo build --release"
        )


class F1Lib:
    """The C ABI in `gym::ffi`, bound once."""

    def __init__(self, path: pathlib.Path):
        self.dll = ctypes.CDLL(str(path))
        d = self.dll
        d.f1_obs_dim.restype = ctypes.c_size_t
        d.f1_action_dim.restype = ctypes.c_size_t
        d.f1_env_spec_default_json.restype = ctypes.c_char_p
        d.f1_env_new.argtypes = [ctypes.c_char_p]
        d.f1_env_new.restype = ctypes.c_void_p
        d.f1_env_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        d.f1_env_reset.restype = ctypes.c_int
        d.f1_env_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
        d.f1_env_step.restype = ctypes.c_int
        d.f1_env_obs.argtypes = [ctypes.c_void_p]
        d.f1_env_obs.restype = ctypes.POINTER(ctypes.c_float)
        d.f1_env_flags.argtypes = [ctypes.c_void_p]
        d.f1_env_flags.restype = ctypes.c_int
        d.f1_env_total_reward.argtypes = [ctypes.c_void_p]
        d.f1_env_total_reward.restype = ctypes.c_double
        d.f1_env_progress_m.argtypes = [ctypes.c_void_p]
        d.f1_env_progress_m.restype = ctypes.c_double
        d.f1_env_metrics_json.argtypes = [ctypes.c_void_p]
        d.f1_env_metrics_json.restype = ctypes.c_char_p
        d.f1_env_log_len.argtypes = [ctypes.c_void_p]
        d.f1_env_log_len.restype = ctypes.c_size_t
        d.f1_env_log_copy.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        d.f1_env_log_copy.restype = ctypes.c_size_t
        d.f1_env_clear_log.argtypes = [ctypes.c_void_p]
        d.f1_env_clear_log.restype = ctypes.c_int
        d.f1_env_close.argtypes = [ctypes.c_void_p]
        d.f1_teacher_action.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        d.f1_teacher_action.restype = ctypes.c_int
        d.f1_env_split_claim = getattr(d, "f1_split_claim", None)
        d.f1_split_claim.argtypes = [ctypes.c_int, ctypes.c_uint64]
        d.f1_split_claim.restype = ctypes.c_int
        d.f1_split_count.argtypes = [ctypes.c_int]
        d.f1_split_count.restype = ctypes.c_size_t
        d.f1_split_seed.argtypes = [ctypes.c_int, ctypes.c_size_t]
        d.f1_split_seed.restype = ctypes.c_uint64
        d.f1_split_reset.argtypes = []
        d.f1_split_reset.restype = None
        d.f1_last_error.restype = ctypes.c_char_p

    def error(self) -> str:
        msg = self.dll.f1_last_error()
        return msg.decode("utf-8", "replace") if msg else "(no error recorded)"


TRAIN, EVAL = 0, 1


class F1Env:
    """One persistent environment. Create once, reset per episode."""

    def __init__(self, lib: F1Lib, spec: dict | None = None):
        self.lib = lib
        if spec is None:
            raw = lib.dll.f1_env_spec_default_json()
            spec = json.loads(raw.decode("utf-8"))
        self.spec = spec
        self.obs_dim = lib.dll.f1_obs_dim()
        self.action_dim = lib.dll.f1_action_dim()
        self._check_spec(spec)
        blob = json.dumps(spec).encode("utf-8")
        self.handle = lib.dll.f1_env_new(blob)
        if not self.handle:
            raise SystemExit(f"f1_env_new refused the spec: {lib.error()}")

    def _check_spec(self, spec: dict) -> None:
        """The environment defines itself; a mismatch here is a broken build,
        not something to paper over."""
        n = len(spec["obs"]["scale"])
        if n != self.obs_dim:
            raise SystemExit(
                f"EnvSpec carries {n} observation scales but the library reports {self.obs_dim}"
            )

    # -- lifecycle --

    def reset(self, seed: int) -> np.ndarray:
        rc = self.lib.dll.f1_env_reset(self.handle, ctypes.c_uint64(seed))
        if rc != 0:
            raise SystemExit(f"reset failed: {self.lib.error()}")
        return self.obs()

    def step(self, action) -> tuple[np.ndarray, float, bool, bool]:
        arr = np.ascontiguousarray(action, dtype=np.float32)
        rc = self.lib.dll.f1_env_step(
            self.handle, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), arr.size
        )
        if rc != 0:
            raise SystemExit(f"step failed: {self.lib.error()}")
        flags = self.lib.dll.f1_env_flags(self.handle)
        return self.obs(), self.lib.dll.f1_env_total_reward(self.handle), bool(flags & 1), bool(flags & 2)

    def obs(self) -> np.ndarray:
        n = self.lib.dll.f1_env_log_len(self.handle)  # touch the handle first
        del n
        ptr = self.lib.dll.f1_env_obs(self.handle)
        return np.ctypeslib.as_array(ptr, shape=(self.obs_dim,)).copy()

    def metrics(self) -> dict:
        return json.loads(self.lib.dll.f1_env_metrics_json(self.handle).decode("utf-8"))

    def log(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The episode's (observations, actions, rewards), as recorded by Rust."""
        n = self.lib.dll.f1_env_log_len(self.handle)
        if n == 0:
            empty = np.zeros((0, self.obs_dim), dtype=np.float32)
            return empty, np.zeros((0, self.action_dim), dtype=np.float32), np.zeros(0)
        obs = np.zeros((n, self.obs_dim), dtype=np.float32)
        act = np.zeros((n, self.action_dim), dtype=np.float32)
        rew = np.zeros(n, dtype=np.float64)
        got = self.lib.dll.f1_env_log_copy(
            self.handle,
            obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rew.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            n,
        )
        if got != n:
            raise SystemExit(f"log copy returned {got} of {n} transitions")
        return obs, act, rew

    def clear_log(self) -> None:
        self.lib.dll.f1_env_clear_log(self.handle)

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.dll.f1_env_close(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# Episode splits — the ledger is Rust's, and it says no
# ---------------------------------------------------------------------------


def claim_split(lib: F1Lib, split: int, seed: int) -> None:
    if lib.dll.f1_split_claim(split, ctypes.c_uint64(seed)) != 0:
        raise SystemExit(f"split ledger refused seed {seed}: {lib.error()}")


def split_seeds(lib: F1Lib, split: int) -> list[int]:
    n = lib.dll.f1_split_count(split)
    return [int(lib.dll.f1_split_seed(split, i)) for i in range(n)]


# ---------------------------------------------------------------------------
# Learned vs baseline, measured in Rust
# ---------------------------------------------------------------------------
#
# The comparison deliberately does not live in Python. It needs a whole 20-car
# race driven to its own conclusion, and the only thing that can do that is the
# engine itself — so `apexsim policy-eval` does it and this script calls it.


def binary_dir() -> pathlib.Path:
    return find_library().parent


def compare_controllers(model_path: pathlib.Path, seeds: list[int], laps: int,
                        cars: int) -> dict:
    """Ask the Rust binary to race baseline and learned on identical seeds."""
    import subprocess

    exe = binary_dir() / ("apexsim.exe" if os.name == "nt" else "apexsim")
    cmd = [
        str(exe), "policy-eval",
        "--model", str(model_path),
        "--seeds", ",".join(str(s) for s in seeds),
        "--laps", str(laps),
        "--cars", str(cars),
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        raise SystemExit(f"policy-eval failed ({proc.returncode}): {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def latency_report(model_path: pathlib.Path, grid: int, runs: int) -> dict:
    """End-to-end inference latency at the intended grid size, from Rust."""
    import subprocess

    exe = binary_dir() / ("apexsim.exe" if os.name == "nt" else "apexsim")
    cmd = [str(exe), "policy-latency", "--model", str(model_path),
           "--cars", str(grid), "--runs", str(runs), "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        raise SystemExit(f"policy-latency failed ({proc.returncode}): {proc.stderr.strip()}")
    return json.loads(proc.stdout)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=40, help="training episodes")
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--laps", type=int, default=4)
    ap.add_argument("--cars", type=int, default=20)
    ap.add_argument("--out", default="artifacts/phase12")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--latency-runs", type=int, default=2000)
    ap.add_argument("--parity-only", action="store_true",
                    help="re-check parity and latency against an existing model")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dll_path = find_library()
    lib = F1Lib(dll_path)
    # Start from a clean ledger so a rerun is reproducible; the ledger still
    # refuses any within-run collision, which is the property that matters.
    lib.dll.f1_split_reset()
    print(f"library   : {dll_path}")
    print(f"obs/action: {lib.dll.f1_obs_dim()} / {lib.dll.f1_action_dim()}")

    model_path = out / "policy.onnx"
    if args.parity_only:
        if not model_path.is_file():
            raise SystemExit(f"no model at {model_path}")
        metrics = parity_report(lib, model_path, n=4000)
        lat = latency_report(model_path, args.cars, args.latency_runs)
        report_latency(lat)
        return 0 if metrics["pass"] and lat.get("within_budget", False) else 1

    import train_lib

    with F1Env(lib) as env:
        spec = env.spec
        print(f"env spec  : {spec['cars']} cars, {spec['laps']} laps, "
              f"action every {spec['action_every_ticks']} ticks, "
              f"budget {spec['max_steps']} steps")
        (out / "env_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

        # ---- seeds, then data ----
        train_seeds = list(range(1_000, 1_000 + args.episodes))
        eval_seeds = list(range(9_000, 9_000 + args.eval_episodes))

        t0 = time.time()
        data = train_lib.collect_expert(lib, env, train_seeds, eval_seeds)
        overlap = set(split_seeds(lib, TRAIN)) & set(split_seeds(lib, EVAL))
        assert not overlap, f"split ledger leaked: {overlap}"
        # Written out, not just asserted here. The ledger proves leakage is
        # impossible; this file is what lets a later reader confirm that *this*
        # run used disjoint seeds without having to trust the assertion that
        # printed alongside the numbers they are judging.
        (out / "splits.json").write_text(
            json.dumps(
                {
                    "train_seeds": train_seeds,
                    "eval_seeds": eval_seeds,
                    "ledger_train": split_seeds(lib, TRAIN),
                    "ledger_eval": split_seeds(lib, EVAL),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"splits    : {len(train_seeds)} train / {len(eval_seeds)} eval, disjoint")
        print(f"collected : {data['n_obs']} expert transitions in {time.time() - t0:.1f}s")

        np.savez_compressed(
            out / "expert.npz",
            obs=data["obs"],
            act=data["act"],
            train_seeds=np.asarray(train_seeds, dtype=np.int64),
            eval_seeds=np.asarray(eval_seeds, dtype=np.int64),
        )

        t0 = time.time()
        model, history, train_info = train_lib.train(
            data["obs"], data["act"], epochs=args.epochs, seed=args.seed,
            episode_lengths=data["episode_lengths"],
        )
        print(f"trained   : {args.epochs} epochs in {time.time() - t0:.1f}s, "
              f"val MSE {history[0]:.5f} -> {history[-1]:.5f} "
              f"(held out {train_info['n_val_episodes']} episodes, "
              f"{train_info['n_val_rows']} rows)")
        (out / "train_history.json").write_text(
            json.dumps({"history": history, **train_info}, indent=2), encoding="utf-8"
        )

        train_lib.export_onnx(model_path, data["obs"].shape[1], data["act"].shape[1],
                              model=model)
        print(f"exported  : {model_path} ({model_path.stat().st_size} bytes)")

    # ---- parity: Rust evaluator vs onnxruntime, identical observations ----
    metrics = parity_report(lib, model_path, n=4000)
    (out / "parity.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ---- latency at the intended grid ----
    lat = latency_report(model_path, args.cars, args.latency_runs)
    report_latency(lat)
    (out / "latency.json").write_text(json.dumps(lat, indent=2), encoding="utf-8")

    # ---- learned vs baseline, on the evaluation split only ----
    comparison = compare_controllers(model_path, eval_seeds, args.laps, args.cars)
    report_comparison(comparison)
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    ok = metrics["pass"] and lat.get("within_budget", False) and comparison["improves"]
    state = "ENABLED" if ok else "NOT enabled"
    print()
    print(f"verdict   : learned driver {state}")
    print(f"             parity {metrics['pass']}, latency within budget {lat.get('within_budget')}, improves {comparison['improves']}")
    return 0 if ok else 1


def report_latency(lat: dict) -> None:
    print(f"latency   : p50 {lat['p50_us']:.1f} us  p99 {lat['p99_us']:.1f} us  "
          f"max {lat['max_us']:.1f} us over {lat['runs']} runs x {lat['cars']} cars "
          f"(budget {lat['budget_us']:.0f} us) -> {'PASS' if lat['within_budget'] else 'FAIL'}")


def report_comparison(c: dict) -> None:
    b, l = c["baseline"], c["learned"]
    print("comparison: identical seeds, same field")
    print(f"  baseline: mean best lap {b['mean_best_lap_s']:.3f}s, "
          f"mean laps {b['mean_laps']:.2f}, mean position {b['mean_position']:.2f}, "
          f"finished {b['finished']}/{b['episodes']}")
    print(f"  learned : mean best lap {l['mean_best_lap_s']:.3f}s, "
          f"mean laps {l['mean_laps']:.2f}, mean position {l['mean_position']:.2f}, "
          f"finished {l['finished']}/{l['episodes']}")
    for k, v in c["deltas"].items():
        print(f"  delta {k:>16}: {v:+.4f}")


def parity_report(lib: F1Lib, model_path: pathlib.Path, n: int) -> dict:
    """Same observations through Rust and through onnxruntime."""
    import numpy as np

    if not model_path.is_file():
        print(f"parity    : SKIPPED, no model at {model_path}")
        return {"pass": False, "reason": "no model"}
    import onnxruntime as ort

    obs_dim = lib.dll.f1_obs_dim()
    rng = np.random.default_rng(7)
    # Observations in the range the environment actually produces, not uniform
    # noise over the whole space: a parity check on inputs no policy will ever
    # see proves nothing about the ones it will.
    obs = rng.uniform(-1.5, 1.5, size=(n, obs_dim)).astype(np.float32)
    # The traffic channels carry -1.0 for "no car", and the parity sample has to
    # include that value: it is a real input the race produces, and skipping it
    # would leave the evaluator untested exactly where a normaliser is easiest
    # to get wrong.
    obs[:, 14] = rng.uniform(-1.0, 1.0, size=n)   # gap ahead
    obs[:, 16] = rng.uniform(-1.0, 1.0, size=n)   # lateral ahead
    obs[:, 17] = rng.uniform(-1.0, 1.0, size=n)   # gap behind
    obs[:, 20] = rng.uniform(0.4, 1.0, size=n)     # grip scale, as tires wear
    obs[:, 21] = rng.uniform(20.0, 75.0, size=n)    # target speed
    absent = rng.random(n) < 0.25
    obs[absent, 14] = -1.0
    obs[absent, 16] = -1.0
    obs[absent, 17] = -1.0

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    ref = sess.run(None, {sess.get_inputs()[0].name: obs})[0].astype(np.float64)

    import train_lib

    rust = np.asarray(train_lib.rust_evaluate(lib, model_path, obs), dtype=np.float64)

    diff = np.abs(rust - ref)
    worst = float(diff.max()) if diff.size else 0.0
    rel = worst / max(1e-9, float(np.abs(ref).max()))
    tol = 1e-4
    ok = worst <= tol
    print(f"parity    : n={n} max|rust-onnxruntime| = {worst:.3e} "
          f"(relative {rel:.3e}, tolerance {tol:.0e}) -> {'PASS' if ok else 'FAIL'}")
    return {
        "n": n,
        "max_abs_diff": worst,
        "max_rel_diff": rel,
        "tolerance": tol,
        "pass": bool(ok),
        "reference": "onnxruntime CPUExecutionProvider",
    }


if __name__ == "__main__":
    sys.path.insert(0, str(_HERE))
    raise SystemExit(main())
