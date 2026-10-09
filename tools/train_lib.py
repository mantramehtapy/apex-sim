"""Expert collection, training, ONNX export and Rust-side evaluation for
Phase 12.

Kept separate from `gym_train.py` so the command-line entry point stays
readable: that file is the argument handling and the reporting, this one is
the work.

The training method is deliberately plain. The plan says to compare a learned
driver against the Phase 1/4 baseline and warns against assuming self-play
produces a good teacher, so the teacher here is the engine's own deterministic
driver and the method is behaviour cloning against it. No RL, no reward
shaping search, no self-play bootstrap: the claim being tested is "can a small
network reproduce the baseline driver, and does that reproduction hold up in
Rust", not "can this project discover a better driver than it already has".
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np
import torch
import torch.nn as nn

# Chosen so the exported graph stays inside the op set `src/onnx.rs`
# implements: Gemm, Tanh, and the Constant nodes the exporter folds into it.
HIDDEN = (64, 64)


class Policy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in HIDDEN:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        # Tanh on the output keeps every action inside [-1, 1], which is where
        # the Rust decoder expects them; a policy that could ask for more would
        # be asking for something the car cannot do.
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Expert collection
# ---------------------------------------------------------------------------


def collect_expert(lib, env, train_seeds, eval_seeds, keep_eval_demos: bool = False):
    """Roll the baseline driver and record (observation, action) pairs.

    The action comes from Rust's own baseline driver for the current state, not
    from a Python copy of it — a Python reimplementation of the teacher is a
    second thing to keep in sync with the engine.

    Every seed is claimed in Rust's split ledger before it runs. Claiming is
    what makes leakage impossible rather than merely absent: the ledger refuses
    a seed either split already holds, so an evaluation episode cannot be
    replayed as a training episode by a later run with different constants.
    """
    obs_dim = env.obs_dim
    act_dim = env.action_dim
    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    demo_obs: list[np.ndarray] = []
    demo_act: list[np.ndarray] = []
    episode_summaries = []

    scratch = (ctypes.c_float * act_dim)()

    def claim(split: int, seed: int) -> None:
        if lib.dll.f1_split_claim(split, ctypes.c_uint64(seed)) != 0:
            raise SystemExit(f"split ledger refused seed {seed}: {lib.error()}")

    def teacher_action() -> np.ndarray:
        rc = lib.dll.f1_teacher_action(env.handle, scratch, act_dim)
        if rc != 0:
            raise SystemExit(f"teacher action failed: {lib.error()}")
        return np.asarray(scratch[:act_dim], dtype=np.float32)

    def run_seed(seed: int, record: bool) -> dict:
        env.reset(seed)
        local_obs, local_act = [], []
        while True:
            o = env.obs()
            a = teacher_action()
            if record:
                local_obs.append(o)
                local_act.append(a)
            env.step(a)
            if lib.dll.f1_env_flags(env.handle) & 3:
                break
        m = env.metrics()
        env.clear_log()
        return m, np.asarray(local_obs, dtype=np.float32), np.asarray(local_act, dtype=np.float32)

    for seed in train_seeds:
        claim(0, seed)
        m, o, a = run_seed(seed, True)
        obs_buf.append(o)
        act_buf.append(a)
        episode_summaries.append(m)

    for seed in eval_seeds:
        claim(1, seed)
        m, o, a = run_seed(seed, keep_eval_demos)
        if keep_eval_demos:
            demo_obs.append(o)
            demo_act.append(a)
        episode_summaries.append(m)

    obs = np.concatenate(obs_buf) if obs_buf else np.zeros((0, obs_dim), dtype=np.float32)
    act = np.concatenate(act_buf) if act_buf else np.zeros((0, act_dim), dtype=np.float32)

    # The teacher's third channel is a placeholder the decoder ignores; a
    # network trained to reproduce a constant channel would just spend capacity
    # learning zero.
    if act.size:
        act[:, 2] = 0.0

    result = {
        "obs": obs,
        "act": act,
        "n_obs": int(obs.shape[0]),
        "episodes": episode_summaries,
        # Row counts per training episode, in order, so the train/validation
        # split can be cut on episode boundaries rather than mid-episode.
        "episode_lengths": [int(x.shape[0]) for x in obs_buf],
    }
    if demo_obs:
        result["eval_obs"] = np.concatenate(demo_obs)
        result["eval_act"] = np.concatenate(demo_act)
        result["eval_act"][:, 2] = 0.0
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def split_by_episode(
    obs: np.ndarray, act: np.ndarray, episode_lengths, val_frac: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Cut (obs, act) into train/validation halves on episode boundaries.

    A random per-row split would put consecutive frames of the same lap on both
    sides, which is the standard way a behavioural-cloning validation number
    becomes meaningless: the network can memorise a corner it will then be
    scored on. Episodes here are held out whole, so validation is always a race
    the model has not seen a single frame of.
    """
    lengths = [int(n) for n in episode_lengths]
    if sum(lengths) != len(obs):
        raise SystemExit(
            f"episode lengths sum to {sum(lengths)} but there are {len(obs)} rows"
        )
    n_val_eps = max(1, int(round(len(lengths) * val_frac)))
    if n_val_eps >= len(lengths):
        raise SystemExit("validation would take every episode; need at least two")
    cut = sum(lengths[:-n_val_eps])
    return obs[:cut], act[:cut], obs[cut:], act[cut:], n_val_eps


def train(obs: np.ndarray, act: np.ndarray, epochs: int = 40, seed: int = 0,
          batch: int = 256, lr: float = 1e-3, val_frac: float = 0.1,
          episode_lengths=None):
    """Behaviour cloning with MSE. Returns (model, per-epoch validation MSE).

    CPU-only and seeded, so a rerun on the same data reproduces the same
    weights — otherwise "the exported model" would not be a fixed artifact that
    anything downstream could pin.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if episode_lengths is None:
        # No boundaries supplied: fall back to one contiguous tail and say so,
        # rather than silently pretending the split is episode-aligned.
        n_val = max(1, int(len(obs) * val_frac))
        episode_lengths = [len(obs) - n_val, n_val]
    x_tr, y_tr, x_va, y_va, n_val_eps = split_by_episode(obs, act, episode_lengths, val_frac)

    x_train = torch.from_numpy(np.ascontiguousarray(x_tr, dtype=np.float32))
    y_train = torch.from_numpy(np.ascontiguousarray(y_tr, dtype=np.float32))
    x_val = torch.from_numpy(np.ascontiguousarray(x_va, dtype=np.float32))
    y_val = torch.from_numpy(np.ascontiguousarray(y_va, dtype=np.float32))

    model = Policy(x_train.shape[1], y_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()

    history = []
    n = len(x_train)
    gen = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            loss = lossf(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(lossf(model(x_val), y_val))
        history.append(v)
    model.eval()
    train_info = {
        "n_val_episodes": n_val_eps,
        "n_train_rows": int(len(x_train)),
        "n_val_rows": int(len(x_val)),
    }
    return model, history, train_info


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_onnx(path: pathlib.Path, in_dim: int, act_dim: int, model: Policy | None = None,
                seed: int = 0):
    """Export to opset 13, keeping initializers in graph.input.

    `keep_initializers_as_inputs` matters here: a TensorProto does not carry
    its own name, so `src/onnx.rs` recovers the weight names from
    `graph.input`. An export without it produces a file whose weights cannot
    be named at all.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if model is None:
        torch.manual_seed(seed)
        model = Policy(in_dim, act_dim).eval()
    torch.onnx.export(
        model,
        torch.zeros(1, in_dim),
        str(path),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "n"}, "action": {0: "n"}},
        opset_version=13,
        dynamo=False,
        keep_initializers_as_inputs=True,
    )
    return path


# ---------------------------------------------------------------------------
# Rust-side evaluation
# ---------------------------------------------------------------------------


def rust_evaluate(lib, model_path: pathlib.Path, obs: np.ndarray) -> np.ndarray:
    """Run the Rust ONNX evaluator over a batch of observations."""
    dll = lib.dll
    dll.f1_onnx_open.argtypes = [ctypes.c_char_p]
    dll.f1_onnx_open.restype = ctypes.c_size_t
    dll.f1_onnx_run.argtypes = [
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    dll.f1_onnx_run.restype = ctypes.c_int
    dll.f1_onnx_close.argtypes = [ctypes.c_size_t]
    dll.f1_onnx_close.restype = None
    dll.f1_onnx_last_error.restype = ctypes.c_char_p

    handle = dll.f1_onnx_open(str(pathlib.Path(model_path).resolve()).encode())
    if not handle:
        raise SystemExit(f"Rust could not load {model_path}: {dll.f1_onnx_last_error()}")
    try:
        width = obs.shape[1]
        out = np.zeros((obs.shape[0], 3), dtype=np.float32)
        for i in range(obs.shape[0]):
            n = dll.f1_onnx_run(
                handle,
                obs[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                width,
                out[i].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                3,
            )
            if n != 3:
                raise SystemExit(f"Rust inference failed on row {i}: {dll.f1_onnx_last_error()}")
        return out
    finally:
        dll.f1_onnx_close(handle)
