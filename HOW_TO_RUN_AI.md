# HOW TO RUN — instructions for an AI agent

You are asked to build, verify, and run this project on this machine. Follow
these steps in order. Do not improvise alternatives unless a step fails and
the Failure Playbook tells you what to do. If you are still stuck after the
playbook, stop and report the exact command and error text.

## Project snapshot

- Rust workspace, single crate at `apexsim/`, binary name `apexsim`.
- Subcommands: `run`, `serve`, `race`, `replay`, `validate`, `dev`,
  `aibench`, `weatherbench`, `tournament`.
- The dashboard is served by `serve` at http://127.0.0.1:8080 (static HTML
  at `apexsim/static/index.html`; car sprites and sound WAVs are committed —
  no asset generation needed).
- Windows, macOS, and Linux are all supported. No GPU, no container, no
  network access required at runtime (only the first `cargo build`, which
  downloads ~40 MB of crates).
- Reference docs once running: `apexsim/README.md` (all options),
  `apexsim/README.md` (what each module does).

## Rules of engagement

1. Do not `git commit`, `git push`, or create branches. Build and run only.
2. Do not modify source files to make things work. The ONE permitted edit is
   Step 2 below (the build-path override), which is an environment fix, not
   a code fix.
3. Do not create, delete, or "clean up" anything under gitignored paths
   (`target/`, `out/`, `bench/`, `*.log`, `*.simlog`,
   `artifacts/validation/`) — some of that is measurement
   evidence.
4. Prefer `--release` for builds, tests, and runs as written; debug builds
   are 10–100× slower and will look like a hang.
5. Long waits are normal: first release build ~2–6 min (full LTO), full test
   suite several minutes. Do not kill a step for being slow; only proceed to
   the playbook on a non-zero exit or an explicit error message.

## Step 0 — Environment check

Run:

```
cargo --version
rustc --version
```

- Both print a version → continue.
- Command not found → install Rust non-interactively:
  - Linux/macOS:
    `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y`
    then source `~/.cargo/env` (or open a new shell) and re-check.
  - Windows: download `https://win.rustup.rs/x86_64` and run
    `rustup-init.exe -y --default-toolchain stable`, then re-check in a new
    shell. If the MSVC linker is reported missing, install the "Desktop
    development with C++" workload via the Visual Studio Build Tools
    installer, or switch the host with
    `rustup set default-host x86_64-pc-windows-gnu`.

Minimum hardware to proceed: any 64-bit dual-core CPU, 4 GB RAM, ~3 GB disk.
No GPU needed.

## Step 1 — Locate the crate

All cargo commands run from the `apexsim/` directory (the one containing
`Cargo.toml`). Confirm it exists before proceeding.

## Step 2 — Remove any leftover build-path override

`apexsim/.cargo/config.toml` is gitignored and normally absent from a fresh
clone. If your copy has one, it pins the build output to the original
author's home directory:

```
[build]
target-dir = "C:/Users/<you>/.local/share/apexsim-target"
```

Symptom if left in place: `cargo build` fails with "could not create target
directory" or a permission error mentioning that path.

Action: delete `apexsim/.cargo/config.toml` (this is the one permitted edit).
Alternatively set `CARGO_TARGET_DIR` to a writable path in the environment.
Verify the file is absent before building.

## Step 3 — Build

```
cargo build --release
```

Success criterion: exit 0 and the binary exists at `target/release/apexsim`
(`target/release/apexsim.exe` on Windows). On failure, go to the playbook.

## Step 4 — Verify with the test suite

```
cargo test --release
```

Success criterion: exit 0, i.e. the final lines contain
`test result: ok` for every suite and no `FAILED`. The exact number of tests
grows with each project phase — do not require a specific count. This step
runs for several minutes; that is expected.

## Step 5 — Acceptance run (headless race)

```
cargo run --release -- race --cars 20 --laps 5 --seed 42
```

Success criteria: exit 0; the output shows a formation start, lap
progression, and a final classification table with 20 cars. No `panic`,
no `error]` lines.

## Step 6 — Acceptance run (dashboard server)

Start the server in the background:

```
cargo run --release -- serve --port 8080 --cars 20 --laps 12 --seed 42
```

Then verify, with an HTTP GET to http://127.0.0.1:8080/ :
- status 200, and the body contains `<html` (the dashboard page).
- http://127.0.0.1:8080/api/state (or the page's WebSocket) is served — a
  200/valid JSON on any `/api/*` endpoint documented in `apexsim/README.md`
  is sufficient evidence.

The browser user view: canvas track map with 20 liveried cars, start
lights, live classification, event feed, telemetry and sound panels.

After verifying, stop the server process you started (Ctrl-C / terminate
that exact process). Do not leave it running.

## Determinism — read before reporting anything about hashes

State hashes are reproducible **per platform and per binary build**, not
universally: rebuilding from identical source on a different OS (or
different compiler) shifts f64 values by 1–2 ULP and every hash moves. This
is measured, documented behavior (see `apexsim/README.md`, Phase 10 notes).

Therefore:
- A hash you compute that differs from one printed in a run log or any
  doc is NOT a failure. Never report it as one.
- `replay <file>.simlog` must report PASS/bit-identical only when the log
  was produced on the same machine and same binary. If asked to replay a
  log you just generated (Step 7), it must pass.

## Step 7 — Optional, only if the user asks: replay a run

```
cargo run --release -- race --cars 20 --laps 5 --seed 42 --log /tmp/check.simlog
cargo run --release -- replay /tmp/check.simlog
```

Expected: `replay` reports a bit-identical PASS. Write the log to a temp
path, never into `bench/` or `artifacts/`.

## Optional subsystems — install only if the user asks for that feature

| Feature | Extra dependency | Command shape |
|---------|------------------|---------------|
| AI strategy advisor | Ollama ≥ 0.35 running locally (`ollama pull qwen2.5:3b`) | `cargo run --release -- race --cars 20 --laps 8 --seed 42 --ai --ai-model qwen2.5:3b` — needs a running local Ollama on 127.0.0.1:11434; if unreachable the race still completes on the deterministic baseline |
| UDP telemetry check | Python 3.8+ | start `python apexsim/tools/telemetry_recv.py --port 20777 --seconds 10` first, then add `--telemetry 127.0.0.1:20777` to a race command |
| Tournament in Docker | Docker | `docker build -t f1-sim-tournament apexsim/` (usage in `apexsim/README.md`) |

Ports this project uses: 8080 dashboard (configurable via `--port`),
8099 tournament dashboard, 20777 outbound telemetry, 11434 Ollama.

## Failure playbook

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| `could not create target directory` or permission error mentioning a `C:/Users/...` build path | Step 2 not done | Delete `apexsim/.cargo/config.toml`, rebuild |
| `error: linking with link.exe not found` / MSVC missing (Windows) | Build tools absent | `rustup set default-host x86_64-pc-windows-gnu`, or install VS Build Tools "Desktop development with C++" |
| `Address already in use` on `serve` | Port 8080 occupied | Retry with `--port 8081` |
| HTTP GET times out but process alive | Windows firewall prompt pending | Allow the binary through the firewall, or bind nothing — server listens on localhost only, so this should not occur; report if it does |
| `cargo test` non-zero exit | Real test failure or toolchain too old | Re-run with `-- --nocapture` for the failing suite; capture the failing test names and full error text; stop and report — do NOT edit source |
| Build hangs > 15 min with no output | Likely a stuck compiler under a synced/AV-scanned folder | Check the path is not under OneDrive/Dropbox; report rather than retry-loop |

## Definition of done

You may report success only when ALL of these hold:
1. `cargo build --release` exit 0.
2. `cargo test --release` exit 0 (no `FAILED`).
3. The Step 5 race completed with a classification.
4. The Step 6 dashboard returned HTTP 200 and was then shut down.
5. No source files were modified (Step 2's config deletion excepted) and
   nothing was committed.

Report format: what you ran, exit status, one line per acceptance criterion,
plus any playbook deviations you had to make.
