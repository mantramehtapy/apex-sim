# How to Run — for humans

This guide gets Apex Sim building and running on your own
machine, from a clean start, with nothing assumed. If you only read one
section, read **Step 2** — it is the one thing that will bite anyone who
isn't the original author.

## What you're running

A deterministic open-wheel race simulator written in Rust: a physics core, a
20-car race engine with pit stops and weather, and a live browser dashboard.
`apexsim` is the binary; everything runs locally on your machine.

## Bare minimum specs

| Component | Minimum | Comfortable |
|-----------|---------|-------------|
| CPU | Any 64-bit dual-core from the last decade | 4+ cores (faster tournament batches) |
| RAM | 4 GB | 8 GB |
| Disk | ~3 GB free | 5 GB |
| OS | Windows 10/11, macOS, or Linux (64-bit) | same |
| Browser | Any current Chrome / Edge / Firefox / Safari | same |
| Internet | Once — the first build downloads crates (~40 MB) | not needed after that |

The simulator itself is light: a 20-car race uses ~13 µs of CPU per 4 ms tick
(under 1% of one core) and ~50 MB of RAM. The 3 GB disk is mostly the Rust
toolchain and build cache, not the sim.

## Step 1 — Install Rust

Install from https://rustup.rs (the standard Rust installer):

- **Windows:** download and run `rustup-init.exe`. Accept the default
  "stable" toolchain and the default MSVC host when asked (if it says the
  Visual Studio C++ build tools are missing, let rustup install them).
- **macOS / Linux:**
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
  Then open a new terminal so `cargo` is on your PATH.

Verify: `cargo --version` prints something (any stable release from the last
few years works).

That is the only required install. Ollama, Python, and Docker below are
optional extras.

## Step 2 — The one gotcha: fix the build directory

Some clones of this repo carry a `apexsim/.cargo/config.toml` that redirects
build output into the original author's home directory — a workaround for
their OneDrive-synced workspace. It is gitignored now, so you may not have it
at all; if you do, it will break your build with "could not create target
dir" or a permission error on a path that isn't yours.

**Fix — if the file exists, delete it:**

```bash
cd apexsim
rm .cargo/config.toml      # Windows PowerShell: Remove-Item .cargo\config.toml
```

Build output then goes to the normal `apexsim/target/` folder. Nothing else in
the project depends on that file. If you'd rather keep your build output off a
synced folder, set `CARGO_TARGET_DIR` to any writable path instead.

## Step 3 — Build and verify

```bash
cd apexsim
cargo build --release
cargo test --release
```

The first release build takes a few minutes (the project builds with full
link-time optimization). Tests take a few more minutes; the suite covers
every phase and prints one `PASS`-style result block per test. As long as
`cargo test` exits without `test result: FAILED`, you're good.

## Step 4 — Run a race (no browser needed)

```bash
cargo run --release -- race --cars 20 --laps 5 --seed 42
```

You'll see the formation start, live standings as laps complete, contacts
and pit stops, and a final classification table. The same seed always
produces the same race — try seed 7, or `--laps 12`.

## Step 5 — Run the live dashboard

```bash
cargo run --release -- serve --port 8080 --cars 20 --laps 12 --seed 42
```

Open **http://127.0.0.1:8080** in your browser. You get a live track map
with all 20 cars in their 2026 liveries, the starting-light sequence, live
classification with tire compounds and pit badges, a race-control event
feed, and (top right) telemetry/sound panels — the sound panel has an
"Enable audio" button that plays the built-in cues. Stop the server with
Ctrl+C in the terminal.

On Windows, the first run may trigger a firewall prompt for the Rust
binary — allow it; the server only listens on your own machine.

## More things to try

```bash
# Re-run any recorded race bit-identically (headless):
cargo run --release -- race --cars 20 --laps 5 --seed 42 --log myrace.simlog
cargo run --release -- replay myrace.simlog

# Physics validation against real Bahrain 2024 qualifying reference data:
cargo run --release -- validate

# Race a different circuit. Four presets ship in the binary, or pass any
# track JSON file (a closed loop of control points + track width):
cargo run --release -- race --track kidney --cars 20 --laps 5 --seed 42
cargo run --release -- race --track tracks/paperclip.json --cars 12 --laps 8
# Bad circuits (self-crossing, impossibly tight, too short) are refused at
# load with the reason, never mid-race. See apexsim/tracks/ for the format.

# Weather: a wet race with mid-race grip changes:
cargo run --release -- race --cars 20 --laps 8 --weather wet --overrides '30:water:0.9,60:grip:0.85'

# Custom circuits: pick a preset or point at a track JSON file:
cargo run --release -- race --cars 20 --laps 5 --seed 42 --track trioval
cargo run --release -- serve --port 8080 --track paperclip
cargo run --release -- race --track mytrack.json
# Presets ship in apexsim/tracks/ (paperclip, kidney, trioval, roadcourse);
# a track file is {"name", "half_width", "points", "pit"} -- the engine
# smooths the points into the circuit and validates it at load.

# Realism pack (on by default): driver mistakes can spin cars, engines and
# gearboxes fail, touches cost grip, brake locks flat-spot tyres, crashes
# deploy a REAL safety car on the map with local yellow sectors, lapped
# cars see blue flags, races open with a rolling formation lap, DRS works
# on the longest straights, and cars carry fuel mass + cold tyres (burn
# per metre, refuel at stops, stop time scales with the fill).
# Tune or step back to the old engine:
cargo run --release -- race --cars 20 --laps 12 --seed 42 --error-rate 0.1
cargo run --release -- race --cars 20 --laps 12 --seed 42 --realism off

# The race weekend (Phase E): one-by-one timed qualifying sets the grid,
# then the race runs on that grid. --quali-s is the qualifying clock per
# car; --no-quali skips straight to a race:
cargo run --release -- weekend --cars 20 --laps 12 --seed 42 --track trioval --quali-s 90
cargo run --release -- weekend --cars 10 --laps 5 --seed 7 --no-quali

# Headless tournament: 5 races, scored report in bench/batch1/:
cargo run --release -- tournament --seeds 1,2,3,4,5 --cars 10 --laps 4 --out bench/batch1

# Phase 12: the learned driver. Train it (needs Python with torch, onnx and
# onnxruntime; takes a few minutes):
python tools/gym_train.py --episodes 40 --eval-episodes 10 --epochs 60                           --laps 4 --latency-runs 3000 --out artifacts/phase12
# Then compare it against the built-in driver on held-out seeds, or measure
# its inference latency across a full grid:
cargo run --release -- policy-eval --model artifacts/phase12/policy.onnx --seeds 9001,9002
cargo run --release -- policy-latency --model artifacts/phase12/policy.onnx
# The trained policy currently loses to the built-in driver on this circuit,
# so the engine races with the built-in driver unless you ask for the policy.
```

Full option reference: `apexsim/README.md`.

## Optional extras

Each of these is off by default and only needed for the feature named:

| To use | Install | Notes |
|--------|---------|-------|
| AI strategy advisor (`--ai`) | [Ollama](https://ollama.com), any model | `cargo run --release -- race --cars 20 --laps 8 --seed 42 --ai --ai-model qwen2.5:3b` |
| UDP telemetry receiver | Python 3.8+ (standard library only) | `python apexsim/tools/telemetry_recv.py --port 20777 --seconds 10` |
| Regenerate sound cues / car sprites | Python 3.8+ (standard library only) | `python apexsim/tools/make_sounds.py`, `python tools/make_sprites.py` — both assets are already committed; only needed if you change them |
| Tournaments in Docker | Docker | `docker build -t f1-sim-tournament apexsim/` then run per `apexsim/README.md` |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Build fails: "could not create target dir" / permission denied on `C:/Users/<you>/...` | A leftover build-path override from Step 2 | Delete `apexsim/.cargo/config.toml`, or set `CARGO_TARGET_DIR` |
| `serve` fails: address already in use | Port 8080 is taken | `--port 8081` (or close the other app) |
| A race log replayed on your machine doesn't match a hash printed in the docs | Hashes are per-machine (floating-point differences between platforms) | Not a fault — replays of logs made **on your machine** always match |
| Debug build (`cargo test` without `--release`) is very slow | The engine is built for release optimization | Use `--release` as shown |
| `cargo` not found | rustup PATH not loaded | Open a new terminal |

Questions the guide doesn't answer: `apexsim/README.md` (what was built and
measured), `apexsim/README.md` (every command and option).
