# apexsim — Rust Simulation Core + Browser Dashboard

Phase 1: one car, one dry stadium track (2×300 m straights, 2×R80 m corners), a
deterministic line-following driver, seeded reset, pause, state logging, and
bit-exact input replay. Phase 2: an axum/tokio server publishing ~25 Hz
snapshots over WebSocket with an embedded canvas dashboard. Phase 3:
reference-based physics validation at 125–1000 Hz. Phase 4: the full race
field — grid scaling 1→4→10→20, traffic awareness with overtaking/defending,
impulse contacts with retirements, classification, and a multi-car dashboard.
Phase 5: the first complete race weekend — formation start with the
five-light sequence, session modes (race / qualifying / practice), tire
compounds with distance-based wear, a pit lane (brake zone, entry, 24 m/s
limit, box service, exit) with deterministic strategies, and a results sheet
with fastest lap and the full stop log. Phase 7: an optional local LLM
strategy advisor (Ollama) that is asked three typed questions — pace, pit
intent, tire choice — on the advised car's lap boundaries, never for steering
and never on a physics tick, with deterministic Rust rules keeping ownership
of legality and fallback and every exchange written to an audit log.
Phase 8: weather and race-director control — deterministic dry/damp/wet
surface zones, water / track-temperature / grip sliders applied only on a
queued simulation tick (never injected into live physics state), every
override logged with its effective tick and replayable through the run log,
and seeded driver errors applied as a pure control-layer perturbation so a
replay reproduces every mistake. See `apexsim weatherbench` for the measured
braking, cornering, and lap-time effects. Phase 9: hardware telemetry and
sound — UDP telemetry in the 2025 packet layout (`m_packetFormat = 2025`, port
20777) published on its own schedule so physics never waits on a socket, and a
validated 13-cue vocabulary (shift, limiter, tyre slip, wind, race events)
extracted deterministically from simulation state and played by a bounded,
priority-aware consumer — in the engine or in the browser dashboard.
Phase 10: headless tournaments and deployment — a saved `BatchSpec` (explicit
seeds, per-run policies, track version and geometry, session setup) that fully
describes a set of races, a scoring report covering finishes, lap consistency,
contacts, pit choices and run failures with a sample size attached to every
figure, a serial-first runner whose concurrency is enabled only after being
measured, a container image, and a dashboard that stays observable and
stoppable while a long batch runs.
Phase 11: an optional in-simulation stewarding aid — an incident recorder
that keeps positions, overlap, contact, track limits, speed and the events
around each incident; three stated simulator rules (contact, shortcut, unsafe
rejoin) that record evidence, rule version, decision and service; and a
no-penalty or inconclusive outcome for every case the evidence does not
support. Review sheets, the penalty ledger, and the `D`/`P` run-log records are
replayable, so any penalty can be audited from its record alone. It is an aid,
not an official rules authority.

The sim core is deterministic and self-contained; the server layer adds axum,
tokio, and serde for the wire protocol.

## Build & run

```bash
cargo build --release
cargo test --release          # 191 tests: unit + Phase 1–12 gate suites

# Race 5 seeded laps as fast as the CPU allows, write the input log:
apexsim run --laps 5 --seed 42

# Race 1 lap locked to wall-clock 250 Hz (deadline compliance check):
apexsim run --laps 1 --seed 42 --realtime

# Verify a saved run reproduces bit-identically:
apexsim replay artifacts/phase1_maxspeed.log

# Phase 3 physics validation (component tests at 125–1000 Hz, lap traces,
# published tolerances, report + replayable case files):
apexsim validate
apexsim validate --replay-case artifacts/validation/cases/braking_50_250hz.casefile

# Phase 4: a full 20-car race (classification, contacts, retirements):
apexsim race --cars 20 --laps 5 --seed 42 --log artifacts/phase4_race_20car.simlog
apexsim replay artifacts/phase4_race_20car.simlog   # whole-field bit-exact replay
apexsim race --cars 20 --scale-bench                # 1/4/10/20 tick-cost ladder
apexsim race --cars 20 --probe 750                  # field state at t = 3 s

# Phase 5: a race weekend — formation start, tires, pit stops, results:
apexsim race --cars 20 --laps 5 --seed 42 --log artifacts/phase5_race_20car.simlog
apexsim replay artifacts/phase5_race_20car.simlog   # formation+pits replay bit-exact
apexsim race --cars 20 --laps 3 --mode qualifying   # timed session, best-lap order
apexsim race --cars 10 --laps 8 --compound soft --stop-lap 4 --switch-to hard
apexsim race --cars 20 --laps 5 --no-pits           # tire-only comparison

# Serve the live 20-car race dashboard (http://127.0.0.1:8080):
apexsim serve --port 8080 --cars 20 --laps 12 --seed 42 --noise 0.002

# Phase 6: one command to start, inspect, and stop a dev server.
apexsim dev --init                                 # write the documented f1_dev.toml
apexsim dev                                        # launch from f1_dev.toml; waits for readiness
apexsim dev --port 8090 --cars 10 --duration 60    # flag overrides; auto-shutdown after 60 s
apexsim dev --config f1_dev.toml                   # explicit config file

# Phase 7: local strategy advisor (needs Ollama running on 127.0.0.1:11434).
# Car 0 is advised; everything else stays deterministic.
apexsim race --cars 20 --laps 8 --seed 42 --ai --ai-model qwen2.5:3b
apexsim race --cars 20 --laps 8 --ai --ai-log bench/exchanges.jsonl

# The gate-1 benchmark: same seeds with and without the advisor.
apexsim aibench --model qwen2.5:3b --seeds 1,2,3,4,5,6 --laps 8 --cars 20
apexsim aibench --model tev1:0.8b --seeds 1,2,3 --quiet

# Phase 8: weather presets, director sliders, and queued overrides.
# --overrides takes seconds or a raw tick: '20:water:0.9,@2500:grip:0.85'
apexsim race --cars 20 --laps 8 --weather wet
apexsim race --cars 20 --laps 8 --weather damp   --overrides '30:water:0.9,60:grip:0.85' --driver-error 0.5 --log bench/wet.log
apexsim replay bench/wet.log

# The gate-2 benchmark: measured braking, cornering, and race-time effects.
apexsim weatherbench --seeds 1,2,3 --laps 8 --cars 20

# Phase 9: publish UDP telemetry and extract sound cues.
apexsim race --cars 20 --laps 8 --telemetry 127.0.0.1:20777 --telemetry-hz 30 --sound
# Start the independent receiver in another shell first:
python tools/telemetry_recv.py --port 20777 --seconds 10

# Phase 10: headless tournament batches. Write a spec, run it, keep the
# artifacts, and reproduce the batch later from the saved configuration alone.
apexsim tournament --write-spec bench/batch.json      # a valid starter spec
apexsim tournament --seeds 1,2,3,4,5 --cars 10 --laps 4 --out bench/batch1
apexsim tournament --spec bench/batch1/spec.json --out bench/batch2
apexsim tournament --replay bench/batch1             # compare batch2 against batch1

# Score the same seed under several policies (seed-major, then policy):
# soft|medium|hard (auto one-stop), no-stop, advised
apexsim tournament --seeds 7 --policies soft,no-stop --cars 10 --laps 6

# Watch a batch on the dashboard, and stop it without killing the process:
apexsim tournament --profile engine+dashboard --seeds 1,2,3,4,5,6 --port 8099
#   GET  /api/batch         -> {"active":true,"done":2,"total":6,...}
#   POST /api/batch/stop    -> {"ok":true,"stop_requested":true}

# Measure before parallelising. Serial is the baseline; the tool checks that
# the parallel run produced the same hashes rather than assuming it.
apexsim tournament --seeds 1,2,3,4,5,6,7,8 --cars 20 --laps 8 --measure 8

# Then run for real across workers. The batch runs once, and the wall time in
# the saved report.json is that one execution -- not a serial re-run.
apexsim tournament --spec bench/batch1/spec.json --out bench/batch3 --threads 8

# Container: same binary, somewhere that is not the laptop.
docker build -t f1-sim-tournament .
docker run --rm -v "$PWD/spec.json:/work/spec.json:ro" -v "$PWD/out:/work/out" \
           f1-sim-tournament --spec /work/spec.json --out /work/out

# Phase 11 (optional): run the stewarding aid. Off by default; turning it on
# does not change the race -- the aid is an observer, never a participant.
apexsim race --steward --cars 12 --laps 8 --seed 3
#   review sheets per incident, then the penalty ledger and time served.

# Re-derive a review from a run log. This replays and re-judges, so it
# reproduces the log's own records or tells you they no longer match.
apexsim steward artifacts/phase11_race.simlog

# Score the aid across a tournament batch. The report gains an
# incidents/run -> penalised, inconclusive, served row, and only when the aid
# actually ran.
apexsim tournament --steward --seeds 1,2,3,4,5,6 --cars 20 --laps 8 --out bench/batch4

# Field: the 2026 championship is the default for `race` and `serve` --
# 11 teams with their own engine, top speed, brakes and tyres from
# apexsim/teams.json (edit that file to tune a team; both cars of a
# team share the machinery, drivers differ by a seeded +/-0.3% form delta).
# `--field classic` restores the identical-cars engine bit-for-bit.
apexsim race --cars 20 --laps 5 --seed 42 --field classic

# Phase C — custom circuits. Presets: paperclip, kidney, trioval, roadcourse.
# Or author a track JSON: {"name": ..., "points": [[x,y],... >= 6], "pit": {...}}
apexsim race --track kidney --cars 20 --laps 5 --seed 42
apexsim race --track mytrack.json --cars 12 --laps 8 --seed 3
apexsim serve --track trioval --port 8080     # dashboard draws the custom shape
# A run log carries the circuit's control points, so a custom-track log
# replays bit-identically on any machine, preset or not.

# Phase 12 (optional): train the learned driver. Needs the release DLL, and
# the trainer refuses to run against one older than apexsim/src -- a stale
# library silently produces a model fitted to engine behaviour that no longer
# exists. Rebuild first if it complains.
cd apexsim && cargo build --release && cd ..
python tools/gym_train.py --episodes 40 --eval-episodes 10 --epochs 60 \
                          --laps 4 --latency-runs 3000 --out artifacts/phase12
#   expert.npz + policy.onnx + splits.json, then parity against onnxruntime,
#   p99 latency over the 20-car grid, and a baseline-vs-learned comparison on
#   held-out seeds. The verdict line says whether the driver was enabled; on
#   this circuit it is not, because it loses to the local controller.

# Phase E: the race weekend -- solo timed qualifying (cars out one by one,
# each run measuring its own car's machinery, form and noise) sets the grid,
# then the race runs on that grid:
cargo run --release -- weekend --cars 20 --laps 12 --seed 42 --quali-s 90
cargo run --release -- weekend --cars 10 --laps 5 --seed 7 --track paperclip --no-quali
```

### Phase 12 notes

- **The learned driver is not enabled, and that is the result.** It was trained
  as a small MLP on expert data from the Phase 1/4 baseline driver, exported to
  ONNX, and compared against that same baseline over 10 held-out seeds. It
  clones the teacher well (steer RMSE 0.0035, pedal RMSE 0.0527) and still loses
  by 12 places, 1.7 laps and 22.9 s per lap. The gate says enable it "only if it
  improves declared metrics", so it ships disabled and `policy::Controller`
  falls back to the local driver. The negative result is recorded in
  `artifacts/phase12/comparison.json` with the reasons, not smoothed over.
- **Cloning is not driving.** A pure-pursuit controller is a feedback loop over
  track geometry; a network fed 22 numbers reproduces the current pedal without
  representing the lookahead that produced it, so it is slow into corners and
  gets collected. Behaviour cloning was the cheap experiment the plan asked for;
  RL on the same `EnvSpec` is the obvious next step and is not built.
- **Python drives Rust through one persistent handle.** `GymEnv` is behind a C
  ABI; the trainer creates it once and calls `f1_env_reset` / `f1_env_obs` /
  `f1_env_step` / `f1_teacher_action` in a loop. No process per action, and
  `step` returns only once the physics has moved.
- **`EnvSpec` is the contract, and `env_spec.json` is its record.** 22
  observation channels each with an explicit SI scale, an action every 5 ticks,
  progress/speed/off-track reward, and stall/off-track/race-end/budget
  termination. The gate compares the artifact against `EnvSpec::default()` so
  the file cannot describe an environment the engine does not run.
- **The split cannot leak.** `SplitLedger` refuses a seed the other split holds,
  in both directions, and a second claim within one split. It lives in Rust
  because the trainer is the thing that would otherwise be trusted with its own
  seeds. The run also writes `splits.json` so the disjointness of *that* run can
  be re-derived rather than taken on trust.
- **The trainer refuses a stale library.** `find_library()` compares the DLL's
  mtime against the newest file in `apexsim/src` and aborts if the engine is
  newer. This is not hypothetical: the first post-fix training run silently
  loaded a DLL predating the driver fix, collected all 354,879 transitions from
  the buggy teacher, and reported the result as current. Validation MSE went
  0.0329 → 0.00090 on rebuild.
- **`src/radio.rs` is an observer over stable events only.** It reads the
  race-control log and the stewarding aid's incident list — both already
  reproducible — and never per-tick car state, so it cannot chatter mid-corner.
  Messages are capped at 96 characters and cite the event they came from. Pit
  rejoins and inconclusive steward findings are deliberately silent. It passes
  independently of everything else in this phase, and attaching it leaves the
  race state hash unchanged.
- **Two more engine bugs came out of this phase.** `Driver::drive()` applied
  seeded noise to both pedal channels before the both-pedals rule, silently
  cancelling full throttle on ~46% of steps — the real cause of the long-standing
  "the network cannot learn the pedals" failure. And `contact_pass` wrote
  `vel_long` on retired cars, so a stopped hazard could be re-launched by a
  passing car (one did, 48 s after the race was decided).
### Phase 11 notes

- **It says what it is.** Official driving standards judge the whole incident and
  are guidelines, not a single apex-overlap formula, so the aid never presents
  itself as a rules authority. `apexsim race --steward` and `apexsim steward`
  both print the disclaimer, and a gate test fails if it goes missing.
- **Three rules, each with its thresholds logged.** `contact` (closing speed
  ≥ 8 m/s, penalise the closing party only if its closing component beats the
  other's by ≥ 3 m/s), `shortcut` (≥ 25 ticks at least 1 m beyond the edge),
  and `unsafe-rejoin` (returns after ≥ 25 ticks off, into a car that closed to
  within 12 m at ≥ 6 m/s while still having ≥ 12 m of room at the rejoin). The
  `D` log header carries the rule set, version and all eleven thresholds, so a
  record says which rules and which numbers produced it.
- **Ambiguity is an outcome, not a failure.** Five branches decline with
  `NoPenalty` or `Inconclusive`, naming no car and serving no time. A 24 000-case
  sweep asserts the structural invariant: only a `Penalised` decision can carry
  a penalty. The aid errs toward recording nothing.
- **Every penalty is auditable from its own evidence.** Re-running the judge on
  a record's stored evidence reproduces the same decision, and a replay of the
  log reproduces every `P` line field for field — a tampered decision line is
  refused rather than shown.
- **The track-limit rules are calibrated, and do not fire on this circuit.**
  Over 20-car × 12-lap and 12-car × 10-lap races the furthest any car gets
  past the edge *outside the pit lane* is 0.46 m, with no gap in the
  distribution. The 1.00 m threshold sits above what these drivers do, so the
  shortcut and rejoin rules are reachable and tested but silent in normal
  racing. Lowering the threshold to make them fire would penalise cars running
  half a metre wide. The pit lane, at a designed −7.5 m against a 7.0 m
  half-width, is excluded from track-limits accounting entirely.
- **It is an observer.** Stewarding is off by default, and the gate suite
  requires an identical trajectory hash with the aid on and off. Penalties
  reorder a derived view; the simulated classification is never edited.
- **This phase fixed a real physics bug.** The contact pass computed car j's
  lateral velocity from `u_j * h_jx` instead of `v_j * h_jx`, which could
  report a closing speed larger than the car's own speed. Because the aid reads
  that value as fault evidence, it could have turned an ambiguous contact into a
  confident penalty against the wrong car. Real penalties fell from 60 to 2 in
  a 20-car × 12-lap race once fixed; the earlier number was an artefact.

### Phase 10 notes

- **The spec is the reproduction recipe.** Seeds, per-run policies, session
  setup, and the track version *with its geometry* are all in `spec.json`. The
  track was not an input at all before this phase — `Race::new` hardcoded
  `Track::default()` — so two batches on different circuits used to be
  indistinguishable. It is now a `RaceConfig` field and a `G` line in the run
  log.
- **Every mean carries its own `n`, and failures are data.** A run that panics,
  is rejected, or hits the cool-down cap is recorded with its status and
  excluded from every aggregate rather than averaged over. A verdict line will
  not print a general claim from fewer than 5 reproducible runs; it says
  "NOT a general claim: N reproducible run(s) < 5" instead.
- **Advised runs are not reproducible** (Phase 7: model latency feeds back),
  so they are counted separately and never counted toward the reproducible
  sample.
- **Concurrency was measured first.** On an 8C/16T box, 8 races × 20 cars × 8
  laps: 1.88× at 2 workers, 2.75× at 4, 4.64× at 8, no further gain at 12.
  Results are bit-identical at every count.
- **State hashes are reproducible per platform and per binary, not universally.**
  Within one binary, repeated runs are bit-identical. Across platforms, and
  even across two builds of *identical source*, the `f64` scores move by 1–2
  ULP and every hash moves — measured with noise disabled at 1 car × 1 lap, so
  it is `libm` differing in the last bits, not drift accumulating. So
  `engine_id()` carries a source hash (not a hand-bumped constant), reports
  record their platform, and `--replay` distinguishes:
  - `IDENTICAL` — same engine, same platform, every hash reproduced;
  - `CROSS-PLATFORM MATCH` — every score agrees, the hashes are not portable;
  - `NOT COMPARABLE` — different engine, spec, track or profile;
  - `DIVERGED` — same everything, and something genuinely moved.
- **A stopped batch keeps what it scored.** The stop flag is checked between
  races, so the race in flight always runs to the flag and the report still
  says how many were planned. A batch stopped before it starts still scores
  one race rather than returning an empty report.

### Phase 9 notes

- **The protocol is one exact named layout and version**: the 2025 UDP telemetry
  UDP, `m_packetFormat = 2025`, port 20777, little-endian, 29-byte header.
  Motion (id 0, 1349 bytes) and Car Telemetry (id 6, 1352 bytes) are emitted;
  every other packet id is left zero. The field table, its provenance, and a
  real bug it caught are in [`docs/f1_2025_udp_spec.md`](docs/f1_2025_udp_spec.md).
- **Port 20777 is not compatibility.** No vendor application — MoTeC, SimHub,
  or any racing game — has been tested, and the spec doc says so plainly. What is
  verified is a byte-level layout agreement with three independent third-party
  decoders on a real loopback capture, which is a *layout* check and not a value
  check.
- Telemetry publishes on its own schedule: the sim thread hands a datagram to a
  bounded queue (`try_send`, depth 4) and moves on. Under back-pressure the
  datagram is dropped and counted rather than queued — a 6-car max-speed race
  drops ~10,800 of 13,042 and the race hash is unchanged, because nothing on
  `Race` is reachable from `mix_state`.
- Sound is **extraction total and playback lossy**, on purpose. Extraction is a
  pure function of `(tick, car state)`, written to an ordered, hashed log that a
  replay reproduces tick for tick. Playback is bounded (default 40 cues/s) and
  sheds by priority, so ~4,300 cues/s from an accelerated race become ~40 —
  green flag and start lights always survive, continuous slip and wind cues
  absorb the losses.
- Both subsystems are off unless asked for, and an inert config normalises back
  to `None`: a run with `--sound` on and nothing listening is the Phase 8 race
  bit-for-bit.
- The dashboard has **telemetry** and **sound** panels. The telemetry panel shows
  offered/sent/dropped separately, because the gap between them is the drop
  counter doing its job and is only legible if both sides are on the page. The
  sound panel's "Enable audio" loads the 13 validated WAVs from `/sounds/…` and
  plays whatever the simulation decided to fire — the browser is a consumer, not
  a second extractor, so pause/replay/accelerated synchronisation comes for free.

### Phase 8 notes

- Weather is off unless asked for: `RaceConfig.weather` is `None` by default
  and an inert config (`--weather dry`, no overrides, no errors) normalises
  back to `None`, so a dry run is the Phase 7 race bit-for-bit.
- Slider values are queued for a known tick and applied at the top of the
  tick — in formation and on the replay path as well as the live one. Out of
  range values are clamped with a printed note, never silently accepted.
- A wet run is slower because the *driver* is slowed by exactly what the
  surface takes away, not just because grip is lower. Measured: wet braking
  is 60% longer from 200 km/h, the cornering limit drops 45% (1.56 g →
  0.86 g), and an 8-lap race costs ~55 s.
- The dashboard has a **race-director panel**: water, track temperature, and a
  grip trim, plus a surface-layout preset for the next session. A move sends
  `slider:<name>:<value>` on the existing command channel, is queued for the
  next simulation tick, and is acknowledged with the exact tick it will take
  effect on — the same queue `--overrides` uses, so a live move replays
  identically to a pre-set one. A served session starts with the panel live but
  grip-neutral, so an untouched panel is the Phase 7 race bit-for-bit.
- Driver errors are pure functions of `(seed, car, lap)` — no RNG state to
  carry — so they replay exactly. They are a control-layer perturbation only:
  physics never sees a mistake, only slightly wrong inputs.
- Surface grip multipliers and the hydroplaning threshold are assumptions,
  now pinned by tests but not validated against real data.

### Phase 7 notes

- `--ai` needs Ollama ≥ 0.35 for the plan's `tev1:0.8b`; any local model
  works via `--ai-model`.
- An advised run is **not reproducible** (model latency feeds back into
  decisions), so `--log` is refused on `--ai` runs. Use `--ai-log` for the
  exchange record.
- In realtime the advisor is fire-and-forget and late answers are
  stale-rejected, so the 250 Hz loop never stalls. In max-speed mode it waits
  at lap boundaries, because the sim covers 250 s of race time in under a
  second and the model would otherwise always answer after the flag.
- Measured with `qwen2.5:3b`: helped in some scenarios, hurt in others, no
  net edge. With the plan's `tev1:0.8b` it produced zero usable answers and
  every race fell back to the deterministic baseline.

`apexsim` binary: `<target-dir>/release/apexsim.exe` under the dev machine's
`CARGO_TARGET_DIR` (a gitignored `.cargo/config.toml` keeps the build output
outside the OneDrive workspace; it is not committed).

Options: `--laps N` `--seed S` `--noise A` (driver-input noise, 0 = silent)
`--realtime` `--log PATH` `--quiet`; `race` adds `--cars N` (1..=30, default 20
= the 2024-season grid — never mixed with 22), `--mode race|qualifying|practice`,
`--compound soft|medium|hard` (start tires), `--no-pits`,
`--stop-lap N --switch-to C` (fixed one-stop strategy), `--scale-bench`,
`--probe N`; Phase 8 adds `--weather dry|damp|wet|mixed`, `--water W`,
`--track-temp C`, `--grip G`, `--overrides 'TICK:SLIDER:VALUE',...`,
`--driver-error RATE`; Phase 9 adds `--telemetry HOST:PORT` (default
127.0.0.1:20777), `--telemetry-hz HZ`, `--telemetry-player N`,
`--telemetry-uid UID`, `--sound`, `--sound-max-cues N`, `--slip-threshold G`;
`serve` adds `--cars N` (1 = the Phase 1–3 single-car session) and `--mode M`;
Phase 10 adds `tournament` with `--spec FILE` `--out DIR` `--replay DIR`
`--write-spec FILE` `--seeds 1,2,3` `--policies soft|medium|hard|no-stop|advised`
`--cars N` `--laps N` `--mode M` `--weather W|--no-weather`
`--profile engine-only|engine+dashboard|engine+ollama` `--port N`
`--threads N` `--measure N` `--logs` `--quiet`.

## Architecture

```
main.rs            CLI: run / serve / dev / race / aibench / weatherbench / tournament /
                   replay / validate / steward
src/
  config.rs        Phase 6: documented f1_dev.toml template + strict
                   key=value parser (unknown keys are errors)
  logging.rs       Phase 6: structured NDJSON logs (ts, severity, run_id,
                   sid, tick) at lifecycle events only, optional file mirror
  sim.rs           authoritative single-car state, 4 ms tick loop, pacing
  physics.rs       planar bicycle model + per-axle friction circle (pure fn)
  driver.rs        50 Hz pure-pursuit steering + curvature-aware speed control,
                   lateral line offset + traffic speed cap inputs (Phase 4)
  track.rs         analytic stadium geometry, projection, curvature; Phase 10
                   adds TrackSpec — version + geometry — so the circuit is a
                   saved input rather than a constant in Race::new
  race.rs          Phase 4: the field — grid spawn, traffic awareness
                   (on-my-path scan, clear-gap safe-follow, overtake/defend/
                   crawl-past offsets), impulse contacts + retirements,
                   classification. Phase 5: session modes, formation +
                   five-light start, tire compounds/wear, pit lane state
                   machine + strategies, fastest lap, race-control events,
                   v5 session log incl. the weather W line (absent = no
                   weather, so a dry run replays as the Phase 5/7 engine) and
                   the Phase 11 steward D header + per-incident P lines,
                   race-control events, deterministic hashing over the whole
                   state machine
  steward.rs       Phase 11: the in-simulation stewarding aid — StewardConfig
                   (the thresholds ARE the rule, so all of them are logged),
                   the incident Recorder (positions, overlap, contact, track
                   limits, speed and events over a history+review window),
                   three stated rules (contact / shortcut / unsafe-rejoin) with
                   NoPenalty and Inconclusive outcomes, the penalty ledger and
                   post-penalty order, the human-review sheet, and the
                   header_line()/incident_line() pair the log writer and the
                   replay checker share so the two cannot drift
  weather.rs       Phase 8: Surface/SurfaceZone layouts, race-director
                   Slider + queued Override schedules, surface_grip()
                   (zone x water x temperature x grip trim, with
                   hydroplaning above a speed threshold), and DriverErrorConfig
                   (seeded, pure per (car, lap) control perturbation)
  telemetry.rs     Phase 9: UDP telemetry (m_packetFormat 2025) Motion + Car
                   Telemetry encoders, and a publisher that hands datagrams to
                   a bounded queue on a sender thread — try_send, drop and
                   count, so physics never waits on a socket
  sound.rs         Phase 9: the 13-cue validated vocabulary, deterministic
                   extraction from (tick, car state), and AudioConsumer — a
                   bounded, priority-shedding playback queue
  tournament.rs    Phase 10: BatchSpec (the saved reproduction recipe — seeds,
                   policies, session, track version + geometry), RunReport and
                   ScoreSummary (every mean carries its n; failures are counted,
                   never averaged over), BatchRunner (serial first, parallel
                   verified against the serial hashes), compare() /
                   CROSS-PLATFORM MATCH, and measure_concurrency()
  replay.rs        numeric run-log writer/reader (H/I/R line format, v1 + v2;
                   the G line added in Phase 10 carries the track spec; Phase 11
                   reads back the steward D/P records and refuses a log whose
                   incidents no longer reproduce)
  rng.rs           seeded LCG for driver noise (and the start-light hold)
  stats.rs         tick-cost percentiles
  protocol.rs      versioned wire messages: v / sid / tick / seq; snapshots
                   carry cars[] + leaderboard[] + contacts_total + session +
                   lights in race mode; race-control event kinds
  server.rs        axum HTTP + WS bridge: watch-channel snapshots (25 Hz),
                   broadcast events (laps, contacts, finish), bounded
                   per-client queues, command acks; race-mode sim thread;
                   Phase 6: readiness polling, structured lifecycle logs,
                   graceful Ctrl-C/duration shutdown, bind-error reporting;
                   Phase 10: /api/batch + /api/batch/stop, and the served race
                   now runs on the circuit /api/track reports
  validation.rs    Phase 3: component tests vs references, 125–1000 Hz rate
                   sweep, lap-trace comparison, case storage + replay
static/
  index.html       embedded dashboard: canvas track map with all cars (2026
                   liveries from images/ via tools/make_sprites.py, served at
                   /cars/*, rotated to heading, retired fade, leader ringed,
                   interpolation), formation start lights, live classification
                   with 2026 numbers / compound chips / stop counts / PIT
                   badges, telemetry HUD, race-control event feed, controls,
                   reconnect logic
  cars/            generated livery sprites (see tools/make_sprites.py)
tests/
  phase1_gate.rs   Phase 1 completion-gate criteria
  phase2_gate.rs   Phase 2 gate: reconnect catch-up, disconnect isolation,
                   versioned flow, command acks, config validation, metrics
  phase3_gate.rs   Phase 3 gate: tolerances met, convergence, report,
                   stored-case replay + model-version guard
  phase4_gate.rs   Phase 4 gate: determinism + v2 replay at every scale,
                   contact/retirement, dense 20-car start, classification,
                   tick-cost ladder, sustained realtime 20-car race with a
                   connected WS client
  phase5_gate.rs   Phase 5 gate: start lights + seed-deterministic green,
                   seeded race → correct classification, pit stops lose time
                   and repeat bit-exactly, compound/wear pace effects,
                   session modes, retired-car consistency, v3 + legacy v2
                   replay round-trips, pit-lane limits/service/compounds
  phase6_gate.rs   Phase 6 gate: drives the real `dev`/`serve` binaries —
                   fresh-start readiness report with structured logs, failed
                   dependency (occupied port / bad config) → visible error +
                   non-zero status, shutdown stops only its own children,
                   documented config init + honored values
  phase7_gate.rs   Phase 7 gate: scripted mock Ollama — good advice applied
                   and recorded; timeout / unload / bad answer / prose /
                   low-confidence / unreachable all fall back to a complete
                   race with the baseline hash; illegal pit, tire, and
                   Fixed-strategy ownership; lap cadence and p99 timing;
                   baseline untouched, full audit trail, stale rejection
  phase8_gate.rs   Phase 8 gate: same seed + override schedule reproduces
                   exactly (and a different schedule does not); measured wet
                   braking and cornering limits; hydroplaning threshold;
                   override replay round-trip, no W line for a dry run,
                   unfired overrides leave no trace, overrides land on their
                   queued tick; seeded driver errors off by default,
                   reproducible, bounded, and control-layer only; p99 timing
  phase9_gate.rs   Phase 9 gate: one named protocol/version with units, rates,
                   headers and car indexing confirmed; a real loopback capture
                   decoded by an INDEPENDENT decoder (shares no code with
                   telemetry.rs) and cross-checked packet-against-packet; a
                   dead receiver leaves the race hash unchanged; pause, replay
                   and accelerated mode each tested separately for cue
                   synchronisation; every cue has a validated asset and is
                   fetchable over HTTP; phase9_on_hashes_identically
                   inside the 4 ms budget; a live panel at neutral changes
                   nothing, a slider move is queued and never injected, lands
                   on exactly the acknowledged tick, and a slider moved
                   mid-race replays bit-identically; bad slider commands are
                   refused with a reason
```

The simulation runs on its own thread (one authoritative physics owner — every
car is stepped sequentially inside the single 4 ms tick; no parallel physics
yet, per the plan) and communicates via channel events; nothing mutates the
race state from outside. Real-time pacing uses absolute deadlines (with a 1 ms
Windows timer period and hybrid sleep/spin), never a bare sleep-per-tick, and
never a dt change — overload shows up as measured deadline misses and start
lag instead.

## Determinism

Identical (seed, config, binary) ⇒ identical trajectory, verified by an
FNV-1a hash over every tick of every car's state (retirement, tire compound,
wear and pit phase included; legacy v2 logs reproduce the Phase 4 hash mix).
`replay` re-feeds the recorded 50 Hz per-car inputs and asserts the hash
matches the original run — for the whole field, formation and pit stops
included.

## Key numbers (defaults)

- dt = 4 ms (250 Hz provisional), driver at 50 Hz (every 5 ticks)
- Car: 720 kg, µ 1.6, 400 kW, CdA 1.3 m²
- Corner speed ≈ 28.8 m/s target, straight cap 75 m/s, observed top ≈ 55 m/s
- Lap ≈ 31.5 s settled on the 1102.65 m centreline; 20-car race laps ≈ 33.5 s
- Pit stop: 2.4 s box service, ~10–15 s total lane loss on a 5-lap race
- Compounds: Soft 1.03 grip / 8 laps, Medium 1.00 / 14, Hard 0.975 / 22
- Formation: five lights at 1 s + seeded 1.0–3.2 s hold ≈ 7.6 s to green
- 20-car field: 13 µs avg / ~100 µs worst per tick ≈ 2.6% of the 4 ms budget
