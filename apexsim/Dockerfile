# Headless tournament runner for Apex Sim.
#
# The plan makes containerization optional and conditional: "optional packaging
# /containerization after the local launcher works". The local launcher works
# and its round trip is checked by tests/phase10_gate.rs, so this image is the
# deployment half of Phase 10 — not a second way to run the engine.
#
# What this image is for: running comparable race batches somewhere that is not
# the laptop. The engine is a pure function of RaceConfig, so a container is a
# reasonable host as long as the spec pins the track version and the report
# records the engine build. Both are already in the artifact, which is why the
# image does not need to reimplement any of it.
#
# Deliberately NOT in this image:
#   - Ollama. Advised runs are non-reproducible by construction (Phase 7), and
#     pulling a multi-GB model would bloat an image whose whole value is being
#     cheap to schedule. For engine+ollama runs, point the host's Ollama at
#     this container over the network instead.
#   - The browser. A batch has no UI. The engine+dashboard profile runs on the
#     host if you want to watch it; inside a container there is no one.
#
# Build:
#   docker build -t apexsim-tournament .
# Run a batch, mounting the spec in and the results out:
#   docker run --rm -v "$PWD/spec.json:/work/spec.json:ro" \
#                    -v "$PWD/out:/work/out" \
#              apexsim-tournament --spec /work/spec.json --out /work/out
# Reproduce a batch somebody else ran:
#   docker run --rm -v "$PWD/other-run:/work/run:ro" \
#              apexsim-tournament --replay /work/run

# --- build stage ------------------------------------------------------------
# Multi-stage so the 400 MB+ of Rust toolchain never reaches the runner.
FROM rust:1.99-slim-bookworm AS build

WORKDIR /src

# axum pulls in ring and friends, which need a C compiler and headers.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential pkg-config \
 && rm -rf /var/lib/apt/lists/*

# static/ must be in the build context: the crate embeds the dashboard HTML,
# the 22 car sprites and the 13 sound cues with include_str!/include_bytes!, so
# a build without it fails with "No such file or directory" rather than
# producing a smaller image. ~950 KB, which is the right trade.
COPY static ./static

# Copy the manifests first and build a stub, so a source-only change does not
# re-download and recompile the dependency tree.
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src \
 && echo 'fn main() {}' > src/main.rs \
 && echo '' > src/lib.rs \
 && cargo build --release \
 && rm -rf src

# build.rs computes the source hash that becomes the engine id stamped into
# every batch report. Without it the crate does not compile.
COPY build.rs ./build.rs
COPY src ./src
# Touch so cargo does not consider main.rs older than the cached stub.
RUN touch src/main.rs src/lib.rs && cargo build --release --locked

# --- runner stage ----------------------------------------------------------
FROM debian:bookworm-slim AS runner

# The engine reads nothing outside its working directory and needs no network,
# so the runtime image stays at the base plus a CA bundle. Run as a non-root
# user: an image that runs batches on someone else's machine should not run
# them as root there.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --shell /usr/sbin/nologin apexsim

WORKDIR /work
COPY --from=build /src/target/release/apexsim /usr/local/bin/apexsim
RUN chown -R apexsim:apexsim /work
USER apexsim

# Report what built the batch. The engine already stamps this into every report;
# failing fast here just makes a mistyped image obvious.
RUN apexsim tournament --write-spec /work/example-spec.json \
 && rm -f /work/example-spec.json

# Default to a short serial batch so a bare `docker run` does something useful
# instead of hanging. Override with your own args.
ENTRYPOINT ["apexsim", "tournament"]
CMD ["--seeds", "1,2,3,4,5", "--cars", "10", "--laps", "4", "--out", "/work/out"]

# Note: `tournament` is CPU-bound and single-batch-per-process. To use several
# cores, pass --threads N (measured on the 8C/16T dev box: ~4.6x at 8 workers,
# results bit-identical) or run several containers against the same core
# budget. Do not oversubscribe: each race is CPU-bound with no I/O wait, so
# threads beyond the physical core count only add contention.