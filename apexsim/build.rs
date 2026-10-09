//! Embed a hash of the crate's own source in the binary.
//!
//! Why this exists: the engine is a pure function of `RaceConfig`, so the same
//! spec gives the same race — but the race is computed in `f64`, and neither
//! the compiler's instruction selection nor the platform's `libm` is
//! contractually identical across builds or hosts. Measured on this machine:
//! a 5-run batch is bit-identical run to run on one binary, and two binaries
//! built from *identical source* differ by 1–2 ULP in the leader's lap CV.
//!
//! Phase 10 stamps an engine id into every batch report and uses it to decide
//! whether two reports may be compared at all. A hand-maintained constant
//! like `p10-r1` cannot tell those two binaries apart, so a comparison across
//! them would confidently print `IDENTICAL`. Hashing the sources makes the
//! build identity mean what the comparison claims it means.

use std::hash::{Hash, Hasher};

fn main() {
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    collect(std::path::Path::new("src"), &mut files);
    files.sort();
    for f in &files {
        // Hash a normalised path, not the platform's own spelling of it.
        // `Path::to_string_lossy` yields "src\\main.rs" on Windows and
        // "src/main.rs" on Linux, which would give the same source two
        // different hashes on two different machines — and this hash exists
        // precisely so a report can be compared across machines.
        f.to_string_lossy().replace('\\', "/").hash(&mut hasher);
        match std::fs::read(f) {
            Ok(bytes) => bytes.hash(&mut hasher),
            Err(_) => continue,
        }
    }
    if let Ok(b) = std::fs::read("Cargo.toml") { b.hash(&mut hasher) }
    println!("cargo:rustc-env=F1_SIM_SOURCE_HASH={:016x}", hasher.finish());
}

fn collect(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(p);
        }
    }
}