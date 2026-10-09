//! Deterministic seeded RNG (64-bit LCG, PCG-style output shift).
//!
//! The engine must produce identical sequences for identical seeds on any
//! run, so it cannot lean on the OS entropy source. A plain LCG is more
//! than enough for driver-input noise; it is not a crypto primitive.

#[derive(Clone, Debug)]
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        // Scatter the seed so seeds 1, 2, 3 diverge immediately.
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xA0761D6478BD642F);
        // Discard the first outputs so the scramble settles.
        for _ in 0..4 {
            s = Self::step(s);
        }
        Lcg { state: s }
    }

    fn step(s: u64) -> u64 {
        s.wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
    }

    /// Raw next value.
    pub fn next_u64(&mut self) -> u64 {
        self.state = Self::step(self.state);
        self.state
    }

    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = Lcg::new(42);
        let mut b = Lcg::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge() {
        let mut a = Lcg::new(1);
        let mut b = Lcg::new(2);
        let differs = (0..10).any(|_| a.next_u64() != b.next_u64());
        assert!(differs);
    }
}
