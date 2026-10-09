//! Small statistics helpers for the tick-cost gate report.

#[derive(Clone, Copy, Debug, Default)]
pub struct Summary {
    pub count: usize,
    pub avg_us: f64,
    pub p50_us: f64,
    pub p99_us: f64,
    pub p999_us: f64,
    pub worst_us: f64,
}

/// Percentile summary of tick costs in microseconds.
/// p99.9 uses the ceil index so it never under-reports the tail.
pub fn summarize_us(mut samples: Vec<f64>) -> Summary {
    if samples.is_empty() {
        return Summary::default();
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples.len();
    let pick = |q: f64| -> f64 {
        let idx = ((q * (n as f64 - 1.0)).ceil() as usize).min(n - 1);
        samples[idx]
    };
    let sum: f64 = samples.iter().sum();
    Summary {
        count: n,
        avg_us: sum / n as f64,
        p50_us: pick(0.50),
        p99_us: pick(0.99),
        p999_us: pick(0.999),
        worst_us: samples[n - 1],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_of_known_vector() {
        // 0..=99 -> p50 = 50, p99 = 99, p99.9 = 99, worst = 99
        let s = summarize_us((0..100).map(|i| i as f64).collect());
        assert_eq!(s.count, 100);
        assert!((s.p50_us - 50.0).abs() < 1e-9);
        assert!((s.p99_us - 99.0).abs() < 1e-9);
        assert!((s.p999_us - 99.0).abs() < 1e-9);
        assert!((s.avg_us - 49.5).abs() < 1e-9);
    }

    #[test]
    fn empty_is_zeroed() {
        let s = summarize_us(vec![]);
        assert_eq!(s.count, 0);
        assert_eq!(s.worst_us, 0.0);
    }
}
