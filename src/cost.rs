//! What a benchmark step cost, from the price of the GPUs serving it.
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Price of the hardware under test. The deployment is assumed to be
/// dedicated to the benchmark while it runs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuCost {
    pub usd_per_gpu_hour: f64,
    pub gpus: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CostResult {
    pub usd_total: f64,
    #[serde(default)]
    pub usd_per_request: Option<f64>,
    #[serde(default)]
    pub usd_per_1m_output_tokens: Option<f64>,
}

impl GpuCost {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.usd_per_gpu_hour < 0.0 || self.gpus == 0 {
            return Err(anyhow::anyhow!(
                "GPU cost needs a non-negative price and at least one GPU"
            ));
        }
        Ok(())
    }

    pub fn for_step(&self, duration: Duration, requests: u64, output_tokens: u64) -> CostResult {
        let usd_total = self.usd_per_gpu_hour * self.gpus as f64 * duration.as_secs_f64() / 3600.0;
        let per = |n: u64, scale: f64| (n > 0).then(|| usd_total * scale / n as f64);
        CostResult {
            usd_total,
            usd_per_request: per(requests, 1.0),
            usd_per_1m_output_tokens: per(output_tokens, 1_000_000.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_of_a_step() {
        let cost = GpuCost {
            usd_per_gpu_hour: 2.0,
            gpus: 2,
        };
        let r = cost.for_step(Duration::from_secs(1800), 100, 500_000);
        assert_eq!(r.usd_total, 2.0);
        assert_eq!(r.usd_per_request, Some(0.02));
        assert_eq!(r.usd_per_1m_output_tokens, Some(4.0));
        assert_eq!(
            cost.for_step(Duration::from_secs(60), 0, 0).usd_per_request,
            None
        );
    }
}
