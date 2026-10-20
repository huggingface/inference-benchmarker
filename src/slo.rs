//! Service level objectives: latency and error targets, and goodput.
use crate::requests::TextGenerationAggregatedResponse;
use serde::{Deserialize, Serialize};

/// Targets a deployment should meet. A request is *good* when it succeeded
/// and meets every latency target that is set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Slo {
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    /// Mean time between tokens of a request.
    #[serde(default)]
    pub itl_ms: Option<f64>,
    #[serde(default)]
    pub e2e_ms: Option<f64>,
    /// Share of requests that must be good, from 0 to 1.
    #[serde(default = "default_attainment")]
    pub attainment: f64,
    /// Highest acceptable share of failed requests, from 0 to 1.
    #[serde(default)]
    pub max_error_rate: Option<f64>,
}

fn default_attainment() -> f64 {
    0.95
}

impl Default for Slo {
    fn default() -> Self {
        Self {
            ttft_ms: None,
            itl_ms: None,
            e2e_ms: None,
            attainment: default_attainment(),
            max_error_rate: None,
        }
    }
}

/// How one benchmark step did against an [`Slo`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SloResult {
    pub good_requests: u64,
    /// Share of all requests (failed ones included) that were good.
    pub attainment: f64,
    /// Good requests per second.
    pub goodput_rps: f64,
    pub passed: bool,
}

impl Slo {
    pub fn validate(&self) -> anyhow::Result<()> {
        let in_unit = |v: f64| (0.0..=1.0).contains(&v);
        if !in_unit(self.attainment) || !self.max_error_rate.is_none_or(in_unit) {
            return Err(anyhow::anyhow!(
                "SLO attainment and max_error_rate must be between 0 and 1"
            ));
        }
        let targets = [self.ttft_ms, self.itl_ms, self.e2e_ms];
        if targets.iter().flatten().any(|t| *t <= 0.0) {
            return Err(anyhow::anyhow!("SLO latency targets must be positive"));
        }
        Ok(())
    }

    pub fn is_good(&self, response: &TextGenerationAggregatedResponse) -> bool {
        let within = |target: Option<f64>, value: Option<std::time::Duration>| match target {
            None => true,
            Some(ms) => value.is_some_and(|v| v.as_secs_f64() * 1000.0 <= ms),
        };
        !response.failed
            && within(self.ttft_ms, response.time_to_first_token())
            && within(self.itl_ms, response.inter_token_latency())
            && within(self.e2e_ms, response.e2e_latency())
    }

    /// Score a step from its good and total request counts.
    pub fn result(
        &self,
        good_requests: u64,
        total_requests: u64,
        error_rate: f64,
        duration_secs: f64,
    ) -> SloResult {
        let attainment = if total_requests == 0 {
            0.0
        } else {
            good_requests as f64 / total_requests as f64
        };
        SloResult {
            good_requests,
            attainment,
            goodput_rps: if duration_secs > 0.0 {
                good_requests as f64 / duration_secs
            } else {
                0.0
            },
            passed: total_requests > 0
                && attainment >= self.attainment
                && self.max_error_rate.is_none_or(|max| error_rate <= max),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_passes_only_within_targets() {
        let slo = Slo {
            attainment: 0.9,
            max_error_rate: Some(0.05),
            ..Default::default()
        };
        let ok = slo.result(95, 100, 0.01, 10.0);
        assert!(ok.passed);
        assert_eq!(ok.goodput_rps, 9.5);
        assert!(!slo.result(85, 100, 0.01, 10.0).passed);
        assert!(!slo.result(95, 100, 0.1, 10.0).passed);
        assert!(!slo.result(0, 0, 0.0, 10.0).passed);
    }

    #[test]
    fn validation_and_serde_defaults() {
        let slo: Slo = serde_json::from_str(r#"{"ttft_ms": 500}"#).unwrap();
        assert_eq!(slo.attainment, 0.95);
        assert!(slo.validate().is_ok());
        assert!(Slo {
            attainment: 1.5,
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(Slo {
            ttft_ms: Some(0.0),
            ..Default::default()
        }
        .validate()
        .is_err());
    }
}
