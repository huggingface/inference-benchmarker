use crate::cost::{CostResult, GpuCost};
use crate::requests::ErrorKind;
use crate::results::{AccuracyResult, BenchmarkReport, BenchmarkResults};
use crate::slo::{Slo, SloResult};
use crate::{executors, BenchmarkConfig};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, System};
use tokio::fs;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PercentilesWriter {
    pub p50: f64,
    pub p60: f64,
    pub p70: f64,
    pub p80: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
    pub avg: f64,
}

impl PercentilesWriter {
    /// Milliseconds from a percentile function and an average; 0 when there is no data.
    fn from_durations(
        percentile: impl Fn(f64) -> anyhow::Result<Duration>,
        avg: anyhow::Result<Duration>,
    ) -> Self {
        let ms = |d: anyhow::Result<Duration>| {
            d.map(|d| d.as_micros() as f64 / 1000.).unwrap_or_default()
        };
        Self {
            p50: ms(percentile(0.5)),
            p60: ms(percentile(0.6)),
            p70: ms(percentile(0.7)),
            p80: ms(percentile(0.8)),
            p90: ms(percentile(0.9)),
            p95: ms(percentile(0.95)),
            p99: ms(percentile(0.99)),
            avg: ms(avg),
        }
    }
}

/// The results of one benchmark step.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkResultsWriter {
    pub id: String,
    pub executor_type: String,
    pub config: executors::ExecutorConfig,
    pub total_requests: u64,
    pub total_tokens: u64,
    pub token_throughput_secs: f64,
    pub duration_ms: u128,
    pub time_to_first_token_ms: PercentilesWriter,
    pub inter_token_latency_ms: PercentilesWriter,
    pub failed_requests: u64,
    pub successful_requests: u64,
    pub request_rate: f64,
    pub total_tokens_sent: u64,
    pub e2e_latency_ms: PercentilesWriter,
    #[serde(default)]
    pub prompt_tokens_avg: f64,
    #[serde(default)]
    pub output_tokens_avg: f64,
    #[serde(default)]
    pub error_rate: f64,
    #[serde(default)]
    pub timeout_rate: f64,
    #[serde(default)]
    pub errors_by_kind: BTreeMap<ErrorKind, u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<AccuracyResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slo: Option<SloResult>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<CostResult>,
}

impl BenchmarkResultsWriter {
    pub fn new(
        results: BenchmarkResults,
        slo: Option<&Slo>,
        gpu_cost: Option<&GpuCost>,
    ) -> anyhow::Result<BenchmarkResultsWriter> {
        let duration = results.duration().unwrap_or_default();
        Ok(BenchmarkResultsWriter {
            id: results.id.clone(),
            executor_type: results.executor_type().to_string(),
            config: results.executor_config(),
            total_requests: results.total_requests() as u64,
            total_tokens: results.total_tokens(),
            token_throughput_secs: results.token_throughput_secs().unwrap_or_default(),
            duration_ms: duration.as_micros() / 1000,
            time_to_first_token_ms: PercentilesWriter::from_durations(
                |p| results.time_to_first_token_percentile(p),
                results.time_to_first_token_avg(),
            ),
            inter_token_latency_ms: PercentilesWriter::from_durations(
                |p| results.inter_token_latency_percentile(p),
                results.inter_token_latency_avg(),
            ),
            failed_requests: results.failed_requests() as u64,
            successful_requests: results.successful_requests() as u64,
            request_rate: results.successful_request_rate().unwrap_or_default(),
            total_tokens_sent: results.total_tokens_sent(),
            e2e_latency_ms: PercentilesWriter::from_durations(
                |p| results.e2e_latency_percentile(p),
                results.e2e_latency_avg(),
            ),
            prompt_tokens_avg: results.prompt_tokens_avg().unwrap_or_default(),
            output_tokens_avg: results.output_tokens_avg(),
            error_rate: results.error_rate(),
            timeout_rate: results.timeout_rate(),
            errors_by_kind: results.errors_by_kind(),
            accuracy: results.accuracy(),
            slo: slo.map(|slo| results.slo(slo)),
            cost: gpu_cost.map(|c| {
                c.for_step(
                    duration,
                    results.successful_requests() as u64,
                    results.total_tokens(),
                )
            }),
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SystemInfo {
    pub cpu: Vec<String>,
    pub memory: String,
    pub os_name: String,
    pub os_version: String,
    pub kernel: String,
    pub hostname: String,
}

impl SystemInfo {
    pub fn new() -> SystemInfo {
        let s = System::new_with_specifics(
            sysinfo::RefreshKind::nothing()
                .with_memory(MemoryRefreshKind::everything())
                .with_cpu(CpuRefreshKind::everything()),
        );
        let cpu_info = s
            .cpus()
            .iter()
            .map(|cpu| format!("{} {}@{:.0}MHz", cpu.brand(), cpu.name(), cpu.frequency()))
            .collect::<Vec<String>>();
        SystemInfo {
            cpu: cpu_info,
            memory: format!(
                "{:.2} GB",
                s.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0
            ),
            os_name: System::name().unwrap_or_else(|| "N/A".into()),
            os_version: System::os_version().unwrap_or_else(|| "N/A".into()),
            kernel: System::kernel_version().unwrap_or_else(|| "N/A".into()),
            hostname: System::host_name().unwrap_or_else(|| "N/A".into()),
        }
    }
}

/// The whole report: what the CLI writes as JSON and what library callers get back.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkReportWriter {
    pub config: BenchmarkConfig,
    pub results: Vec<BenchmarkResultsWriter>,
    pub start_time: String,
    pub end_time: String,
    pub system: SystemInfo,
    #[serde(skip)]
    report: BenchmarkReport,
}

impl BenchmarkReportWriter {
    pub fn try_new(
        config: BenchmarkConfig,
        report: BenchmarkReport,
    ) -> anyhow::Result<BenchmarkReportWriter> {
        let mut results: Vec<BenchmarkResultsWriter> = Vec::new();
        for result in report.get_results() {
            let writer =
                BenchmarkResultsWriter::new(result, config.slo.as_ref(), config.gpu_cost.as_ref())?;
            results.push(writer);
        }
        Ok(BenchmarkReportWriter {
            config,
            results,
            start_time: report
                .start_time()
                .ok_or(anyhow::anyhow!("start_time not set"))?
                .to_rfc3339(),
            end_time: report
                .end_time()
                .ok_or(anyhow::anyhow!("end_time not set"))?
                .to_rfc3339(),
            system: SystemInfo::new(),
            report,
        })
    }
    pub async fn json(&self, path: &Path) -> anyhow::Result<()> {
        // write the benchmark report to json
        let report = serde_json::to_string(&self)?;

        // create path hierarchy if it doesn't exist
        if !path.exists() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).await?;
            }
        }
        fs::write(path, report).await?;
        Ok(())
    }

    /// Every request of every step, for callers that keep the raw responses.
    pub fn benchmark_report(&self) -> &BenchmarkReport {
        &self.report
    }

    #[cfg(feature = "cli")]
    pub async fn stdout(&self) -> anyhow::Result<()> {
        use crate::table;
        let param_table = table::parameters_table(self.config.clone())?;
        println!("\n{param_table}\n");
        let results_table = table::results_table(self.report.clone())?;
        println!("\n{results_table}\n");
        Ok(())
    }
}
