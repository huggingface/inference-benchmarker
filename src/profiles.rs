use std::string::ToString;
use crate::{RunConfiguration, TokenizeOptions};

pub fn apply_profile(profile: &str, run_configuration: RunConfiguration) -> anyhow::Result<RunConfiguration> {
    match profile {
        "fixed-length"=>{
            Ok(RunConfiguration {
                max_vus: 128,
                duration: std::time::Duration::from_secs(120),
                rates: None,
                num_rates: 10,
                benchmark_kind: "sweep".to_string(),
                warmup_duration: std::time::Duration::from_secs(30),
                prompt_options: Some(TokenizeOptions {
                    num_tokens: Some(200),
                    min_tokens: 200,
                    max_tokens: 200,
                    variance: 0,
                }),
                decode_options: Some(TokenizeOptions {
                    num_tokens: Some(800),
                    min_tokens: 50,
                    max_tokens: 800,
                    variance: 100,
                }),
                dataset: "hlarcher/inference-benchmarker".to_string(),
                dataset_file: "share_gpt_0_turns.json".to_string(),
                ..run_configuration
            })
        }
        "chat" => {
            Ok(RunConfiguration {
                max_vus: 128,
                duration: std::time::Duration::from_secs(120),
                rates: None,
                num_rates: 10,
                benchmark_kind: "sweep".to_string(),
                warmup_duration: std::time::Duration::from_secs(30),
                prompt_options: None, // use prompts from dataset
                decode_options: Some(TokenizeOptions {
                    num_tokens: Some(800), // decode up to 800 tokens
                    min_tokens: 50,
                    max_tokens: 800,
                    variance: 100,
                }),
                dataset: "hlarcher/inference-benchmarker".to_string(),
                dataset_file: "share_gpt_turns.json".to_string(),
                ..run_configuration
            })
        },
        "code-generation"=>{
            Ok(RunConfiguration {
                max_vus: 128,
                duration: std::time::Duration::from_secs(120),
                rates: None,
                num_rates: 10,
                benchmark_kind: "throughput".to_string(),
                warmup_duration: std::time::Duration::from_secs(30),
                prompt_options: Some(TokenizeOptions {
                    num_tokens: Some(4096),
                    min_tokens: 3000,
                    max_tokens: 6000,
                    variance: 1000,
                }),
                decode_options: Some(TokenizeOptions {
                    num_tokens: Some(50),
                    min_tokens: 30,
                    max_tokens: 80,
                    variance: 10,
                }),
                dataset: "hlarcher/inference-benchmarker".to_string(),
                dataset_file: "github_code.json".to_string(),
                ..run_configuration
            })
        }
        _ => {
            Err(anyhow::anyhow!("Unknown profile: {}", profile))
        }
    }
}
