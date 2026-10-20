//! The command-line tool: profiles, Hub downloads, terminal UI, result files.
use crate::api::{execute, Cancelled, RunOptions};
use crate::app::run_console;
use crate::benchmark::{BenchmarkConfig, BenchmarkKind, Event, MessageEvent};
use crate::hub::{self, RepoType};
use crate::profiles::apply_profile;
use crate::requests::{
    ConversationTextRequestGenerator, OpenAITextGenerationBackend, TokenizeOptions,
};
use chrono::Local;
use crossterm::ExecutableCommand;
use log::{debug, error, info, warn, Level, LevelFilter};
use reqwest::Url;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::broadcast::Sender;
use tokio::sync::Mutex;

pub struct RunConfiguration {
    pub url: Url,
    pub api_key: String,
    pub tokenizer_name: String,
    pub profile: Option<String>,
    pub max_vus: u64,
    pub duration: std::time::Duration,
    pub rates: Option<Vec<f64>>,
    pub num_rates: u64,
    pub benchmark_kind: String,
    pub warmup_duration: std::time::Duration,
    pub interactive: bool,
    pub prompt_options: Option<TokenizeOptions>,
    pub decode_options: Option<TokenizeOptions>,
    pub dataset: String,
    pub dataset_file: String,
    pub hf_token: Option<String>,
    pub extra_metadata: Option<HashMap<String, String>>,
    pub model_name: String,
    pub run_id: String,
}

pub async fn run(mut run_config: RunConfiguration, stop_sender: Sender<()>) -> anyhow::Result<()> {
    info!("Starting benchmark");
    // apply profile if needed
    run_config = match run_config.profile.clone() {
        None => run_config,
        Some(profile) => match apply_profile(profile.as_str(), run_config) {
            Ok(config) => {
                info!("Profile applied: {}", profile);
                config
            }
            Err(e) => {
                error!("Failed to apply profile: {:?}", e);
                return Err(e);
            }
        },
    };
    let cache_dir = hub::default_cache_dir();
    let tokenizer = hub::load_tokenizer(
        &run_config.tokenizer_name,
        run_config.hf_token.as_deref(),
        Some(&cache_dir),
    )
    .await?;
    let tokenizer = Arc::new(tokenizer);

    let config = BenchmarkConfig {
        max_vus: run_config.max_vus,
        duration: run_config.duration,
        benchmark_kind: match run_config.benchmark_kind.to_lowercase().as_str() {
            "throughput" => BenchmarkKind::Throughput,
            "sweep" => BenchmarkKind::Sweep,
            "rate" => BenchmarkKind::Rate,
            other => return Err(anyhow::anyhow!("Unknown benchmark kind: {other}")),
        },
        warmup_duration: run_config.warmup_duration,
        rates: run_config.rates,
        num_rates: run_config.num_rates,
        prompt_options: run_config.prompt_options.clone(),
        decode_options: run_config.decode_options.clone(),
        tokenizer: run_config.tokenizer_name.clone(),
        model_name: run_config.model_name.clone(),
        profile: run_config.profile.clone(),
        extra_metadata: run_config.extra_metadata.clone(),
        run_id: run_config.run_id.clone(),
        ..Default::default()
    };
    config.validate()?;
    let backend = OpenAITextGenerationBackend::try_new(
        run_config.api_key,
        run_config.url,
        run_config.model_name.clone(),
        tokenizer.clone(),
        config.request_timeout(),
    )?;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    if run_config.interactive {
        // send logs to file
        let target = Box::new(File::create("log.txt")?);
        env_logger::Builder::new()
            .target(env_logger::Target::Pipe(target))
            .filter(Some("inference_benchmarker"), LevelFilter::Debug)
            .format(|buf, record| {
                writeln!(
                    buf,
                    "[{} {} {}:{}] {}",
                    Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
                    record.level(),
                    record.file().unwrap_or("unknown"),
                    record.line().unwrap_or(0),
                    record.args()
                )
            })
            .try_init()?;
    } else {
        env_logger::try_init()?;
    }
    let config_clone = config.clone();
    let mut stop_receiver = stop_sender.subscribe();
    let stop_sender_clone = stop_sender.clone();
    let ui_thread = tokio::spawn(async move {
        tokio::select! {
            _ = stop_receiver.recv() => {
                debug!("Received stop signal, stopping benchmark");
            }
            _ = async{
                if run_config.interactive {
                    run_console(config_clone, rx, stop_sender_clone).await;
                } else {
                    // consume the channel to avoid closed channel error
                    while rx.recv().await.is_some() {}
                }
            } => {}
        }
    });

    // download prompts dataset
    info!("Downloading dataset");
    let _ = tx.send(Event::Message(MessageEvent {
        message: "Downloading dataset".to_string(),
        timestamp: chrono::Utc::now(),
        level: Level::Info,
    }));
    let filepath = hub::download_cached(
        RepoType::Dataset,
        &run_config.dataset,
        &run_config.dataset_file,
        run_config.hf_token.as_deref(),
        &cache_dir,
    )
    .await?;
    let prompt_options = run_config.prompt_options;
    let decode_options = run_config.decode_options;
    let requests = tokio::task::spawn_blocking(move || {
        ConversationTextRequestGenerator::load(filepath, tokenizer, prompt_options, decode_options)
    })
    .await??;

    let result = execute(
        config.clone(),
        Box::new(backend),
        Arc::new(Mutex::new(requests)),
        Vec::new(),
        RunOptions {
            events: Some(tx.clone()),
            records: None,
        },
        stop_sender.clone(),
    )
    .await;
    let writer = match result {
        Ok(writer) => {
            let path = format!(
                "results/{}_{}.json",
                run_config
                    .tokenizer_name
                    .replace("/", "_")
                    .replace(".", "_"),
                chrono::Utc::now().format("%Y-%m-%d-%H-%M-%S")
            );
            let path = Path::new(&path);
            writer.json(path).await?;
            info!("Report saved to {:?}", path);
            let _ = tx.send(Event::BenchmarkReportEnd(format!("{:?}", path)));
            Some(writer)
        }
        Err(e) => match e.downcast::<Cancelled>() {
            Ok(cancelled) => {
                debug!("Received stop signal, stopping benchmark");
                cancelled.partial.map(|w| *w)
            }
            Err(e) => {
                error!("Error running benchmark: {:?}", e.to_string());
                let _ = tx.send(Event::BenchmarkError(e.to_string()));
                None
            }
        },
    };
    info!("Benchmark finished");
    if !run_config.interactive {
        // quit app if not interactive
        let _ = stop_sender.send(());
    }
    ui_thread.await?;

    if run_config.interactive {
        // Revert terminal to original view
        io::stdout().execute(ratatui::crossterm::terminal::LeaveAlternateScreen)?;
        ratatui::crossterm::terminal::disable_raw_mode()?;
        io::stdout().execute(ratatui::crossterm::cursor::Show)?;
    }

    match writer {
        Some(writer) => writer.stdout().await?,
        None => warn!("No results to report."),
    }

    Ok(())
}
