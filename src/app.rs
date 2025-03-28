use crate::benchmark::Event as BenchmarkEvent;
use crate::event::{terminal_event_task, AppEvent};
use crate::flux::{Action, AppState, Dispatcher, Store};
use crate::scheduler::ExecutorType;
use crate::BenchmarkConfig;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::text::Span;
use ratatui::widgets::ListDirection::BottomToTop;
use ratatui::widgets::{Cell, Dataset, List, ListItem, Row, Table};
use ratatui::{
    buffer::Buffer,
    layout::{Alignment, Rect},
    style::Stylize as OtherStylize,
    symbols,
    symbols::border,
    text::{Line, Text},
    widgets::{Block, Paragraph, Widget},
    DefaultTerminal, Frame,
};
use std::collections::HashMap;
use std::io;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast::Sender;
use tokio::sync::mpsc::{Receiver, UnboundedReceiver};
use tokio::sync::{broadcast, mpsc};

pub struct App {
    exit: bool,
    store: Arc<Mutex<Store>>,
    dispatcher: Arc<Mutex<Dispatcher>>,
    receiver: Receiver<AppEvent>,
    benchmark_config: BenchmarkConfig,
    stop_sender: broadcast::Sender<()>,
}

pub async fn run_console(
    benchmark_config: BenchmarkConfig,
    mut receiver: UnboundedReceiver<BenchmarkEvent>,
    stop_sender: broadcast::Sender<()>,
) {
    let (app_tx, app_rx) = mpsc::channel(8);
    // Create event task
    let stop_receiver_signal = stop_sender.subscribe();
    tokio::spawn(terminal_event_task(250, app_tx, stop_receiver_signal));

    let mut app = App::new(benchmark_config, app_rx, stop_sender.clone());
    app.dispatcher
        .lock()
        .expect("lock")
        .dispatch(Action::LogMessage(LogMessageUI {
            message: "Starting benchmark".to_string(),
            level: LogLevel::Info,
            timestamp: chrono::Utc::now(),
        }));
    let dispatcher = app.dispatcher.clone();
    let mut stop_receiver_signal = stop_sender.subscribe();
    let event_thread = tokio::spawn(async move {
        tokio::select! {
            _=async{
                while let Some(event) = receiver.recv().await {
                    match event {
                        BenchmarkEvent::BenchmarkStart(event) => {
                            dispatcher.lock().expect("lock").dispatch(Action::AddBenchmark(BenchmarkUI {
                                id: event.id,
                                status: BenchmarkStatus::Running,
                                progress: 0.0,
                                throughput: "0".to_string(),
                                successful_requests: 0,
                                failed_requests: 0,
                            }));
                        }
                        BenchmarkEvent::BenchmarkProgress(event) => {
                            let (successful_requests,failed_requests) = (event.successful_requests,event.failed_requests);
                            dispatcher.lock().expect("lock").dispatch(Action::AddBenchmark(BenchmarkUI {
                                id: event.id,
                                status: BenchmarkStatus::Running,
                                progress: event.progress,
                                throughput: event.request_throughput.map_or("0".to_string(), |e| format!("{e:.2}")),
                                successful_requests,
                                failed_requests,
                            }));
                        }
                        BenchmarkEvent::BenchmarkEnd(event) => {
                            dispatcher.lock().expect("lock").dispatch(Action::LogMessage(LogMessageUI {
                                message: format!("Benchmark {} ended", event.id),
                                level: LogLevel::Info,
                                timestamp: chrono::Utc::now(),
                            }));
                            if let Some(results) = event.results {
                                let (successful_requests,failed_requests) = (results.successful_requests() as u64,results.failed_requests() as u64);
                                dispatcher.lock().expect("lock").dispatch(Action::AddBenchmark(BenchmarkUI {
                                    id: event.id,
                                    status: BenchmarkStatus::Completed,
                                    progress: 100.0,
                                    throughput: event.request_throughput.map_or("0".to_string(), |e| format!("{e:.2}")),
                                    successful_requests,
                                    failed_requests,
                                }));
                                dispatcher.lock().expect("lock").dispatch(Action::AddBenchmarkResults(results));
                            }
                        }
                        BenchmarkEvent::Message(event) => {
                            dispatcher.lock().expect("lock").dispatch(Action::LogMessage(LogMessageUI {
                                message: event.message,
                                level: LogLevel::Info,
                                timestamp: event.timestamp,
                            }));
                        }
                        BenchmarkEvent::BenchmarkReportEnd(path) => {
                            dispatcher.lock().expect("lock").dispatch(Action::LogMessage(LogMessageUI {
                                message: format!("Benchmark report saved to {}", path),
                                level: LogLevel::Info,
                                timestamp: chrono::Utc::now(),
                            }));
                            break;
                        }
                        BenchmarkEvent::BenchmarkError(event) => {
                            dispatcher.lock().expect("lock").dispatch(Action::LogMessage(LogMessageUI {
                                message: format!("Error running benchmark: {:?}", event),
                                level: LogLevel::Error,
                                timestamp: chrono::Utc::now(),
                            }));
                            break;
                        }
                    }
                }
            }=>{}
            _ = stop_receiver_signal.recv() => {}
        }
    });
    let mut stop_receiver_signal = stop_sender.subscribe();
    let app_thread = tokio::spawn(async move {
        tokio::select! {
            _ = async {
                let _ = app.run(&mut ratatui::init()).await;
                ratatui::restore();
            }=>{}
            _ = stop_receiver_signal.recv() => {}
        }
    });
    let _ = event_thread.await;
    let _ = app_thread.await;
}

impl App {
    pub fn new(
        benchmark_config: BenchmarkConfig,
        receiver: Receiver<AppEvent>,
        stop_sender: Sender<()>,
    ) -> App {
        let store = Arc::from(Mutex::new(Store::new()));
        let dispatcher = Arc::from(Mutex::new(Dispatcher::new(store.clone())));
        App {
            exit: false,
            store: store.clone(),
            dispatcher: dispatcher.clone(),
            receiver,
            benchmark_config,
            stop_sender,
        }
    }
    pub async fn run(&mut self, terminal: &mut DefaultTerminal) -> io::Result<()> {
        while !self.exit {
            terminal.draw(|frame| self.draw(frame))?;
            self.handle_events().await?;
        }
        // signal everybody to stop
        let _ = self.stop_sender.send(());
        Ok(())
    }
    fn draw(&self, frame: &mut Frame) {
        frame.render_widget(self, frame.area())
    }
    async fn handle_events(&mut self) -> io::Result<()> {
        match self.receiver.recv().await {
            None => Err(io::Error::new(io::ErrorKind::Other, "No event")),
            Some(event) => match event {
                AppEvent::Tick => Ok(()),
                AppEvent::Key(key_event) => self.handle_key_event(key_event),
                AppEvent::Resize => Ok(()),
            },
        }
    }

    fn handle_key_event(&mut self, key_event: KeyEvent) -> io::Result<()> {
        match key_event {
            KeyEvent {
                code: KeyCode::Char('q'),
                ..
            } => self.exit(),
            KeyEvent {
                code: KeyCode::Char('c'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => self.exit(),
            _ => {}
        }
        Ok(())
    }
    fn exit(&mut self) {
        self.exit = true;
    }

    fn create_datasets(&self, state: AppState) -> HashMap<String, Vec<(f64, f64)>> {
        let token_throughput_rate = state
            .results
            .iter()
            .filter_map(|r| match r.executor_type() {
                ExecutorType::ConstantArrivalRate => {
                    let throughput = r.token_throughput_secs().unwrap_or(0.0);
                    Some((r.executor_config().rate.unwrap(), throughput))
                }
                ExecutorType::ConstantVUs => None,
            })
            .collect::<Vec<_>>();
        let token_throughput_vus = state
            .results
            .iter()
            .filter_map(|r| match r.executor_type() {
                ExecutorType::ConstantVUs => {
                    let throughput = r.token_throughput_secs().unwrap_or(0.0);
                    Some((r.executor_config().max_vus as f64, throughput))
                }
                ExecutorType::ConstantArrivalRate => None,
            })
            .collect::<Vec<_>>();
        let inter_token_latency_rate = state
            .results
            .iter()
            .filter_map(|r| match r.executor_type() {
                ExecutorType::ConstantArrivalRate => {
                    let latency = r
                        .inter_token_latency_avg()
                        .unwrap_or_default()
                        .as_secs_f64();
                    Some((r.executor_config().rate.unwrap(), latency))
                }
                ExecutorType::ConstantVUs => None,
            })
            .collect::<Vec<_>>();
        let inter_token_latency_vus = state
            .results
            .iter()
            .filter_map(|r| match r.executor_type() {
                ExecutorType::ConstantVUs => {
                    let latency = r
                        .inter_token_latency_avg()
                        .unwrap_or_default()
                        .as_secs_f64();
                    Some((r.executor_config().max_vus as f64, latency))
                }
                ExecutorType::ConstantArrivalRate => None,
            })
            .collect::<Vec<_>>();
        HashMap::from([
            ("token_throughput_rate".to_string(), token_throughput_rate),
            ("token_throughput_vus".to_string(), token_throughput_vus),
            (
                "inter_token_latency_rate".to_string(),
                inter_token_latency_rate,
            ),
            (
                "inter_token_latency_vus".to_string(),
                inter_token_latency_vus,
            ),
        ])
    }
}

impl Widget for &App {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let state = self.store.lock().unwrap().get_state();
        let data = self.create_datasets(state.clone());

        let main_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(20)])
            .split(area);
        let bottom_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(main_layout[1]);
        let steps_graph_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(35), Constraint::Percentage(65)])
            .split(bottom_layout[0]);
        // LOGS
        let logs_title = Line::from("Logs".bold()).centered();
        let logs_block = Block::bordered()
            .title_top(logs_title)
            .border_set(border::THICK);
        List::new(
            state
                .messages
                .iter()
                .rev()
                .map(|m| {
                    let level_span = match m.level {
                        LogLevel::Info => {
                            Span::raw(m.level.to_string().to_uppercase()).green().bold()
                        }
                        LogLevel::Warning => Span::raw(m.level.to_string().to_uppercase())
                            .yellow()
                            .bold(),
                        LogLevel::Error => {
                            Span::raw(m.level.to_string().to_uppercase()).red().bold()
                        }
                    };
                    let content = Line::from(vec![
                        m.formatted_timestamp().clone().gray(),
                        Span::raw(" "),
                        level_span,
                        Span::raw(" "),
                        Span::raw(m.message.to_string()).bold(),
                    ]);
                    ListItem::new(content)
                })
                .collect::<Vec<_>>(),
        )
        .direction(BottomToTop)
        .block(logs_block)
        .render(bottom_layout[1], buf);

        // BENCHMARK config
        let rate_mode = match self.benchmark_config.rates {
            None => "Automatic".to_string(),
            Some(_) => "Manual".to_string(),
        };
        let config_text = Text::from(vec![Line::from(vec![
            format!("Profile: {profile} | Benchmark: {kind} | Max VUs: {max_vus} | Duration: {duration} sec | Rates: {rates} | Warmup: {warmup} sec",
                    profile = self.benchmark_config.profile.clone().unwrap_or("N/A".to_string()),
                    kind = self.benchmark_config.benchmark_kind,
                    max_vus = self.benchmark_config.max_vus,
                    duration = self.benchmark_config.duration.as_secs_f64(),
                    rates = rate_mode,
                    warmup = self.benchmark_config.warmup_duration.as_secs_f64()).white().bold(),
        ])]);
        Paragraph::new(config_text.clone()).render(main_layout[0], buf);

        // STEPS
        let steps_block_title = Line::from("Benchmark steps".bold()).centered();
        let steps_block = Block::bordered()
            .title(steps_block_title.alignment(Alignment::Center))
            .border_set(border::THICK);
        let step_rows = state
            .benchmarks
            .iter()
            .map(|b| {
                let error_rate = if b.failed_requests > 0 {
                    format!(
                        "{:4.0}%",
                        b.failed_requests as f64
                            / (b.failed_requests + b.successful_requests) as f64
                            * 100.
                    )
                    .light_red()
                    .bold()
                } else {
                    format!("{:4.0}%", 0).to_string().white()
                };
                let cells = vec![
                    b.id.clone().white(),
                    b.status.to_string().white(),
                    format!("{:4.0}%", b.progress).white(),
                    error_rate,
                    format!("{:>6.6} req/sec avg", b.throughput).green().bold(),
                ];
                Row::new(cells)
            })
            .collect::<Vec<_>>();
        let widths = [
            Constraint::Length(30),
            Constraint::Length(10),
            Constraint::Length(5),
            Constraint::Length(5),
            Constraint::Length(20),
        ];
        // steps table
        Table::new(step_rows, widths)
            .header(Row::new(vec![
                Cell::new(Line::from("Bench").alignment(Alignment::Left)),
                Cell::new(Line::from("Status").alignment(Alignment::Left)),
                Cell::new(Line::from("%").alignment(Alignment::Left)),
                Cell::new(Line::from("Err").alignment(Alignment::Left)),
                Cell::new(Line::from("Throughput").alignment(Alignment::Left)),
            ]))
            .block(steps_block)
            .render(steps_graph_layout[0], buf);

        // CHARTS
        let graphs_block_title = Line::from("Token throughput rate".bold()).centered();
        let graphs_block = Block::bordered()
            .title(graphs_block_title.alignment(Alignment::Center))
            .border_set(border::THICK);
        let binding = data.get("token_throughput_rate").unwrap().clone();
        let datasets = vec![Dataset::default()
            .name("Token throughput rate".to_string())
            .marker(symbols::Marker::Dot)
            .graph_type(ratatui::widgets::GraphType::Scatter)
            .style(ratatui::style::Style::default().fg(ratatui::style::Color::LightMagenta))
            .data(&binding)];
        let (xmax, ymax) = get_max_bounds(&binding, (10.0, 100.0));
        let x_axis = ratatui::widgets::Axis::default()
            .title("Arrival rate (req/s)".to_string())
            .style(ratatui::style::Style::default().white())
            .bounds([0.0, xmax])
            .labels(get_axis_labels(0.0, xmax, 5));
        let y_axis = ratatui::widgets::Axis::default()
            .title("Throughput (tokens/s)".to_string())
            .style(ratatui::style::Style::default().white())
            .bounds([0.0, ymax])
            .labels(get_axis_labels(0.0, ymax, 5));
        ratatui::widgets::Chart::new(datasets)
            .x_axis(x_axis)
            .y_axis(y_axis)
            .block(graphs_block)
            .legend_position(None)
            .render(steps_graph_layout[1], buf);
    }
}

fn get_max_bounds(data: &[(f64, f64)], default_max: (f64, f64)) -> (f64, f64) {
    let xmax = data
        .iter()
        .map(|(x, _)| x)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&default_max.0);
    let ymax = data
        .iter()
        .map(|(_, y)| y)
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&default_max.1);
    (*xmax, *ymax)
}

fn get_axis_labels(min: f64, max: f64, num_labels: u32) -> Vec<String> {
    let step = (max - min) / num_labels as f64;
    (0..num_labels)
        .map(|i| format!("{:.2}", min + i as f64 * step))
        .collect()
}

#[allow(dead_code)]
#[derive(Clone, strum_macros::Display)]
enum LogLevel {
    Info,
    Warning,
    Error,
}

#[derive(Clone)]
pub(crate) struct LogMessageUI {
    message: String,
    level: LogLevel,
    timestamp: chrono::DateTime<chrono::Utc>,
}

impl LogMessageUI {
    fn formatted_timestamp(&self) -> String {
        self.timestamp.to_rfc3339()
    }
}

#[derive(Clone)]
pub(crate) struct BenchmarkUI {
    pub(crate) id: String,
    status: BenchmarkStatus,
    progress: f64,
    throughput: String,
    successful_requests: u64,
    failed_requests: u64,
}

#[derive(Clone, strum_macros::Display)]
enum BenchmarkStatus {
    Running,
    Completed,
}
