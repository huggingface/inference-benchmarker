import os
from cProfile import label
from contextlib import ExitStack
from dataclasses import dataclass

import gradio as gr
import pandas as pd


@dataclass
class PlotConfig:
    x_title: str
    y_title: str
    title: str


css = '''
.summary span {
    font-size: 10px;
    padding-top:0;
    padding-bottom:0;
}
'''

summary_desc = '''
## Summary
This table shows the average of the metrics for each model and QPS rate.

Benchmark are run with:
- Prompts: 200±10 tokens length (normal distribution)
- Generation: 200±10 tokens length (normal distribution)
- 120s duration 

Each benchmark is run using a constant arrival rate of requests per second (QPS), 
independently of the number of requests that are being processed (open loop).

The metrics are:
* Inter token latency: Time to generate a new output token for each user querying the system. 
  It translates as the “speed” perceived by the end-user. We aim for at least 300 words per minute (average reading speed), so ITL<150ms
* Time to First Token: Time the user has to wait before seeing the first token of its answer. 
  Lower waiting time are essential for real-time interactions, less so for offline workloads.
* End-to-end latency: The overall time the system took to generate the full response to the user.
* Throughput: The number of tokens per second the system can generate across all requests
* Successful requests: The number of requests the system was able to honor in the benchmark timeframe
* Error rate: The percentage of requests that ended up in error, as the system could not process them in time or failed to process them. 
  
⚠️ TGI has a rate-limiting mechanism that will throttle requests, so a high error rate can be a sign of rate limit hit.

'''

df_bench = pd.DataFrame()
df_ci = pd.DataFrame()
summary = pd.DataFrame()
line_plots = []
column_mappings = {'inter_token_latency_ms_p90': 'ITL P90 (ms)', 'time_to_first_token_ms_p90': 'TTFT P90 (ms)',
                   'e2e_latency_ms_p90': 'E2E P90 (ms)', 'token_throughput_secs': 'Throughput (tokens/s)',
                   'successful_requests': 'Successful requests', 'error_rate': 'Error rate (%)', 'model': 'Model',
                   'rate': 'QPS'}


def plot(model, device) -> pd.DataFrame:
    d = df_bench[(df_bench['model'] == model) & (df_bench['device'] == device)]
    return d


def update_app(device_bench, device_ci, model, commit_ref, commit_compare):
    res = []
    for plot in line_plots:
        res.append(df_bench[(df_bench['model'] == model) & (df_bench['device'] == device_bench)])
    return res + [summary_table(device_bench), compare_table(device_ci, commit_ref, commit_compare)]


def summary_table(device) -> pd.DataFrame:
    rates = [4., 8., 16.]
    data = df_bench[(df_bench['device'] == device) & (df_bench['rate'].isin(rates))]
    data = data.groupby(['model', 'rate', 'engine']).agg(
        {'inter_token_latency_ms_p90': 'mean', 'time_to_first_token_ms_p90': 'mean',
         'e2e_latency_ms_p90': 'mean', 'token_throughput_secs': 'mean',
         'successful_requests': 'mean', 'error_rate': 'mean'}).reset_index()
    data = data[
        ['model', 'engine', 'rate', 'inter_token_latency_ms_p90', 'time_to_first_token_ms_p90', 'e2e_latency_ms_p90',
         'token_throughput_secs']]
    for metric in ['inter_token_latency_ms_p90', 'time_to_first_token_ms_p90', 'e2e_latency_ms_p90',
                   'token_throughput_secs']:
        data[metric] = data[metric].apply(lambda x: f"{x:.2f}")
    data = data.rename(
        columns=column_mappings)
    return data


def compare_table(device, commit_ref, commit_compare) -> pd.DataFrame:
    rates = [4., 8., 16.]
    metrics = ['inter_token_latency_ms_p90', 'time_to_first_token_ms_p90', 'e2e_latency_ms_p90',
               'token_throughput_secs']
    data = df_ci[(df_ci['device'] == device) & (df_ci['rate'].isin(rates))]
    ref = data[(data['device'] == device) & (data['version'] == commit_ref) & (data['engine'] == 'TGI')]
    compare = data[(data['device'] == device) & (data['version'] == commit_compare) & (data['engine'] == 'TGI')]
    data = ref.merge(compare, on=['model', 'rate'], suffixes=('_ref', '_compare'))
    data = data.rename(
        columns=column_mappings)
    for metric in metrics:
        name = column_mappings[metric]
        data[f'∆ {name}'] = (data[f'{metric}_compare'] - data[f'{metric}_ref']) / data[f'{metric}_ref'] * 100.0
        data[f'∆ {name}'] = data[f'∆ {name}'].apply(lambda x: f"{x:.2f}%")
    data = data[['Model', 'QPS'] + [f'∆ {column_mappings[metric]}' for metric in metrics]]

    return data


def load_bench_results(source) -> pd.DataFrame:
    data = pd.read_parquet(source)
    # remove warmup and throughput
    data = data[(data['id'] != 'warmup') & (data['id'] != 'throughput')]
    # only keep constant rate
    data = data[data['executor_type'] == 'ConstantArrivalRate']
    # sanity check: we should have only one version per engine
    # assert data.groupby(['engine', 'version']).size().reset_index().shape[0] == 2
    return data


def load_ci_results(source) -> pd.DataFrame:
    data = pd.read_parquet(source)
    return data


def select_region(selection: gr.SelectData, device, model):
    min_w, max_w = selection.index
    data = df_bench[(df_bench['model'] == model) & (df_bench['device'] == device) & (df_bench['rate'] >= min_w) & (
            df_bench['rate'] <= max_w)]
    res = []
    for plot in line_plots:
        # find the y values for the selected region
        metric = plot["metric"]
        y_min = data[metric].min()
        y_max = data[metric].max()
        res.append(gr.LinePlot(x_lim=[min_w, max_w], y_lim=[y_min, y_max]))
    return res


def reset_region():
    res = []
    for _ in line_plots:
        res.append(gr.LinePlot(x_lim=None, y_lim=None))
    return res


def load_datasource(datasource, fn):
    if datasource.startswith('file://'):
        return fn(datasource)
    elif datasource.startswith('s3://'):
        return fn(datasource)
    else:
        raise ValueError(f"Unknown datasource: {datasource}")


if __name__ == '__main__':
    datasource_bench = os.environ.get('DATASOURCE_BENCH', 'file://benchmarks.parquet')
    datasource_ci = os.environ.get('DATASOURCE_CI', 'file://ci.parquet')
    df_bench = load_datasource(datasource_bench, load_bench_results)
    df_ci = load_datasource(datasource_ci, load_ci_results)

    metrics = {
        "inter_token_latency_ms_p90": PlotConfig(title="Inter Token Latency P90 (lower is better)", x_title="QPS",
                                                 y_title="Time (ms)"),
        "time_to_first_token_ms_p90": PlotConfig(title="TTFT P90 (lower is better)", x_title="QPS",
                                                 y_title="Time (ms)"),
        "e2e_latency_ms_p90": PlotConfig(title="End to End Latency P90 (lower is better)", x_title="QPS",
                                         y_title="Time (ms)"),
        "token_throughput_secs": PlotConfig(title="Request Output Throughput P90 (higher is better)", x_title="QPS",
                                            y_title="Tokens/s"),
        "successful_requests": PlotConfig(title="Successful requests (higher is better)", x_title="QPS",
                                          y_title="Count"),
        "error_rate": PlotConfig(title="Error rate", x_title="QPS", y_title="%")
    }
    default_df = pd.DataFrame.from_dict(
        {"rate": [1, 2], "inter_token_latency_ms_p90": [10, 20], "engine": ["tgi", "vllm"]})
    # df_bench = load_bench_results()
    # df_ci = load_ci_results()
    models = df_bench["model"].unique()
    devices_bench = df_bench["device"].unique()
    devices_ci = df_ci["device"].unique()
    commits = df_ci[df_ci["engine"] == "TGI"]["version"].unique()
    with gr.Blocks(css=css, title="TGI benchmarks") as demo:
        with gr.Row():
            header = gr.Markdown("# TGI benchmarks\nBenchmark results for Hugging Face TGI 🤗")
        with gr.Tab(label="TGI benchmarks"):
            with gr.Row():
                device_bench = gr.Radio(devices_bench, label="Select device", value="H100")
            with gr.Row():
                summary_desc = gr.Markdown(summary_desc)
                versions = df_bench.groupby(['engine', 'version']).size().reset_index()
            with gr.Row():
                versions_md = "**Versions**\n"
                for engine in versions['engine'].unique():
                    versions_md += f"* **{engine}**: {versions[versions['engine'] == engine]['version'].values[0]}\n"
                versions_desc = gr.Markdown(versions_md)
            with gr.Row():
                table = gr.DataFrame(
                    pd.DataFrame(),
                    elem_classes=["summary"],
                )
            with gr.Row():
                details_desc = gr.Markdown("## Details")
            with gr.Row():
                model = gr.Dropdown(list(models), label="Select model", value=models[0])
            colors = ['#640D5F', '#D91656', '#EE66A6', '#FFEB55']
            colormap = {}
            for idx, engine in enumerate(df_bench['engine'].unique()):
                colormap[engine] = colors[idx % len(colors)]
            colormap['vLLM'] = '#2F5BA1'
            colormap['TGI'] = '#FF9D00'
            i = 0
            with ExitStack() as stack:
                for k, v in metrics.items():
                    if i % 2 == 0:
                        stack.close()
                        gs = stack.enter_context(gr.Row())
                    line_plots.append(
                        {"component": gr.LinePlot(default_df, label=f'{v.title}', x="rate", y=k,
                                                  color="engine", y_title=v.y_title, x_title=v.x_title,
                                                  color_map=colormap),
                         "model": model.value,
                         "device": device_bench,
                         "metric": k
                         },
                    )
                    i += 1
        with gr.Tab(label="CI results"):
            with gr.Row():
                header = gr.Markdown("# CI results\nSummary of the benchmarks")
            with gr.Row():
                device_ci = gr.Radio(list(devices_ci), label="Select device", value=devices_ci[0])
            with gr.Row():
                commit_ref = gr.Dropdown(list(commits), label="Reference commit", value=commits[0])
                commit_compare = gr.Dropdown(list(commits), label="Commit to compare", value=commits[0])
            with gr.Row():
                comparison_table = gr.DataFrame(
                    pd.DataFrame(),
                    elem_classes=["summary"],
                )
        for component in [device_bench, device_ci, model, commit_ref, commit_compare]:
            component.change(update_app, [device_bench, device_ci, model, commit_ref, commit_compare],
                             [item["component"] for item in line_plots] + [table, comparison_table])
        gr.on([plot["component"].select for plot in line_plots], select_region, [device_bench, model],
              outputs=[item["component"] for item in line_plots])
        gr.on([plot["component"].double_click for plot in line_plots], reset_region, None,
              outputs=[item["component"] for item in line_plots])
        demo.load(update_app, [device_bench, device_ci, model, commit_ref, commit_compare],
                  [item["component"] for item in line_plots] + [table, comparison_table])

    demo.launch()