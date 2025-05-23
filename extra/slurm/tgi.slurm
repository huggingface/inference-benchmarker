#!/usr/bin/env bash
#SBATCH --job-name tgi-benchmark
#SBATCH --output /fsx/%u/logs/%x-%j.log
#SBATCH --time 1:50:00
#SBATCH --qos normal
#SBATCH --partition hopper-prod
#SBATCH --gpus 1 --ntasks 1 --cpus-per-task 11 --mem-per-cpu 20G --nodes=1
#SBATCH hetjob
#SBATCH --gpus 1 --ntasks 1 --cpus-per-task 11 --mem-per-cpu 20G --nodes=1

if [ -z "$MODEL" ]; then
    echo "MODEL environment variable is not set"
    exit 1
fi

if [ -z "$TP" ]; then
    echo "TP environment variable is not set"
    exit 1
fi

echo "Starting TGI benchmark for $MODEL"
export RUST_BACKTRACE=full
export RUST_LOG=inference_benchmarker=info

# set a random available port to avoid conflicts
PORT=$(shuf -i 8000-9999 -n 1)
export PORT
export PREFILL_CHUNKING=1

echo "Model will run on ${SLURM_JOB_NODELIST_HET_GROUP_0}:${PORT}"
echo "Benchmark will run on ${SLURM_JOB_NODELIST_HET_GROUP_1}"

# start TGI
srun --het-group=0 \
     -u \
     -n 1 \
     --container-image='ghcr.io#huggingface/text-generation-inference' \
     --container-env=PORT \
     --container-mounts="/scratch:/data" \
     --container-workdir='/usr/src' \
     --no-container-mount-home \
     /usr/local/bin/text-generation-launcher \
      --model-id $MODEL \
      --max-concurrent-requests 1024&

# wait until /health is available, die after 5 minutes
timeout 600 bash -c "while [[ \"\$(curl -s -o /dev/null -w '%{http_code}' http://localhost:${PORT}/health)\" != \"200\" ]]; do sleep 1 && echo \"Waiting for TGI to start...\"; done" || exit 1
exit_code=$?

RESULTS_DIR="/fsx/$USER/benchmarks_results/tgi"
mkdir -p "${RESULTS_DIR}"

if [[ $exit_code != 124 ]]; then
    # run benchmark
    echo "Starting benchmark"
    VERSION=$(curl -s http://${SLURM_JOB_NODELIST_HET_GROUP_0}:${PORT}/info | jq -r '.sha')
    srun --het-group=1 \
         -u \
         -n 1 \
         --container-image="ghcr.io#huggingface/inference-benchmarker:latest" \
         --container-mounts="${RESULTS_DIR}:/opt/inference-benchmarker/results" \
         --no-container-mount-home \
         inference-benchmarker \
             --tokenizer-name "$MODEL" \
             --max-vus 128 \
             --url "http://${SLURM_JOB_NODELIST_HET_GROUP_0}:${PORT}" \
             --duration 120s \
             --warmup 30s \
             --benchmark-kind rate \
             --rates 0.8 --rates 2.4 --rates 4.0 --rates 5.6 --rates 7.2 --rates 8.8 --rates 10.4 --rates 12.0 --rates 13.6 --rates 15.2 --rates 16.8 --rates 18.4 --rates 20.0 --rates 21.6 --rates 23.2 --rates 24.0 \
             --extra-meta "version=$VERSION,engine=TGI,tp=$TP" \
             --decode-options "num_tokens=800,max_tokens=800,min_tokens=50,variance=100" \
             --dataset-file share_gpt_turns.json \
             --no-console
fi

# stop TGI
scancel --signal=TERM "$SLURM_JOB_ID+0"

echo "End of benchmark"