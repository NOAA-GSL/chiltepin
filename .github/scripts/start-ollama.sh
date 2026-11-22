#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Start a local model for the MPAS mesh agent's prompt test, on the Slurm
# cluster's frontend, where the compute nodes running the agent's tasks can
# reach it. No credentials are needed.
#
# usage: start-ollama.sh DIR MODEL VERSION SHA256
#
#   DIR      where Ollama, its models and its logs are kept, on the home
#            volume every node shares
#   MODEL    the Ollama model to test with, such as qwen3:8b
#   VERSION  the Ollama release, such as 0.40.16
#   SHA256   that release's checksum for ollama-linux-amd64.tar.zst
#
# The model is copied as "chiltepin-test" at temperature 0, so its answers
# repeat, and loaded before the test, which then needs no time to load it.
# DIR/ollama.env sets OLLAMA_API_BASE for the tasks that call it.
set -euo pipefail

dir=$1
model=$2
version=$3
sha256=$4
api="localhost:11434"

mkdir -p "$dir"
cd "$dir"
if [ ! -x bin/ollama ]; then
    curl -fsSL --retry 5 --retry-all-errors -o ollama.tar.zst \
        "https://github.com/ollama/ollama/releases/download/v${version}/ollama-linux-amd64.tar.zst"
    echo "${sha256}  ollama.tar.zst" | sha256sum -c -
    tar --use-compress-program=zstd -xf ollama.tar.zst
    rm ollama.tar.zst
fi

# Listen on every interface, so the compute nodes reach the frontend. One
# request at a time, with the model kept loaded until the server stops.
OLLAMA_MODELS="$dir/models" OLLAMA_HOST=0.0.0.0:11434 OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=-1 \
    setsid nohup bin/ollama serve > serve.log 2>&1 < /dev/null &
for _ in $(seq 60); do
    if curl -sf "$api/api/version" > /dev/null; then
        break
    fi
    sleep 1
done
curl -sf "$api/api/version"
echo

OLLAMA_HOST="$api" bin/ollama pull "$model"
printf 'FROM %s\nPARAMETER temperature 0\n' "$model" > Modelfile
OLLAMA_HOST="$api" bin/ollama create chiltepin-test -f Modelfile
start=$SECONDS
curl -sf "$api/api/generate" \
    -d '{"model": "chiltepin-test", "prompt": "Hi", "stream": false, "options": {"num_predict": 1}}' \
    > /dev/null
echo "Loaded the model in $((SECONDS - start)) s"

echo "export OLLAMA_API_BASE=http://$(hostname):11434" > ollama.env
echo "Wrote $dir/ollama.env"
