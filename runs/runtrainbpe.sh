#!/bin/bash

# Running a full training run of BNE on the CPU (sufficient RAM recommended)

# Run as:
# bash runs/runbne_cpu.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# You may also want to run this script manually and one by one, copy pasting commands into your terminal.

# all the setup stuff
export NANOCHAT_BASE_DIR="$HOME/nanochat/data/BPE"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# train tokenizer on ~2B characters (~34 seconds on my MacBook Pro M3 Max)
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000 --tokenizer-type=BPE --vocab-size=262144 --tokenizer-name=256k

python -m scripts.tok_train --max-chars=2000000000 --tokenizer-type=BPE --vocab-size=131072 --tokenizer-name=128k

python -m scripts.tok_train --max-chars=2000000000 --tokenizer-type=BPE --vocab-size=65536 --tokenizer-name=64k

python -m scripts.tok_train --max-chars=2000000000 --tokenizer-type=BPE --vocab-size=32768 --tokenizer-name=32k

python -m scripts.tok_train --max-chars=2000000000 --tokenizer-type=BPE --vocab-size=16384 --tokenizer-name=16k

python -m scripts.tok_train --max-chars=2000000000 --tokenizer-type=BPE --vocab-size=8192 --tokenizer-name=8k

python -m scripts.tok_train --max-chars=2000000000 --tokenizer-type=BPE --vocab-size=4096 --tokenizer-name=4k