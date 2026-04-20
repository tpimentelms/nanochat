# nanochat

![nanochat logo](dev/nanochat.png)

This readme file aims to give a quick guide on how to run jobs for full model training using the BNE tokenizer.
Refer to the [readme.md](README.md) file for guidance on installation.

# Setup

## Pretrained Tokenizers

A set of pretrained unrestricted BNE tokenizers can be found in [data/BNE_unrest](data/BNE_unrest/).
To setup a BNE trainingrun using pretrained tokenizers just copy the `tokenizer.json` and the `token_bytes.pt` files into a folder `tokenizer` in your nanochat base dir.

The file [runs/runpretrainedbne_spu.sh](runs/runpretrainedbne_cpu.sh) shows an example run using a pretrained BNE tokenizer. The script [scripts/tok_gen_bytes.py](scripts/tok_gen_bytes.py) is only executed if the `tokenizer` folder contains no file `token_bytes.pt`.

## Architecture specifics

The uv installation contains three modes/configurations:

- cpu: for runs on a CPU and testing
- gpu: for runs on GPUs with architectures capable of CUDA 12.8 or higher
- cu118: for runs on GPUs using CUDA 11.8 or higher. (not quite functional yet)

