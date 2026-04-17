"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""
import os
import torch
from nanochat.tokenizer import HuggingFaceTokenizer, get_tokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# load tokenizer
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer = get_tokenizer()

# -----------------------------------------------------------------------------
# generate Bytes files
# -----------------------------------------------------------------------------
# we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
if os.path.exists(os.path.join(tokenizer_dir, "token_bytes.pt")):
    print("token_bytes.pt already exists, skipping generation.")
else:
    vocab_size = tokenizer.get_vocab_size()
    special_set = set(tokenizer.get_special_tokens())
    token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
    token_bytes = []
    for token_id in range(vocab_size):
        token_str = token_strings[token_id] # the Python string representation of this token
        if token_str in special_set:
            token_bytes.append(0) # special characters are not counted
        else:
            id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
            token_bytes.append(id_bytes)
    token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    with open(token_bytes_path, "wb") as f:
        torch.save(token_bytes, f)
    print(f"Saved token_bytes to {token_bytes_path}")

    # Log to report
    from nanochat.report import get_report
    token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
    get_report().log(section="Tokenizer training", data=[
        {"num_special_tokens": len(special_set)},
        {
            "token_bytes_min": int(token_bytes_nonzero.min().item()),
            "token_bytes_max": int(token_bytes_nonzero.max().item()),
            "token_bytes_mean": token_bytes_nonzero.mean().item(),
            "token_bytes_std": token_bytes_nonzero.std().item(),
        }
    ])
