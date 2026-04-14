#!/usr/bin/env python3
"""
Python wrapper for inference_gpt2cu.
Handles prompt tokenization, then delegates to the CUDA binary.

Usage:
  python infer.py -p "Once upon a time"
  python infer.py -p "Hello world" -g 128 -e gpt2_124M_bf16.bin
"""

import argparse
import subprocess
import sys

try:
    import tiktoken
except ImportError:
    print("tiktoken not found. Install with: pip install tiktoken")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="GPT-2 inference with prompt support")
    parser.add_argument("-p", "--prompt",  default="",                  help="prompt text")
    parser.add_argument("-e", "--model",   default="gpt2_124M_bf16.bin", help="model checkpoint")
    parser.add_argument("-g", "--gen",     type=int, default=256,        help="tokens to generate")
    parser.add_argument("-s", "--seed",    type=int, default=1337,       help="random seed")
    parser.add_argument("-x", "--stop-eot", type=int, default=0,        help="stop at <|endoftext|> (1=yes)")
    parser.add_argument("--binary",        default="./inference_gpt2cu", help="path to compiled binary")
    args = parser.parse_args()

    enc = tiktoken.get_encoding("gpt2")

    cmd = [args.binary, "-e", args.model, "-g", str(args.gen),
           "-s", str(args.seed), "-x", str(args.stop_eot)]

    if args.prompt:
        token_ids = enc.encode(args.prompt)
        cmd += ["-p", " ".join(map(str, token_ids))]
        print(f"prompt tokens ({len(token_ids)}): {token_ids}")

    try:
        subprocess.run(cmd)
    except FileNotFoundError:
        print(f"binary not found: {args.binary}")
        print("build with: make inference_gpt2cu")
        sys.exit(1)


if __name__ == "__main__":
    main()
