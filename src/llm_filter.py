#!/usr/bin/env python3

import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

SYSTEM_PROMPT = """You are a genomic data quality filter. Your task is to identify accessions that should be REMOVED from a metagenomic database based ONLY on strong evidence in the accession title.

BE CONSERVATIVE: Only output "Remove" if the title contains clear indicators of:
- Host Contamination
- Artificial Origin (Vectors, Plasmids, Constructs, Synthetic)
- Recombinant
- Low quality / UNVERIFIED
- Chimeric

DO NOT EVER remove based on:
- Viral names that include host names
- Complete or Partial Sequences
- Uncultured organisms
- Isolates / Clones
- Ambiguous cases

Output ONLY one word: "Keep" or "Remove". No explanation needed."""

def user_message(accession_id, title):
    return "Accession ID: {}\nTitle: {}\n\nDecision:".format(accession_id, title)

def parse_decision(text):
    return "Remove" if "remove" in text.strip().lower() else "Keep"

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    ap.add_argument("--model", default="google/medgemma-27b-text-it",
                    help="HF model id or local path")
    ap.add_argument("--quantization", default=None,
                    help="vLLM quantization, e.g. awq / awq_marlin / bitsandbytes. Default: none (BF16).")
    ap.add_argument("--dtype", default="bfloat16",
                    help="Model dtype. Default: bfloat16")
    ap.add_argument("--chunk-size", type=int, default=5000,
                    help="Rows per checkpoint save. Default: 5000")
    ap.add_argument("--max-model-len", type=int, default=2048,
                    help="vLLM max context length (titles are short). Default: 2048")
    ap.add_argument("--gpu-mem-util", type=float, default=0.95)
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME"),
                    help="HF cache dir (sets HF_HOME / cache env).")
    args = ap.parse_args()

    if args.hf_home:
        os.makedirs(args.hf_home, exist_ok=True)
        os.environ["HF_HOME"] = args.hf_home
        os.environ["TRANSFORMERS_CACHE"] = args.hf_home
        os.environ["HF_DATASETS_CACHE"] = args.hf_home
        os.environ["VLLM_CACHE_ROOT"] = args.hf_home
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print("Reading {}...".format(args.input_csv))
    df = pd.read_csv(args.input_csv)
    if "Accession" not in df.columns or "Title" not in df.columns:
        sys.exit("ERROR: input CSV must have 'Accession' and 'Title' columns")
    df["Title"] = df["Title"].fillna("").astype(str)
    if "Decision" not in df.columns:
        df["Decision"] = None

    temp_output = args.output_csv + ".tmp"
    if os.path.exists(temp_output):
        print("Resuming from checkpoint {}".format(temp_output))
        df_temp = pd.read_csv(temp_output)
        df.update(df_temp)

    todo = df[df["Decision"].isna()].index.tolist()
    print("Total: {} | done: {} | remaining: {}".format(len(df), len(df) - len(todo), len(todo)))
    if not todo:
        df[["Accession", "Decision"]].to_csv(args.output_csv, index=False)
        return

    tok = AutoTokenizer.from_pretrained(args.model)

    def render(acc, title):

        content = SYSTEM_PROMPT + "\n\n" + user_message(acc, title)
        return tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True,
        )

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        trust_remote_code=True,
        enable_prefix_caching=True,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    print("Loading vLLM model: {} (quantization={})".format(args.model, args.quantization))
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=0, max_tokens=5)

    chunks = [todo[i:i + args.chunk_size] for i in range(0, len(todo), args.chunk_size)]
    for chunk in tqdm(chunks, desc="Chunks"):
        prompts = [render(df.at[i, "Accession"], df.at[i, "Title"]) for i in chunk]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        for i, out in zip(chunk, outputs):
            df.at[i, "Decision"] = parse_decision(out.outputs[0].text)
        df[["Accession", "Decision"]].to_csv(temp_output, index=False)

    df[["Accession", "Decision"]].to_csv(args.output_csv, index=False)
    if os.path.exists(temp_output):
        os.remove(temp_output)
    kept = (df["Decision"] == "Keep").sum()
    removed = (df["Decision"] == "Remove").sum()
    print("Final -- Keep: {} | Remove: {}".format(kept, removed))

if __name__ == "__main__":
    main()
