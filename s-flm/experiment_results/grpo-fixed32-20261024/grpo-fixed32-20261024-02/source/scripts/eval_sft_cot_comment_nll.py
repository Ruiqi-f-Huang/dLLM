#!/usr/bin/env python3
"""Compare validation diffusion NLL on GSM8K reasoning comments vs code tokens.

Uses the same full validation split, corruption mask, and random timestep/noise
draws for each checkpoint. Reports token-weighted MDLM objective separately
for actual Python COMMENT tokens and all other supervised target tokens.
"""
import json
import os
import tokenize
from io import StringIO
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import algo
import dataloader


def _comment_spans(source):
    spans = []
    lines = source.splitlines(keepends=True)
    starts = []
    cursor = 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line)
    for token in tokenize.generate_tokens(StringIO(source).readline):
        if token.type != tokenize.COMMENT:
            continue
        (start_line, start_col), (end_line, end_col) = token.start, token.end
        start = starts[start_line - 1] + start_col
        end = starts[end_line - 1] + end_col
        spans.append((start, end))
    return spans


def _target_masks(rows, tokenizer, dataset, separator, length, device):
    comment_masks, target_masks = [], []
    sep = tokenizer(separator, add_special_tokens=False).input_ids
    # Verify the source text reconstructs exactly the target token segment in
    # the cached validation dataset, then map character comment spans to BPEs.
    for i, row in enumerate(rows):
        code = row["code"].strip()
        encoded = tokenizer(code, add_special_tokens=False,
                            return_offsets_mapping=True)
        offsets = encoded.offset_mapping
        ds_ids = dataset[i]["input_ids"].tolist()
        prompt_len = 1 + len(tokenizer(row["question"].strip(),
                                      add_special_tokens=False).input_ids) + len(sep)
        target_ids = tokenizer(code, add_special_tokens=False).input_ids
        if ds_ids[prompt_len:prompt_len + len(target_ids)] != target_ids:
            raise ValueError(f"Token mismatch for validation row {i}")
        if ds_ids[prompt_len + len(target_ids)] != tokenizer.eos_token_id:
            raise ValueError(f"Missing EOS after target for validation row {i}")
        spans = _comment_spans(code)
        comment = [False] * length
        target = [False] * length
        for token_index, (start, end) in enumerate(offsets):
            position = prompt_len + token_index
            if position >= length:
                raise ValueError(f"Target exceeds configured length for row {i}")
            target[position] = True
            if end > start and any(start < span_end and end > span_start
                                   for span_start, span_end in spans):
                comment[position] = True
        eos_position = prompt_len + len(target_ids)
        target[eos_position] = True
        comment_masks.append(comment)
        target_masks.append(target)
    return (torch.tensor(comment_masks, dtype=torch.bool, device=device),
            torch.tensor(target_masks, dtype=torch.bool, device=device))


def _load_model(checkpoint, lora_enabled, config, tokenizer, device):
    config.lora.enabled = lora_enabled
    config.eval.checkpoint_path = checkpoint
    model = algo.MDLM.load_from_checkpoint(
        checkpoint, tokenizer=tokenizer, config=config,
        strict=config.eval.strict_loading, map_location="cpu")
    model.to(device)
    model._eval_mode()  # score EMA weights, matching prior evaluations
    return model


@hydra.main(version_base=None, config_path="../configs",
            config_name="sft-a800-lora-result-cot-comments-pilot")
def main(config: DictConfig):
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if not torch.cuda.is_available():
        raise RuntimeError("This diagnostic requires the rented CUDA GPU.")
    torch.backends.cuda.matmul.allow_tf32 = True
    config.model.length = 512
    config.data.train_on_prompt = False
    config.data.train_on_pad = True
    config.eval.disable_ema = False
    config.algo.name = "mdlm"
    device = torch.device("cuda:0")

    checkpoints = [
        ("base", os.environ["BASE_CKPT"], False),
        ("code_only_200", os.environ["CONTROL_CKPT"], True),
        ("cot_comments_200", os.environ["COTCOMMENTS_CKPT"], True),
    ]
    out_path = Path(os.environ.get("COMMENT_NLL_OUT", "comment_code_nll.json"))
    batch_size = int(os.environ.get("COMMENT_NLL_BATCH_SIZE", "8"))
    draws = int(os.environ.get("COMMENT_NLL_DRAWS", "3"))

    tokenizer = dataloader.get_tokenizer(config)
    dataset = dataloader.get_gsm8k_pot_dataset(config, tokenizer, "validation")
    rows = [json.loads(line) for line in Path(config.data.validation_file)
            .read_text().splitlines() if line.strip()]
    if len(rows) != len(dataset):
        raise ValueError("Raw validation rows and tokenized dataset differ.")
    comment_mask, target_mask = _target_masks(
        rows, tokenizer, dataset, config.data.separator,
        config.model.length, device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)

    output = {
        "validation_file": str(config.data.validation_file),
        "examples": len(rows),
        "batch_size": batch_size,
        "matched_random_draws": draws,
        "weights": "EMA",
        "loss": "MDLM validation diffusion NLL (same mask/timestep draws per checkpoint)",
        "regions": {
            "cot_comment": "tokens overlapping Python COMMENT spans inserted from GSM8K reference explanations",
            "code_and_eos": "all other supervised target tokens, including the copied docstring and EOS",
        },
        "models": {},
    }

    for label, checkpoint, lora in checkpoints:
        print(f"Scoring {label}: {checkpoint}", flush=True)
        model = _load_model(checkpoint, lora, config, tokenizer, device)
        sums = {"cot_comment": 0.0, "code_and_eos": 0.0}
        counts = {"cot_comment": 0, "code_and_eos": 0}
        row_sums = {
            name: [0.0] * len(rows)
            for name in ("cot_comment", "code_and_eos")
        }
        row_counts = {
            name: [0] * len(rows)
            for name in ("cot_comment", "code_and_eos")
        }
        for draw in range(draws):
            for batch_index, batch in enumerate(loader):
                # Same seed for the same draw/batch across all checkpoints.
                seed = 20261022 + draw * 100000 + batch_index
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                x0 = batch["input_ids"].to(device)
                loss_mask = batch["attention_mask"].to(device).bool()
                noise_mask = batch["noise_mask"].to(device).bool()
                begin = batch_index * batch_size
                end = begin + x0.shape[0]
                comments = comment_mask[begin:end]
                targets = target_mask[begin:end]
                with torch.no_grad():
                    per_token, _ = model.nll(
                        x0=x0, output_tokens=None, context=None,
                        current_accumulation_step=None, train_mode=False,
                        valid_tokens=noise_mask)
                regions = {
                    "cot_comment": comments & loss_mask & targets,
                    "code_and_eos": (~comments) & loss_mask & targets,
                }
                for name, region in regions.items():
                    count = int(region.sum().item())
                    sums[name] += float((per_token.float() * region).sum().item())
                    counts[name] += count
                    item_sums = (per_token.float() * region).sum(dim=1).cpu().tolist()
                    item_counts = region.sum(dim=1).cpu().tolist()
                    for offset, (item_sum, item_count) in enumerate(
                            zip(item_sums, item_counts)):
                        row_sums[name][begin + offset] += float(item_sum)
                        row_counts[name][begin + offset] += int(item_count)
        output["models"][label] = {
            "checkpoint": checkpoint,
            "token_counts_across_draws": counts,
            "mean_nll_per_token": {
                name: sums[name] / max(counts[name], 1) for name in sums
            },
            "weighted_nll_sum": sums,
            "per_example_mean_nll": {
                name: [
                    (row_sums[name][i] / row_counts[name][i]
                     if row_counts[name][i] else None)
                    for i in range(len(rows))
                ]
                for name in row_sums
            },
        }
        del model
        torch.cuda.empty_cache()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
