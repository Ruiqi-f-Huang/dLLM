"""Trim only trailing reference-solution comments to fit a token budget."""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
COMMENT = re.compile(r"^(\s*)# ")


def token_length(tokenizer, row, separator_ids):
    question_ids = tokenizer(
        row["question"].strip(), add_special_tokens=False).input_ids
    code_ids = tokenizer(row["code"].strip(), add_special_tokens=False).input_ids
    return 1 + len(question_ids) + len(separator_ids) + len(code_ids) + 1


def trailing_comment_indices(code):
    lines = code.splitlines(keepends=True)
    module = ast.parse(code)
    function = module.body[0]
    docstring = function.body[0]
    start = docstring.end_lineno
    while start < len(lines) and not lines[start].strip():
        start += 1
    indices = []
    for index in range(start, len(lines)):
        if COMMENT.match(lines[index]):
            indices.append(index)
        else:
            break
    return lines, indices


def trim_row(row, tokenizer, separator_ids, max_length):
    lines, indices = trailing_comment_indices(row["code"])
    before = len(indices)
    length = token_length(tokenizer, row, separator_ids)
    removed = 0
    while length > max_length and indices:
        index = indices.pop()
        del lines[index]
        row["code"] = "".join(lines)
        removed += 1
        length = token_length(tokenizer, row, separator_ids)
    if length > max_length:
        raise ValueError(
            f"{row['sample_id']} remains {length} tokens after removing all comments")

    row["cot_comment_lines_before_length_trim"] = before
    row["cot_comment_lines_removed_for_length"] = removed
    row["cot_comment_lines"] = before - removed
    return length, removed


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--separator", default=r"\n")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True)
    separator_ids = tokenizer(
        args.separator, add_special_tokens=False).input_ids
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    ids_by_split = {}
    for split in ("train", "validation"):
        src = args.input_dir / f"sft_{split}.jsonl"
        dst = args.output_dir / f"sft_{split}.jsonl"
        rows = [json.loads(line) for line in src.read_text().splitlines() if line]
        lengths = []
        removed_rows = removed_lines = 0
        for row in rows:
            original_code = row["code"]
            length, removed = trim_row(
                row, tokenizer, separator_ids, args.max_length)
            if ast.dump(ast.parse(original_code), include_attributes=False) != ast.dump(
                    ast.parse(row["code"]), include_attributes=False):
                raise ValueError(f"Executable AST changed for {row['sample_id']}")
            lengths.append(length)
            removed_rows += int(removed > 0)
            removed_lines += removed
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate IDs in {split}")
        ids_by_split[split] = set(ids)
        dst.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n"
                                       for row in rows))
        stats[split] = {
            "rows": len(rows),
            "max_tokens": max(lengths),
            "over_max": sum(length > args.max_length for length in lengths),
            "rows_with_trimmed_comments": removed_rows,
            "comment_lines_removed": removed_lines,
            "sha256": sha256(dst),
        }
    if ids_by_split["train"] & ids_by_split["validation"]:
        raise ValueError("Train/validation IDs overlap")

    manifest = {
        "source_dir": str(args.input_dir),
        "source_hashes": {
            f"sft_{split}.jsonl": sha256(
                args.input_dir / f"sft_{split}.jsonl")
            for split in ("train", "validation")
        },
        "transformation": (
            "Remove trailing GSM8K reference-solution comment lines only "
            "until each complete [BOS] question separator target [EOS] "
            "sequence fits max_length. Preserve the question, executable "
            "Python AST, answer, and split."),
        "tokenizer": tokenizer.name_or_path,
        "separator": args.separator,
        "max_length": args.max_length,
        "split_method": json.loads(
            (args.input_dir / "split_manifest.json").read_text()
        )["split_method"],
        "test": json.loads(
            (args.input_dir / "split_manifest.json").read_text()
        )["test"],
        "splits": stats,
    }
    (args.output_dir / "length_trim_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# GSM8K PoT：参考解答注释版（512-token 安全版）\n\n"
        "该目录由 `scripts/data/trim_gsm8k_pot_cot_comments.py` 从完整注释版 "
        "生成。每条样本使用 GSM8K 原有参考解答作为 Python 注释；为满足 "
        f"{args.max_length}-token 上限，只从超长样本末尾删除完整注释行。"
        "题目、可执行 Python AST、答案以及 train/validation 划分保持不变。\n\n"
        "`length_trim_manifest.json` 记录 tokenizer、长度上限、删减数量和文件哈希；"
        "`tokenization_report.json` 记录最终长度统计。\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
