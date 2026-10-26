"""Add GSM8K reference-solution steps as Python comments to PoT targets.

The source data is never modified. The answer explanation is taken from the
existing GSM8K reference_answer field, stripped of GSM8K's inline calculation
markup and final #### answer line, then inserted as comments immediately
after the question docstring. The executable program body is preserved.
"""

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

try:
    from .audit_gsm8k_pot_final import execute, numbers_equal, to_number
except ImportError:  # Direct execution adds this script's directory to sys.path.
    from audit_gsm8k_pot_final import execute, numbers_equal, to_number


ROOT = Path(__file__).resolve().parents[2]
INLINE_ANSWER = re.compile(r"<<.*?>>")
FINAL_ANSWER = re.compile(r"^\s*####\s*.*$", re.MULTILINE)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_steps(reference_answer):
    """Return readable explanation lines without GSM8K answer annotations."""
    cleaned = INLINE_ANSWER.sub("", reference_answer)
    cleaned = FINAL_ANSWER.sub("", cleaned)
    return [line.strip() for line in cleaned.splitlines() if line.strip()]


def add_reference_comments(code, reference_answer):
    """Insert explanation comments after the function docstring."""
    steps = reference_steps(reference_answer)
    if not steps:
        raise ValueError("Reference solution has no explanation steps")

    module = ast.parse(code)
    functions = [node for node in module.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if (len(module.body) != 1 or len(functions) != 1
            or not functions[0].body
            or not isinstance(functions[0].body[0], ast.Expr)
            or not isinstance(functions[0].body[0].value, ast.Constant)
            or not isinstance(functions[0].body[0].value.value, str)):
        raise ValueError("Expected one function whose first statement is a docstring")

    docstring = functions[0].body[0]
    lines = code.splitlines(keepends=True)
    # Insert after the docstring's closing line; preserve the executable body.
    insertion = docstring.end_lineno
    indent = " " * docstring.col_offset
    comment_lines = [f"{indent}# {step}\n" for step in steps]
    if insertion < len(lines) and not lines[insertion].strip():
        insertion += 1
    lines[insertion:insertion] = comment_lines
    transformed = "".join(lines)

    # Confirm the function and its full body still parse. Comments do not alter
    # the executable AST, so compare the ASTs after removing source locations.
    transformed_module = ast.parse(transformed)
    if ast.dump(module, include_attributes=False) != ast.dump(
            transformed_module, include_attributes=False):
        raise ValueError("Comment insertion changed the executable AST")
    return transformed, steps


def transform_split(source, destination, timeout):
    output = []
    total_steps = 0
    max_steps = 0
    for line_number, line in enumerate(source.read_text().splitlines(), 1):
        if not line:
            continue
        record = json.loads(line)
        original = record["code"]
        transformed, steps = add_reference_comments(
            original, record["reference_answer"])

        original_answer = execute(original, timeout)
        transformed_answer = execute(transformed, timeout)
        gold_answer = to_number(record["answer"])
        if original_answer is None or transformed_answer is None:
            raise ValueError(f"{source}:{line_number}: non-numeric return")
        if not numbers_equal(original_answer, transformed_answer):
            raise ValueError(f"{source}:{line_number}: comments changed answer")
        if not numbers_equal(transformed_answer, gold_answer):
            raise ValueError(f"{source}:{line_number}: answer differs from gold")

        record["code"] = transformed
        record["target_transform"] = "reference_solution_as_python_comments"
        record["cot_comment_lines"] = len(steps)
        output.append(record)
        total_steps += len(steps)
        max_steps = max(max_steps, len(steps))

    destination.write_text("".join(
        json.dumps(row, ensure_ascii=True) + "\n" for row in output))
    return output, total_steps, max_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "data/gsm8k_pot_result")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/gsm8k_pot_result_cot_comments")
    parser.add_argument("--timeout", type=float, default=0.25)
    args = parser.parse_args()

    source_manifest_path = args.input_dir / "split_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    split_ids = {}
    total_steps = 0
    max_steps = 0
    for split in ("train", "validation"):
        source = args.input_dir / f"sft_{split}.jsonl"
        destination = args.output_dir / f"sft_{split}.jsonl"
        rows, steps, split_max_steps = transform_split(
            source, destination, args.timeout)
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample ID in {split}")
        split_ids[split] = set(ids)
        total_steps += steps
        max_steps = max(max_steps, split_max_steps)
        files[destination.name] = {
            "rows": len(rows),
            "mean_comment_lines": steps / len(rows),
            "max_comment_lines": split_max_steps,
            "sha256": sha256(destination),
        }
    if split_ids["train"] & split_ids["validation"]:
        raise ValueError("Train and validation IDs overlap")

    manifest = {
        "source_manifest_sha256": sha256(source_manifest_path),
        "source_files": {
            f"sft_{split}.jsonl": sha256(
                args.input_dir / f"sft_{split}.jsonl")
            for split in ("train", "validation")
        },
        "transformation": (
            "Convert each GSM8K reference_answer explanation line into a "
            "Python comment after the existing question docstring. Remove "
            "inline <<calculation>> markers and the final #### answer line. "
            "Preserve all executable PoT source text and the original split."),
        "verification": (
            "For every row, source and transformed targets parse and execute "
            "under the restricted audit runtime, return the same numeric "
            "answer, and match the stored GSM8K gold answer. The executable "
            "Python AST is identical before and after comment insertion."),
        "split_method": source_manifest["split_method"],
        "test": source_manifest["test"],
        "total_rows": sum(item["rows"] for item in files.values()),
        "total_comment_lines": total_steps,
        "max_comment_lines": max_steps,
        "files": files,
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# GSM8K PoT：参考解答注释版\n\n"
        "该目录由 `scripts/data/prepare_gsm8k_pot_cot_comments.py` 从 "
        "`data/gsm8k_pot_result/` 确定性生成。原题、docstring、可执行 PoT "
        "代码、答案及 train/validation 划分均保留；GSM8K `reference_answer` "
        "中的解释步骤经过去除 `<<...>>` 标记和最终 `####` 答案行后，作为 "
        "Python 注释放在题目 docstring 之后。原始数据不会被覆盖。\n\n"
        "每条改造后的程序均通过 AST 解析与受限执行，执行结果与改造前相同，且与 gold 答案一致。\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
