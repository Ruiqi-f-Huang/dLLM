"""Create a GSM8K PoT variant ending in `result = ...; return result`."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

try:
    from .audit_gsm8k_pot_final import execute, numbers_equal, to_number
except ImportError:  # Direct execution adds this script's directory to sys.path.
    from audit_gsm8k_pot_final import execute, numbers_equal, to_number


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_strict_result_return(function):
    """Return whether the last two statements assign and return `result`."""
    if len(function.body) < 2:
        return False
    assignment, returned = function.body[-2:]
    if not isinstance(returned, ast.Return):
        return False
    if not isinstance(returned.value, ast.Name) or returned.value.id != "result":
        return False
    if isinstance(assignment, ast.Assign):
        return (
            len(assignment.targets) == 1
            and isinstance(assignment.targets[0], ast.Name)
            and assignment.targets[0].id == "result"
        )
    return (
        isinstance(assignment, ast.AnnAssign)
        and isinstance(assignment.target, ast.Name)
        and assignment.target.id == "result"
    )


def normalize_result_return(code):
    """Replace only the final return line, preserving all earlier source text."""
    module = ast.parse(code)
    functions = [
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(module.body) != 1 or len(functions) != 1:
        raise ValueError("Expected exactly one top-level function")
    function = functions[0]
    if function.name != "simple_math_problem":
        raise ValueError("Expected simple_math_problem")
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    if len(returns) != 1 or not function.body:
        raise ValueError("Expected exactly one return")
    returned = returns[0]
    if function.body[-1] is not returned:
        raise ValueError("Return must be the final top-level function statement")
    if returned.value is None:
        raise ValueError("Return has no value")
    if has_strict_result_return(function):
        return code, False
    if returned.lineno != returned.end_lineno:
        raise ValueError("Multiline return is not supported")

    expression = ast.get_source_segment(code, returned.value)
    if expression is None or "\n" in expression:
        raise ValueError("Could not recover the return expression")
    lines = code.splitlines(keepends=True)
    source_line = lines[returned.lineno - 1]
    indentation = source_line[:len(source_line) - len(source_line.lstrip())]
    line_ending = "\r\n" if source_line.endswith("\r\n") else "\n"
    replacement = (
        f"{indentation}result = {expression}{line_ending}"
        f"{indentation}return result{line_ending}"
    )
    lines[returned.lineno - 1:returned.end_lineno] = [replacement]
    transformed = "".join(lines)

    transformed_module = ast.parse(transformed)
    transformed_function = transformed_module.body[0]
    if not has_strict_result_return(transformed_function):
        raise ValueError("Result-return normalization failed")
    return transformed, True


def transform_split(source, destination, timeout):
    output = []
    changed = 0
    for line_number, line in enumerate(source.read_text().splitlines(), 1):
        if not line:
            continue
        record = json.loads(line)
        original = record["code"]
        transformed, was_changed = normalize_result_return(original)
        original_answer = execute(original, timeout)
        transformed_answer = execute(transformed, timeout)
        gold_answer = to_number(record["answer"])
        if original_answer is None or transformed_answer is None:
            raise ValueError(f"{source}:{line_number}: non-numeric return")
        if not numbers_equal(original_answer, transformed_answer):
            raise ValueError(f"{source}:{line_number}: answer changed")
        if not numbers_equal(transformed_answer, gold_answer):
            raise ValueError(f"{source}:{line_number}: answer differs from gold")

        record["code"] = transformed
        if "unpadded_total_tokens" in record:
            record["original_unpadded_total_tokens"] = record.pop(
                "unpadded_total_tokens")
        record["target_transform"] = "normalize_result_return"
        output.append(record)
        changed += int(was_changed)

    destination.write_text("".join(
        json.dumps(row, ensure_ascii=True) + "\n" for row in output))
    return output, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=ROOT / "data/gsm8k_pot")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/gsm8k_pot_result")
    parser.add_argument("--timeout", type=float, default=0.25)
    args = parser.parse_args()

    input_manifest = args.input_dir / "split_manifest.json"
    source_hashes = {
        f"sft_{split}.jsonl": sha256(
            args.input_dir / f"sft_{split}.jsonl")
        for split in ("train", "validation")
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    split_ids = {}
    total_changed = 0
    for split in ("train", "validation"):
        source = args.input_dir / f"sft_{split}.jsonl"
        destination = args.output_dir / f"sft_{split}.jsonl"
        rows, changed = transform_split(source, destination, args.timeout)
        ids = [row["sample_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample ID in {split}")
        split_ids[split] = set(ids)
        total_changed += changed
        files[destination.name] = {
            "rows": len(rows),
            "changed_rows": changed,
            "sha256": sha256(destination),
        }
    if split_ids["train"] & split_ids["validation"]:
        raise ValueError("Train and validation IDs overlap")

    source_manifest = json.loads(input_manifest.read_text())
    manifest = {
        "source_manifest_sha256": sha256(input_manifest),
        "source_files": source_hashes,
        "transformation": (
            "Preserve the function and all preceding source text; normalize "
            "the final return to `result = <original expression>` followed "
            "by `return result`."),
        "verification": (
            "Every source and transformed program parses, executes with the "
            "restricted audit runtime, returns the same numeric value, and "
            "matches the stored gold answer."),
        "token_metadata": (
            "The source unpadded_total_tokens value is retained as "
            "original_unpadded_total_tokens. The transformed length must be "
            "recomputed by the fixed checkpoint tokenizer before training."),
        "split_method": source_manifest["split_method"],
        "test": source_manifest["test"],
        "total_changed_rows": total_changed,
        "total_unchanged_rows": sum(
            item["rows"] for item in files.values()) - total_changed,
        "files": files,
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    (args.output_dir / "README.md").write_text(
        "# GSM8K PoT：统一 result-return 的 SFT 变体\n\n"
        "该目录由 `scripts/data/prepare_gsm8k_pot_result.py` 从 "
        "`data/gsm8k_pot/` 确定性生成。原题、docstring、数学计算、答案及 "
        "train/validation 划分不变；函数结尾统一为：\n\n"
        "```python\nresult = <原 return 表达式>\nreturn result\n```\n\n"
        "转换前后逐条受限执行，要求返回相同数值并与 GSM8K gold 一致。"
        "原始数据不会被修改。转换后的 token 长度由训练数据加载器使用固定 tokenizer "
        "重新计算。\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
