"""Create a GSM8K PoT SFT variant without duplicated question docstrings."""

import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
import re
import signal
import warnings
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def remove_function_docstring(code):
    """Remove only the leading docstring from simple_math_problem()."""
    module = ast.parse(code)
    functions = [
        node for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(functions) != 1 or functions[0].name != 'simple_math_problem':
        raise ValueError('Expected exactly one simple_math_problem function')
    function = functions[0]
    if not function.body:
        raise ValueError('Function has no body')
    docstring = function.body[0]
    if not (
        isinstance(docstring, ast.Expr)
        and isinstance(docstring.value, ast.Constant)
        and isinstance(docstring.value.value, str)
    ):
        raise ValueError('Function has no leading docstring')
    if docstring.lineno <= function.lineno or docstring.end_lineno is None:
        raise ValueError('Unsupported inline docstring layout')

    lines = code.splitlines(keepends=True)
    del lines[docstring.lineno - 1:docstring.end_lineno]
    transformed = ''.join(lines)
    parsed = ast.parse(transformed)
    transformed_function = parsed.body[0]
    if ast.get_docstring(transformed_function, clean=False) is not None:
        raise ValueError('Docstring removal failed')
    return transformed


class ExecutionTimeout(Exception):
    pass


def timeout_handler(signum, frame):
    del signum, frame
    raise ExecutionTimeout()


@contextlib.contextmanager
def time_limit(seconds):
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def safe_exec(code):
    safe_builtins = {
        'abs': abs, 'min': min, 'max': max, 'sum': sum,
        'len': len, 'range': range, 'enumerate': enumerate,
        'int': int, 'float': float, 'str': str, 'bool': bool,
        'round': round, 'print': print,
    }

    def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'math':
            return __import__(name, globals, locals, fromlist, level)
        raise ImportError(f'Import blocked: {name}')

    safe_builtins['__import__'] = limited_import
    namespace = {'__builtins__': safe_builtins, 'math': math}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                exec(code, namespace, namespace)
    return namespace


def to_number(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(round(value)) if abs(value - round(value)) < 1e-6 else value
    if isinstance(value, str):
        match = re.search(r'[-+]?\$?\d[\d,]*\.?\d*', value)
        if match is None:
            return None
        text = match.group(0).replace(',', '').replace('$', '')
        return to_number(float(text)) if '.' in text else int(text)
    return None


def numbers_equal(left, right):
    if left is None or right is None:
        return False
    if isinstance(left, float) or isinstance(right, float):
        return abs(float(left) - float(right)) <= 1e-3
    return int(left) == int(right)


def execute_answer(code, timeout):
    with time_limit(timeout):
        namespace = safe_exec(code)
        function = namespace.get('simple_math_problem')
        if function is None:
            raise ValueError('simple_math_problem is missing after execution')
        return to_number(function())


def transform_split(source, destination, timeout):
    output = []
    for line_number, line in enumerate(source.read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        original = record['code']
        transformed = remove_function_docstring(original)
        original_answer = execute_answer(original, timeout)
        transformed_answer = execute_answer(transformed, timeout)
        gold_answer = to_number(record['answer'])
        if original_answer is None or transformed_answer is None or gold_answer is None:
            raise ValueError(f'{source}:{line_number}: non-numeric answer')
        if not numbers_equal(original_answer, transformed_answer):
            raise ValueError(f'{source}:{line_number}: transformation changed answer')
        if not numbers_equal(transformed_answer, gold_answer):
            raise ValueError(f'{source}:{line_number}: program does not match gold')

        record['code'] = transformed
        if 'unpadded_total_tokens' in record:
            record['original_unpadded_total_tokens'] = record.pop(
                'unpadded_total_tokens')
        record['target_transform'] = 'remove_function_docstring'
        output.append(record)

    destination.write_text(
        ''.join(json.dumps(row, ensure_ascii=True) + '\n' for row in output))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input-dir', type=Path, default=ROOT / 'data/gsm8k_pot')
    parser.add_argument(
        '--output-dir', type=Path, default=ROOT / 'data/gsm8k_pot_nodoc')
    parser.add_argument('--timeout', type=float, default=1.0)
    args = parser.parse_args()

    source_manifest = json.loads(
        (args.input_dir / 'split_manifest.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    split_ids = {}
    for split in ('train', 'validation'):
        source = args.input_dir / f'sft_{split}.jsonl'
        destination = args.output_dir / f'sft_{split}.jsonl'
        rows = transform_split(source, destination, args.timeout)
        ids = [row['sample_id'] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate sample ID in {split}')
        split_ids[split] = set(ids)
        files[destination.name] = {
            'rows': len(rows),
            'sha256': sha256(destination),
        }
    if split_ids['train'] & split_ids['validation']:
        raise ValueError('Train and validation IDs overlap')

    manifest = {
        'source_manifest_sha256': sha256(args.input_dir / 'split_manifest.json'),
        'source_files': {
            f'sft_{split}.jsonl': sha256(
                args.input_dir / f'sft_{split}.jsonl')
            for split in ('train', 'validation')
        },
        'transformation': (
            'Remove the leading function docstring from simple_math_problem; '
            'leave the remaining program unchanged.'),
        'verification': (
            'Every source and transformed program parses, executes with restricted '
            'builtins, returns the same numeric answer, and matches the stored gold.'),
        'split_method': source_manifest['split_method'],
        'test': source_manifest['test'],
        'files': files,
    }
    manifest_path = args.output_dir / 'split_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    readme = args.output_dir / 'README.md'
    readme.write_text(
        '# GSM8K PoT SFT：无重复题目 docstring 变体\n\n'
        '该目录由 `scripts/data/prepare_gsm8k_pot_nodoc.py` 从 '
        '`data/gsm8k_pot/` 确定性生成。题目、答案、训练/验证划分及程序计算逻辑不变；'
        '只删除 `simple_math_problem()` 开头重复题目的函数 docstring。\n\n'
        '生成过程逐条解析并执行转换前后的程序，要求二者返回相同数值且与标准答案一致。'
        '原始数据不会被修改。\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
