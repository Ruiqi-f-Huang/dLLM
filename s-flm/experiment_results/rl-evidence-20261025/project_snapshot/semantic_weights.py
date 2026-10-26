"""Deterministic Python span labels; never inspect question/gold answers."""
import ast
import io
import tokenize


VERSION = 1


def critical_spans(code):
    tree = ast.parse(code)
    lines = code.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(line, column, byte=False):
        text = lines[line - 1]
        if byte:
            column = len(text.encode('utf-8')[:column].decode('utf-8'))
        return starts[line - 1] + column

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign,
                             ast.Return, ast.BinOp, ast.UnaryOp, ast.Compare)):
            names.update((offset(n.lineno, n.col_offset, True),
                          offset(n.end_lineno, n.end_col_offset, True))
                         for n in ast.walk(node) if isinstance(n, ast.Name))
    spans = []
    # tokenize excludes docstring contents automatically (one STRING token).
    operators = {'+', '-', '*', '/', '//', '%', '**', '<', '>', '<=', '>=', '==', '!='}
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        if (token.type == tokenize.NUMBER
                or (token.type == tokenize.OP and token.string in operators)
                or (token.type == tokenize.NAME and
                    (offset(*token.start), offset(*token.end)) in names)):
            spans.append((offset(*token.start), offset(*token.end)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            value = node.value
            spans.append((offset(value.lineno, value.col_offset, True),
                          offset(value.end_lineno, value.end_col_offset, True)))
    return sorted(set(spans))


def code_weights(code, tokenizer, weight):
    encoded = tokenizer(code, add_special_tokens=False, return_offsets_mapping=True)
    spans = critical_spans(code)
    weights = [float(weight) if any(a < d and b > c for c, d in spans)
               else 1.0 for a, b in encoded.offset_mapping]
    return encoded.input_ids, weights


def weighted_mean(loss, valid_tokens, weights):
    effective = valid_tokens * weights
    return (loss * effective).sum() / effective.sum().clamp(min=1)
