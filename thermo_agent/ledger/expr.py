"""Safe arithmetic for ledger lines. Only numbers, + - * /, parentheses, the constant T0 and table lookups such as m[s5] are
accepted; nothing is executed. Evaluation errors become ExprError."""
from __future__ import annotations
import ast, math

FAMILIES = {"m", "h", "s", "g0", "ech", "E", "W"}
_OK = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Subscript, ast.Name, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)


class ExprError(ValueError):
    pass


def parse(expr: str):
    if not isinstance(expr, str) or len(expr) > 300: raise ExprError("expression too long")
    try: tree = ast.parse(expr.strip(), mode="eval")
    except (SyntaxError, ValueError): raise ExprError("syntax error")
    count = 0
    for node in ast.walk(tree):
        count += 1
        if count > 120: raise ExprError("expression too complex")
        if not isinstance(node, _OK): raise ExprError(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or not isinstance(node.value, (int, float))): raise ExprError("only numbers are allowed")
        if isinstance(node, ast.Subscript) and not (isinstance(node.value, ast.Name) and node.value.id in FAMILIES and isinstance(node.slice, ast.Name)):
            raise ExprError("lookups must look like m[s5]")
    return tree


def _ev(node, ctx):
    if isinstance(node, ast.Expression): return _ev(node.body, ctx)
    if isinstance(node, ast.Constant): return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in ctx["consts"]: return ctx["consts"][node.id]
        raise ExprError(f"unknown name {node.id}")
    if isinstance(node, ast.Subscript):
        fam, key = node.value.id, node.slice.id
        v = ctx["tables"].get(fam, {}).get(key)
        if v is None or v != v: raise ExprError(f"{fam}[{key}] does not exist")
        return v
    if isinstance(node, ast.UnaryOp):
        v = _ev(node.operand, ctx); return -v if isinstance(node.op, ast.USub) else v
    a, b = _ev(node.left, ctx), _ev(node.right, ctx)
    if isinstance(node.op, ast.Add): return a + b
    if isinstance(node.op, ast.Sub): return a - b
    if isinstance(node.op, ast.Mult): return a * b
    if b == 0: raise ExprError("division by zero")
    return a / b


def evaluate(expr: str, ctx: dict) -> float:
    try: v = _ev(parse(expr), ctx)
    except OverflowError: raise ExprError("overflow")
    if not math.isfinite(v): raise ExprError("result is not finite")
    return v
