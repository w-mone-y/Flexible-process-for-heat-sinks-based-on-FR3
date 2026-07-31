"""Safe arithmetic duration models evaluated from capability parameters.

Capability definitions carry a ``duration_model`` expression such as
``"2.4 + 0.9 * path_count"`` instead of a hard-coded constant.  The expression
is parsed once with :mod:`ast` and evaluated against validated operation
parameters, so a routing YAML can change process tempo without touching Python
and without ever handing arbitrary text to :func:`eval`.
"""

from __future__ import annotations

import ast
from math import ceil, floor, isfinite
from typing import Any, Mapping

_ALLOWED_FUNCTIONS: dict[str, Any] = {
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "ceil": lambda value: float(ceil(value)),
    "floor": lambda value: float(floor(value)),
}

_ALLOWED_BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_ALLOWED_UNARY = (ast.UAdd, ast.USub)


class DurationModelError(ValueError):
    """Raised when a duration expression is malformed or references unknowns."""


class DurationModel:
    """A compiled, side-effect-free arithmetic expression over named parameters."""

    __slots__ = ("expression", "_tree", "_names")

    def __init__(self, expression: str, *, allowed_names: frozenset[str] | None = None) -> None:
        self.expression = str(expression).strip()
        if not self.expression:
            raise DurationModelError("节拍表达式不能为空")
        try:
            tree = ast.parse(self.expression, mode="eval")
        except SyntaxError as exc:
            raise DurationModelError(f"节拍表达式语法错误：{exc.msg}") from exc
        names: set[str] = set()
        self._validate(tree.body, names)
        if allowed_names is not None:
            unknown = names - set(allowed_names)
            if unknown:
                raise DurationModelError(
                    f"节拍表达式引用了未声明的参数：{sorted(unknown)}，"
                    f"已声明参数为 {sorted(allowed_names)}"
                )
        self._tree = tree
        self._names = frozenset(names)

    @property
    def parameter_names(self) -> frozenset[str]:
        """Parameter names the expression depends on."""

        return self._names

    @classmethod
    def _validate(cls, node: ast.AST, names: set[str]) -> None:
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINARY):
                raise DurationModelError(f"不支持的运算符：{type(node.op).__name__}")
            cls._validate(node.left, names)
            cls._validate(node.right, names)
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARY):
                raise DurationModelError(f"不支持的一元运算符：{type(node.op).__name__}")
            cls._validate(node.operand, names)
            return
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise DurationModelError("节拍表达式只允许数值常量")
            return
        if isinstance(node, ast.Name):
            names.add(node.id)
            return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
                allowed = sorted(_ALLOWED_FUNCTIONS)
                raise DurationModelError(f"只允许调用 {allowed} 中的函数")
            if node.keywords:
                raise DurationModelError("节拍表达式函数不支持关键字参数")
            for argument in node.args:
                cls._validate(argument, names)
            return
        raise DurationModelError(f"节拍表达式包含不允许的语法节点：{type(node).__name__}")

    def evaluate(self, params: Mapping[str, Any]) -> float:
        """Evaluate the expression and return a finite, non-negative duration."""

        missing = self._names - set(params)
        if missing:
            raise DurationModelError(f"缺少节拍参数：{sorted(missing)}")
        value = self._evaluate(self._tree.body, params)
        result = float(value)
        if not isfinite(result):
            raise DurationModelError(f"节拍表达式 {self.expression!r} 求值结果不是有限数值")
        if result < 0.0:
            raise DurationModelError(f"节拍表达式 {self.expression!r} 求值结果为负：{result}")
        return result

    @classmethod
    def _evaluate(cls, node: ast.AST, params: Mapping[str, Any]) -> float:
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            raw = params[node.id]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise DurationModelError(f"节拍参数 {node.id} 必须是数值，实际 {type(raw).__name__}")
            return float(raw)
        if isinstance(node, ast.UnaryOp):
            operand = cls._evaluate(node.operand, params)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)  # guaranteed by _validate
            arguments = [cls._evaluate(item, params) for item in node.args]
            try:
                return float(_ALLOWED_FUNCTIONS[node.func.id](*arguments))
            except TypeError as exc:
                raise DurationModelError(f"函数 {node.func.id} 参数不合法：{exc}") from exc
        if isinstance(node, ast.BinOp):
            left = cls._evaluate(node.left, params)
            right = cls._evaluate(node.right, params)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and abs(right) < 1e-12:
                raise DurationModelError("节拍表达式出现除以零")
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return float(left // right)
            if isinstance(node.op, ast.Mod):
                return float(left % right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > 8.0:
                    raise DurationModelError("节拍表达式指数过大")
                return float(left**right)
        raise DurationModelError(f"节拍表达式包含不允许的语法节点：{type(node).__name__}")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"DurationModel({self.expression!r})"


__all__ = ["DurationModel", "DurationModelError"]
