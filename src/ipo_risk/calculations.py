"""声明式数值计算：从结构化资料取值并生成可复算 Calculation。"""

from __future__ import annotations

import ast
import math
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from ipo_risk.domain import Calculation


class CalculationInput(BaseModel):
    """一个公式输入在结构化资料中的路径及其审计来源。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    path: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)


class CalculationSpec(BaseModel):
    """风险规则声明的公式、输入、单位和期间来源。"""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    tool: str = Field(min_length=1)
    formula: str = Field(min_length=1)
    inputs: dict[str, CalculationInput] = Field(min_length=1)
    unit: str = Field(min_length=1)
    period_path: str = Field(min_length=1)


_JSON_INPUTS = TypeAdapter(dict[str, JsonValue])


def _numeric(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def evaluate_formula(formula: str, inputs: Mapping[str, object]) -> float:
    """只允许常量、已声明输入和四则运算的安全公式求值。"""
    tree = ast.parse(formula, mode="eval")

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            return _numeric(node.value, "formula constant")
        if isinstance(node, ast.Name):
            if node.id not in inputs:
                raise ValueError(f"formula input is missing: {node.id}")
            return _numeric(inputs[node.id], node.id)
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)
            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return -value
        raise ValueError(f"unsupported formula element: {type(node).__name__}")

    result = evaluate(tree)
    if not math.isfinite(result):
        raise ValueError("formula result must be finite")
    return result


def _resolve_path(source: Mapping[str, object], path: str) -> object:
    value: object = source
    for segment in path.split("."):
        if not isinstance(value, Mapping) or segment not in value:
            raise ValueError(f"calculation source path is missing: {path}")
        value = value[segment]
    return value


def build_calculation(
    spec: CalculationSpec,
    source: Mapping[str, object],
) -> Calculation:
    """按声明路径解析输入并返回包含来源的可复算计算对象。"""
    inputs = _JSON_INPUTS.validate_python(
        {
            name: _resolve_path(source, input_spec.path)
            for name, input_spec in spec.inputs.items()
        }
    )
    result = evaluate_formula(spec.formula, inputs)
    period = str(_resolve_path(source, spec.period_path)).strip()
    if not period:
        raise ValueError(f"calculation period is empty: {spec.period_path}")
    return Calculation(
        tool=spec.tool,
        formula=spec.formula,
        inputs=inputs,
        result=result,
        unit=spec.unit,
        period=period,
        source_refs={
            name: input_spec.source_ref
            for name, input_spec in spec.inputs.items()
        },
    )


def reproduce_calculation(calculation: Calculation) -> tuple[float, float]:
    """返回声明结果和安全公式复算结果，供核验模块比较。"""
    expected = _numeric(calculation.result, "calculation result")
    reproduced = evaluate_formula(calculation.formula, calculation.inputs)
    return expected, reproduced


__all__ = [
    "CalculationInput",
    "CalculationSpec",
    "build_calculation",
    "evaluate_formula",
    "reproduce_calculation",
]
