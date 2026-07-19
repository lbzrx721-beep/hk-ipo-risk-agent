"""证据存在性、原文、页码、坐标和文件版本的确定性核验。"""

from __future__ import annotations

import ast
import math
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

import fitz

from ipo_risk.domain import (
    Evidence,
    EvidenceVerification,
    FindingVerification,
    RiskFinding,
    VerificationCheck,
)
from ipo_risk.storage.files import sha256_file


def _normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", "", normalized)


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _result(evidence_id: str, checks: list[VerificationCheck]) -> EvidenceVerification:
    return EvidenceVerification(
        evidence_id=evidence_id,
        passed=all(check.passed for check in checks),
        checks=tuple(checks),
    )


def verify_evidence(evidence: Evidence, *, project_root: Path) -> EvidenceVerification:
    """Reproduce one evidence location without changing source files or artifacts."""
    source_path = _resolve_path(evidence.source_file, project_root)
    source_exists = source_path.is_file()
    checks = [
        VerificationCheck(
            name="source_exists",
            passed=source_exists,
            detail=str(source_path) if source_exists else f"missing source file: {source_path}",
        )
    ]
    if not source_exists:
        return _result(evidence.evidence_id, checks)

    actual_hash = sha256_file(source_path)
    checks.append(
        VerificationCheck(
            name="source_hash",
            passed=actual_hash == evidence.source_hash,
            detail=f"expected={evidence.source_hash}; actual={actual_hash}",
        )
    )

    try:
        document = fitz.open(source_path)
    except Exception as exc:
        checks.append(
            VerificationCheck(
                name="page_range",
                passed=False,
                detail=f"cannot open source PDF: {exc}",
            )
        )
        return _result(evidence.evidence_id, checks)

    try:
        page_in_range = evidence.page <= document.page_count
        checks.append(
            VerificationCheck(
                name="page_range",
                passed=page_in_range,
                detail=f"page={evidence.page}; page_count={document.page_count}",
            )
        )
        if not page_in_range:
            return _result(evidence.evidence_id, checks)

        page = document[evidence.page - 1]
        evidence_rect = fitz.Rect(evidence.bbox)
        bbox_valid = page.rect.contains(evidence_rect)
        checks.append(
            VerificationCheck(
                name="bbox",
                passed=bbox_valid,
                detail=f"bbox={tuple(evidence_rect)}; page_rect={tuple(page.rect)}",
            )
        )

        page_text = _normalized_text(page.get_text("text"))
        quote = _normalized_text(evidence.quote)
        quote_present = bool(quote) and quote in page_text
        checks.append(
            VerificationCheck(
                name="quote",
                passed=quote_present,
                detail="quote found on physical PDF page" if quote_present else "quote not found",
            )
        )
    finally:
        document.close()

    if evidence.screenshot_path is not None:
        screenshot_path = _resolve_path(evidence.screenshot_path, project_root)
        screenshot_exists = screenshot_path.is_file() and screenshot_path.stat().st_size > 0
        checks.append(
            VerificationCheck(
                name="screenshot_exists",
                passed=screenshot_exists,
                detail=(
                    str(screenshot_path)
                    if screenshot_exists
                    else f"missing screenshot: {screenshot_path}"
                ),
            )
        )

    return _result(evidence.evidence_id, checks)


_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
}
_UNARY_OPERATORS = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}


def _numeric(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _evaluate_formula(formula: str, inputs: dict[str, object]) -> float:
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
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            binary_operator = _BINARY_OPERATORS[type(node.op)]
            return binary_operator(evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            unary_operator = _UNARY_OPERATORS[type(node.op)]
            return unary_operator(evaluate(node.operand))
        raise ValueError(f"unsupported formula element: {type(node).__name__}")

    result = evaluate(tree)
    if not math.isfinite(result):
        raise ValueError("formula result must be finite")
    return result


def _verify_calculation(finding: RiskFinding) -> VerificationCheck:
    has_numeric_claim = re.search(r"\d", finding.claim) is not None
    calculation = finding.calculation
    if calculation is None:
        return VerificationCheck(
            name="calculation",
            passed=not has_numeric_claim,
            detail=(
                "qualitative claim does not require a calculation"
                if not has_numeric_claim
                else "numeric claim is missing a reproducible calculation"
            ),
        )
    try:
        reproduced = _evaluate_formula(calculation.formula, dict(calculation.inputs))
        expected = _numeric(calculation.result, "calculation result")
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return VerificationCheck(name="calculation", passed=False, detail=str(exc))
    matches = math.isclose(reproduced, expected, rel_tol=1e-9, abs_tol=1e-9)
    return VerificationCheck(
        name="calculation",
        passed=matches,
        detail=f"expected={expected}; reproduced={reproduced}",
    )


def verify_finding(
    finding: RiskFinding,
    evidences: Sequence[Evidence],
    *,
    project_root: Path,
) -> FindingVerification:
    """Verify references, source locations, company ownership and arithmetic."""
    evidence_by_id = {evidence.evidence_id: evidence for evidence in evidences}
    missing = [
        evidence_id
        for evidence_id in finding.evidence_ids
        if evidence_id not in evidence_by_id
    ]
    referenced = [
        evidence_by_id[evidence_id]
        for evidence_id in finding.evidence_ids
        if evidence_id in evidence_by_id
    ]
    checks = [
        VerificationCheck(
            name="evidence_references",
            passed=not missing,
            detail="all evidence references resolve" if not missing else f"missing={missing}",
        ),
        VerificationCheck(
            name="evidence_company",
            passed=all(evidence.company_id == finding.company_id for evidence in referenced),
            detail=f"expected company={finding.company_id}",
        ),
        _verify_calculation(finding),
    ]
    evidence_results = tuple(
        verify_evidence(evidence, project_root=project_root) for evidence in referenced
    )
    passed = all(check.passed for check in checks) and all(
        result.passed for result in evidence_results
    )
    return FindingVerification(
        finding_id=finding.finding_id,
        passed=passed,
        checks=tuple(checks),
        evidence_verifications=evidence_results,
    )


__all__ = ["verify_evidence", "verify_finding"]
