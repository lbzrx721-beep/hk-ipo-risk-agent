"""证据存在性、原文、页码、坐标和文件版本的确定性核验。"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

import fitz

from ipo_risk.calculations import reproduce_calculation
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


def _layout_fragment_match(
    quote: str,
    clipped_text: str,
) -> tuple[bool, str] | None:
    """Match multiline layout text without trusting PDF reading order.

    Tables, diagrams and two-column definitions can contain the correct text
    inside the declared bbox while PyMuPDF emits columns in a different order.
    This fallback remains strict: at least three distinct source fragments and
    90% of their characters must be present inside the evidence bbox.
    """
    fragments = tuple(
        dict.fromkeys(
            fragment for line in quote.splitlines() if len(fragment := _normalized_text(line)) >= 2
        )
    )
    if len(fragments) < 3:
        return None
    normalized_clip = _normalized_text(clipped_text)
    matched = tuple(fragment for fragment in fragments if fragment in normalized_clip)
    total_chars = sum(len(fragment) for fragment in fragments)
    matched_chars = sum(len(fragment) for fragment in matched)
    coverage = matched_chars / total_chars if total_chars else 0.0
    passed = len(matched) >= 3 and matched_chars >= min(20, total_chars) and coverage >= 0.9
    detail = (
        f"layout fragments found in source bbox: {len(matched)}/{len(fragments)} "
        f"({coverage:.0%} character coverage)"
        if passed
        else (
            f"layout fragment coverage is insufficient: "
            f"{len(matched)}/{len(fragments)} ({coverage:.0%})"
        )
    )
    return passed, detail


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

        if evidence.evidence_type.value == "table" and evidence.table_cell_values:
            clipped_text = _normalized_text(page.get_textbox(evidence_rect))
            required_values = tuple(
                value for value in evidence.table_cell_values if _normalized_text(value)
            )
            missing_values = tuple(
                value for value in required_values if _normalized_text(value) not in clipped_text
            )
            quote_present = bool(required_values) and not missing_values
            quote_detail = (
                f"{len(required_values)} table cells found in source bbox"
                if quote_present
                else f"table cells missing from source bbox: {missing_values[:5]}"
            )
        else:
            page_text = _normalized_text(page.get_text("text"))
            quote = _normalized_text(evidence.quote)
            quote_present = bool(quote) and quote in page_text
            if quote_present:
                quote_detail = "quote found on physical PDF page"
            else:
                layout_match = _layout_fragment_match(
                    evidence.quote,
                    page.get_textbox(evidence_rect),
                )
                if layout_match is None:
                    quote_detail = "quote not found"
                else:
                    quote_present, quote_detail = layout_match
        checks.append(
            VerificationCheck(
                name="quote",
                passed=quote_present,
                detail=quote_detail,
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


def _verify_calculation(finding: RiskFinding) -> VerificationCheck:
    calculation = finding.calculation
    if calculation is None:
        return VerificationCheck(
            name="calculation",
            passed=True,
            detail=(
                "not applicable: disclosed facts and quoted numbers are verified "
                "through their evidence references"
            ),
        )
    try:
        expected, reproduced = reproduce_calculation(calculation)
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
        evidence_id for evidence_id in finding.evidence_ids if evidence_id not in evidence_by_id
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
