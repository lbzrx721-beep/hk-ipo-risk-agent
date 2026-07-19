from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from ipo_risk.agents.verifier import verify_finding
from ipo_risk.pipeline import analyze_prospectus
from ipo_risk.storage.files import read_json


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PDF_PATH = (
    _PROJECT_ROOT
    / "data"
    / "raw"
    / "prospectus"
    / "03696_英矽智能"
    / "招股章程_2025-12-18_繁体中文.pdf"
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _PDF_PATH.exists(), reason="公开招股书样本不存在"),
]


def test_pdf_to_grounded_risk_json(tmp_path: Path) -> None:
    company_info = read_json(
        _PROJECT_ROOT
        / "data"
        / "raw"
        / "company_info"
        / "03696_英矽智能"
        / "公司信息.json"
    )
    artifacts = analyze_prospectus(
        pdf_path=_PDF_PATH,
        company_id="03696",
        company_name="英矽智能",
        company_folder="03696_英矽智能",
        document_id="prospectus_2025-12-18",
        available_at=date(2025, 12, 18),
        project_root=_PROJECT_ROOT,
        interim_root=tmp_path / "interim",
        artifacts_root=tmp_path / "artifacts",
        risk_rules_path=_PROJECT_ROOT / "configs" / "risks.yaml",
        min_findings=5,
        expected_source_hash=(
            "c44e3c20c02747bb8768eefedd8703ff841b242d1b8cb96adc50b18b5a51809b"
        ),
        company_info=company_info,
    )

    report = artifacts.report
    assert report.source_document.page_count == 672
    assert report.source_document.chunk_count > 5_000
    assert report.summary.finding_count >= 5
    assert report.summary.evidence_count == report.summary.finding_count
    assert report.summary.screenshot_count == report.summary.evidence_count
    assert all(evidence.quote for evidence in report.evidences)
    assert all(evidence.page >= 1 for evidence in report.evidences)
    assert all(evidence.printed_page_label for evidence in report.evidences)
    assert all(
        evidence.screenshot_path and Path(evidence.screenshot_path).is_file()
        for evidence in report.evidences
    )
    assert all(finding.evidence_ids for finding in report.risk_findings)

    issuance_finding = next(
        finding
        for finding in report.risk_findings
        if finding.risk_type == "issuance_structure"
    )
    assert issuance_finding.calculation is not None
    assert issuance_finding.calculation.formula == (
        "cornerstone_shares / offer_shares * 100"
    )
    assert issuance_finding.calculation.inputs == {
        "cornerstone_shares": 37_204_500,
        "offer_shares": 94_690_500,
    }
    assert issuance_finding.calculation.result == pytest.approx(39.2906363486)
    assert issuance_finding.calculation.unit == "percent"
    assert issuance_finding.calculation.period == "2025-12-29"
    assert issuance_finding.calculation.source_refs == {
        "cornerstone_shares": "company_info:offering.cornerstone_shares",
        "offer_shares": "company_info:offering.offer_shares",
    }
    assert verify_finding(
        issuance_finding,
        report.evidences,
        project_root=_PROJECT_ROOT,
    ).passed

    payload = read_json(artifacts.report_path)
    assert payload["summary"]["finding_count"] >= 5
    assert len(payload["evidence_display_zh_hans"]) == payload["summary"]["evidence_count"]
    assert artifacts.pages_path.is_file()
    assert artifacts.chunks_path.is_file()
    assert artifacts.pages_path.parent.name == "prospectus_2025-12-18"
    assert artifacts.chunks_path.parent.name == "prospectus_2025-12-18"
    assert artifacts.report_path.parent.name == "prospectus_2025-12-18"
