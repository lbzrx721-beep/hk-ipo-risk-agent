"""证据与风险核验的公共行为测试。"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import fitz

from ipo_risk.agents.verifier import verify_evidence, verify_finding
from ipo_risk.domain import (
    AgentRole,
    Calculation,
    Evidence,
    EvidenceType,
    RiskFinding,
    RiskSeverity,
    VerificationStatus,
)
from ipo_risk.storage.files import sha256_file


def test_valid_evidence_can_be_reproduced_from_its_source_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    page.insert_text((50, 60), "Auditable evidence quote for 2025")
    document.save(pdf_path)
    document.close()

    evidence = Evidence(
        evidence_id="E-TEST-0001",
        company_id="TEST",
        document_id="doc_test_prospectus_2025-01-01_aaaaaaaa",
        source_file=pdf_path.name,
        source_hash=sha256_file(pdf_path),
        page=1,
        bbox=(40.0, 40.0, 250.0, 80.0),
        quote="Auditable evidence quote for 2025",
        evidence_type=EvidenceType.TEXT,
        extractor=AgentRole.DOCUMENT_PARSER,
        confidence=0.95,
        available_at=date(2025, 1, 1),
    )

    result = verify_evidence(evidence, project_root=tmp_path)

    assert result.passed is True
    assert {check.name for check in result.checks} == {
        "source_exists",
        "source_hash",
        "page_range",
        "bbox",
        "quote",
    }


def test_structured_table_evidence_is_verified_by_its_source_cells(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "table-source.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    page.insert_text((50, 50), "Customer")
    page.insert_text((150, 50), "Revenue")
    page.insert_text((50, 75), "Customer A")
    page.insert_text((150, 75), "120")
    document.save(pdf_path)
    document.close()

    evidence = Evidence(
        evidence_id="E-TEST-TABLE-0001",
        company_id="TEST",
        document_id="doc_test_prospectus_2025-01-01_aaaaaaaa",
        source_file=pdf_path.name,
        source_hash=sha256_file(pdf_path),
        page=1,
        bbox=(40.0, 30.0, 220.0, 90.0),
        quote=(
            "[表格证据]\n|Customer|Revenue|\n|---|---|\n"
            "|Customer A|120|\n逐行语义:\n- Customer=Customer A；Revenue=120"
        ),
        evidence_type=EvidenceType.TABLE,
        extractor=AgentRole.DOCUMENT_PARSER,
        confidence=0.95,
        available_at=date(2025, 1, 1),
        table_id="T-P0001-001",
        table_cell_refs=(
            "T-P0001-001:R1:C1",
            "T-P0001-001:R1:C2",
        ),
        table_cell_values=("Customer A", "120"),
    )

    result = verify_evidence(evidence, project_root=tmp_path)

    assert result.passed is True
    quote_check = next(check for check in result.checks if check.name == "quote")
    assert "table cells" in quote_check.detail


def test_layout_text_can_be_verified_when_pdf_reading_order_differs(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "layout-source.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=260)
    page.insert_text((50, 55), "Priority offering")
    page.insert_text((50, 85), "15,000,000 reserved shares")
    page.insert_text((50, 115), "10% of the global offering")
    page.insert_text((50, 145), "Eligible shareholders")
    document.save(pdf_path)
    document.close()

    evidence = Evidence(
        evidence_id="E-TEST-LAYOUT-0001",
        company_id="TEST",
        document_id="doc_test_prospectus_2025-01-01_aaaaaaaa",
        source_file=pdf_path.name,
        source_hash=sha256_file(pdf_path),
        page=1,
        bbox=(35.0, 30.0, 300.0, 170.0),
        quote=(
            "15,000,000 reserved shares\n"
            "10% of the global offering\n"
            "Eligible shareholders\n"
            "Priority offering"
        ),
        evidence_type=EvidenceType.TEXT,
        extractor=AgentRole.DOCUMENT_PARSER,
        confidence=0.95,
        available_at=date(2025, 1, 1),
    )

    result = verify_evidence(evidence, project_root=tmp_path)

    assert result.passed is True
    quote_check = next(check for check in result.checks if check.name == "quote")
    assert "layout fragments" in quote_check.detail


def test_layout_text_still_fails_when_material_fragments_are_missing(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "incomplete-layout-source.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=260)
    page.insert_text((50, 55), "Priority offering")
    page.insert_text((50, 85), "15,000,000 reserved shares")
    document.save(pdf_path)
    document.close()

    evidence = Evidence(
        evidence_id="E-TEST-LAYOUT-0002",
        company_id="TEST",
        document_id="doc_test_prospectus_2025-01-01_aaaaaaaa",
        source_file=pdf_path.name,
        source_hash=sha256_file(pdf_path),
        page=1,
        bbox=(35.0, 30.0, 300.0, 170.0),
        quote=(
            "15,000,000 reserved shares\n"
            "10% of the global offering\n"
            "Eligible shareholders\n"
            "Priority offering"
        ),
        evidence_type=EvidenceType.TEXT,
        extractor=AgentRole.DOCUMENT_PARSER,
        confidence=0.95,
        available_at=date(2025, 1, 1),
    )

    result = verify_evidence(evidence, project_root=tmp_path)

    assert result.passed is False
    quote_check = next(check for check in result.checks if check.name == "quote")
    assert "layout fragment coverage" in quote_check.detail


def test_finding_verification_reproduces_a_numeric_calculation(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    page.insert_text((50, 60), "Cash balance 100 and annual cash burn 80")
    document.save(pdf_path)
    document.close()
    evidence = Evidence(
        evidence_id="E-TEST-0001",
        company_id="TEST",
        document_id="doc_test_prospectus_2025-01-01_aaaaaaaa",
        source_file=pdf_path.name,
        source_hash=sha256_file(pdf_path),
        page=1,
        bbox=(40.0, 40.0, 280.0, 80.0),
        quote="Cash balance 100 and annual cash burn 80",
        evidence_type=EvidenceType.TEXT,
        extractor=AgentRole.DOCUMENT_PARSER,
        confidence=0.95,
        available_at=date(2025, 1, 1),
    )
    finding = RiskFinding(
        finding_id="F-TEST-0001",
        company_id="TEST",
        risk_type="cash_burn",
        title="现金覆盖期",
        severity=RiskSeverity.HIGH,
        claim="按披露数据计算，现金覆盖期为15个月。",
        evidence_ids=(evidence.evidence_id,),
        status=VerificationStatus.PENDING_VERIFICATION,
        confidence=0.85,
        created_by=AgentRole.FINANCE,
        calculation=Calculation(
            tool="calculate_cash_runway",
            formula="cash_balance / annual_cash_burn * 12",
            inputs={"cash_balance": 100, "annual_cash_burn": 80},
            result=15.0,
            unit="month",
            period="FY2025",
        ),
    )

    result = verify_finding(finding, (evidence,), project_root=tmp_path)

    assert result.passed is True
    assert {check.name for check in result.checks} == {
        "evidence_references",
        "evidence_company",
        "calculation",
    }


def test_formula_verification_rejects_function_calls(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    page.insert_text((50, 60), "Reported value 1")
    document.save(pdf_path)
    document.close()
    evidence = Evidence(
        evidence_id="E-TEST-0001",
        company_id="TEST",
        document_id="doc_test_prospectus_2025-01-01_aaaaaaaa",
        source_file=pdf_path.name,
        source_hash=sha256_file(pdf_path),
        page=1,
        bbox=(40.0, 40.0, 180.0, 80.0),
        quote="Reported value 1",
        evidence_type=EvidenceType.TEXT,
        extractor=AgentRole.DOCUMENT_PARSER,
        confidence=0.95,
        available_at=date(2025, 1, 1),
    )
    finding = RiskFinding(
        finding_id="F-TEST-0001",
        company_id="TEST",
        risk_type="unsafe_formula",
        title="不安全公式",
        severity=RiskSeverity.HIGH,
        claim="报告数值为1。",
        evidence_ids=(evidence.evidence_id,),
        status=VerificationStatus.PENDING_VERIFICATION,
        confidence=0.85,
        created_by=AgentRole.FINANCE,
        calculation=Calculation(
            tool="unsafe",
            formula="max(value, 0)",
            inputs={"value": 1},
            result=1,
            unit="unit",
            period="FY2025",
        ),
    )

    result = verify_finding(finding, (evidence,), project_root=tmp_path)

    calculation_check = next(check for check in result.checks if check.name == "calculation")
    assert result.passed is False
    assert calculation_check.passed is False
    assert "Call" in calculation_check.detail
