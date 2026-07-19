from __future__ import annotations

from ipo_risk.calculations import (
    CalculationInput,
    CalculationSpec,
    build_calculation,
)


def test_declarative_calculation_uses_values_and_lineage_from_context() -> None:
    spec = CalculationSpec(
        tool="calculate_percentage",
        formula="part / total * 100",
        inputs={
            "part": CalculationInput(
                path="offering.part",
                source_ref="company_info:offering.part",
            ),
            "total": CalculationInput(
                path="offering.total",
                source_ref="company_info:offering.total",
            ),
        },
        unit="percent",
        period_path="as_of_date",
    )

    calculation = build_calculation(
        spec,
        {
            "offering": {"part": 20, "total": 80},
            "as_of_date": "2025-12-29",
        },
    )

    assert calculation.inputs == {"part": 20, "total": 80}
    assert calculation.result == 25.0
    assert calculation.period == "2025-12-29"
    assert calculation.source_refs == {
        "part": "company_info:offering.part",
        "total": "company_info:offering.total",
    }
