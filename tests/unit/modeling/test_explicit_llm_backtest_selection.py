import pandas as pd
import pytest

from scripts.backtest_llm_case_rag import (
    _parse_company_ids,
    _select_explicit_companies,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "company_id": ["00300", "00500", "00700"],
            "company_name": ["A", "B", "C"],
            "prospectus_date": ["2024-01-01", "2024-02-01", "2025-01-01"],
            "split": ["validation", "validation", "test"],
            "target_significant_drop_5d": [0, 1, 0],
        }
    )


def test_explicit_selection_preserves_requested_order() -> None:
    selected = _select_explicit_companies(
        _frame(), split="validation", company_ids=("00500", "00300")
    )

    assert selected["company_id"].tolist() == ["00500", "00300"]


def test_explicit_selection_rejects_company_outside_split() -> None:
    with pytest.raises(ValueError, match="00700"):
        _select_explicit_companies(
            _frame(), split="validation", company_ids=("00700",)
        )


def test_company_id_parser_normalizes_and_deduplicates() -> None:
    assert _parse_company_ids("300,00300,700.HK") == ("00300", "00700")
