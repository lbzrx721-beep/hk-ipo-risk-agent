"""东吴证券赛事官方数据的检查、标准化和标签生成。"""

from .ingestion import (
    CompetitionImportResult,
    CompetitionProspectus,
    MarketObservation,
    audit_competition_delivery,
    build_horizon_labels,
    discover_competition_root,
    import_competition_delivery,
    materialize_delivery_subset,
    normalize_stock_code,
    parse_prospectus_filename,
    reconcile_backed_up_market_dataset,
)

__all__ = [
    "CompetitionImportResult",
    "CompetitionProspectus",
    "MarketObservation",
    "audit_competition_delivery",
    "build_horizon_labels",
    "discover_competition_root",
    "import_competition_delivery",
    "materialize_delivery_subset",
    "normalize_stock_code",
    "parse_prospectus_filename",
    "reconcile_backed_up_market_dataset",
]
