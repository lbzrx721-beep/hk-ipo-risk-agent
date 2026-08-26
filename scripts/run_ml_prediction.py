"""用上市日前口径的 enhanced_blend4_rf_v4 对公司做五日显著下跌预测。

用法：
  python scripts/run_ml_prediction.py --company-id 00300
  python scripts/run_ml_prediction.py --company-id 00300 --model artifacts/models/enhanced_blend4_rf_v4_listing/model.joblib

输入：agent_augmented_feature_matrix_v5_compatible.csv（已含全部派生特征）
输出：ML 融合概率、预警标志、与阈值的对比。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from ipo_risk.modeling.predict import load_model_bundle, predict_feature_rows

app = typer.Typer(add_completion=False)

DEFAULT_MODEL = "artifacts/models/enhanced_blend4_rf_v4_listing/model.joblib"
DEFAULT_MATRIX = "data/processed/modeling/agent_augmented_feature_matrix_v5_compatible.csv"


@app.command()
def main(
    company_id: str = typer.Option(..., "--company-id", help="五位港股代码"),
    project_root: Path = typer.Option(Path.cwd(), "--project-root"),
    model_path: Path = typer.Option(DEFAULT_MODEL, "--model"),
    matrix_path: Path = typer.Option(DEFAULT_MATRIX, "--matrix"),
) -> None:
    root = project_root.resolve()
    normalized = company_id.strip().zfill(5)
    bundle = load_model_bundle((root / model_path).resolve())
    frame = pd.read_csv((root / matrix_path).resolve(), dtype={"company_id": str})
    frame["company_id"] = frame["company_id"].astype(str).str.zfill(5)
    frame = frame.copy()
    dates = pd.to_datetime(frame["prospectus_date"], errors="coerce")
    frame["modeling_year"] = dates.dt.year.astype(int)

    row = frame.loc[frame["company_id"] == normalized]
    if row.empty:
        raise typer.BadParameter(f"特征矩阵中找不到公司：{normalized}")
    if len(row) > 1:
        row = row.sort_values("prospectus_date").iloc[[-1]]

    prediction = predict_feature_rows(bundle, row)
    prob = float(prediction["predicted_probability"].iloc[0])
    flag = int(prediction["predicted_significant_drop_5d"].iloc[0])
    threshold = float(bundle["threshold"])

    typer.echo(f"公司：{normalized} {row['company_name'].iloc[0]}（招股书 {row['prospectus_date'].iloc[0]}）")
    typer.echo(f"ML 模型：{bundle['model_name']}（{len(bundle['features'])} 特征）")
    typer.echo(f"五日显著下跌概率：{prob:.4f}（{prob * 100:.1f}%）")
    typer.echo(f"预警阈值：{threshold:.4f}")
    typer.echo(f"预警判定：{'⚠️ 预警' if flag else '不预警'}（概率{'≥' if flag else '<'}阈值）")
    if "predicted_decision" in prediction:
        decision = str(prediction["predicted_decision"].iloc[0])
        decision_text = {
            "high_risk": "高置信度高风险",
            "low_risk": "高置信度低风险",
            "uncertain": "不确定，转 Agent/人工复核",
        }.get(decision, decision)
        typer.echo(f"自动裁决：{decision_text}")
    typer.echo("模型训练年份：2020-2023（2024/2025 未参与拟合）")


if __name__ == "__main__":
    app()
