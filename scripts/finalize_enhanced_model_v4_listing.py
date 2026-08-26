"""固化以上市日为预测截止点的增强模型 v4。"""

from __future__ import annotations

from finalize_enhanced_model_v3 import main


if __name__ == "__main__":
    import typer

    typer.run(main)
