"""无人值守批量生成训练所需的Agent分析JSON。"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from threading import Event, Thread

from ipo_risk.experiments import run_agent_feature_experiment
from ipo_risk.experiments.unattended import (
    select_unattended_companies,
    write_agent_training_dataset,
)


def _default_experiment_name() -> str:
    return datetime.now().astimezone().strftime("agent_train_auto_%Y%m%dT%H%M%S")


def _status_counts(path: Path) -> tuple[int, int, int, int, str] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    companies = payload.get("companies")
    if not isinstance(companies, dict):
        return None
    states = [
        str(item.get("state"))
        for item in companies.values()
        if isinstance(item, dict)
    ]
    return (
        len(states),
        states.count("completed"),
        states.count("failed"),
        sum(state in {"running", "preparing"} for state in states),
        str(payload.get("queue_state") or "pending"),
    )


def _monitor_status(path: Path, stop: Event) -> None:
    last: tuple[int, int, int, int, str] | None = None
    while not stop.wait(3):
        current = _status_counts(path)
        if current is None or current == last:
            continue
        total, completed, failed, running, queue_state = current
        print(
            f"进度：完成{completed}/{total} · 运行中{running} · "
            f"失败{failed} · 队列{queue_state}",
            flush=True,
        )
        last = current


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="自动跳过人工范围确认并批量生成Agent训练JSON。"
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--company-concurrency", type=int, default=2)
    parser.add_argument("--min-findings", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.project_root.resolve()
    experiment_name = args.experiment_name or _default_experiment_name()
    selected = select_unattended_companies(
        root,
        split=args.split,
        limit=args.limit,
    )
    if not selected:
        print("没有符合条件的未分析公司。", flush=True)
        return 0

    print(
        f"实验：{experiment_name} · 分组{args.split} · "
        f"选择{len(selected)}家 · 人工确认：自动跳过",
        flush=True,
    )
    for index, row in enumerate(selected, start=1):
        print(
            f"  {index:02d}. {row['company_id']} · "
            f"{row.get('company_name') or row['company_folder']}",
            flush=True,
        )
    if args.dry_run:
        return 0

    experiment_dir = root / "artifacts" / "experiments" / experiment_name
    status_path = experiment_dir / "status.json"
    stop_monitor = Event()
    monitor = Thread(
        target=_monitor_status,
        args=(status_path, stop_monitor),
        name=f"monitor-{experiment_name}",
        daemon=True,
    )
    monitor.start()
    try:
        result = run_agent_feature_experiment(
            root,
            experiment_name=experiment_name,
            splits=(args.split,),
            company_folders=tuple(
                str(row["company_folder"]) for row in selected
            ),
            min_findings=args.min_findings,
            continue_on_error=True,
            risk_scope_review_controllers=None,
            agent_company_concurrency=max(1, args.company_concurrency),
        )
    finally:
        stop_monitor.set()
        monitor.join(timeout=5)

    dataset_path = write_agent_training_dataset(result.experiment_dir)
    print(
        f"实验结束：成功{result.completed_count}家，失败{result.failed_count}家。",
        flush=True,
    )
    print(f"最终JSON：{dataset_path}", flush=True)
    print(f"逐公司JSON：{result.experiment_dir / 'companies'}", flush=True)
    print(f"状态文件：{result.status_path}", flush=True)
    return 0 if result.failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
