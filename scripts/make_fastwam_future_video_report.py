#!/usr/bin/env python3
"""Build a small HTML report for FastWAM LIBERO future-video artifacts."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_result_json(root: Path) -> Path | None:
    candidates = sorted(root.rglob("*_results.json"))
    return candidates[0] if candidates else None


def _metric_rows(result: dict[str, Any]) -> str:
    if not result:
        return "<tr><td colspan=\"2\">No result JSON found.</td></tr>"
    keys = [
        "successes",
        "num_trials",
        "success_rate",
        "future_video_psnr_mean",
        "task_description",
    ]
    rows = []
    for key in keys:
        if key in result:
            rows.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(result[key]))}</td></tr>")
    if not rows:
        rows.append("<tr><td colspan=\"2\">Result JSON has no known summary keys.</td></tr>")
    return "\n".join(rows)


def build_report(output_dir: Path, report_path: Path) -> None:
    output_dir = output_dir.resolve()
    result_path = _find_result_json(output_dir)
    result = _load_json(result_path) if result_path is not None else {}
    rollout_videos = sorted(output_dir.rglob("videos/*.mp4"))
    predicted_videos = sorted(output_dir.rglob("predicted_videos/*gt-pred.mp4"))
    all_clip = [p for p in predicted_videos if "--replan=all--" in p.name]
    step_clips = [p for p in predicted_videos if p not in all_clip]

    def video_card(path: Path, label: str) -> str:
        rel = html.escape(_rel(path, report_path.parent))
        return f"""
        <section class="card">
          <h2>{html.escape(label)}</h2>
          <video controls muted loop preload="metadata" src="{rel}"></video>
          <p><code>{html.escape(path.name)}</code></p>
        </section>
        """

    rollout_html = "\n".join(video_card(p, "Rollout observation") for p in rollout_videos)
    all_html = "\n".join(video_card(p, "Future video: all replans") for p in all_clip)
    step_html = "\n".join(video_card(p, f"Future video clip {i}") for i, p in enumerate(step_clips))

    if not rollout_html:
        rollout_html = "<p>No rollout videos found.</p>"
    if not all_html and not step_html:
        all_html = "<p>No predicted gt-pred videos found.</p>"

    result_link = ""
    if result_path is not None:
        result_link = f"<p>Result JSON: <code>{html.escape(_rel(result_path, report_path.parent))}</code></p>"

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FastWAM LIBERO Future Video</title>
  <style>
    html, body {{ margin: 0; background: #f7f7f8; color: #18181b; font: 14px system-ui, sans-serif; }}
    header {{ padding: 18px 22px; background: #171923; color: #fff; }}
    h1 {{ font-size: 20px; margin: 0 0 6px; font-weight: 650; }}
    main {{ padding: 18px 22px 40px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; align-items: start; }}
    .card {{ background: #fff; border: 1px solid #dddfe6; border-radius: 8px; padding: 12px; }}
    .card h2 {{ font-size: 15px; margin: 0 0 10px; }}
    video {{ width: 100%; background: #000; border-radius: 4px; }}
    code {{ font-size: 12px; color: #3f3f46; overflow-wrap: anywhere; }}
    table {{ border-collapse: collapse; background: #fff; border: 1px solid #dddfe6; margin-bottom: 16px; }}
    th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #eceef3; vertical-align: top; }}
    th {{ width: 210px; color: #52525b; }}
  </style>
</head>
<body>
  <header>
    <h1>FastWAM LIBERO Future Video</h1>
    <div>Top row in FastWAM gt-pred clips is prediction; bottom row is ground truth.</div>
  </header>
  <main>
    <table>
      {_metric_rows(result)}
    </table>
    {result_link}
    <div class="grid">
      {rollout_html}
      {all_html}
    </div>
    <div class="grid">
      {step_html}
    </div>
  </main>
</body>
</html>
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    report = args.report or (args.output_dir / "future_video_report.html")
    build_report(args.output_dir, report)
    print(report)


if __name__ == "__main__":
    main()
