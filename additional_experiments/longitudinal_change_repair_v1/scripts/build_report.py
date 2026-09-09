#!/usr/bin/env python3
"""Build a Chinese aggregate-only report after analysis has supplied aggregate.csv."""
from __future__ import annotations
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main() -> None:
 report=ROOT/"publication/REPORT_ZH.md"; report.parent.mkdir(exist_ok=True)
 report.write_text("# 纵向学习修复实验报告\n\n本报告只发布汇总指标；患者级结果保留在私有目录。\n\n"
 "实验按冻结方案区分已观察变化识别与未来变化预测。Stage F 仅在 Stage R 验证门通过后执行。\n\n"
 "请以 `metrics/aggregate.csv` 和 `metrics/paired_bootstrap.csv` 的已生成结果补充结论；不得用 pCR 选择模型。\n",encoding="utf-8")
 print(report)
if __name__=="__main__":main()
