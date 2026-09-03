# 报告状态

`final_report.md` 由 aggregate-only generator 生成。本实验的正式 C1–C5 pCR 矩阵、两 seed、五折评估和 fold-safe fusion 已完成；报告只使用本矩阵结果，不能把前置实验结果冒充为本实验结果。重算报告使用：

```bash
python scripts/evaluate.py --predictions predictions --population full_808
python scripts/decision.py
python scripts/generate_report.py
```

交付分支、parent SHA、experiment commit SHA 和 push 状态记录在
`delivery_provenance.json`。
