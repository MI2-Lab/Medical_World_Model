# 报告状态

`final_report.md` 由 aggregate-only generator 生成。正式 C1–C5 pCR 矩阵尚未运行时，报告必须保持 `NOT_RUN`，不能把前置实验结果冒充为本实验结果。运行后使用：

```bash
python scripts/evaluate.py --predictions predictions --population full_808
python scripts/decision.py
python scripts/generate_report.py
```

