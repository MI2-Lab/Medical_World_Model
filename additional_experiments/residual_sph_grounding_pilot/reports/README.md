# Report contract

No scientific outcome is recorded here until the preregistered representation matrix, freeze audit, and post-freeze evaluation have completed.

The formal deliverable is `final_report.md`, written in Chinese. It must answer, separately and without substituting sensitivity results for the primary arm:

1. Was LOCAL response performance preserved?
2. Did raw-SPH grounding work?
3. Did residual-SPH grounding work?
4. Was residual grounding better than raw grounding?
5. Did S2 improve FTV-independent morphology representation?
6. What happened to static FTV?
7. What happened to observed delta FTV?
8. Did MRI-only pCR improve?
9. Did MRI add beyond clinical variables?
10. Did MRI add beyond clinical+FTV?
11. Is SPH worth retaining as an auxiliary target?
12. Is five-seed confirmation justified?

The report must link aggregate target-residualization, optimization-safety, FTV, observed-delta-FTV, SPH/SPH-res, pCR, seed-effect, and paired-bootstrap tables. It must state all four gates and the classification hierarchy, distinguish minimum from strong-form passes, label T3 late/pre-surgery, and state that all pCR analyses occurred after representation freeze. Missing or incomplete computation remains `NOT_RUN`/`INCOMPLETE`; it must not be narrated as a negative result.

No patient identifier, row-level prediction, feature tensor, checkpoint, source-workbook content, raw MRI, or private absolute path may appear in a public report.

