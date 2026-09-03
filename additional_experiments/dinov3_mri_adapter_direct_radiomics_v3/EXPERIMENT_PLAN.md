# Frozen experiment protocol V3

V3 decouples direct residual-radiomics grounding from FTV grounding. The C0
baseline uses JEPA+SIGReg. R025/R050/R100 clone the selected C0 checkpoint and
add only radiomics SmoothL1 with weights 0.25/0.50/1.00. Pilot selection uses no
outcome and chooses the smallest weight passing all preregistered gates.

Mechanism evaluation includes both the trained direct head and matched
outer-fold Ridge probes fitted to the 192-D state plus visit one-hot. The
matched probes are trained identically for baseline and candidate so direct
head training cannot masquerade as representation transfer. FTV is assessed
only through a matched diagnostic probe and remains part of the downstream
clinical+FTV baseline.

Pilot seed 2026 is development-only. Formal confirmation uses fresh seeds
7026/8026/9026/10026/11026 and 50 C0/RAD cells. No pCR-bearing file may be
opened before the complete representation matrix and outcome-blind mechanism
gate have each been hash-locked.
