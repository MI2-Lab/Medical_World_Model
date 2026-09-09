# Frozen bilateral DINO cache readout amendment

This is a pre-specified, low-cost diagnostic amendment to the bilateral ceiling
experiment. It does not change the original V1 decision and does not train a
DINO adapter. It tests whether the locked bilateral cache contains pCR signal
or signal beyond the locked V6 radiomics target.

The readout uses outer-fold-only preprocessing, three-fold inner cross-fitting,
and the existing clinical+FTV offset-fusion contract. No new DINO extraction,
PyRadiomics extraction, morphology audit, or foundation-model swap is allowed.
