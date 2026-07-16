# Included Phase Metadata

`BreastDCEDL_metadata_min_crop.csv` is copied from the public BreastDCEDL repository:

```text
https://github.com/naomifridman/BreastDCEDL
```

It is included here so DCE phase selection does not depend on a private `/home` path. The clean tensor builder reads the `pre`, `post_early`, and `post_late` fields and records the final clipped indices in every patient cache.

The upstream project distributes this material under CC BY 4.0. The upstream license is retained as `LICENSE-BreastDCEDL`.
