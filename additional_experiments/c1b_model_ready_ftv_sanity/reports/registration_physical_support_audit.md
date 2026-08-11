# Registration physical-support sensitivity

Localization support只在全部image-only transform拟合完成后打开，用于独立QC；从未进入registration objective、ROI或初始化。

- C1B-H exact containment：97.800%；C1B-R（失败pair按预冻结identity/header fallback）为98.267%，差值+0.467%。
- FTV retention Q05：H=1.000，R=1.000。
- registration失败/identity fallback：267/1125；失败transform从不采样。
- R相对H exact containment下降不超过0.5 point：PASS；R retention Q05 >=0.95：PASS。
- lesion centroid displacement仅为post-hoc apparent-motion audit，不用于拟合或选择phase。与独立whole-anatomy residual合并后，`anatomy >5 mm && localization <2 mm` 为22/858（2.564%）evaluable successful transforms，因此R-specific residual gate按fail-closed规则为FAIL；最终策略为C1B-H。
