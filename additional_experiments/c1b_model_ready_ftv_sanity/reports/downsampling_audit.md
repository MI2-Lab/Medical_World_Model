# C1B fixed-grid downsampling audit

## 结论

C1B保持统一 `0.9/0.9/2.0 mm` spacing与`112x176x160 ZYX` grid，不因少数outlier改变FOV、tensor shape或patient-specific scale。所有轴 factor `>1.5` 的volume在builder中先执行source-domain Gaussian anti-alias，再做一次4-D spatial linear interpolation；phase轴从不插值。

- 全model-input：597/3792 visit任一轴factor `>2`；正式FTV：35/1500。
- extreme轴计数（visit可重复计轴）：`{"X": 6, "Y": 60, "Z": 591}`。
- 全队列最大factor：X=2.880，Y=2.880，Z=5.689。
- 每个extreme visit的source spacing、axis factor、anti-alias disposition和padding均保存在private QC表；统一处置为`ANTIALIAS_THEN_LINEAR_FIXED_GRID`，没有静默忽略或动态扩大tensor。

是否存在catastrophic case由完整builder finite/nonconstant/cache验收最终合并判定；这里不通过模型performance更改spacing。
