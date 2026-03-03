# 实验补充说明

运行方式（MATLAB）：

```matlab
run('代码/experiment_suite.m')
```

该脚本补充以下内容：

1. 窗函数对比：Hamming / Kaiser / Taylor / Chebyshev。
2. 多次随机统计：FA 与 FA+精修的 PSLR、MW、PAPR 均值/方差。
3. 收敛曲线：平均 FA 曲线与 FA+精修曲线，并打印平均运行时间。
4. 参数敏感性：`alpha`、`gamma`、`lambda_pslr`、`pslr_margin` 扫描趋势。
5. 场景变化：TBP 变化、不同采样点数、SNR=20/10/0 dB。
6. 复现实验细节：窗长度、LFM 参数、脉冲压缩方法、指标定义。

> 说明：若环境缺失 `fmincon` / `taylorwin` / `chebwin`，脚本会采用回退策略（跳过精修或使用 Kaiser 近似窗）以保证流程可运行。
