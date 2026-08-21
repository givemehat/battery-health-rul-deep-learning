# Battery Health RUL Deep Learning Model Benchmark Comparison

Zero-shot evaluation on unseen test cell B0018 (Trained on B0005, B0006; Validated on B0007):

| Model                                                   |    MAE |   RMSE |   MAPE (%) |    R2 |   Max Error |   Params |   Train Time (s) |   Inference (ms/sample) |
|---------------------------------------------------------|--------|--------|------------|-------|-------------|----------|------------------|-------------------------|
| TCN                                                     |  5.050 |  6.049 |     41.267 | 0.952 |      13.084 |   107554 |            5.200 |                   1.541 |
| Random Forest Regressor                                 |  7.100 |  7.947 |     43.616 | 0.916 |      18.210 |    50000 |            0.360 |                   0.050 |
| Proposed Hybrid SOTA (CEEMDAN-TCN-BiLSTM-DualAttention) | 14.397 | 15.809 |    109.873 | 0.669 |      22.050 |   293090 |            4.960 |                   0.277 |
| Temporal Transformer                                    | 14.507 | 16.041 |    115.676 | 0.659 |      23.317 |    67714 |           11.390 |                   1.440 |
| Empirical Double-Exp Baseline                           | 17.432 | 19.196 |    133.879 | 0.512 |      22.000 |        4 |            0.010 |                   0.104 |
| BiLSTM-Attention                                        | 17.536 | 20.209 |    142.109 | 0.459 |      30.944 |   220674 |            9.410 |                   1.349 |
| LSTM                                                    | 20.637 | 22.721 |    158.048 | 0.316 |      31.405 |    52610 |           11.300 |                   1.124 |
| GRU                                                     | 21.666 | 23.597 |    159.926 | 0.262 |      32.298 |    39490 |           22.640 |                   0.346 |
