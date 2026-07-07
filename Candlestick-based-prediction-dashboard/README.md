# Candlestick-based-prediction-dashboard
deep learning framework for stock trend prediction by transforming financial time-series data into candlestick chart images and solving the problem as an image classification task. Using Vision Transformer (ViT) and Explainable AI (XAI)

# 8. Final Report & Observations

## Stock Trend Prediction using Candlestick Images — ViT & XAI

---

## Overview

This project developed an end-to-end pipeline for predicting short-term stock price movement by framing the problem as image classification over candlestick chart images. Historical 1-hour intraday data for 15 NSE-listed banking stocks was collected using `yfinance`, converted into uniform 224×224 PNG candlestick images (each encoding exactly 18 candles spanning 3 trading days), labelled with a stock-specific volatility-adaptive threshold, and fed into a Vision Transformer (ViT) trained from pretrained ImageNet weights using PyTorch. Explainability was provided via attention map visualization and Grad-CAM.

---

## Key Findings

### Dataset & Labelling
- **15 NSE banking stocks** were downloaded at 1-hour granularity covering approximately 2 years of trading history.
- Each sample encodes **18 candles (6 per trading day × 3 days)**, ensuring a consistent and uniform input representation.
- A **volatility-adaptive labelling strategy** was adopted:
  - Threshold = `max(0.5 × rolling_20d_volatility, 0.3%)`
  - Classes: `up`, `down`, `neutral`
  - This avoided a hard fixed threshold, making labels more meaningful across different stocks and market regimes.
- A **purged chronological 70/20/10 split** was applied. A gap of `LOOKBACK_DAYS − 1` samples was dropped at each split boundary to prevent overlapping-window leakage between train, validation, and test sets.

### Model Performance
- The ViT model, initialized with pretrained weights, demonstrated the ability to learn discriminative visual features from candlestick images beyond random chance.
- As expected for a three-class imbalanced financial dataset, the model showed higher recall on the dominant `neutral` class and variable performance on `up` and `down` classes.
- Weighted F1 was the primary metric given class imbalance; macro F1 provided an additional view of per-class fairness.
- Confidence calibration was partially meaningful: a portion of incorrect predictions carried lower confidence scores, suggesting the model has some uncertainty awareness, though overconfidence on hard examples was also observed.

### XAI Insights (Attention & Grad-CAM)
- Attention maps highlighted regions corresponding to **candle bodies and wick extremes**, particularly around the most recent candles in the 18-candle window — consistent with the intuition that recent price action is most predictive.
- Grad-CAM activations occasionally highlighted **inter-day transitions** (the boundary between day 2 and day 3 candle groups), suggesting the model captures multi-day momentum implicitly.
- Some activation patterns corresponded loosely to known classical structures (e.g., activations near doji-like candles and engulfing patterns), though this correspondence was not universal.

### Comparative Study: Classical Patterns vs. Deep Model
| Pattern | Alignment with Deep Model |
|---|---|
| Doji (last candle) | Mixed — maps loosely to `neutral` but inconsistent |
| Hammer (last candle) | Partial alignment with `up` labels; model not always in agreement |
| Bullish Engulfing | Moderate correlation with `up` label; ViT predictions partially agree |
| Bearish Engulfing | Moderate correlation with `down` label; ViT predictions partially agree |

- In cases where no classical pattern was detected yet the model predicted correctly, it likely leveraged **non-classical structures** such as multi-candle slope, wick cluster geometry, or volatility contraction — visual features that are difficult to formalise as rules but are learnable by a deep network.
- In cases where an obvious pattern was present but the model predicted incorrectly, the most common failure was confusion between `neutral` and a directional class, suggesting the threshold band is visually difficult to separate.

---

## Effectiveness

The pipeline demonstrates that:

1. **Candlestick images are a viable input modality** for sequence-based financial prediction — spatial patterns in rendered charts carry sufficient signal for a ViT to learn from.
2. **Pretrained ViT significantly reduces training time** and outperforms a randomly initialized equivalent given the relatively small dataset size (~several thousand images per stock).
3. **Volatility-adaptive labelling** produces more balanced and economically meaningful classes compared to a fixed return threshold, reducing label noise in quiet vs. volatile periods.
4. **XAI techniques (attention maps, Grad-CAM) provide genuine interpretability** — the model focuses on price-relevant regions rather than background artefacts, building trust in the learned representations.
5. The deep model **partially rediscovers classical candlestick wisdom** (hammer → up, bearish engulfing → down) but also captures patterns that go beyond the classical rule set, supporting the hypothesis that DL can identify novel predictive structures.

---

## Limitations

1. **Market regime dependency**: The model is trained on approximately 2 years of NSE banking data. Performance may degrade significantly in unseen market regimes (e.g., a sharp systemic crisis or a prolonged low-volatility consolidation phase not present in the training window).

2. **Limited stock universe**: Only 15 banking sector stocks were used. The learned representations may not generalise to other sectors (IT, FMCG, energy) where price dynamics and chart structures differ substantially.

3. **No volume information in images**: Volume was excluded from the rendered charts. Classical technical analysis treats price-volume confirmation as essential (e.g., a hammer on low volume is less reliable). Adding a volume panel could improve signal quality.

4. **Label lookahead horizon**: Labels are based on the next trading day's close. This is a single-step forecast, and the pipeline provides no insight into multi-day or multi-week trends. Extending the label horizon introduces additional noise.

5. **Class imbalance**: The `neutral` class typically dominates the dataset because most days fall within the volatility-adaptive threshold band. Despite class-weighted training, the model is biased toward predicting `neutral`, which limits practical utility for directional trading signals.

6. **Survivorship and delisting bias**: `yfinance` returns data for currently listed tickers. Stocks that were delisted or restructured during the 2-year window are absent, introducing a mild survivorship bias.

7. **Transaction costs not modelled**: The evaluation is purely classification-based. No backtesting was performed to assess whether the model's predictions are profitable after brokerage fees, slippage, and impact costs — the critical test for practical deployment.

8. **XAI remains qualitative**: Attention maps and Grad-CAM show *where* the model looks, but do not provide a causal explanation of *why* a particular pattern led to a specific prediction. Quantitative attribution (e.g., SHAP over derived features) was not implemented.

---

## Future Scope

1. **Multi-sector generalisation**: Extend the dataset to cover Nifty 50 stocks across all sectors and evaluate cross-sector transfer learning — train on banking, fine-tune on IT, etc.

2. **Volume & technical overlays**: Render multi-panel images incorporating volume bars, RSI, MACD, or Bollinger Bands as additional visual channels. This could be achieved by adding indicator sub-plots via `mplfinance` without changing the ViT input resolution.

3. **Longer context windows**: Experiment with 5-day and 10-day windows (30 and 60 candles respectively) to capture weekly momentum and mean-reversion dynamics.

4. **Multi-class hierarchy**: Introduce a finer label granularity — for example, `strong_up`, `weak_up`, `neutral`, `weak_down`, `strong_down` — mapped to return quintiles. This provides more actionable signals than a three-class scheme.

5. **Backtesting integration**: Connect model predictions to a vectorised backtesting framework (e.g., `backtesting.py` or `zipline`) to measure Sharpe ratio, max drawdown, and win rate under realistic transaction cost assumptions.

6. **Ensemble & hybrid models**: Combine ViT image predictions with LSTM/Transformer predictions over the raw OHLCV time series. Ensembling image-based and sequence-based models may reduce error on hard examples where one modality is uncertain.

7. **Online / continual learning**: Deploy the model in a rolling-retrain regime — weekly retraining on the most recent window — to adapt to evolving market regimes without complete retraining from scratch.

8. **Quantitative XAI**: Apply TCAV (Testing with Concept Activation Vectors) or SHAP on hand-crafted candlestick features to move from qualitative attention visualisation to quantitative feature attribution, enabling rigorous comparison with classical pattern signals.

9. **Alternative architectures**: Benchmark ViT against DeiT, Swin Transformer, and ConvNeXt on the same dataset. Hierarchical vision models (Swin) may better capture the multi-scale structure of candlestick charts (individual wick tips, candle bodies, multi-day trends).

10. **Cross-market validation**: Apply the trained model (with fine-tuning) to other markets — BSE, NYSE, or crypto exchanges — to assess how much of the learned representation is market-specific vs. universal.

---

## Conclusion

This project successfully demonstrates that stock trend prediction can be formulated as a visual image classification task, with Vision Transformers serving as a powerful backbone. The pipeline — from raw intraday data download through candlestick image generation, purged chronological splitting, ViT training, XAI analysis, and classical pattern comparison — is modular, reproducible, and extensible. While the model shows genuine learning and partial rediscovery of classical candlestick patterns, practical deployment would require addressing the limitations above, particularly backtesting under realistic market conditions and generalisation across sectors and market regimes.

> *"Deep learning does not merely memorise classical patterns — it discovers a richer geometric vocabulary of price behaviour that partially overlaps with, and partially extends beyond, the centuries-old tradition of candlestick analysis."*
