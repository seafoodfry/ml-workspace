# Quantitative Finance

# Fork & Merge Capital Management (not yet LLC)

What We Need to Figure Out:

1. Directional bias - When to be bullish/bearish/neutral before entering
1. Trend strength - When can you ride a move longer vs. quick scalp
1. Risk signals - Early warning when price might suddenly reverse on you

**Phase 1: Feature Discovery**

1. Simple correlation analysis - Which of your current indicators (RSI, MACD, SMA, etc.) actually correlate with 5-15 minute price moves >$10?
1. Market regime identification - Can we identify when BKNG is in "choppy/ranging" vs "trending" mode? (This affects whether you should scalp quickly vs ride longer)
1. Volume pattern analysis - let's see which volume patterns predict good bid-ask spread opportunities

**Phase 2: ML Models (After we know which features matter)**

1. Binary classifier - "Will price move >$10 in next 5 minutes?" (Your scalping decision)
1. Direction classifier - "Up, Down, or Sideways?" (Your directional bias)
1. Trend strength predictor - "Should I exit immediately or ride this move?" (Your hold time decision)

---
# Upgrading deps

```bash
uv lock --upgrade
uv sync
```
