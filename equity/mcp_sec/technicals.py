"""
Technical Analysis

Calculates technical indicators from price data:
  - Moving Averages (SMA, EMA)
  - RSI (Relative Strength Index)
  - MACD
  - Bollinger Bands
  - Volume Analysis
  - Support/Resistance Levels
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import httpx
import math


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators."""
    ticker: str
    last_price: float
    last_updated: str
    
    # Moving Averages
    sma_20: float = 0
    sma_50: float = 0
    sma_200: float = 0
    ema_12: float = 0
    ema_26: float = 0
    
    # Trend Signals
    above_sma_20: bool = False
    above_sma_50: bool = False
    above_sma_200: bool = False
    golden_cross: bool = False  # SMA50 > SMA200
    death_cross: bool = False   # SMA50 < SMA200
    
    # RSI
    rsi_14: float = 0
    rsi_signal: str = ""  # "Oversold", "Neutral", "Overbought"
    
    # MACD
    macd_line: float = 0
    macd_signal: float = 0
    macd_histogram: float = 0
    macd_trend: str = ""  # "Bullish", "Bearish"
    
    # Bollinger Bands
    bb_upper: float = 0
    bb_middle: float = 0
    bb_lower: float = 0
    bb_position: str = ""  # "Above Upper", "Middle", "Below Lower"
    
    # Volume
    avg_volume_20: int = 0
    volume_trend: str = ""  # "Above Average", "Below Average"
    
    # Support/Resistance
    support_1: float = 0
    support_2: float = 0
    resistance_1: float = 0
    resistance_2: float = 0
    
    # Overall Signal
    trend: str = ""  # "Bullish", "Neutral", "Bearish"
    strength: str = ""  # "Strong", "Moderate", "Weak"


class TechnicalAnalyzer:
    """Calculates technical indicators from price data."""
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        }
    
    async def get_price_history(
        self,
        ticker: str,
        period: str = "1y"
    ) -> Tuple[List[float], List[float], List[int]]:
        """
        Get historical prices from Yahoo Finance.
        
        Returns: (closes, highs_lows, volumes)
        """
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "interval": "1d",
            "range": period
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, params=params, headers=self.headers, timeout=15.0)
            resp.raise_for_status()
            data = resp.json()
        
        result = data.get("chart", {}).get("result", [{}])[0]
        indicators = result.get("indicators", {}).get("quote", [{}])[0]
        
        closes = [c for c in indicators.get("close", []) if c is not None]
        highs = [h for h in indicators.get("high", []) if h is not None]
        lows = [l for l in indicators.get("low", []) if l is not None]
        volumes = [v for v in indicators.get("volume", []) if v is not None]
        
        return closes, list(zip(highs, lows)), volumes
    
    def calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return 0
        return sum(prices[-period:]) / period
    
    def calculate_ema(self, prices: List[float], period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return 0
        
        multiplier = 2 / (period + 1)
        ema = sum(prices[:period]) / period  # Start with SMA
        
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(
        self,
        prices: List[float]
    ) -> Tuple[float, float, float]:
        """Calculate MACD (12, 26, 9)."""
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        
        macd_line = ema_12 - ema_26
        
        # Calculate MACD line history for signal line
        macd_history = []
        for i in range(26, len(prices)):
            e12 = self.calculate_ema(prices[:i+1], 12)
            e26 = self.calculate_ema(prices[:i+1], 26)
            macd_history.append(e12 - e26)
        
        if len(macd_history) >= 9:
            signal_line = self.calculate_ema(macd_history, 9)
        else:
            signal_line = macd_line
        
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(
        self,
        prices: List[float],
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands."""
        if len(prices) < period:
            return 0, 0, 0
        
        sma = self.calculate_sma(prices, period)
        
        # Calculate standard deviation
        squared_diff = [(p - sma) ** 2 for p in prices[-period:]]
        variance = sum(squared_diff) / period
        std = math.sqrt(variance)
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        return upper, sma, lower
    
    def find_support_resistance(
        self,
        highs_lows: List[Tuple[float, float]],
        current_price: float
    ) -> Tuple[float, float, float, float]:
        """Find support and resistance levels."""
        if not highs_lows:
            return 0, 0, 0, 0
        
        highs = [h for h, l in highs_lows]
        lows = [l for h, l in highs_lows]
        
        # Find levels below current price (support)
        supports = sorted([l for l in lows if l < current_price], reverse=True)
        support_1 = supports[0] if supports else current_price * 0.95
        support_2 = supports[1] if len(supports) > 1 else current_price * 0.90
        
        # Find levels above current price (resistance)
        resistances = sorted([h for h in highs if h > current_price])
        resistance_1 = resistances[0] if resistances else current_price * 1.05
        resistance_2 = resistances[1] if len(resistances) > 1 else current_price * 1.10
        
        return support_1, support_2, resistance_1, resistance_2
    
    async def analyze(self, ticker: str) -> TechnicalIndicators:
        """Perform full technical analysis."""
        ticker = ticker.upper()
        
        closes, highs_lows, volumes = await self.get_price_history(ticker)
        
        if not closes:
            return TechnicalIndicators(
                ticker=ticker,
                last_price=0,
                last_updated=datetime.now().isoformat()
            )
        
        last_price = closes[-1]
        
        # Moving Averages
        sma_20 = self.calculate_sma(closes, 20)
        sma_50 = self.calculate_sma(closes, 50)
        sma_200 = self.calculate_sma(closes, 200)
        ema_12 = self.calculate_ema(closes, 12)
        ema_26 = self.calculate_ema(closes, 26)
        
        # RSI
        rsi = self.calculate_rsi(closes)
        if rsi < 30:
            rsi_signal = "Oversold"
        elif rsi > 70:
            rsi_signal = "Overbought"
        else:
            rsi_signal = "Neutral"
        
        # MACD
        macd_line, macd_signal, macd_hist = self.calculate_macd(closes)
        macd_trend = "Bullish" if macd_hist > 0 else "Bearish"
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(closes)
        if last_price > bb_upper:
            bb_position = "Above Upper"
        elif last_price < bb_lower:
            bb_position = "Below Lower"
        else:
            bb_position = "Middle"
        
        # Volume
        avg_volume = sum(volumes[-20:]) // 20 if len(volumes) >= 20 else 0
        current_volume = volumes[-1] if volumes else 0
        volume_trend = "Above Average" if current_volume > avg_volume else "Below Average"
        
        # Support/Resistance
        s1, s2, r1, r2 = self.find_support_resistance(highs_lows[-60:], last_price)
        
        # Overall trend determination
        bullish_signals = 0
        bearish_signals = 0
        
        if last_price > sma_20:
            bullish_signals += 1
        else:
            bearish_signals += 1
        if last_price > sma_50:
            bullish_signals += 1
        else:
            bearish_signals += 1
        if last_price > sma_200:
            bullish_signals += 1
        else:
            bearish_signals += 1
        if rsi < 30:
            bullish_signals += 1  # Oversold = potential reversal up
        elif rsi > 70:
            bearish_signals += 1  # Overbought = potential reversal down
        if macd_hist > 0:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if bullish_signals >= 4:
            trend = "Bullish"
            strength = "Strong" if bullish_signals == 5 else "Moderate"
        elif bearish_signals >= 4:
            trend = "Bearish"
            strength = "Strong" if bearish_signals == 5 else "Moderate"
        else:
            trend = "Neutral"
            strength = "Weak"
        
        return TechnicalIndicators(
            ticker=ticker,
            last_price=last_price,
            last_updated=datetime.now().isoformat(),
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            ema_12=ema_12,
            ema_26=ema_26,
            above_sma_20=last_price > sma_20,
            above_sma_50=last_price > sma_50,
            above_sma_200=last_price > sma_200,
            golden_cross=sma_50 > sma_200,
            death_cross=sma_50 < sma_200,
            rsi_14=rsi,
            rsi_signal=rsi_signal,
            macd_line=macd_line,
            macd_signal=macd_signal,
            macd_histogram=macd_hist,
            macd_trend=macd_trend,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            bb_position=bb_position,
            avg_volume_20=avg_volume,
            volume_trend=volume_trend,
            support_1=s1,
            support_2=s2,
            resistance_1=r1,
            resistance_2=r2,
            trend=trend,
            strength=strength
        )


def format_technicals_markdown(t: TechnicalIndicators) -> str:
    """Format technical analysis as Markdown."""
    
    def fmt(val, decimals=2):
        if val == 0:
            return "-"
        return f"${val:,.{decimals}f}"
    
    trend_emoji = "🟢" if t.trend == "Bullish" else "🔴" if t.trend == "Bearish" else "🟡"
    
    return f"""# {t.ticker} Technical Analysis
*Last Price: {fmt(t.last_price)} | Updated: {t.last_updated}*

---

## Overall Signal

| Metric | Value |
|:-------|------:|
| **Trend** | {trend_emoji} **{t.trend}** |
| **Strength** | {t.strength} |

---

## Moving Averages

| Indicator | Value | Signal |
|:----------|------:|:-------|
| SMA 20 | {fmt(t.sma_20)} | {"✅ Above" if t.above_sma_20 else "❌ Below"} |
| SMA 50 | {fmt(t.sma_50)} | {"✅ Above" if t.above_sma_50 else "❌ Below"} |
| SMA 200 | {fmt(t.sma_200)} | {"✅ Above" if t.above_sma_200 else "❌ Below"} |
| EMA 12 | {fmt(t.ema_12)} | - |
| EMA 26 | {fmt(t.ema_26)} | - |

**Cross Signal:** {"🟢 Golden Cross (Bullish)" if t.golden_cross else "🔴 Death Cross (Bearish)" if t.death_cross else "➖ No Cross"}

---

## RSI (Relative Strength Index)

| Metric | Value |
|:-------|------:|
| RSI (14) | {t.rsi_14:.1f} |
| Signal | **{t.rsi_signal}** |

*Interpretation: <30 = Oversold (Buy signal), >70 = Overbought (Sell signal)*

---

## MACD

| Metric | Value |
|:-------|------:|
| MACD Line | {t.macd_line:.3f} |
| Signal Line | {t.macd_signal:.3f} |
| Histogram | {t.macd_histogram:.3f} |
| Trend | **{t.macd_trend}** |

---

## Bollinger Bands

| Band | Value |
|:-----|------:|
| Upper | {fmt(t.bb_upper)} |
| Middle (SMA 20) | {fmt(t.bb_middle)} |
| Lower | {fmt(t.bb_lower)} |
| Position | **{t.bb_position}** |

---

## Volume

| Metric | Value |
|:-------|------:|
| Avg Volume (20D) | {t.avg_volume_20:,} |
| Trend | {t.volume_trend} |

---

## Support & Resistance

| Level | Price | Distance |
|:------|------:|:---------|
| Resistance 2 | {fmt(t.resistance_2)} | +{((t.resistance_2/t.last_price)-1)*100:.1f}% |
| Resistance 1 | {fmt(t.resistance_1)} | +{((t.resistance_1/t.last_price)-1)*100:.1f}% |
| **Current** | **{fmt(t.last_price)}** | - |
| Support 1 | {fmt(t.support_1)} | {((t.support_1/t.last_price)-1)*100:.1f}% |
| Support 2 | {fmt(t.support_2)} | {((t.support_2/t.last_price)-1)*100:.1f}% |

---

*Technical analysis based on price action. Not investment advice.*
"""


async def get_technicals(ticker: str) -> str:
    """Get technical analysis as Markdown."""
    analyzer = TechnicalAnalyzer()
    indicators = await analyzer.analyze(ticker)
    return format_technicals_markdown(indicators)


# CLI
if __name__ == "__main__":
    import sys
    
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print(asyncio.run(get_technicals(ticker)))
