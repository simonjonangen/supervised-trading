"""
Interactive Yahoo Finance Stock Annotator + Feature Pipeline (cleaned)

Key fixes vs your file
- Single CLI & single main() (removed the earlier copy that forced --ticker).
- Default ticker VOLV-B.ST (no arg needed).
- Click → timezone-safe using matplotlib.dates.
- Pending marker = green; accepted = red; accepted points persist on chart.
- Duplicate prevention; auto-save on Accept.
- Notes removed entirely (annotation = timestamp+price only).  <-- now stores log-return in 'price'
- Feature pipeline & exporter included; buttons: Save Features / Save + Merge.

Quick start
-----------
pip install yfinance pandas matplotlib pyarrow numpy
python main.py --features_out features.csv --merged_out merged.csv --tz Europe/Stockholm
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timezone
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.dates as mdates

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("Missing dependency 'yfinance'. Install with: pip install yfinance")


# ---------------------------- Data Model ------------------------------------
@dataclass
class StockConfig:
    ticker: str
    period: str = "10d"   # Yahoo limit for 5m
    interval: str = "5m"  # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 1wk, 1mo


class YahooStock:
    """Represents a single stock with methods to load/refresh OHLCV data."""

    def __init__(self, config: StockConfig):
        self.config = config
        self._hist: Optional[pd.DataFrame] = None

    @property
    def data(self) -> pd.DataFrame:
        if self._hist is None or self._hist.empty:
            self.refresh()
        return self._hist

    def refresh(self) -> pd.DataFrame:
        t = yf.Ticker(self.config.ticker)
        df = t.history(period=self.config.period, interval=self.config.interval, auto_adjust=False)
        if df is None or df.empty:
            raise RuntimeError(
                f"No data returned for {self.config.ticker} with period={self.config.period}, interval={self.config.interval}"
            )
        # Ensure timezone-aware index in UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize(timezone.utc)
        else:
            df.index = df.index.tz_convert(timezone.utc)
        df = df.rename_axis("timestamp")
        self._hist = df
        return df


# ---------------------------- Annotations -----------------------------------
@dataclass
class Annotation:
    timestamp: pd.Timestamp
    price: float  # NOTE: now holds log-return value


class AnnotationStore:
    """Manages accepted annotations and persistence to/from disk."""

    def __init__(self, out_path: Path):
        self.out_path = Path(out_path)
        self._df = self._load_existing()

    def _load_existing(self) -> pd.DataFrame:
        if not self.out_path.exists():
            return pd.DataFrame(columns=["timestamp", "price"]).astype({
                "timestamp": "datetime64[ns, UTC]",
                "price": "float64",
            })
        suffix = self.out_path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(self.out_path, parse_dates=["timestamp"])
        elif suffix == ".json":
            df = pd.read_json(self.out_path, convert_dates=["timestamp"])  # records handled on save
        elif suffix == ".parquet":
            df = pd.read_parquet(self.out_path)
        else:
            raise ValueError(f"Unsupported output format: {suffix}")
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "price"]).astype({
                "timestamp": "datetime64[ns, UTC]",
                "price": "float64",
            })
        # Normalize tz
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(timezone.utc)
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(timezone.utc)
        # Drop legacy 'note' column if present
        if "note" in df.columns:
            df = df.drop(columns=["note"])
        return df

    @property
    def df(self) -> pd.DataFrame:
        return self._df.copy()

    def add(self, ann: Annotation):
        new_row = pd.DataFrame({
            "timestamp": [pd.Timestamp(ann.timestamp).tz_convert(timezone.utc)],
            "price": [float(ann.price)],
        })
        self._df = pd.concat([self._df, new_row], ignore_index=True)

    def has_timestamp(self, ts: pd.Timestamp) -> bool:
        if self._df.empty:
            return False
        ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize(timezone.utc)
        else:
            ts = ts.tz_convert(timezone.utc)
        return (self._df["timestamp"] == ts).any()

    def save(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = self.out_path.suffix.lower()
        if suffix == ".csv" or suffix == "":
            self._df.to_csv(self.out_path, index=False)
        elif suffix == ".json":
            self._df.to_json(self.out_path, orient="records", date_unit="ms")
        elif suffix == ".parquet":
            self._df.to_parquet(self.out_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {suffix}")


# ---------------------------- Indicators ------------------------------------
class IndicatorBlock:
    """Base for an indicator block. compute(self, df)->DataFrame with SAME index."""
    name: str = "base"
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class TAClassicBlock(IndicatorBlock):
    name = "ta_classic"
    def __init__(self, sma=(10, 20, 50), ema=(12, 26), rsi_period=14, bb_period=20, bb_k=2.0):
        self.sma = sma; self.ema = ema
        self.rsi_period = rsi_period; self.bb_period = bb_period; self.bb_k = bb_k
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        close = df['Close']
        delta = close.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/self.rsi_period, adjust=False).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        out[f'RSI_{self.rsi_period}'] = 100 - (100 / (1 + rs))
        m = close.rolling(self.bb_period, min_periods=1).mean()
        s = close.rolling(self.bb_period, min_periods=1).std(ddof=0)
        out[f'BB_up_{self.bb_period}'] = m + self.bb_k * s
        out[f'BB_dn_{self.bb_period}'] = m - self.bb_k * s
        out[f'BB_width_{self.bb_period}'] = (out[f'BB_up_{self.bb_period}'] - out[f'BB_dn_{self.bb_period}']) / m.replace(0,np.nan)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        out['MACD'] = macd; out['MACD_signal'] = signal; out['MACD_hist'] = macd - signal
        return out


class KalmanTrendBlock(IndicatorBlock):
    name = "kalman_trend"
    def __init__(self, q=1e-3, r=1e-2):
        self.q = float(q); self.r = float(r)
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df['Close'].astype(float).to_numpy(); idx = df.index
        if len(idx) >= 2:
            dt_minutes = np.median(np.diff(idx.view('i8'))) / 1e9 / 60.0
            if not np.isfinite(dt_minutes) or dt_minutes <= 0: dt_minutes = 1.0
        else:
            dt_minutes = 1.0
        dt = float(dt_minutes)
        A = np.array([[1.0, dt],[0.0, 1.0]]); H = np.array([[1.0, 0.0]])
        Q = self.q * np.array([[dt**3/3, dt**2/2],[dt**2/2, dt]]); R = np.array([[self.r]])
        n = len(close); x = np.zeros((2,)); P = np.eye(2)
        kf_trend = np.empty(n); kf_gain0 = np.empty(n)
        for t in range(n):
            x = A @ x; P = A @ P @ A.T + Q
            y = np.array([[close[t]]]); S = H @ P @ H.T + R
            K = (P @ H.T) @ np.linalg.inv(S)
            innovation = y - (H @ x).reshape(1,1)
            x = x + (K @ innovation).ravel(); P = (np.eye(2) - K @ H) @ P
            kf_trend[t] = x[1]; kf_gain0[t] = K[0,0]
        return pd.DataFrame({'KF_trend': kf_trend,'KF_gain': kf_gain0}, index=df.index)


class StochasticCalcBlock(IndicatorBlock):
    name = "stoch_calc"
    def __init__(self, window=78):
        self.window = int(window)
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        close = df['Close'].astype(float)
        logp = np.log(close.replace(0, np.nan)).ffill()
        r = logp.diff().fillna(0.0)
        out['log_ret'] = r
        w = self.window
        rv = r.pow(2).rolling(w, min_periods=1).sum(); out[f'RVol_{w}'] = np.sqrt(rv)
        mu1 = np.sqrt(2/np.pi)
        bpv = (r.abs() * r.abs().shift(1)).rolling(w, min_periods=2).sum() / (mu1**2)
        out[f'BPV_{w}'] = bpv
        med = (r.shift(1).abs().rolling(w, min_periods=3).apply(lambda x: np.median(np.abs(x)), raw=True))
        out[f'MedRV_{w}'] = np.sqrt(np.pi/2) * med
        x = close.astype(float); x_tm1 = x.shift(1)
        a = pd.Series(index=x.index, dtype=float); b = pd.Series(index=x.index, dtype=float); sig = pd.Series(index=x.index, dtype=float)
        arr_x = x.to_numpy(); arr_xm1 = x_tm1.to_numpy()
        def _ols_ab(y, x):
            if len(y) < 2: return np.nan, np.nan, np.nan
            X = np.vstack([x, np.ones_like(x)]).T
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            a_hat, b_hat = beta[0], beta[1]
            resid = y - (a_hat*x + b_hat)
            s2 = (resid**2).mean(); return a_hat, b_hat, np.sqrt(s2)
        for i in range(len(x)):
            lo = max(0, i - w + 1); sl = slice(lo, i+1)
            y = arr_x[sl]; xv = arr_xm1[sl]
            mask = np.isfinite(y) & np.isfinite(xv); y = y[mask]; xv = xv[mask]
            a_i, b_i, s_i = _ols_ab(y[1:], xv[1:]) if len(y) > 2 else (np.nan, np.nan, np.nan)
            a.iloc[i] = a_i; b.iloc[i] = b_i; sig.iloc[i] = s_i
        out['OU_b'] = b; out['OU_sigma'] = sig
        with np.errstate(divide='ignore', invalid='ignore'):
            hl = -np.log(2) / np.log(a.clip(upper=0.999999))
        out['OU_halflife'] = hl.replace([np.inf, -np.inf], np.nan)
        return out


class IndicatorPipeline:
    def __init__(self, blocks: List[IndicatorBlock]):
        self.blocks = list(blocks)
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        pieces = []
        for blk in self.blocks:
            feat = blk.compute(df).reindex(df.index)
            pieces.append(feat)
        return pd.concat(pieces, axis=1)


# ---------------------------- Feature Export & Merge -------------------------
@dataclass
class FeatureExportConfig:
    out_features: Path
    out_merged: Optional[Path] = None


class FeatureExporter:
    def __init__(self, stock: YahooStock, store: AnnotationStore, pipeline: IndicatorPipeline, cfg: FeatureExportConfig):
        self.stock = stock; self.store = store; self.pipeline = pipeline; self.cfg = cfg
    def build_feature_frame(self) -> pd.DataFrame:
        df = self.stock.data.copy()
        feats = self.pipeline.compute(df)
        base = df[['Open','High','Low','Close','Volume']].copy()
        features = pd.concat([base, feats], axis=1); features.index.name = 'timestamp'
        return features

    def save_features(self, features: pd.DataFrame):
        # Ensure timestamp is preserved as a column
        if "timestamp" not in features.columns and features.index.name == "timestamp":
            features = features.reset_index()

        # Force drop OHLC before saving (inplace guaranteed)
        for col in ["Open", "High", "Low"]:
            if col in features.columns:
                features = features.drop(columns=col)

        path = Path(self.cfg.out_features)
        path.parent.mkdir(parents=True, exist_ok=True)
        suf = path.suffix.lower()
        if suf in ('', '.csv'):
            features.to_csv(path, index=False)
        elif suf == '.parquet':
            features.to_parquet(path, index=False)
        elif suf == '.json':
            features.to_json(path, orient='records', date_unit='ms')
        else:
            raise ValueError(f"Unsupported feature output format: {path.suffix}")

    def save_merged(self, features: pd.DataFrame):
        if not self.cfg.out_merged:
            return

        # Drop OHLC for the merged export as well
        features = features.drop(columns=["Open", "High", "Low"], errors="ignore")

        ann = self.store.df.set_index('timestamp').copy()
        ann['label'] = 1

        # features’ index is already 'timestamp'; join on index
        merged = features.join(ann[['label']], how='left')
        merged['label'] = merged['label'].fillna(0).astype(int)

        mpath = Path(self.cfg.out_merged)
        mpath.parent.mkdir(parents=True, exist_ok=True)

        suf = mpath.suffix.lower()
        if suf in ('', '.csv'):
            # write index so 'timestamp' remains as the first column name
            merged.to_csv(mpath, index=True)
        elif suf == '.parquet':
            merged.to_parquet(mpath, index=True)
        elif suf == '.json':
            merged.reset_index().to_json(mpath, orient='records', date_unit='ms')
        else:
            raise ValueError(f"Unsupported merged output format: {mpath.suffix}")


# ---------------------------- Interactive Plot ------------------------------
class InteractiveAnnotator:
    """Interactive annotator with Accept/Reject + optional feature export buttons."""
    def __init__(self, stock: YahooStock, store: AnnotationStore, tz_display: str = "UTC", feature_exporter: Optional[FeatureExporter] = None):
        self.stock = stock; self.store = store; self.tz_display = tz_display
        self.feature_exporter = feature_exporter
        self.pending_idx: Optional[int] = None
        self.pending_marker = None
        self.accepted_markers: list = []
        self.textbox = None

        # prepared on init
        self._ret_series = None  # pandas Series of log-returns

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        try:
            self.fig.canvas.manager.set_window_title(f"{stock.config.ticker} — {stock.config.interval} ({stock.config.period})")
        except Exception:
            pass
        self._init_plot()
        self._draw_existing_annotations()
        self._init_widgets()
        self._connect_events()

    def _compute_log_returns(self, df: pd.DataFrame) -> pd.Series:
        logp = np.log(df['Close'].astype(float).replace(0, np.nan)).ffill()
        r = logp.diff().fillna(0.0)
        r.name = "log_return"
        return r

    def _init_plot(self):
        df = self.stock.data
        self._ret_series = self._compute_log_returns(df)
        self.ax.plot(df.index, self._ret_series, lw=1.2, picker=5)
        self.ax.set_title(f"{self.stock.config.ticker} Log-Returns — interval {self.stock.config.interval}")
        self.ax.set_xlabel("Time"); self.ax.set_ylabel("Log-return"); self.ax.grid(True, alpha=0.3)
        self.fig.autofmt_xdate()

    def _init_widgets(self):
        ax_accept = self.fig.add_axes([0.80, 0.90, 0.08, 0.05])
        ax_reject = self.fig.add_axes([0.69, 0.90, 0.08, 0.05])
        ax_save = self.fig.add_axes([0.58, 0.90, 0.08, 0.05])
        self.btn_accept = Button(ax_accept, "Accept")
        self.btn_reject = Button(ax_reject, "Reject")
        self.btn_save = Button(ax_save, "Save")
        if self.feature_exporter is not None:
            ax_feat = self.fig.add_axes([0.46, 0.90, 0.10, 0.05])
            ax_merge = self.fig.add_axes([0.33, 0.90, 0.12, 0.05])
            self.btn_features = Button(ax_feat, "Save Features")
            self.btn_merge = Button(ax_merge, "Save + Merge")
        self.textbox = self.fig.text(0.01, 0.95, "Click a point on the chart to annotate…", ha="left", va="center")

    def _connect_events(self):
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.btn_accept.on_clicked(lambda evt: self._accept_pending())
        self.btn_reject.on_clicked(lambda evt: self._reject_pending())
        self.btn_save.on_clicked(lambda evt: self._save())
        if self.feature_exporter is not None:
            self.btn_features.on_clicked(lambda evt: self._save_features_only())
            self.btn_merge.on_clicked(lambda evt: self._save_features_and_merge())

    def _draw_existing_annotations(self):
        ann_df = self.store.df
        if ann_df.empty: return
        df = self.stock.data
        rs = self._ret_series if self._ret_series is not None else self._compute_log_returns(df)
        xs, ys = [], []
        for _, row in ann_df.iterrows():
            ts_saved = row['timestamp']
            idx = df.index.get_indexer([ts_saved], method='nearest')[0]
            ts_plot = df.index[idx]; val = float(rs.iloc[idx])
            xs.append(ts_plot); ys.append(val)
        if xs:
            sc = self.ax.scatter(xs, ys, s=60, marker='o', zorder=6, color='red')
            self.accepted_markers.append(sc)
        self.fig.canvas.draw_idle()

    def _add_accepted_marker(self, ts: pd.Timestamp, value: float):
        sc = self.ax.scatter([ts], [value], s=60, marker='o', zorder=6, color='red')
        self.accepted_markers.append(sc)
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        df = self.stock.data
        rs = self._ret_series if self._ret_series is not None else self._compute_log_returns(df)
        clicked_dt = mdates.num2date(event.xdata)  # usually naive UTC
        clicked_ts = pd.Timestamp(clicked_dt)
        if df.index.tz is not None:
            if clicked_ts.tzinfo is None: clicked_ts = clicked_ts.tz_localize("UTC")
            else: clicked_ts = clicked_ts.tz_convert(df.index.tz)
        else:
            if clicked_ts.tzinfo is not None: clicked_ts = clicked_ts.tz_localize(None)
        idx = df.index.get_indexer([clicked_ts], method='nearest')[0]
        self.pending_idx = int(idx)
        ts = df.index[idx]; value = float(rs.iloc[idx])
        self._update_pending_marker(ts, value)
        disp_ts = ts.tz_convert(self.tz_display) if ts.tz is not None else ts
        self._set_status(f"Selected: {disp_ts.strftime('%Y-%m-%d %H:%M:%S %Z')} → {value:.6f}. Accept? (a) or Reject (r)")

    def _on_key(self, event):
        if event.key == 'a': self._accept_pending()
        elif event.key == 'r': self._reject_pending()
        elif event.key == 's': self._save()
        elif event.key == 'q': plt.close(self.fig)

    def _update_pending_marker(self, ts: pd.Timestamp, value: float):
        if self.pending_marker is not None:
            self.pending_marker.remove(); self.pending_marker = None
        self.pending_marker = self.ax.scatter([ts], [value], s=70, marker='o', zorder=7, color='green')
        self.fig.canvas.draw_idle()

    def _set_status(self, text: str):
        self.textbox.set_text(text); self.fig.canvas.draw_idle()

    def _accept_pending(self):
        if self.pending_idx is None:
            self._set_status("No point selected. Click on the chart first."); return
        df = self.stock.data
        rs = self._ret_series if self._ret_series is not None else self._compute_log_returns(df)
        ts = df.index[self.pending_idx]; value = float(rs.iloc[self.pending_idx])
        if self.store.has_timestamp(ts):
            disp_ts = ts.tz_convert(self.tz_display) if ts.tz is not None else ts
            self._set_status(f"Already annotated: {disp_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}."); return
        ann = Annotation(timestamp=ts, price=value)  # stores log-return in 'price'
        self.store.add(ann); self.store.save()
        self._add_accepted_marker(ts, value)
        if self.pending_marker is not None: self.pending_marker.remove(); self.pending_marker = None
        self.pending_idx = None
        disp_ts = ts.tz_convert(self.tz_display) if ts.tz is not None else ts
        self._set_status(f"Added & saved: {disp_ts.strftime('%Y-%m-%d %H:%M:%S %Z')} @ {value:.6f}")

    def _reject_pending(self):
        self.pending_idx = None
        if self.pending_marker is not None: self.pending_marker.remove(); self.pending_marker = None
        self._set_status("Selection cleared. Click another point…")

    def _save(self):
        self.store.save(); self._set_status(f"Saved annotations → {self.store.out_path}")

    # Feature export buttons
    def _save_features_only(self):
        if self.feature_exporter is None: return
        try:
            features = self.feature_exporter.build_feature_frame(); self.feature_exporter.save_features(features)
            self._set_status(f"Saved features → {self.feature_exporter.cfg.out_features}")
        except Exception as e:
            self._set_status(f"Feature save error: {e}")
    def _save_features_and_merge(self):
        if self.feature_exporter is None: return
        try:
            features = self.feature_exporter.build_feature_frame(); self.feature_exporter.save_features(features)
            self.feature_exporter.save_merged(features)
            if self.feature_exporter.cfg.out_merged:
                self._set_status(f"Saved features → {self.feature_exporter.cfg.out_features} and merged → {self.feature_exporter.cfg.out_merged}")
        except Exception as e:
            self._set_status(f"Merge save error: {e}")


    def show(self):
        """Open the interactive window (required on some environments)."""
        plt.show()

# ---------------------------- CLI -------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Interactive Yahoo Finance Stock Annotator")
    p.add_argument("--ticker", default="VOLV-B.ST", help="Ticker symbol, e.g., AAPL (default: VOLV-B.ST)")
    p.add_argument("--period", default="60d", help="History period, e.g., 5d, 60d, 1y")
    p.add_argument("--interval", default="5m", help="Bar interval, e.g., 1m, 5m, 15m, 1h, 1d")
    p.add_argument("--out", default="data/annotations.csv", help="Output file (.csv, .json, .parquet)")
    p.add_argument("--tz", default="UTC", help="Time zone string for display, e.g., Europe/Stockholm")
    p.add_argument("--features_out", default="data/features.csv", help="Where to save computed indicators")
    p.add_argument("--merged_out", default="data/merged.csv", help="Where to save features joined with annotations (0/1 label)")
    return p.parse_args()


def main():
    args = parse_args()
    stock = YahooStock(StockConfig(ticker=args.ticker, period=args.period, interval=args.interval))
    store = AnnotationStore(Path(args.out))
    pipeline = IndicatorPipeline([
        TAClassicBlock(),
        KalmanTrendBlock(q=1e-3, r=1e-2),
        StochasticCalcBlock(window=78),
    ])
    exporter = FeatureExporter(stock, store, pipeline, FeatureExportConfig(out_features=Path(args.features_out), out_merged=Path(args.merged_out) if args.merged_out else None))
    annot = InteractiveAnnotator(stock, store, tz_display=args.tz, feature_exporter=exporter)
    annot.show()


if __name__ == "__main__":
    main()
