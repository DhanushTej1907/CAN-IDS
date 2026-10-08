"""CAN-IDS live demo: replay a CAN file through the causal feature extractor and the final model.

Run from the repo root:  streamlit run app/dashboard.py

Smoothness: only the live section reruns (st.fragment(run_every=TICK_S)); each tick processes a batch
of frames within a compute budget adapted to the render time (src/live.py ReplayEngine.advance),
charts keep a rolling 150 s window, and model/frames are cached once per process. Any error during a
tick pauses the replay instead of crashing.
"""
from __future__ import annotations

import html
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from src.live import (ALERT_WINDOW_S, ATTACK_NAMES, ReplayEngine, list_streams,  # noqa: E402
                      load_detector, load_frames)
from src.train import REPORT_DIR  # noqa: E402
from src.train_stats import load_splits, load_train_stats  # noqa: E402

DEFAULT_STEM = "Pre_submit_D"
DEFAULT_SPEED = 10
TICK_S = 0.5           # UI refresh interval of the live section
TICK_SHARE = 0.6       # compute + render must fit in this share of TICK_S: a run still going when the
                       # next timed rerun arrives is cut short, and the elements at the bottom go stale
BUDGET_RANGE = (0.05, 0.30)  # compute budget per tick, adapted to the measured render time
# One meaning per color, everywhere: cyan = normal/safe, red = attack/alert, amber = warning/missed.
CYAN, RED, AMBER = "#00E5FF", "#FF1744", "#FFC400"
INK, INK_2, INK_3 = "#FFFFFF", "#9CA3AF", "#6B7280"  # text, secondary text, axis ticks
SURFACE = "#050505"
GRID = "rgba(255,255,255,0.07)"
TRANSPARENT = "rgba(0,0,0,0)"
# Space Grotesk for text, JetBrains Mono for numbers; system fallbacks keep it readable offline.
FONT = "'Space Grotesk', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Consolas, 'Courier New', monospace"
PLOT_FONT = "Space Grotesk, system-ui, Segoe UI, sans-serif"
PLOT_MONO = "JetBrains Mono, Consolas, monospace"


def rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


st.set_page_config(page_title="CAN-IDS Live", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
.stApp {{
  font-family: {FONT};
  background: #000000;
  color: {INK};
}}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px; }}
/* no dimming of elements while a fragment reruns */
[data-stale="true"] {{ opacity: 1 !important; transition: none !important; }}

section[data-testid="stSidebar"] {{
  background: {SURFACE}; backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border-right: 1px solid rgba(255,255,255,0.08);
}}

.stApp [class*="st-key-glass"] {{
  background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px; padding: 18px 20px 14px 20px;
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.6);
}}
.card-title {{ font-size: 0.78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  color: {INK_2}; margin: 0 0 6px 2px; }}

.hero h1 {{ font-size: 1.65rem; font-weight: 700; margin: 0; color: {INK}; letter-spacing: -0.01em; }}
.hero p {{ color: {INK_2}; margin: 4px 0 18px 0; font-size: 0.92rem; }}

.kpi-row {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin-bottom: 16px; }}
@media (max-width: 900px) {{ .kpi-row {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.kpi {{
  background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.12); border-radius: 16px;
  padding: 16px 20px; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.6); min-height: 112px;
}}
.kpi-label {{ font-size: 0.74rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: {INK_2}; }}
.kpi-value {{ font-family: {MONO}; font-size: 1.75rem; font-weight: 700; color: {INK}; margin-top: 6px;
  letter-spacing: -0.02em; }}
.mono {{ font-family: {MONO}; }}
.kpi-sub {{ font-size: 0.8rem; color: {INK_2}; margin-top: 2px; font-variant-numeric: tabular-nums; }}
.warn {{ color: {AMBER}; }}
.badge {{ display: inline-flex; align-items: center; gap: 10px; margin-top: 8px; padding: 8px 16px;
  border-radius: 999px; font-weight: 700; font-size: 1.15rem; letter-spacing: .06em; }}
.badge .dot {{ width: 11px; height: 11px; border-radius: 50%; }}
.badge.normal {{ background: {rgba(CYAN, 0.10)}; border: 1px solid {rgba(CYAN, 0.65)}; color: {CYAN};
  box-shadow: 0 0 18px {rgba(CYAN, 0.18)}; }}
.badge.normal .dot {{ background: {CYAN}; box-shadow: 0 0 8px {CYAN}; }}
.badge.attack {{ background: {rgba(RED, 0.14)}; border: 1px solid {RED}; color: {RED};
  animation: pulse 1.2s ease-in-out infinite; }}
.badge.attack .dot {{ background: {RED}; box-shadow: 0 0 8px {RED}; }}
.kpi.attack {{ border-color: {rgba(RED, 0.7)}; box-shadow: 0 0 30px {rgba(RED, 0.35)}; }}
@keyframes pulse {{ 0%,100% {{ box-shadow: 0 0 0 0 {rgba(RED, 0.55)}; }} 50% {{ box-shadow: 0 0 0 10px {rgba(RED, 0)}; }} }}

.side-brand {{ font-weight: 700; font-size: 1.1rem; color: {INK}; margin-bottom: 2px; }}
.side-muted {{ color: {INK_2}; font-size: 0.8rem; }}
.speed-note {{ color: {INK_2}; font-size: 0.8rem; margin: -8px 0 10px 0; }}
.speed-note b {{ font-family: {MONO}; color: {INK}; font-weight: 500; }}
.side-kv {{ display: flex; justify-content: space-between; gap: 14px; font-size: 0.86rem; padding: 3px 0; color: {INK_2}; }}
.side-kv span {{ flex: none; }}
.side-kv b {{ color: {INK}; font-weight: 600; text-align: right; }}
.side-kv b.mono {{ font-weight: 500; }}

.alert-log {{ max-height: 400px; overflow-y: auto; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; }}
.alert-log table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; }}
.alert-log th {{ position: sticky; top: 0; background: #0B0B0B; color: {INK_2}; font-weight: 500; text-align: left;
  padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,0.12); white-space: nowrap; }}
.alert-log td {{ padding: 7px 10px; border-bottom: 1px solid rgba(255,255,255,0.06); color: {INK}; white-space: nowrap; }}
.alert-log td.mono {{ font-size: 0.82rem; }}
.alert-log td.num {{ text-align: right; }}
.alert-log .atk {{ color: {RED}; font-weight: 600; }}
.alert-log .fa {{ color: {AMBER}; }}
div[data-testid="stProgress"] > div > div > div > div {{ background: {CYAN}; box-shadow: 0 0 10px {CYAN}; }}
[data-testid="stBaseButton-primary"]:not(:disabled), [data-testid="stBaseButton-primary"]:not(:disabled) p {{
  color: #000000 !important; font-weight: 600; }}
</style>
"""


# ---------------------------------------------------------------- cached resources

@st.cache_resource(show_spinner="Loading model…")
def get_detector():
    return load_detector()


@st.cache_resource(show_spinner=False)
def get_stats():
    return load_train_stats()


@st.cache_resource(show_spinner="Loading CAN frames…", max_entries=2)
def get_frames(stem: str):
    return load_frames(stem)


@st.cache_resource(show_spinner=False)
def held_out_streams() -> dict[str, str]:
    """Replayable files: splits.json test + extra_test only (never training files), stem -> label."""
    splits = load_splits()
    labels = {**{s: "Test" for s in splits["test"]}, **{s: "Extra test" for s in splits["extra_test"]}}
    available = set(list_streams())
    return {s: lab for s, lab in labels.items() if s in available}


@st.cache_resource(show_spinner=False)
def reference_latency_us(model: str, feature_set: str) -> float | None:
    """Single-frame total p50 from the Phase 4 benchmark (optional; read-only)."""
    try:
        d = pd.read_csv(REPORT_DIR / "latency.csv")
        r = d[(d["model"] == model) & (d["feature_set"] == feature_set)]
        return float(r["total_p50_us"].iloc[0]) if len(r) else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------- state + controls

def fmt_t(s: float) -> str:
    s = max(0.0, float(s))
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


def init_state() -> None:
    ss = st.session_state
    for k, v in {"running": False, "engine": None, "last_wall": 0.0, "eff_speed": 0.0,
                 "proc_fps": 0.0, "render_s": 0.1, "error": None}.items():
        ss.setdefault(k, v)


def ensure_engine(stem: str, det) -> ReplayEngine:
    ss = st.session_state
    eng = ss.engine
    if eng is None or eng.frames.stem != stem or eng.detector is not det:
        ss.running = False
        ss.engine = ReplayEngine(get_frames(stem), det, get_stats(), n_alert=ss.get("n_alert", 3))
        ss.eff_speed = ss.proc_fps = 0.0
    return ss.engine


def on_start() -> None:
    ss = st.session_state
    if ss.engine is not None and ss.engine.done:
        ss.engine.reset()
    ss.error = None
    ss.running = True
    ss.last_wall = time.perf_counter()


def on_pause() -> None:
    st.session_state.running = False


def on_reset() -> None:
    ss = st.session_state
    ss.running = False
    ss.error = None
    ss.eff_speed = ss.proc_fps = 0.0
    if ss.engine is not None:
        ss.engine.reset()


def on_file_change() -> None:
    st.session_state.running = False
    st.session_state.engine = None


# ---------------------------------------------------------------- rendering

def kpi_html(eng: ReplayEngine, speed: int) -> str:
    ss = st.session_state
    n = len(eng.frames)
    pct = 100 * eng.i / n if n else 0
    cd_load = eng.chart_data()["load"]
    bus_fps = cd_load[-10:].sum() if len(cd_load) else 0  # last ~1 s of stream
    eff = ss.eff_speed
    cpu_note = ' · <span class="warn">CPU-bound</span>' if ss.running and eff and eff < 0.8 * speed else ""
    status = eng.status
    last = eng.alerts[-1] if eng.alerts else None
    if status == "ATTACK" and last:
        status_sub = f"{last['can_id']} · true: {last['true']}"
    else:
        status_sub = f"last alert at {fmt_t(last['time_s'])}" if last else "no alerts yet"
    cls = "attack" if status == "ATTACK" else "normal"
    return f"""
<div class="kpi-row">
  <div class="kpi"><div class="kpi-label">Frames processed</div>
    <div class="kpi-value">{eng.i:,}</div>
    <div class="kpi-sub">{pct:.1f}% of file · t = {fmt_t(eng.sim_t)} / {fmt_t(eng.frames.duration)}</div></div>
  <div class="kpi"><div class="kpi-label">Frames / s</div>
    <div class="kpi-value">{ss.proc_fps:,.0f}</div>
    <div class="kpi-sub">effective {eff:.1f}× real time{cpu_note} · bus {bus_fps:,.0f}/s</div></div>
  <div class="kpi"><div class="kpi-label">Alerts</div>
    <div class="kpi-value">{eng.n_alerts:,}</div>
    <div class="kpi-sub">{int(eng.flagged.sum()):,} flagged frames · N ≥ {eng.n_alert} in {ALERT_WINDOW_S * 1000:.0f} ms</div></div>
  <div class="kpi {cls}"><div class="kpi-label">Status</div>
    <div class="badge {cls}"><span class="dot"></span>{status}</div>
    <div class="kpi-sub">{status_sub}</div></div>
</div>"""


def _layout(fig: go.Figure, height: int) -> None:
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
        font=dict(family=PLOT_FONT, color=INK_2, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor="rgba(255,255,255,0.25)", font=dict(color=INK)),
        uirevision="keep")
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=INK_3, family=PLOT_MONO, size=11))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=INK_3, family=PLOT_MONO, size=11))


# Figures are built once per session and only their data is replaced each tick: building a plotly
# figure (validation) costs ~150 ms per tick, updating the arrays of an existing one a few ms.

def _build_live_figure(thr: float) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.58, 0.42])
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Attack score", line=dict(color=CYAN, width=2),
                             hovertemplate="t=%{x:.1f}s<br>score %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Above threshold", line=dict(color=RED, width=2.5),
                             connectgaps=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(color=RED, size=8, line=dict(color=SURFACE, width=2))), row=1, col=1)
    fig.add_hline(y=thr, line=dict(color=INK_2, width=1.5, dash="dash"), row=1, col=1,
                  annotation_text=f"threshold {thr:.2f}", annotation_position="top left",
                  annotation_font=dict(color=INK_2, size=11))
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Bus load", fill="tozeroy",
                             line=dict(color=CYAN, width=1.5), fillcolor=rgba(CYAN, 0.12),
                             hovertemplate="t=%{x:.1f}s<br>%{y:.0f} frames<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Bar(x=[], y=[], name="Ground-truth attack", width=0.1,
                         marker=dict(color=rgba(RED, 0.45), line=dict(width=0)),
                         hovertemplate="t=%{x:.1f}s<br>%{y:.0f} frames<extra>true attack</extra>"), row=2, col=1)
    _layout(fig, 470)
    fig.update_layout(hovermode="x unified", bargap=0, showlegend=True)
    fig.update_xaxes(ticksuffix=" s", row=2, col=1)
    axis_title = dict(font=dict(color=INK_2, size=11))
    fig.update_yaxes(title=dict(text="Attack score (max / 100 ms)", **axis_title), row=1, col=1)
    fig.update_yaxes(title=dict(text="Frames / 100 ms", **axis_title), row=2, col=1)
    return fig


def _cached_figure(name: str, key, build) -> go.Figure:
    ss = st.session_state
    if ss.get(f"{name}_key") != key or ss.get(name) is None:
        ss[name], ss[f"{name}_key"] = build(), key
    return ss[name]


def live_figure(eng: ReplayEngine) -> go.Figure:
    thr = eng.detector.threshold
    fig = _cached_figure("fig_live", thr, lambda: _build_live_figure(thr))
    cd = eng.chart_data()
    t, score, load, att = cd["t"], cd["score"], cd["load"], cd["attack"]
    # Above-threshold stretches drawn in red; isolated bins (no red neighbour) get a marker to stay visible.
    above = score > thr
    lone = above & ~np.r_[False, above[:-1]] & ~np.r_[above[1:], False]
    with fig.batch_update():
        fig.data[0].update(x=t, y=score)
        fig.data[1].update(x=t, y=np.where(above, score, np.nan))
        fig.data[2].update(x=t[lone], y=score[lone])
        fig.data[3].update(x=t, y=load)
        fig.data[4].update(x=t[att], y=load[att])
    return fig


def _gauge_layout(fig: go.Figure, height: int, margin: dict) -> None:
    fig.update_layout(template="plotly_dark", height=height, margin=margin,
                      paper_bgcolor=TRANSPARENT, plot_bgcolor=TRANSPARENT,
                      font=dict(family=PLOT_FONT, color=INK_2, size=12))


def _band_color(pct: float) -> str:
    return RED if pct < 50 else AMBER if pct < 80 else CYAN


# 2x2 grid of per-attack gauges (paper coordinates); the label under each sits just below its domain.
GAUGE_DOMAINS = [((0.0, 0.44), (0.63, 1.0)), ((0.56, 1.0), (0.63, 1.0)),
                 ((0.0, 0.44), (0.09, 0.46)), ((0.56, 1.0), (0.09, 0.46))]


def _build_attack_gauges() -> go.Figure:
    fig = go.Figure()
    steps = [dict(range=[0, 50], color=rgba(RED, 0.30)), dict(range=[50, 80], color=rgba(AMBER, 0.30)),
             dict(range=[80, 100], color=rgba(CYAN, 0.30))]
    for name, (dx, dy) in zip(ATTACK_NAMES, GAUGE_DOMAINS):
        fig.add_trace(go.Indicator(
            mode="gauge", value=0, domain=dict(x=dx, y=dy),
            title=dict(text=name, font=dict(color=INK, size=14)),
            number=dict(suffix="%", valueformat=".0f", font=dict(color=INK, size=24, family=PLOT_MONO)),
            gauge=dict(shape="angular", bgcolor=TRANSPARENT, borderwidth=0, steps=steps,
                       bar=dict(color=TRANSPARENT, thickness=0.32),
                       axis=dict(range=[0, 100], tickvals=[50, 80], ticksuffix="%",
                                 tickcolor=INK_3, tickwidth=1, tickfont=dict(color=INK_3, size=10, family=PLOT_MONO)))))
    _gauge_layout(fig, 420, dict(l=24, r=24, t=30, b=24))
    return fig


def attack_gauges(eng: ReplayEngine) -> go.Figure:
    fig = _cached_figure("fig_gauges", "gauges", _build_attack_gauges)
    pa = eng.per_attack()  # ATTACK_NAMES order, same as the traces
    notes = []
    with fig.batch_update():
        for trace, r, (dx, dy) in zip(fig.data, pa.itertuples(), GAUGE_DOMAINS):
            pct = 100 * r.detected / r.total if r.total else 0.0
            trace.update(value=pct, mode="gauge+number" if r.total else "gauge",
                         gauge_bar_color=_band_color(pct) if r.total else TRANSPARENT)
            text = f"{r.detected:,} / {r.total:,}<br>frames detected" if r.total else "not seen yet"
            notes.append(dict(x=(dx[0] + dx[1]) / 2, y=dy[0], xref="paper", yref="paper",
                              xanchor="center", yanchor="top",
                              text=text, showarrow=False, font=dict(color=INK_2, size=12)))
        fig.layout.annotations = notes
    return fig


def alert_table(eng: ReplayEngine) -> pd.DataFrame:
    rows = list(eng.alerts)[::-1][:50]
    return pd.DataFrame({
        "Time": [fmt_t(a["time_s"]) for a in rows],
        "CAN ID": [a["can_id"] for a in rows],
        "Predicted": [a["predicted"] for a in rows],
        "Peak": [round(a["peak_score"], 2) for a in rows],
        "True SubClass": [a["true"] for a in rows],
        "Dur.": [f"{a['end_s'] - a['time_s']:.1f} s" for a in rows],
    })


def alert_log_html(df: pd.DataFrame) -> str:
    """Predicted ATTACK in red; true SubClass red for real attacks, amber for false alarms."""
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for r in df.itertuples(index=False):
        true_cls = "fa" if r[4].startswith("Normal") else "atk"
        rows.append(f'<tr><td class="mono">{r[0]}</td><td class="mono">{html.escape(r[1])}</td>'
                    f'<td class="atk">{html.escape(r[2])}</td><td class="mono num">{r[3]:.2f}</td>'
                    f'<td class="{true_cls}">{html.escape(r[4])}</td><td class="mono num">{r[5]}</td></tr>')
    return f'<div class="alert-log"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def render(eng: ReplayEngine, speed: int) -> None:
    ss = st.session_state
    st.markdown(kpi_html(eng, speed), unsafe_allow_html=True)
    st.progress(min(1.0, eng.i / max(1, len(eng.frames))))
    if ss.error:
        st.error(f"Replay paused after an error: {ss.error}")
    elif eng.done:
        st.success("Replay finished. Press Start to replay again.")

    with st.container(key="glass_live"):
        st.plotly_chart(live_figure(eng), key="live_chart", width="stretch",
                        config={"displayModeBar": False, "responsive": True})

    left, right = st.columns([1.35, 1], gap="medium")
    with left, st.container(key="glass_log"):
        st.markdown('<div class="card-title">Alert log (newest first)</div>', unsafe_allow_html=True)
        df = alert_table(eng)
        if df.empty:
            st.caption("No alerts yet.")
        else:
            st.markdown(alert_log_html(df), unsafe_allow_html=True)
    with right, st.container(key="glass_gauges"):
        st.markdown('<div class="card-title">Per-attack detection (frame level)</div>', unsafe_allow_html=True)
        st.plotly_chart(attack_gauges(eng), key="attack_gauges", width="stretch",
                        config={"displayModeBar": False, "staticPlot": True, "responsive": True})
        fp = int(eng.flagged[0])
        st.caption(f"Flag = score > threshold (before N-aggregation). Normal false positives: "
                   f"{fp:,} (FPR {eng.normal_fpr:.3%}).")


# ---------------------------------------------------------------- page

def main() -> None:
    init_state()
    ss = st.session_state
    st.markdown(CSS, unsafe_allow_html=True)
    det = get_detector()
    labels = held_out_streams()
    stems = list(labels)
    if not stems:
        st.error("No held-out files (splits.json test / extra_test) in data/processed. Run `python -m src.load`.")
        return

    with st.sidebar:
        st.markdown('<div class="side-brand">🛡️ CAN-IDS</div>'
                    '<div class="side-muted">Real-time CAN bus intrusion detection</div>', unsafe_allow_html=True)
        st.divider()
        stem = st.selectbox("Replay file", stems, key="stem", on_change=on_file_change,
                            format_func=lambda s: f"{s} · {labels[s]}",
                            index=stems.index(DEFAULT_STEM) if DEFAULT_STEM in stems else 0)
        st.slider("Replay speed", 1, 100, DEFAULT_SPEED, key="speed", format="%d×")
        speed_panel()
        st.slider("Alert when ≥ N flagged frames within 100 ms", 1, 20, 3, key="n_alert")
        c1, c2, c3 = st.columns(3)
        c1.button("Start", on_click=on_start, type="primary", width="stretch", disabled=ss.running)
        c2.button("Pause", on_click=on_pause, width="stretch", disabled=not ss.running)
        c3.button("Reset", on_click=on_reset, width="stretch")
        st.divider()
        st.markdown('<div class="card-title">Model</div>', unsafe_allow_html=True)
        kind = "rule" if det.fallback else "xgboost"
        ref = reference_latency_us(kind, "behavioral")
        st.markdown(
            f'<div class="side-kv"><span>Model</span><b>{det.name}</b></div>'
            f'<div class="side-kv"><span>Threshold</span><b class="mono">{det.threshold:.4f}</b></div>'
            f'<div class="side-kv"><span>Target FPR</span><b class="mono">{det.target_fpr:.1%}</b></div>'
            + (f'<div class="side-kv"><span>1-frame p50 (bench)</span><b class="mono">{ref:,.0f} µs</b></div>'
               if ref else ""),
            unsafe_allow_html=True)
        if det.fallback:
            st.warning(det.source)
        else:
            st.caption(f"Source: {det.source}")

    try:
        eng = ensure_engine(stem, det)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load {stem}: {type(e).__name__}: {e}")
        return

    st.markdown(f'<div class="hero"><h1>Live CAN intrusion detection</h1>'
                f'<p>Replaying <b>{stem}</b> (mode {eng.frames.mode}) · {len(eng.frames):,} frames · '
                f'{fmt_t(eng.frames.duration)} · causal features → {det.name} → threshold → alert</p></div>',
                unsafe_allow_html=True)
    if det.fallback:
        st.warning("Final XGBoost model or threshold not found: running the dt_ratio rule fallback.")

    with st.sidebar:
        latency_panel()
    live_panel()


def speed_panel() -> None:
    run_every = 1.0 if st.session_state.running else None

    @st.fragment(run_every=run_every)
    def _speed() -> None:
        ss = st.session_state
        requested = int(ss.get("speed", DEFAULT_SPEED))
        achieved = ss.eff_speed if ss.running else 0.0
        if not ss.running:
            note = f"requested <b>{requested}×</b> · paused"
        else:
            note = f"requested <b>{requested}×</b> · achieved <b>{achieved:.1f}×</b>"
            if achieved < 0.8 * requested:
                note += ' · <span class="warn">CPU-bound</span>'
        st.markdown(f'<div class="speed-note">{note}</div>', unsafe_allow_html=True)

    _speed()


@st.fragment(run_every=1.0)
def latency_panel() -> None:
    eng = st.session_state.get("engine")
    lt = eng.last_tick if eng is not None else None
    st.markdown('<div class="card-title" style="margin-top:14px">Live per-frame latency</div>', unsafe_allow_html=True)
    if lt is None or not lt.n_frames:
        st.caption("Starts with the replay.")
        return
    st.markdown(
        f'<div class="side-kv"><span>Features (step)</span><b class="mono">{lt.feature_us:,.1f} µs</b></div>'
        f'<div class="side-kv"><span>Model (batched)</span><b class="mono">{lt.model_us:,.1f} µs</b></div>'
        f'<div class="side-kv"><span>Total</span><b class="mono">{lt.feature_us + lt.model_us:,.1f} µs</b></div>'
        f'<div class="side-muted">last tick: {lt.n_frames:,} frames</div>', unsafe_allow_html=True)


def live_panel() -> None:
    run_every = TICK_S if st.session_state.running else None

    @st.fragment(run_every=run_every)
    def _live() -> None:
        ss = st.session_state
        eng = ss.engine
        speed = int(ss.get("speed", DEFAULT_SPEED))
        finished = False
        if ss.running:
            try:
                now = time.perf_counter()
                dt = min(max(now - ss.last_wall, 1e-3), 1.0)  # long pauses never cause a jump
                ss.last_wall = now
                before = eng.sim_t
                eng.set_n_alert(ss.get("n_alert", 3))
                budget = float(np.clip(TICK_S * TICK_SHARE - ss.render_s, *BUDGET_RANGE))
                tick = eng.advance(eng.sim_t + speed * dt, budget_s=budget)
                ss.eff_speed = 0.7 * ss.eff_speed + 0.3 * (eng.sim_t - before) / dt
                ss.proc_fps = 0.7 * ss.proc_fps + 0.3 * tick.n_frames / dt
                finished = eng.done
            except Exception as e:  # noqa: BLE001 - never crash the demo
                ss.error = f"{type(e).__name__}: {e}"
                finished = True
            if finished:
                ss.running = False
        try:
            t_render = time.perf_counter()
            render(eng, speed)
            ss.render_s = 0.7 * ss.render_s + 0.3 * (time.perf_counter() - t_render)
        except Exception as e:  # noqa: BLE001 - a drawing bug must not take the page down
            st.error(f"Display error (replay state is intact): {type(e).__name__}: {e}")
        if finished:
            st.rerun(scope="app")  # stops the fragment timer and refreshes the buttons

    _live()


main()
