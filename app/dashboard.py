"""CAN-IDS live demo: replay a CAN file through the causal feature extractor and the final model.

Run from the repo root:  streamlit run app/dashboard.py

Smoothness: only the live section reruns (st.fragment(run_every=TICK_S)); each tick processes a batch
of frames within a compute budget adapted to the render time (src/live.py ReplayEngine.advance),
charts keep a rolling 150 s window, and model/frames are cached once per process. Any error during a
tick pauses the replay instead of crashing.
"""
from __future__ import annotations

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

from src.live import (ALERT_WINDOW_S, ReplayEngine, list_streams, load_detector,  # noqa: E402
                      load_frames)
from src.train import REPORT_DIR  # noqa: E402
from src.train_stats import load_train_stats  # noqa: E402

DEFAULT_STEM = "Pre_submit_D"
SPEEDS = {"1×": 1, "10×": 10, "100×": 100}
TICK_S = 0.5           # UI refresh interval of the live section
TICK_SHARE = 0.6       # compute + render must fit in this share of TICK_S: a run still going when the
                       # next timed rerun arrives is cut short, and the elements at the bottom go stale
BUDGET_RANGE = (0.05, 0.30)  # compute budget per tick, adapted to the measured render time
TEAL, RED, SLATE = "#0F9F94", "#F43F5E", "#475569"  # normal / attack / neutral (validated on #0E1526)
INK, INK_2, INK_3 = "#E6ECF5", "#A9B6CF", "#6B7A96"
GRID = "rgba(255,255,255,0.06)"

st.set_page_config(page_title="CAN-IDS Live", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp {{
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  background:
    radial-gradient(1100px 600px at 8% -10%, rgba(15,159,148,0.13), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(244,63,94,0.07), transparent 60%),
    linear-gradient(180deg, #0A1020 0%, #05070D 100%);
  background-attachment: fixed;
  color: {INK};
}}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px; }}
/* no dimming of elements while a fragment reruns */
[data-stale="true"] {{ opacity: 1 !important; transition: none !important; }}

section[data-testid="stSidebar"] {{
  background: rgba(10,16,32,0.72); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
  border-right: 1px solid rgba(255,255,255,0.08);
}}

.stApp [class*="st-key-glass"] {{
  background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px; padding: 18px 20px 14px 20px;
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}}
.card-title {{ font-size: 0.78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  color: {INK_2}; margin: 0 0 6px 2px; }}

.hero h1 {{ font-size: 1.65rem; font-weight: 700; margin: 0; color: {INK}; letter-spacing: -0.01em; }}
.hero p {{ color: {INK_2}; margin: 4px 0 18px 0; font-size: 0.92rem; }}

.kpi-row {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; margin-bottom: 16px; }}
@media (max-width: 900px) {{ .kpi-row {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.kpi {{
  background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); border-radius: 16px;
  padding: 16px 20px; backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.35); min-height: 112px;
}}
.kpi-label {{ font-size: 0.74rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: {INK_2}; }}
.kpi-value {{ font-size: 1.9rem; font-weight: 700; color: {INK}; margin-top: 6px; font-variant-numeric: tabular-nums; }}
.kpi-sub {{ font-size: 0.8rem; color: {INK_3}; margin-top: 2px; font-variant-numeric: tabular-nums; }}
.badge {{ display: inline-flex; align-items: center; gap: 10px; margin-top: 8px; padding: 8px 16px;
  border-radius: 999px; font-weight: 700; font-size: 1.15rem; letter-spacing: .06em; }}
.badge .dot {{ width: 11px; height: 11px; border-radius: 50%; }}
.badge.normal {{ background: rgba(15,159,148,0.16); border: 1px solid rgba(15,159,148,0.55); color: #8EF0E4; }}
.badge.normal .dot {{ background: {TEAL}; }}
.badge.attack {{ background: rgba(244,63,94,0.18); border: 1px solid rgba(244,63,94,0.7); color: #FFC2CD;
  animation: pulse 1.2s ease-in-out infinite; }}
.badge.attack .dot {{ background: {RED}; }}
.kpi.attack {{ border-color: rgba(244,63,94,0.55); box-shadow: 0 0 28px rgba(244,63,94,0.25); }}
@keyframes pulse {{ 0%,100% {{ box-shadow: 0 0 0 0 rgba(244,63,94,0.45); }} 50% {{ box-shadow: 0 0 0 10px rgba(244,63,94,0); }} }}

.side-brand {{ font-weight: 700; font-size: 1.1rem; color: {INK}; margin-bottom: 2px; }}
.side-muted {{ color: {INK_3}; font-size: 0.8rem; }}
.side-kv {{ display: flex; justify-content: space-between; gap: 14px; font-size: 0.86rem; padding: 3px 0; color: {INK_2}; }}
.side-kv span {{ flex: none; }}
.side-kv b {{ color: {INK}; font-weight: 600; font-variant-numeric: tabular-nums; text-align: right; }}
div[data-testid="stProgress"] > div > div > div > div {{ background: {TEAL}; }}
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
    cpu_note = " · CPU-bound" if ss.running and eff and eff < 0.8 * speed else ""
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
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, Segoe UI, sans-serif", color=INK_2, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor="#0E1526", bordercolor="rgba(255,255,255,0.2)", font=dict(color=INK)),
        uirevision="keep")
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=INK_3))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont=dict(color=INK_3))


# Figures are built once per session and only their data is replaced each tick: building a plotly
# figure (validation) costs ~150 ms per tick, updating the arrays of an existing one a few ms.

def _build_live_figure(thr: float) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.58, 0.42])
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Attack score", line=dict(color=TEAL, width=2),
                             hovertemplate="t=%{x:.1f}s<br>score %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Above threshold", line=dict(color=RED, width=2.5),
                             connectgaps=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(color=RED, size=8, line=dict(color="#0E1526", width=2))), row=1, col=1)
    fig.add_hline(y=thr, line=dict(color=INK_2, width=1.5, dash="dash"), row=1, col=1,
                  annotation_text=f"threshold {thr:.2f}", annotation_position="top left",
                  annotation_font=dict(color=INK_2, size=11))
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", name="Bus load", fill="tozeroy",
                             line=dict(color=TEAL, width=1.5), fillcolor="rgba(15,159,148,0.18)",
                             hovertemplate="t=%{x:.1f}s<br>%{y:.0f} frames<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Bar(x=[], y=[], name="Ground-truth attack", width=0.1,
                         marker=dict(color="rgba(244,63,94,0.55)", line=dict(width=0)),
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


def _build_bar_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(y=[], x=[], orientation="h", name="Detected",
                         marker=dict(color=TEAL, line=dict(color="#0E1526", width=2)),
                         hovertemplate="%{y}: %{x:,} detected<extra></extra>"))
    fig.add_trace(go.Bar(y=[], x=[], orientation="h", name="Missed",
                         marker=dict(color=SLATE, line=dict(color="#0E1526", width=2)),
                         hovertemplate="%{y}: %{x:,} missed<extra></extra>"))
    _layout(fig, 300)
    fig.update_layout(barmode="stack", bargap=0.35, hovermode="closest", legend_traceorder="normal")
    fig.update_xaxes(title=dict(text="attack frames", font=dict(color=INK_3, size=11)))
    return fig


def per_attack_figure(eng: ReplayEngine) -> go.Figure:
    fig = _cached_figure("fig_bar", "bar", _build_bar_figure)
    pa = eng.per_attack().iloc[::-1]  # Flooding on top
    labels = [f"{r.detected / r.total:.0%}" if r.total else "not seen yet" for r in pa.itertuples()]
    with fig.batch_update():
        fig.data[0].update(y=pa["subclass"].tolist(), x=pa["detected"].to_numpy())
        fig.data[1].update(y=pa["subclass"].tolist(), x=pa["missed"].to_numpy())
        fig.layout.annotations = [dict(x=int(tot), y=sc, text=lab, showarrow=False, xanchor="left", xshift=6,
                                       font=dict(color=INK, size=12))
                                  for sc, tot, lab in zip(pa["subclass"], pa["total"], labels)]
        fig.layout.xaxis.range = [0, max(1, int(pa["total"].max())) * 1.18]
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
            st.dataframe(df, hide_index=True, height=300, width="stretch", column_config={
                "Time": st.column_config.TextColumn(width="small"),
                "CAN ID": st.column_config.TextColumn(width="small"),
                "Predicted": st.column_config.TextColumn(width="small"),
                "Peak": st.column_config.NumberColumn(width="small", format="%.2f"),
                "True SubClass": st.column_config.TextColumn(width="medium"),
                "Dur.": st.column_config.TextColumn(width="small")})
    with right, st.container(key="glass_bars"):
        st.markdown('<div class="card-title">Per-attack detection (frame level)</div>', unsafe_allow_html=True)
        st.plotly_chart(per_attack_figure(eng), key="bar_chart", width="stretch",
                        config={"displayModeBar": False, "responsive": True})
        fp = int(eng.flagged[0])
        st.caption(f"Flag = score > threshold (before N-aggregation). Normal false positives: "
                   f"{fp:,} (FPR {eng.normal_fpr:.3%}).")


# ---------------------------------------------------------------- page

def main() -> None:
    init_state()
    ss = st.session_state
    st.markdown(CSS, unsafe_allow_html=True)
    det = get_detector()
    stems = list_streams()
    if not stems:
        st.error("No files in data/processed. Run `python -m src.load` first.")
        return

    with st.sidebar:
        st.markdown('<div class="side-brand">🛡️ CAN-IDS</div>'
                    '<div class="side-muted">Real-time CAN bus intrusion detection</div>', unsafe_allow_html=True)
        st.divider()
        stem = st.selectbox("Replay file", stems, key="stem", on_change=on_file_change,
                            index=stems.index(DEFAULT_STEM) if DEFAULT_STEM in stems else 0)
        speed_label = st.radio("Replay speed", list(SPEEDS), index=1, horizontal=True, key="speed_label")
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
            f'<div class="side-kv"><span>Threshold</span><b>{det.threshold:.4f}</b></div>'
            f'<div class="side-kv"><span>Target FPR</span><b>{det.target_fpr:.1%}</b></div>'
            + (f'<div class="side-kv"><span>Single-frame p50 (benchmark)</span><b>{ref:,.0f} µs</b></div>' if ref else ""),
            unsafe_allow_html=True)
        if det.fallback:
            st.warning(det.source)
        else:
            st.caption(f"Source: {det.source}")

    speed = SPEEDS[speed_label]
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
    live_panel(speed)


@st.fragment(run_every=1.0)
def latency_panel() -> None:
    eng = st.session_state.get("engine")
    lt = eng.last_tick if eng is not None else None
    st.markdown('<div class="card-title" style="margin-top:14px">Live per-frame latency</div>', unsafe_allow_html=True)
    if lt is None or not lt.n_frames:
        st.caption("Starts with the replay.")
        return
    st.markdown(
        f'<div class="side-kv"><span>Features (step)</span><b>{lt.feature_us:,.1f} µs</b></div>'
        f'<div class="side-kv"><span>Model (batched)</span><b>{lt.model_us:,.1f} µs</b></div>'
        f'<div class="side-kv"><span>Total</span><b>{lt.feature_us + lt.model_us:,.1f} µs</b></div>'
        f'<div class="side-muted">last tick: {lt.n_frames:,} frames</div>', unsafe_allow_html=True)


def live_panel(speed: int) -> None:
    run_every = TICK_S if st.session_state.running else None

    @st.fragment(run_every=run_every)
    def _live() -> None:
        ss = st.session_state
        eng = ss.engine
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
