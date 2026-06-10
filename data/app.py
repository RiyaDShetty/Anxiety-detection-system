import streamlit as st
import numpy as np
import pandas as pd
import time
import datetime
from collections import deque

# ===== CONFIG =====
LIVE_OUTPUT_FILE  = "live_output.txt"
ANXIETY_THRESHOLD = 200                 # 20s * 10Hz = 200 consecutive ticks
MAX_HISTORY       = 120
WINDOW_SIZE       = 25
BREATHING_PHASES  = [
    ("INHALE",  4, "#4ade80"),
    ("HOLD",    4, "#facc15"),
    ("EXHALE",  6, "#60a5fa"),
]

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="ANXIO · Anxiety Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===== GLOBAL CSS =====
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root {
    --bg:        #080c14;
    --surface:   #0e1420;
    --surface2:  #141c2e;
    --border:    #1e2d45;
    --accent:    #38bdf8;
    --accent2:   #818cf8;
    --success:   #34d399;
    --warn:      #fb923c;
    --danger:    #f87171;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --mono:      'Space Mono', monospace;
    --sans:      'DM Sans', sans-serif;
}
html, body, [class*="css"] { background-color: var(--bg) !important; color: var(--text) !important; font-family: var(--sans) !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="metric-container"] { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; padding: 18px 22px !important; }
[data-testid="stMetricValue"]  { font-family: var(--mono) !important; font-size: 2rem !important; color: var(--accent) !important; }
[data-testid="stMetricLabel"]  { color: var(--muted) !important; font-size: 0.75rem !important; letter-spacing: 0.1em; text-transform: uppercase; }
[data-testid="stMetricDelta"]  { font-family: var(--mono) !important; }
div.stButton > button { background: linear-gradient(135deg, var(--accent), var(--accent2)) !important; color: #080c14 !important; border: none !important; border-radius: 8px !important; font-family: var(--mono) !important; font-weight: 700 !important; letter-spacing: 0.05em !important; padding: 0.5rem 1.5rem !important; transition: opacity .2s, transform .1s !important; }
div.stButton > button:hover { opacity: .85 !important; transform: translateY(-1px) !important; }
.stAlert { border-radius: 10px !important; font-family: var(--sans) !important; }
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; border-radius: 4px !important; }
details { background: var(--surface2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
summary { padding: 10px 16px !important; }
.section-title { font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted); margin: 1.5rem 0 0.5rem; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
.badge { display: inline-block; padding: 4px 14px; border-radius: 999px; font-family: var(--mono); font-size: 0.75rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
.badge-calm    { background: #0d2d22; color: var(--success); border: 1px solid #166534; }
.badge-anxiety { background: #2d1010; color: var(--danger);  border: 1px solid #991b1b; animation: pulse 1.2s infinite; }
.badge-idle    { background: var(--surface2); color: var(--muted); border: 1px solid var(--border); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .55; } }
.breath-ring { width: 140px; height: 140px; border-radius: 50%; border: 3px solid var(--accent); display: flex; align-items: center; justify-content: center; margin: 0 auto 8px; font-family: var(--mono); font-size: 0.9rem; font-weight: 700; color: var(--accent); animation: breathe 14s ease-in-out infinite; }
@keyframes breathe { 0%,100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(56,189,248,.35); } 30% { transform: scale(1.3); box-shadow: 0 0 24px 8px rgba(56,189,248,.2); } 50% { transform: scale(1.3); box-shadow: 0 0 24px 8px rgba(56,189,248,.2); } 80% { transform: scale(.9); box-shadow: 0 0 0 0 rgba(56,189,248,.0); } }
.log-row { display: flex; justify-content: space-between; padding: 6px 10px; border-bottom: 1px solid var(--border); font-family: var(--mono); font-size: 0.75rem; }
.log-row:nth-child(even) { background: var(--surface2); }
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ===== SESSION STATE =====
defaults = {
    "running":         False,
    "history":         deque(maxlen=MAX_HISTORY),
    "session_start":   None,
    "total_readings":  0,
    "anxiety_events":  0,
    "calm_streak":     0,
    "anxiety_counter": 0,
    "log":             [],
    "breath_step":     0,
    "breath_tick":     0,
    "last_line":       "",   # KEY FIX: track last processed line to avoid re-processing
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ===== HELPERS =====
def fmt_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def anxiety_risk_score(counter, threshold):
    return min(100, int(counter / threshold * 100))

def read_live_output():
    """Read latest line from live_output.txt.
    Returns (raw_line, status, p, r, counter) or None on failure."""
    try:
        with open(LIVE_OUTPUT_FILE, "r") as f:
            line = f.read().strip()
        if not line:
            return None
        parts   = line.split(',')
        status  = parts[0]
        p       = float(parts[1])
        r       = float(parts[2])
        counter = int(parts[3])
        return line, status, p, r, counter
    except (FileNotFoundError, ValueError, IndexError):
        return None

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown('<p class="section-title">⚙ Settings</p>', unsafe_allow_html=True)
    threshold_sec     = st.slider("Anxiety alert threshold (sec)", 5, 60, 20, 1)
    ANXIETY_THRESHOLD = threshold_sec * 10

    show_raw = st.checkbox("Show raw sensor values", value=True)
    show_log = st.checkbox("Show event log",         value=True)

    st.markdown('<p class="section-title">ℹ About</p>', unsafe_allow_html=True)
    st.caption(
        "**ANXIO** reads predictions from **live_predict.py** in real-time. "
        "Alert fires only after the set consecutive seconds of anxiety — "
        "any calm tick resets the streak to zero."
    )
    st.caption("Model: Random Forest · Features: 19 · Window: 25 samples · ~10 Hz")

# ===== HEADER =====
col_title, col_badge = st.columns([6, 2])
with col_title:
    st.markdown("## 🧠 ANXIO")
    st.caption("Real-time anxiety detection via wrist-worn BLE IMU sensor")
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    if not st.session_state.running:
        st.markdown('<span class="badge badge-idle">● IDLE</span>', unsafe_allow_html=True)
    elif st.session_state.anxiety_counter >= ANXIETY_THRESHOLD:
        st.markdown('<span class="badge badge-anxiety">⚠ ANXIETY</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-calm">● CALM</span>', unsafe_allow_html=True)

st.markdown("---")

# ===== START / STOP =====
c1, c2, _ = st.columns([2, 2, 4])
with c1:
    if not st.session_state.running:
        if st.button("▶  Start Monitoring"):
            st.session_state.running        = True
            st.session_state.session_start  = time.time()
            st.session_state.anxiety_events = 0
            st.session_state.total_readings = 0
            st.session_state.calm_streak    = 0
            st.session_state.log            = []
            st.session_state.last_line      = ""
            st.rerun()
    else:
        if st.button("■  Stop"):
            st.session_state.running = False
            st.rerun()
with c2:
    if st.button("🗑  Reset Session"):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()

# ===== KPI ROW =====
st.markdown('<p class="section-title">Session Overview</p>', unsafe_allow_html=True)
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
elapsed = (time.time() - st.session_state.session_start) if st.session_state.session_start else 0
with kpi1: st.metric("Session Duration",    fmt_duration(elapsed))
with kpi2: st.metric("Total Readings",      st.session_state.total_readings)
with kpi3: st.metric("Anxiety Events",      st.session_state.anxiety_events)
with kpi4:
    denom = max(1, st.session_state.total_readings // WINDOW_SIZE)
    st.metric("Anxiety Rate", f"{st.session_state.anxiety_events / denom * 100:.1f}%")
with kpi5: st.metric("Calm Streak (ticks)", st.session_state.calm_streak)

# ===== MAIN COLUMNS =====
left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown('<p class="section-title">Live Sensor Feed</p>', unsafe_allow_html=True)
    pitch_ph = st.empty()
    risk_ph  = st.empty()
    chart_ph = st.empty()
with right:
    st.markdown('<p class="section-title">Status & Intervention</p>', unsafe_allow_html=True)
    status_ph = st.empty()
    breath_ph = st.empty()
    help_ph   = st.empty()
    if show_log:
        st.markdown('<p class="section-title">Event Log</p>', unsafe_allow_html=True)
        log_ph = st.empty()

# ===== RENDER HELPERS =====
def render_chart():
    if len(st.session_state.history) < 2:
        return
    hist = list(st.session_state.history)
    df   = pd.DataFrame(hist, columns=["ts", "pitch", "roll", "label"])
    chart_ph.area_chart(
        df[["pitch", "roll"]].rename(columns={"pitch": "Pitch", "roll": "Roll"}),
        use_container_width=True, height=160,
    )

def render_log():
    if not show_log:
        return
    if not st.session_state.log:
        log_ph.caption("No events yet.")
        return
    rows = ""
    for ev in reversed(st.session_state.log[-20:]):
        color = "#f87171" if ev["type"] == "ANXIETY" else "#34d399"
        rows += (
            f'<div class="log-row">'
            f'<span style="color:var(--muted)">{ev["time"]}</span>'
            f'<span style="color:{color};font-weight:700">{ev["type"]}</span>'
            f'<span>{ev["detail"]}</span>'
            f'</div>'
        )
    log_ph.markdown(
        f'<div style="max-height:220px;overflow-y:auto;border:1px solid var(--border);border-radius:8px">{rows}</div>',
        unsafe_allow_html=True,
    )

def render_breathing():
    phase_label, phase_sec, phase_color = BREATHING_PHASES[
        st.session_state.breath_step % len(BREATHING_PHASES)
    ]
    st.session_state.breath_tick += 1
    if st.session_state.breath_tick >= phase_sec * 10:
        st.session_state.breath_tick = 0
        st.session_state.breath_step += 1
    breath_ph.markdown(f"""
<div style="text-align:center;padding:10px 0">
  <div class="breath-ring" style="border-color:{phase_color};color:{phase_color}">{phase_label}</div>
  <p style="font-family:var(--mono);font-size:.75rem;color:var(--muted);margin:0">
    {phase_sec}s · box breathing 4-4-6
  </p>
</div>""", unsafe_allow_html=True)

def render_status():
    """Render status panel using current session state — no new data needed."""
    file_counter = st.session_state.anxiety_counter
    risk         = anxiety_risk_score(file_counter, ANXIETY_THRESHOLD)
    risk_color   = "#34d399" if risk < 40 else "#facc15" if risk < 75 else "#f87171"

    risk_ph.markdown(f"""
<div style="margin-top:6px">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px">
    <span style="font-family:var(--mono);font-size:.7rem;color:var(--muted)">
      ANXIETY RISK &nbsp;·&nbsp; {file_counter} / {ANXIETY_THRESHOLD} ticks
    </span>
    <span style="font-family:var(--mono);font-size:.7rem;color:{risk_color};font-weight:700">{risk}%</span>
  </div>
</div>""", unsafe_allow_html=True)
    risk_ph.progress(risk / 100)

    if file_counter >= ANXIETY_THRESHOLD:
        status_ph.error(
            f"⚠️  Anxiety sustained for {threshold_sec}s+ — please take a moment to breathe."
        )
        render_breathing()
        help_ph.markdown(f"""
<div style="background:#1a1010;border:1px solid #7f1d1d;border-radius:10px;padding:14px 18px;margin-top:8px">
  <p style="font-family:var(--mono);font-size:.75rem;color:#f87171;margin:0 0 6px">INTERVENTION ACTIVE</p>
  <p style="font-size:.9rem;margin:0 0 10px">
    Follow the breathing guide above.<br>
    Inhale <b>4s</b> → Hold <b>4s</b> → Exhale <b>6s</b>
  </p>
  <details>
    <summary style="cursor:pointer;font-size:.82rem;color:var(--accent)">📞 Need more help?</summary>
    <p style="font-size:.82rem;margin:8px 0 0">
      • Talk to someone you trust<br>
      • iCall (India): <b>9152987821</b><br>
      • Vandrevala Foundation: <b>1860-2662-345</b>
    </p>
  </details>
</div>""", unsafe_allow_html=True)
    else:
        if risk > 40:
            status_ph.warning(
                f"🟡 Anxiety building ({risk}% toward {threshold_sec}s threshold). Stay relaxed."
            )
        else:
            status_ph.success("✅  All clear — calm state detected.")
        breath_ph.empty()
        help_ph.empty()

# ===== MAIN LOOP =====
if st.session_state.running:
    result = read_live_output()

    if result is None:
        status_ph.warning("⏳ Waiting for `live_predict.py` to write data... Make sure it's running.")
        time.sleep(0.2)
        st.rerun()

    raw_line, file_status, p, r, file_counter = result

    # ── KEY FIX: only process if live_predict.py wrote a NEW line ──
    if raw_line != st.session_state.last_line:
        st.session_state.last_line = raw_line

        prev_counter                     = st.session_state.anxiety_counter
        st.session_state.anxiety_counter = file_counter
        st.session_state.total_readings += 1
        ts = datetime.datetime.now().strftime("%H:%M:%S")

        # Calm streak: counter==0 means live_predict just hard-reset on a calm tick
        if file_counter == 0:
            st.session_state.calm_streak += 1
            if prev_counter >= ANXIETY_THRESHOLD:
                st.session_state.log.append({
                    "time": ts, "type": "CALM",
                    "detail": "Anxiety streak broken — counter reset to 0",
                })
        else:
            st.session_state.calm_streak = 0

        # Log new anxiety alert — rising edge only
        if file_counter >= ANXIETY_THRESHOLD and prev_counter < ANXIETY_THRESHOLD:
            st.session_state.anxiety_events += 1
            st.session_state.log.append({
                "time": ts, "type": "ANXIETY",
                "detail": f"P={p:.1f}° R={r:.1f}° — {threshold_sec}s sustained",
            })

        # Sensor metrics
        if show_raw:
            hist = list(st.session_state.history)
            prev = hist[-1] if hist else None
            mc1, mc2 = pitch_ph.columns(2)
            mc1.metric("Pitch (°)", f"{p:.2f}", delta=f"{p - prev[1]:.2f}" if prev else None)
            mc2.metric("Roll  (°)", f"{r:.2f}", delta=f"{r - prev[2]:.2f}" if prev else None)

        # History
        final_pred = 1 if file_status == "ANXIETY" else 0
        st.session_state.history.append((ts, p, r, final_pred))

    # Always render UI from current state (even if no new data)
    render_status()
    render_chart()
    render_log()

    time.sleep(0.1)
    st.rerun()

else:
    render_chart()
    if show_log:
        render_log()
    if not st.session_state.history:
        pitch_ph.info("Press **▶ Start Monitoring** to begin a session.")