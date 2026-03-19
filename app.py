import re, io, os, warnings, zipfile
import pandas as pd
import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

st.set_page_config(page_title="TalentLens", page_icon="◎", layout="wide", initial_sidebar_state="collapsed")

# ─────────────────────────────────────────────────────────────────────────────
#  FULL CSS + CURSOR FIX + NO DUPLICATE BUTTONS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@300;400;500&display=swap');

/* ══ HIDE STREAMLIT CHROME ══ */
#MainMenu, footer, header,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebarNav"], [data-testid="collapsedControl"],
[data-testid="stHeader"], [data-testid="stStatusWidget"],
.stDeployButton { display:none !important; visibility:hidden !important; }

/* ══ TOKENS ══ */
:root {
  --bg:#080810; --bg1:#0d0d18; --bg2:#111120; --bg3:#161628; --bg4:#1c1c32;
  --p:#6366f1; --p2:#818cf8; --p3:#a5b4fc;
  --p-glow:rgba(99,102,241,0.3); --p-dim:rgba(99,102,241,0.1); --p-line:rgba(99,102,241,0.2);
  --cyan:#22d3ee; --cyan-dim:rgba(34,211,238,0.08); --cyan-line:rgba(34,211,238,0.18);
  --emerald:#10b981; --em-dim:rgba(16,185,129,0.08); --em-line:rgba(16,185,129,0.2);
  --amber:#f59e0b; --am-dim:rgba(245,158,11,0.08); --am-line:rgba(245,158,11,0.2);
  --rose:#f43f5e; --ro-dim:rgba(244,63,94,0.08);
  --t0:#ffffff; --t1:#e2e2f0; --t2:#9090b0; --t3:#55556a; --t4:#2a2a42;
  --b1:rgba(255,255,255,0.08); --b2:rgba(255,255,255,0.04); --b3:rgba(255,255,255,0.02);
  --r1:6px; --r2:12px; --r3:18px; --r4:24px;
  --spring:cubic-bezier(0.34,1.56,0.64,1); --smooth:cubic-bezier(0.4,0,0.2,1); --out:cubic-bezier(0,0,0.2,1);
  --sha:0 1px 3px rgba(0,0,0,.5),0 4px 16px rgba(0,0,0,.3);
  --shb:0 4px 20px rgba(0,0,0,.6),0 2px 6px rgba(0,0,0,.4);
  --shc:0 16px 64px rgba(0,0,0,.7),0 4px 16px rgba(0,0,0,.5);
  --shg:0 0 40px rgba(99,102,241,.15),0 0 80px rgba(99,102,241,.08);
}

/* ══ BASE ══ */
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],section.main,.stApp {
  background:var(--bg) !important; color:var(--t1) !important;
  font-family:'Geist',system-ui,sans-serif !important;
  -webkit-font-smoothing:antialiased;
}
.main .block-container { padding:0 2.5rem 8rem !important; max-width:1280px !important; }
::-webkit-scrollbar { width:2px; } ::-webkit-scrollbar-thumb { background:var(--p); border-radius:99px; }

/* ══ CURSOR — dot + ring, always on top ══ */
#tl-dot {
  position:fixed; z-index:2147483647; pointer-events:none;
  width:10px; height:10px; border-radius:50%;
  background:#818cf8;
  box-shadow:0 0 0 2px rgba(129,140,248,0.35),0 0 16px rgba(99,102,241,0.9),0 0 32px rgba(99,102,241,0.35);
  transform:translate(-50%,-50%);
  left:-100px; top:-100px;
  transition:width .15s,height .15s,background .15s,box-shadow .15s,opacity .2s;
  opacity:0;
}
#tl-ring {
  position:fixed; z-index:2147483646; pointer-events:none;
  width:36px; height:36px; border-radius:50%;
  border:1.5px solid rgba(129,140,248,0.55);
  transform:translate(-50%,-50%);
  left:-100px; top:-100px;
  transition:width .3s var(--spring),height .3s var(--spring),border-color .2s,background .2s,opacity .2s;
  background:transparent; opacity:0;
}

/* ══ MESH BG ══ */
.tl-mesh { position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden; }
.tl-mesh-g {
  position:absolute;inset:0;
  background:
    radial-gradient(ellipse 60% 50% at 20% 20%,rgba(99,102,241,.07) 0%,transparent 60%),
    radial-gradient(ellipse 50% 40% at 80% 80%,rgba(34,211,238,.05) 0%,transparent 60%),
    radial-gradient(ellipse 40% 35% at 60% 10%,rgba(99,102,241,.04) 0%,transparent 50%);
  animation:mesh 20s ease-in-out infinite alternate;
}
@keyframes mesh { 0%{opacity:1;transform:scale(1)} 50%{opacity:.7;transform:scale(1.05) translate(-1%,1%)} 100%{opacity:1;transform:scale(1) translate(1%,-.5%)} }
.tl-mesh-grid {
  position:absolute;inset:0;
  background-image:linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);
  background-size:72px 72px;
  mask-image:radial-gradient(ellipse 100% 100% at 50% 0%,black 30%,transparent 80%);
}
.tl-mesh-noise {
  position:absolute;inset:0;opacity:.022;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  background-size:200px 200px;
}

/* ══ NAVBAR ══ */
.tl-nav {
  display:flex; align-items:center; justify-content:space-between;
  padding:1.4rem 0 1.2rem; border-bottom:1px solid var(--b1);
  position:relative; z-index:20;
  animation:slideDown .5s var(--out) both;
}
@keyframes slideDown { from{opacity:0;transform:translateY(-10px)} to{opacity:1;transform:none} }
.tl-nav::after {
  content:''; position:absolute; bottom:-1px; left:0;
  width:180px; height:1px;
  background:linear-gradient(90deg,var(--p),transparent);
}
.tl-brand { display:flex; align-items:center; gap:12px; }
.tl-logomark { width:36px; height:36px; flex-shrink:0; }
.tl-logomark svg { width:36px; height:36px; display:block; }
.tl-brand-text { display:flex; flex-direction:column; gap:3px; }
.tl-brand-name {
  font-family:'Instrument Serif',serif;
  font-size:1.25rem; font-weight:400; color:var(--t0);
  letter-spacing:-.01em; line-height:1;
}
.tl-brand-name em { color:var(--p2); font-style:italic; }
/* FIXED: tagline now clearly visible */
.tl-brand-tag {
  font-family:'Geist Mono',monospace;
  font-size:0.6rem; letter-spacing:.18em; text-transform:uppercase;
  color:var(--t2); /* was --t3 (too dark), now --t2 */
  line-height:1;
}

/* Nav tab pills */
.tl-nav-tabs {
  display:flex; align-items:center; gap:2px;
  background:var(--bg2); border:1px solid var(--b1);
  border-radius:var(--r2); padding:3px;
}
.tl-nav-tab {
  padding:.45rem 1.2rem; border-radius:10px;
  font-family:'Geist',sans-serif; font-size:.82rem; font-weight:500;
  color:var(--t2); border:1px solid transparent;
  transition:all .2s var(--smooth); white-space:nowrap;
  display:flex; align-items:center; gap:7px;
}
.tl-nav-tab:hover { color:var(--t1); background:var(--bg3); }
.tl-nav-tab.on {
  background:var(--bg3); color:var(--t0);
  border-color:var(--b1); box-shadow:var(--sha);
}
.tl-tab-dot {
  width:6px; height:6px; border-radius:50%; background:var(--p2);
  box-shadow:0 0 6px var(--p-glow);
  animation:pulse 2.5s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{box-shadow:0 0 6px var(--p-glow)} 50%{box-shadow:0 0 12px var(--p-glow),0 0 20px rgba(99,102,241,.1)} }

/* Nav stats */
.tl-nav-stats { display:flex; align-items:center; }
.tl-stat { display:flex; flex-direction:column; align-items:flex-end; gap:2px; padding:0 1.4rem; border-right:1px solid var(--b1); }
.tl-stat:last-child { border-right:none; padding-right:0; }
.tl-stat-n { font-family:'Geist Mono',monospace; font-size:1rem; font-weight:500; color:var(--t0); letter-spacing:-.04em; line-height:1; }
.tl-stat-k { font-family:'Geist Mono',monospace; font-size:.48rem; letter-spacing:.18em; text-transform:uppercase; color:var(--t3); }

/* ══ STREAMLIT BUTTON BASE STYLE ══ */
[data-testid="stButton"] button {
  background:transparent !important; border:1px solid var(--b1) !important;
  color:var(--t2) !important; font-family:'Geist',sans-serif !important;
  font-size:.82rem !important; font-weight:500 !important;
  padding:.55rem 1.2rem !important; border-radius:10px !important;
  transition:all .2s !important; width:100% !important;
  letter-spacing:-.01em !important;
}
[data-testid="stButton"] button:hover {
  color:var(--t1) !important; background:var(--bg3) !important;
}
[data-testid="stButton"] button[kind="primary"] {
  background:var(--bg3) !important; color:var(--t0) !important;
  border-color:var(--p-line) !important;
  box-shadow:inset 0 0 0 1px var(--p-line),0 0 12px rgba(99,102,241,.1) !important;
}

/* ══ HERO ══ */
.hero-main {
  background:var(--bg1); border:1px solid var(--b1); border-radius:var(--r4);
  padding:3rem 3.2rem 2.8rem; position:relative; overflow:hidden; margin-top:2rem;
  animation:heroIn .6s var(--out) .1s both;
}
@keyframes heroIn { from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:none} }
.hero-main::before { content:''; position:absolute; top:-100px; right:-60px; width:360px; height:360px; background:radial-gradient(circle,rgba(99,102,241,.12) 0%,transparent 65%); pointer-events:none; }
.hero-main::after { content:''; position:absolute; top:0;left:0;right:0; height:1px; background:linear-gradient(90deg,transparent,var(--p-line) 30%,var(--p) 55%,var(--cyan-line) 80%,transparent); }
.hero-badge { display:inline-flex; align-items:center; gap:8px; background:var(--p-dim); border:1px solid var(--p-line); border-radius:100px; padding:5px 14px 5px 8px; margin-bottom:1.6rem; }
.hero-badge-dot { width:6px; height:6px; border-radius:50%; background:var(--p2); box-shadow:0 0 8px var(--p-glow); animation:pulse 2.5s ease-in-out infinite; }
.hero-badge-txt { font-family:'Geist Mono',monospace; font-size:.6rem; letter-spacing:.12em; text-transform:uppercase; color:var(--p2); }
.hero-h1 { font-family:'Instrument Serif',serif !important; font-size:clamp(2.8rem,4.2vw,4.6rem) !important; font-weight:400 !important; line-height:1.02 !important; color:var(--t0) !important; letter-spacing:-.02em !important; margin:0 0 1.4rem !important; }
.hero-h1 em { color:transparent !important; background:linear-gradient(135deg,var(--p2),var(--cyan)); -webkit-background-clip:text; background-clip:text; font-style:italic; }
.hero-p { font-size:1rem; color:var(--t2); line-height:1.7; max-width:460px; margin-bottom:2rem; font-weight:300; }
.hero-chips { display:flex; flex-wrap:wrap; gap:6px; }
.hero-chip { display:inline-flex; align-items:center; gap:6px; background:var(--bg2); border:1px solid var(--b1); border-radius:100px; padding:5px 12px; font-family:'Geist Mono',monospace; font-size:.62rem; color:var(--t2); transition:all .2s; }
.hero-chip:hover { border-color:var(--p-line); color:var(--p2); background:var(--p-dim); }
.hero-chip-dot { width:4px; height:4px; border-radius:50%; background:var(--p2); }
.hero-chip b { color:var(--t0); font-weight:600; }

/* ══ UPLOAD PANEL ══ */
.upload-panel { background:var(--bg1); border:1px solid var(--b1); border-radius:var(--r4); padding:2rem; display:flex; flex-direction:column; gap:1.2rem; position:relative; overflow:hidden; margin-top:2rem; animation:heroIn .6s var(--out) .2s both; }
.upload-panel::after { content:''; position:absolute; top:0;left:0;right:0; height:1px; background:linear-gradient(90deg,transparent,var(--cyan-line),transparent); }
.up-label { font-family:'Geist Mono',monospace; font-size:.55rem; letter-spacing:.2em; text-transform:uppercase; color:var(--t2); display:flex; align-items:center; gap:8px; }
.up-label::before { content:''; display:block; width:16px; height:1px; background:var(--p-line); }
.up-title { font-family:'Instrument Serif',serif; font-size:1.55rem; font-weight:400; color:var(--t0); line-height:1.15; }
.up-title span { color:var(--p2); font-style:italic; }
.up-meta { font-size:.76rem; color:var(--t3); line-height:1.6; }
.up-steps { display:flex; flex-direction:column; border:1px solid var(--b2); border-radius:var(--r2); overflow:hidden; }
.up-step { display:flex; align-items:center; gap:12px; padding:.7rem 1rem; border-bottom:1px solid var(--b2); }
.up-step:last-child { border-bottom:none; }
.up-step-n { width:22px; height:22px; border-radius:50%; flex-shrink:0; border:1px solid var(--p-line); background:var(--p-dim); display:flex; align-items:center; justify-content:center; font-family:'Geist Mono',monospace; font-size:.58rem; color:var(--p2); }
.up-step-t { font-size:.78rem; color:var(--t2); }

/* File uploader */
[data-testid="stFileUploader"] { background:linear-gradient(135deg,rgba(99,102,241,.06),rgba(34,211,238,.03)) !important; border:1px dashed var(--p-line) !important; border-radius:var(--r3) !important; transition:all .3s !important; }
[data-testid="stFileUploader"]:hover { border-color:var(--p) !important; background:var(--p-dim) !important; box-shadow:0 0 30px rgba(99,102,241,.08) !important; }
[data-testid="stFileUploaderDropzone"] { background:transparent !important; border:none !important; }
[data-testid="stFileUploader"] p,[data-testid="stFileUploader"] span,[data-testid="stFileUploader"] small,[data-testid="stFileUploader"] div { color:var(--t2) !important; font-family:'Geist',sans-serif !important; }
[data-testid="stFileUploader"] button { background:linear-gradient(135deg,var(--p),#4f46e5) !important; color:white !important; border:none !important; font-weight:600 !important; border-radius:var(--r2) !important; font-family:'Geist',sans-serif !important; font-size:.8rem !important; box-shadow:0 4px 16px rgba(99,102,241,.3) !important; }

/* ══ DIVIDER ══ */
.tl-divide { display:flex; align-items:center; gap:1.2rem; margin:2.5rem 0 1.8rem; position:relative; z-index:5; }
.tl-divide-line { flex:1; height:1px; background:var(--b1); }
.tl-divide-label { font-family:'Geist Mono',monospace; font-size:.52rem; letter-spacing:.22em; text-transform:uppercase; color:var(--t4); white-space:nowrap; display:flex; align-items:center; gap:8px; }
.tl-divide-gem { width:4px; height:4px; border-radius:1px; background:var(--p); transform:rotate(45deg); box-shadow:0 0 6px var(--p-glow); }
hr { display:none !important; }

/* ══ SECTION HEADERS ══ */
.tl-kicker { font-family:'Geist Mono',monospace; font-size:.55rem; letter-spacing:.2em; text-transform:uppercase; color:var(--t3); display:flex; align-items:center; gap:8px; margin-bottom:.5rem; position:relative; z-index:5; }
.tl-kicker::before { content:''; width:3px; height:14px; border-radius:2px; background:linear-gradient(180deg,var(--p),var(--cyan)); }
.tl-h { font-family:'Instrument Serif',serif !important; font-size:1.9rem !important; font-weight:400 !important; color:var(--t0) !important; letter-spacing:-.02em !important; line-height:1.1 !important; margin:0 0 1.2rem !important; position:relative; z-index:5; }
.tl-h em { color:var(--p2); font-style:italic; }

/* ══ TERMINAL ══ */
.tl-term { background:var(--bg); border:1px solid var(--b1); border-radius:var(--r3); overflow:hidden; position:relative; z-index:5; box-shadow:var(--shc); }
.tl-term::before { content:''; position:absolute; top:0;left:0;right:0; height:1px; background:linear-gradient(90deg,transparent,var(--p-line) 25%,var(--p-line) 75%,transparent); }
.tl-chrome { display:flex; align-items:center; justify-content:space-between; padding:.75rem 1.2rem; background:var(--bg1); border-bottom:1px solid var(--b1); }
.tl-dots { display:flex; gap:6px; }
.tl-dot { width:10px; height:10px; border-radius:50%; }
.d-r{background:#ff5f57}.d-y{background:#ffbd2e}.d-g{background:#28c941}
.tl-chrome-id { font-family:'Geist Mono',monospace; font-size:.58rem; color:var(--t3); }
.tl-chrome-ok { font-family:'Geist Mono',monospace; font-size:.54rem; padding:2px 8px; border-radius:var(--r1); background:var(--em-dim); border:1px solid var(--em-line); color:var(--emerald); }
.tl-term-body { padding:1.4rem 1.6rem; font-family:'Geist Mono',monospace; font-size:.72rem; color:var(--t2); line-height:1.85; max-height:200px; overflow-y:auto; white-space:pre-wrap; }
.tl-prompt { color:var(--p2); }
.tl-blink { display:inline-block; width:7px; height:13px; background:var(--p2); vertical-align:middle; margin-left:2px; animation:blink 1s step-end infinite; }
@keyframes blink { 50%{opacity:0} }

/* ══ CLASSIFICATION ══ */
.cls-panel { background:var(--bg1); border:1px solid var(--b1); border-radius:var(--r3); overflow:hidden; z-index:5; box-shadow:var(--shb); }
.cls-hdr { padding:1.6rem 1.8rem 1.2rem; background:linear-gradient(135deg,var(--bg2),var(--bg1)); border-bottom:1px solid var(--b1); position:relative; }
.cls-hdr::after { content:''; position:absolute; bottom:-1px; left:1.8rem; width:40px; height:1px; background:var(--p); }
.cls-kicker { font-family:'Geist Mono',monospace; font-size:.52rem; letter-spacing:.18em; text-transform:uppercase; color:var(--p2); margin-bottom:.4rem; }
.cls-role { font-family:'Instrument Serif',serif; font-size:1.8rem; font-weight:400; color:var(--t0); }
.cls-body { padding:1.2rem 1.8rem 1.6rem; }
.kw-row { display:flex; align-items:center; gap:10px; padding:.6rem 0; border-bottom:1px solid var(--b2); }
.kw-row:last-child { border-bottom:none; }
.kw-i { font-family:'Geist Mono',monospace; font-size:.52rem; color:var(--t4); min-width:18px; }
.kw-n { font-size:.82rem; color:var(--t2); min-width:160px; font-weight:300; }
.kw-n.top { color:var(--p2); font-weight:500; }
.kw-track { flex:1; height:2px; background:var(--b1); border-radius:99px; overflow:hidden; }
.kw-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,var(--p),var(--cyan)); }
.kw-v { font-family:'Geist Mono',monospace; font-size:.62rem; color:var(--p2); }

/* ══ CHART ══ */
.chart-panel { background:var(--bg1); border:1px solid var(--b1); border-radius:var(--r3); padding:1.6rem 1.6rem 1.2rem; z-index:5; box-shadow:var(--shb); }
.chart-panel::after { content:''; position:absolute; top:0;left:0;right:0; height:1px; background:linear-gradient(90deg,transparent,var(--cyan-line),transparent); }
.chart-k { font-family:'Geist Mono',monospace; font-size:.52rem; letter-spacing:.18em; text-transform:uppercase; color:var(--t3); margin-bottom:.3rem; }
.chart-t { font-family:'Instrument Serif',serif; font-size:1.1rem; color:var(--t0); margin-bottom:1rem; }

/* ══ JOB CARDS ══ */
.jcard { display:grid; grid-template-columns:60px 1fr auto; grid-template-rows:auto auto; background:var(--bg1); border:1px solid var(--b1); border-radius:var(--r3); overflow:hidden; margin-bottom:8px; box-shadow:var(--sha); position:relative; transition:border-color .25s,box-shadow .25s,transform .25s var(--spring); }
.jcard::before { content:''; position:absolute; left:0;top:0;bottom:0; width:2px; background:linear-gradient(180deg,var(--p),var(--cyan)); opacity:0; transition:opacity .25s; }
.jcard:hover { border-color:var(--p-line); box-shadow:var(--shg),var(--shb); transform:translateX(4px) translateY(-1px); }
.jcard:hover::before { opacity:1; }
.jcard-rank { grid-row:1/3; background:var(--bg2); border-right:1px solid var(--b2); display:flex; align-items:center; justify-content:center; }
.jcard-rank-n { font-family:'Instrument Serif',serif; font-size:1.5rem; color:var(--t4); }
.jcard-body { padding:1rem 1.2rem .35rem; }
.jcard-t { font-family:'Instrument Serif',serif; font-size:1.05rem; color:var(--t0); margin-bottom:2px; }
.jcard-c { font-family:'Geist Mono',monospace; font-size:.58rem; color:var(--t3); }
.jcard-score { padding:1rem 1.2rem .35rem; }
.jbadge { display:inline-flex; align-items:center; padding:4px 10px; border-radius:100px; font-family:'Geist Mono',monospace; font-size:.6rem; white-space:nowrap; }
.jb-hi { background:var(--em-dim); color:var(--emerald); border:1px solid var(--em-line); }
.jb-mid { background:var(--am-dim); color:var(--amber); border:1px solid var(--am-line); }
.jb-lo { background:var(--b2); color:var(--t3); border:1px solid var(--b1); }
.jcard-desc { grid-column:2/-1; padding:.6rem 1.2rem 1rem; font-size:.78rem; color:var(--t3); line-height:1.7; border-top:1px solid var(--b2); font-weight:300; }

/* ══ DATAFRAME ══ */
[data-testid="stDataFrame"] > div { border:1px solid var(--b1) !important; border-radius:var(--r3) !important; overflow:hidden !important; box-shadow:var(--shb) !important; }
[data-testid="stDataFrame"] thead tr th { background:var(--bg2) !important; color:var(--t3) !important; font-family:'Geist Mono',monospace !important; font-size:.54rem !important; letter-spacing:.18em !important; text-transform:uppercase !important; padding:12px 18px !important; border-bottom:1px solid var(--b1) !important; border-right:none !important; font-weight:400 !important; }
[data-testid="stDataFrame"] tbody tr td { background:var(--bg1) !important; color:var(--t1) !important; font-family:'Geist',sans-serif !important; font-size:.84rem !important; padding:11px 18px !important; border-bottom:1px solid var(--b2) !important; border-right:none !important; font-weight:300 !important; }
[data-testid="stDataFrame"] tbody tr:hover td { background:var(--bg2) !important; }

/* ══ RECRUITER HERO ══ */
.rec-hero { background:var(--bg1); border:1px solid var(--b1); border-radius:var(--r4); padding:3rem 3.2rem; margin-top:2rem; position:relative; overflow:hidden; z-index:5; animation:heroIn .6s var(--out) .1s both; }
.rec-hero::before { content:''; position:absolute; top:-80px; right:-60px; width:340px; height:340px; background:radial-gradient(circle,rgba(34,211,238,.08) 0%,transparent 60%); pointer-events:none; }
.rec-hero::after { content:''; position:absolute; top:0;left:0;right:0; height:1px; background:linear-gradient(90deg,transparent,var(--cyan-line),transparent); }
.rec-h1 { font-family:'Instrument Serif',serif !important; font-size:clamp(2.5rem,3.8vw,4rem) !important; font-weight:400 !important; line-height:1.02 !important; color:var(--t0) !important; letter-spacing:-.02em !important; margin:1.4rem 0 1rem !important; }
.rec-h1 em { color:transparent !important; background:linear-gradient(135deg,var(--cyan),var(--p2)); -webkit-background-clip:text; background-clip:text; font-style:italic; }

/* ══ STAT CARDS ══ */
.stat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:1.6rem 0; z-index:5; }
.stat-card { background:var(--bg1); border:1px solid var(--b1); border-radius:var(--r3); padding:1.3rem 1.5rem; position:relative; overflow:hidden; transition:border-color .25s,transform .25s var(--spring); }
.stat-card:hover { border-color:var(--p-line); transform:translateY(-3px); }
.stat-accent { position:absolute; top:0;left:0;right:0; height:2px; }
.sa1{background:linear-gradient(90deg,var(--p),var(--p2))}.sa2{background:linear-gradient(90deg,var(--emerald),var(--cyan))}.sa3{background:linear-gradient(90deg,var(--cyan),var(--p2))}.sa4{background:linear-gradient(90deg,var(--amber),var(--rose))}
.stat-n { font-family:'Instrument Serif',serif; font-size:2rem; color:var(--t0); line-height:1; margin-bottom:4px; letter-spacing:-.03em; }
.stat-l { font-family:'Geist Mono',monospace; font-size:.52rem; letter-spacing:.16em; text-transform:uppercase; color:var(--t3); }
.stat-tag { position:absolute; top:1rem; right:1.2rem; font-family:'Geist Mono',monospace; font-size:.5rem; color:var(--t4); }

/* ══ INFO STRIP ══ */
.info-strip { display:flex; align-items:flex-start; gap:10px; background:rgba(34,211,238,.04); border:1px solid var(--cyan-line); border-radius:var(--r2); padding:.9rem 1.2rem; margin:1rem 0; z-index:5; }
.info-strip-line { width:2px; flex-shrink:0; border-radius:2px; background:linear-gradient(180deg,var(--cyan),transparent); align-self:stretch; }
.info-strip-txt { font-size:.79rem; color:var(--t2); line-height:1.6; font-weight:300; }
.info-strip-txt b { color:var(--cyan); font-weight:500; }

/* ══ FORMS ══ */
[data-testid="stTextArea"] textarea { background:var(--bg1) !important; border:1px solid var(--b1) !important; color:var(--t1) !important; font-family:'Geist',sans-serif !important; border-radius:var(--r2) !important; font-size:.86rem !important; font-weight:300 !important; line-height:1.7 !important; transition:border-color .2s,box-shadow .2s !important; }
[data-testid="stTextArea"] textarea:focus { border-color:var(--p) !important; box-shadow:0 0 0 3px var(--p-dim) !important; }
[data-testid="stTextArea"] label,[data-testid="stTextInput"] label,[data-testid="stNumberInput"] label,[data-testid="stSlider"] label,[data-testid="stSelectbox"] label { color:var(--t2) !important; font-family:'Geist Mono',monospace !important; font-size:.56rem !important; letter-spacing:.16em !important; text-transform:uppercase !important; }
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input { background:var(--bg1) !important; border:1px solid var(--b1) !important; color:var(--t1) !important; border-radius:var(--r2) !important; font-family:'Geist',sans-serif !important; }
[data-testid="stSelectbox"] > div > div { background:var(--bg1) !important; border:1px solid var(--b1) !important; color:var(--t1) !important; border-radius:var(--r2) !important; }

/* ══ RUN BUTTON ══ */
.run-btn [data-testid="stButton"] button {
  display:flex !important; align-items:center !important; justify-content:center !important;
  width:100% !important; padding:1rem 2rem !important;
  background:linear-gradient(135deg,var(--p) 0%,#4f46e5 100%) !important;
  color:white !important; border:none !important; border-radius:var(--r3) !important;
  font-family:'Geist',sans-serif !important; font-weight:600 !important;
  font-size:.95rem !important; letter-spacing:-.01em !important;
  box-shadow:0 8px 32px rgba(99,102,241,.35),0 2px 8px rgba(99,102,241,.2) !important;
  transition:all .25s var(--spring) !important;
}
.run-btn [data-testid="stButton"] button:hover { box-shadow:0 12px 48px rgba(99,102,241,.5) !important; transform:translateY(-2px) !important; }

/* ══ DOWNLOAD BUTTON ══ */
[data-testid="stDownloadButton"] button { background:transparent !important; border:1px solid var(--b1) !important; color:var(--t2) !important; font-family:'Geist Mono',monospace !important; font-size:.68rem !important; border-radius:var(--r2) !important; padding:.7rem 1.5rem !important; transition:all .2s !important; width:100% !important; }
[data-testid="stDownloadButton"] button:hover { border-color:var(--p-line) !important; color:var(--p2) !important; background:var(--p-dim) !important; }

/* ══ CANDIDATE CARDS ══ */
.rcard { display:grid; grid-template-columns:72px 1fr 80px; background:var(--bg1); border:1px solid var(--b1); border-radius:var(--r3); overflow:hidden; margin-bottom:8px; box-shadow:var(--sha); position:relative; z-index:5; transition:border-color .25s,box-shadow .25s,transform .25s var(--spring); }
.rcard::before { content:''; position:absolute; left:0;top:0;bottom:0; width:2px; background:linear-gradient(180deg,var(--emerald),var(--cyan)); opacity:0; transition:opacity .25s; }
.rcard:hover { border-color:var(--p-line); box-shadow:var(--shg),var(--shb); transform:translateX(4px) translateY(-1px); }
.rcard:hover::before { opacity:1; }
.rcard-rank { background:var(--bg2); border-right:1px solid var(--b2); display:flex; flex-direction:column; align-items:center; justify-content:center; gap:4px; }
.rcard-rank-n { font-family:'Instrument Serif',serif; font-size:1.7rem; color:var(--t4); }
.rcard-rank-m { font-family:'Geist Mono',monospace; font-size:.48rem; color:var(--p2); }
.rcard-body { padding:1.1rem 1.4rem; }
.rcard-name { font-family:'Instrument Serif',serif; font-size:1.1rem; color:var(--t0); margin-bottom:3px; }
.rcard-meta { font-family:'Geist Mono',monospace; font-size:.56rem; color:var(--t3); margin-bottom:7px; }
.tag-row { display:flex; flex-wrap:wrap; gap:4px; margin-top:5px; }
.rtag { display:inline-block; padding:2px 8px; border-radius:var(--r1); font-family:'Geist Mono',monospace; font-size:.56rem; border:1px solid var(--b1); color:var(--t3); }
.rtag.match { border-color:var(--em-line); color:var(--emerald); background:var(--em-dim); }
.rcard-score { border-left:1px solid var(--b2); display:flex; align-items:center; justify-content:center; }
.score-ring { width:54px; height:54px; border-radius:50%; border:1.5px solid var(--b1); display:flex; flex-direction:column; align-items:center; justify-content:center; gap:1px; }
.score-ring.hi { border-color:var(--emerald); box-shadow:0 0 16px rgba(16,185,129,.2); }
.score-ring.mid { border-color:var(--amber); box-shadow:0 0 16px rgba(245,158,11,.2); }
.score-ring-v { font-family:'Instrument Serif',serif; font-size:1.05rem; color:var(--t0); line-height:1; }
.score-ring-l { font-family:'Geist Mono',monospace; font-size:.4rem; color:var(--t3); letter-spacing:.12em; text-transform:uppercase; }

/* ══ KW PANEL ══ */
.kw-panel { background:var(--bg1); border:1px solid var(--b1); border-radius:var(--r3); padding:1.5rem 1.6rem; margin-bottom:1.2rem; z-index:5; }
.kw-panel-k { font-family:'Geist Mono',monospace; font-size:.52rem; letter-spacing:.18em; text-transform:uppercase; color:var(--t3); margin-bottom:.4rem; }
.kw-panel-t { font-family:'Instrument Serif',serif; font-size:1.1rem; color:var(--t0); margin-bottom:.9rem; }
.kw-cloud { display:flex; flex-wrap:wrap; gap:6px; }
.kw-ctag { padding:5px 12px; border-radius:100px; font-family:'Geist Mono',monospace; font-size:.6rem; border:1px solid var(--em-line); color:var(--emerald); background:var(--em-dim); }

/* ══ EMPTY STATE ══ */
.tl-empty { display:flex; flex-direction:column; align-items:center; justify-content:center; padding:6rem 2rem 5rem; text-align:center; z-index:5; }
.tl-empty-h { font-family:'Instrument Serif',serif; font-size:1.9rem; color:var(--t0); margin-bottom:.6rem; letter-spacing:-.02em; }
.tl-empty-p { font-size:.86rem; color:var(--t3); line-height:1.75; font-weight:300; max-width:340px; }

/* ══ PROGRESS & ALERTS ══ */
[data-testid="stProgress"] > div > div { background:linear-gradient(90deg,var(--p),var(--cyan)) !important; border-radius:99px !important; }
[data-testid="stProgress"] > div { background:var(--bg2) !important; border-radius:99px !important; }
[data-testid="stSpinner"] p { color:var(--p2) !important; font-family:'Geist Mono',monospace !important; font-size:.72rem !important; }
[data-testid="stAlert"] { background:var(--bg1) !important; border:1px solid var(--b1) !important; border-radius:var(--r2) !important; color:var(--t1) !important; font-family:'Geist',sans-serif !important; }

/* ══ ANIMATIONS ══ */
@keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:none} }
.anim { animation:fadeUp .5s var(--out) both; }
.a1{animation-delay:.06s}.a2{animation-delay:.12s}.a3{animation-delay:.18s}
</style>

<!-- MESH BACKGROUND -->
<div class="tl-mesh" aria-hidden="true">
  <div class="tl-mesh-g"></div>
  <div class="tl-mesh-grid"></div>
  <div class="tl-mesh-noise"></div>
</div>

<!-- CUSTOM CURSOR ELEMENTS -->
<div id="tl-dot"></div>
<div id="tl-ring"></div>

<script>
(function(){
  /* ── Grab cursor elements ── */
  var dot  = document.getElementById('tl-dot');
  var ring = document.getElementById('tl-ring');
  if(!dot || !ring) return;

  var mx=-200,my=-200,rx=-200,ry=-200;
  var visible=false, onHover=false;

  /* ── Smoothly hide native cursor everywhere via injected style ── */
  function hideCursorIn(doc){
    try{
      if(!doc||!doc.head) return;
      if(doc.querySelector('#__tl_hide')) return;
      var s=doc.createElement('style');
      s.id='__tl_hide';
      s.textContent='*,*::before,*::after{cursor:none!important}';
      doc.head.appendChild(s);
    }catch(e){}
  }

  /* ── Patch an iframe: inject hide style + relay mouse events ── */
  function patchFrame(fr){
    try{
      var d=fr.contentDocument||fr.contentWindow.document;
      if(!d||d.__tlDone) return;
      d.__tlDone=true;
      hideCursorIn(d);
      d.addEventListener('mousemove',function(e){
        var r=fr.getBoundingClientRect();
        mx=r.left+e.clientX; my=r.top+e.clientY;
        show(); moveDot();
      },{passive:true});
      d.addEventListener('mousedown',onDown);
      d.addEventListener('mouseup',onUp);
      /* recurse into nested frames */
      Array.from(d.querySelectorAll('iframe')).forEach(patchFrame);
    }catch(e){}
  }

  /* ── Show / hide ── */
  function show(){
    if(visible) return;
    visible=true;
    dot.style.opacity='1';
    ring.style.opacity='1';
  }
  function hide(){
    visible=false;
    dot.style.opacity='0';
    ring.style.opacity='0';
  }
  function moveDot(){
    dot.style.left=mx+'px';
    dot.style.top=my+'px';
  }

  /* ── Outer window events ── */
  window.addEventListener('mousemove',function(e){
    mx=e.clientX; my=e.clientY;
    show(); moveDot();
  },{passive:true});
  window.addEventListener('mouseleave',hide);
  window.addEventListener('mouseenter',show);

  /* ── Click feedback ── */
  function onDown(){
    dot.style.width='16px'; dot.style.height='16px';
    dot.style.background='#c7d2fe';
    dot.style.boxShadow='0 0 0 3px rgba(129,140,248,.4),0 0 24px rgba(99,102,241,1)';
    ring.style.width='18px'; ring.style.height='18px';
    ring.style.background='rgba(99,102,241,.15)';
  }
  function onUp(){
    var sz=onHover?'7px':'10px', rsz=onHover?'50px':'36px';
    dot.style.width=sz; dot.style.height=sz;
    dot.style.background='#818cf8';
    dot.style.boxShadow='0 0 0 2px rgba(129,140,248,.35),0 0 16px rgba(99,102,241,.9),0 0 32px rgba(99,102,241,.35)';
    ring.style.width=rsz; ring.style.height=rsz;
    ring.style.background='transparent';
  }
  window.addEventListener('mousedown',onDown);
  window.addEventListener('mouseup',onUp);

  /* ── RAF: smooth ring lerp ── */
  (function raf(){
    rx+=(mx-rx)*0.1; ry+=(my-ry)*0.1;
    ring.style.left=rx+'px'; ring.style.top=ry+'px';
    requestAnimationFrame(raf);
  })();

  /* ── Magnetic hover binding ── */
  function bindHover(doc){
    try{
      var sel='button,a,input,textarea,select,label,[role="button"],[data-testid="stFileUploader"]';
      doc.querySelectorAll(sel).forEach(function(el){
        if(el.__tlH) return; el.__tlH=true;
        el.addEventListener('mouseenter',function(){
          onHover=true;
          ring.style.width='50px'; ring.style.height='50px';
          ring.style.borderColor='rgba(129,140,248,.8)';
          ring.style.background='rgba(99,102,241,.07)';
          dot.style.width='6px'; dot.style.height='6px';
          dot.style.boxShadow='0 0 0 2px rgba(129,140,248,.5),0 0 20px rgba(99,102,241,1)';
        });
        el.addEventListener('mouseleave',function(){
          onHover=false;
          ring.style.width='36px'; ring.style.height='36px';
          ring.style.borderColor='rgba(129,140,248,.55)';
          ring.style.background='transparent';
          dot.style.width='10px'; dot.style.height='10px';
          dot.style.boxShadow='0 0 0 2px rgba(129,140,248,.35),0 0 16px rgba(99,102,241,.9),0 0 32px rgba(99,102,241,.35)';
        });
      });
    }catch(e){}
  }

  /* ── Poll to patch new iframes & bind new elements ── */
  function poll(){
    hideCursorIn(document);
    Array.from(document.querySelectorAll('iframe')).forEach(patchFrame);
    bindHover(document);
    Array.from(document.querySelectorAll('iframe')).forEach(function(fr){
      try{ bindHover(fr.contentDocument||fr.contentWindow.document); }catch(e){}
    });
  }
  poll();
  setInterval(poll,600);
})();
</script>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_jobs():
    p = os.path.join(BASE,"jobs.csv")
    if not os.path.exists(p): return pd.DataFrame(columns=["Job_Title","Category","Job_Description"])
    df = pd.read_csv(p); df.drop_duplicates(inplace=True); df.reset_index(drop=True,inplace=True); return df

@st.cache_data
def load_skill_profiles():
    p = os.path.join(BASE,"skill_profiles.csv")
    if not os.path.exists(p): return {}
    df = pd.read_csv(p); return dict(zip(df["Job_Title"].str.strip(), df["Skills"].fillna("")))

@st.cache_data
def load_category_keywords():
    p = os.path.join(BASE,"category_keywords.csv")
    if not os.path.exists(p): return {}
    df = pd.read_csv(p); result = {}
    for _,r in df.iterrows():
        cat = r.get("Category","")
        if pd.isna(cat) or str(cat).strip().lower() in ("","nan","none"): continue
        raw = r.get("Keywords","")
        kws = [k.strip().lower() for k in str(raw if not pd.isna(raw) else "").split(",") if k.strip()]
        if kws: result[str(cat).strip()] = kws
    return result

jobs_df = load_jobs(); SKP = load_skill_profiles(); CATK = load_category_keywords()
TJ = len(jobs_df); TT = len(SKP)
TC = jobs_df["Category"].nunique() if "Category" in jobs_df.columns and len(jobs_df)>0 else len(CATK)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def clean(txt):
    t=str(txt).lower(); t=re.sub(r'http\S+|\S+@\S+',' ',t); t=re.sub(r'[^a-zA-Z ]',' ',t); return re.sub(r'\s+',' ',t).strip()

def pdf_text(f):
    try:
        with pdfplumber.open(f) as pdf: return ' '.join(pg.extract_text() or '' for pg in pdf.pages)
    except: return ""

def predict_cat(rc):
    if not CATK: return "Unknown",[]
    rl=rc.lower()
    sc={c:sum(1 for k in ks if k in rl) for c,ks in CATK.items() if c and str(c).strip().lower() not in ("nan","none","")}
    if not sc: return "Unknown",[]
    ranked=sorted(sc.items(),key=lambda x:-x[1])
    return ("No Strong Match",ranked[:5]) if ranked[0][1]==0 else (ranked[0][0],ranked[:5])

def top_jobs(rc,df,n=10):
    if df.empty: return df
    df=df.copy()
    df["_r"]=df.apply(lambda r:f"{r.get('Job_Title','')} {r.get('Job_Title','')} {clean(r.get('Job_Description',''))} {SKP.get(r.get('Job_Title',''),'')}".lower(),axis=1)
    corp=[rc]+df["_r"].tolist()
    mat=TfidfVectorizer(stop_words="english",max_features=15000,ngram_range=(1,2)).fit_transform(corp)
    sims=cosine_similarity(mat[0:1],mat[1:]).flatten()
    df["Match Score"]=(sims*100).round(2)
    return df.sort_values("Match Score",ascending=False).head(n).reset_index(drop=True)

def rank_resumes(jd,res,n=50):
    if not res: return pd.DataFrame()
    corp=[jd]+[r["text"] for r in res]
    mat=TfidfVectorizer(stop_words="english",max_features=20000,ngram_range=(1,2)).fit_transform(corp)
    sims=cosine_similarity(mat[0:1],mat[1:]).flatten()
    df=pd.DataFrame({"File Name":[r["name"] for r in res],"Match Score":(sims*100).round(2),"_text":[r["text"] for r in res]})
    df["Rank"]=df["Match Score"].rank(ascending=False,method="min").astype(int)
    return df.sort_values("Match Score",ascending=False).head(n).reset_index(drop=True)

def top_kw(txt,n=8):
    from sklearn.feature_extraction.text import CountVectorizer
    try:
        v=CountVectorizer(stop_words="english",max_features=200); v.fit([txt])
        w=v.get_feature_names_out(); c=v.transform([txt]).toarray()[0]
        return [x for x,_ in sorted(zip(w,c),key=lambda x:-x[1])[:n]]
    except: return []

def medal(r): return {1:"GOLD",2:"SILVER",3:"BRONZE"}.get(r,"")

D  = lambda lbl: f'<div class="tl-divide"><div class="tl-divide-gem"></div><div class="tl-divide-line"></div><div class="tl-divide-label">{lbl}</div><div class="tl-divide-line"></div><div class="tl-divide-gem"></div></div>'
KK = lambda k,h: f'<div class="tl-kicker">{k}</div><div class="tl-h">{h}</div>'


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "pg" not in st.session_state: st.session_state.pg = "cand"


# ─────────────────────────────────────────────────────────────────────────────
#  NAVBAR  (pure HTML — no Streamlit buttons here)
# ─────────────────────────────────────────────────────────────────────────────
pg = st.session_state.pg
c_on = "on" if pg=="cand" else ""
r_on = "on" if pg=="rec"  else ""
c_dot = '<span class="tl-tab-dot"></span>' if pg=="cand" else ""
r_dot = '<span class="tl-tab-dot"></span>' if pg=="rec"  else ""

st.markdown(f"""
<div class="tl-nav">
  <div class="tl-brand">
    <div class="tl-logomark">
      <svg viewBox="0 0 36 36" xmlns="http://www.w3.org/2000/svg">
        <defs><linearGradient id="nlg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#6366f1"/><stop offset="100%" stop-color="#22d3ee"/>
        </linearGradient></defs>
        <circle cx="18" cy="18" r="16" fill="rgba(99,102,241,0.08)" stroke="rgba(99,102,241,0.2)" stroke-width="1"/>
        <circle cx="18" cy="18" r="10" fill="none" stroke="url(#nlg)" stroke-width="1.5" stroke-dasharray="4 3">
          <animateTransform attributeName="transform" type="rotate" from="0 18 18" to="360 18 18" dur="12s" repeatCount="indefinite"/>
        </circle>
        <circle cx="18" cy="18" r="4" fill="url(#nlg)"/>
        <circle cx="18" cy="8" r="1.5" fill="#6366f1" opacity="0.7"/>
        <circle cx="26" cy="24" r="1.5" fill="#22d3ee" opacity="0.7"/>
        <circle cx="10" cy="24" r="1.5" fill="#818cf8" opacity="0.7"/>
      </svg>
    </div>
    <div class="tl-brand-text">
      <div class="tl-brand-name">Talent<em>Lens</em></div>
      <div class="tl-brand-tag">AI Career Intelligence Platform</div>
    </div>
  </div>

  <div class="tl-nav-tabs">
    <div class="tl-nav-tab {c_on}">
      {c_dot}Candidate Portal
    </div>
    <div class="tl-nav-tab {r_on}">
      {r_dot}Recruiter Portal
    </div>
  </div>

  <div class="tl-nav-stats">
    <div class="tl-stat"><div class="tl-stat-n">{TJ:,}</div><div class="tl-stat-k">Positions</div></div>
    <div class="tl-stat"><div class="tl-stat-n">{TT}</div><div class="tl-stat-k">Skill Tracks</div></div>
    <div class="tl-stat"><div class="tl-stat-n">{TC}</div><div class="tl-stat-k">Verticals</div></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE SWITCHER — Streamlit buttons ONLY, styled as compact strip
#  These are the ONLY buttons. CSS hides them by default via .stButton{display:none}
#  We re-show them here inside .sw-strip which overrides that rule.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sw-strip" style="margin:.75rem 0 0;">
""", unsafe_allow_html=True)

sc1, sc2 = st.columns(2)
with sc1:
    if st.button("Candidate Portal  —  Find My Role", use_container_width=True,
                 type="primary" if st.session_state.pg=="cand" else "secondary", key="sw_c"):
        st.session_state.pg="cand"; st.rerun()
with sc2:
    if st.button("Recruiter Portal  —  Screen Résumés", use_container_width=True,
                 type="primary" if st.session_state.pg=="rec" else "secondary", key="sw_r"):
        st.session_state.pg="rec"; st.rerun()

st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  CANDIDATE PAGE
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.pg == "cand":
    ch, cu = st.columns([1.4, 1], gap="small")

    with ch:
        st.markdown(f"""
        <div class="hero-main">
          <div class="hero-badge">
            <div class="hero-badge-dot"></div>
            <span class="hero-badge-txt">Live &middot; TF-IDF Semantic Engine</span>
          </div>
          <h1 class="hero-h1">Find the role<br>built for <em>exactly</em><br>who you are</h1>
          <p class="hero-p">Drop your résumé. Our engine reads every skill signal, weights every keyword, and ranks {TJ:,} positions by true semantic fit in under three seconds. Private. Instant. Precise.</p>
          <div class="hero-chips">
            <span class="hero-chip"><span class="hero-chip-dot"></span><b>{TJ:,}</b>&nbsp;positions</span>
            <span class="hero-chip"><span class="hero-chip-dot"></span><b>{TT}</b>&nbsp;skill profiles</span>
            <span class="hero-chip"><span class="hero-chip-dot"></span><b>{TC}</b>&nbsp;verticals</span>
            <span class="hero-chip"><span class="hero-chip-dot"></span>Zero data stored</span>
            <span class="hero-chip"><span class="hero-chip-dot"></span>No sign-up</span>
          </div>
          <div style="position:absolute;top:1.6rem;right:1.8rem;opacity:0.1;width:72px;height:72px;">
            <svg viewBox="0 0 72 72" xmlns="http://www.w3.org/2000/svg">
              <defs><linearGradient id="hg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#6366f1"/><stop offset="100%" stop-color="#22d3ee"/></linearGradient></defs>
              <circle cx="36" cy="36" r="32" fill="none" stroke="url(#hg)" stroke-width="1"><animateTransform attributeName="transform" type="rotate" from="0 36 36" to="360 36 36" dur="25s" repeatCount="indefinite"/></circle>
              <circle cx="36" cy="36" r="20" fill="none" stroke="url(#hg)" stroke-width="0.8" stroke-dasharray="6 4"><animateTransform attributeName="transform" type="rotate" from="360 36 36" to="0 36 36" dur="15s" repeatCount="indefinite"/></circle>
              <circle cx="36" cy="36" r="8" fill="none" stroke="url(#hg)" stroke-width="1.2"/>
              <circle cx="36" cy="36" r="3" fill="#6366f1"/>
            </svg>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with cu:
        st.markdown("""
        <div class="upload-panel">
          <div class="up-label">Step 01 of 01</div>
          <div class="up-title">Upload your <span>résumé</span></div>
          <div class="up-meta">PDF format &middot; Processed locally &middot; Never stored</div>
          <div class="up-steps">
            <div class="up-step"><div class="up-step-n">1</div><div class="up-step-t">Select or drag your PDF résumé</div></div>
            <div class="up-step"><div class="up-step-n">2</div><div class="up-step-t">Engine extracts and vectorises all text</div></div>
            <div class="up-step"><div class="up-step-n">3</div><div class="up-step-t">Top roles surface instantly</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        uf = st.file_uploader("", type=["pdf"], label_visibility="collapsed", key="c_resume")

    if uf:
        with st.spinner("Extracting résumé text..."):
            raw=pdf_text(uf); rc=clean(raw)

        st.markdown(D("Document Analysis"), unsafe_allow_html=True)
        st.markdown(KK("01 · Parsed Document","Résumé <em>Preview</em>"), unsafe_allow_html=True)
        prev=(raw[:900]+"...") if len(raw)>900 else raw
        prev=prev.replace("<","&lt;").replace(">","&gt;")
        st.markdown(f"""
        <div class="tl-term anim">
          <div class="tl-chrome">
            <div class="tl-dots"><div class="tl-dot d-r"></div><div class="tl-dot d-y"></div><div class="tl-dot d-g"></div></div>
            <div class="tl-chrome-id">RESUME_PARSER &middot; {len(raw):,} chars</div>
            <div class="tl-chrome-ok">PARSED</div>
          </div>
          <div class="tl-term-body"><span class="tl-prompt">&gt; </span>{prev}<span class="tl-blink"></span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown(D("Profile Classification"), unsafe_allow_html=True)
        pc,top5=predict_cat(rc); ms=max((s for _,s in top5),default=1) or 1
        st.markdown(KK("02 · Classification","Career <em>Profile</em>"), unsafe_allow_html=True)
        cm,cc=st.columns([1,1.05],gap="small")

        with cm:
            rows=""
            for i,(c,s) in enumerate(top5):
                pct=int(s/ms*100); ist=c==pc
                nl=f'<span class="kw-n top">&#9658; {c}</span>' if ist else f'<span class="kw-n">{c}</span>'
                rows+=f'<div class="kw-row"><span class="kw-i">0{i+1}</span>{nl}<div class="kw-track"><div class="kw-fill" style="width:{pct}%"></div></div><span class="kw-v">{s}</span></div>'
            st.markdown(f'<div class="cls-panel anim a1"><div class="cls-hdr"><div class="cls-kicker">Predicted Category</div><div class="cls-role">{pc}</div></div><div class="cls-body">{rows}</div></div>', unsafe_allow_html=True)

        with cc:
            cdf=pd.DataFrame(top5,columns=["Category","Score"])
            st.markdown('<div class="chart-panel anim a2"><div class="chart-k">Keyword Hit Rate</div><div class="chart-t">Score Distribution</div>', unsafe_allow_html=True)
            st.bar_chart(cdf.set_index("Category"),use_container_width=True,height=250)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(D("Role Matching"), unsafe_allow_html=True)
        st.markdown(KK("03 · Top Matches","Top <em>Recommendations</em>"), unsafe_allow_html=True)
        with st.spinner("Ranking all positions..."):
            tj=top_jobs(rc,jobs_df,10)

        if not tj.empty:
            dp=tj[["Job_Title","Category","Match Score"]].copy(); dp.index=range(1,len(dp)+1)
            st.dataframe(dp,use_container_width=True)
            st.markdown(D("Role Details"), unsafe_allow_html=True)
            st.markdown(KK("04 · Breakdown","Job <em>Breakdown</em>"), unsafe_allow_html=True)
            for i,row in tj.iterrows():
                sc=row["Match Score"]
                bc,bl=("jb-hi",f"Strong {sc}%") if sc>=6 else (("jb-mid",f"Moderate {sc}%") if sc>=3 else ("jb-lo",f"Weak {sc}%"))
                desc=str(row.get("Job_Description","")).strip() or "No description available."
                short=((desc[:280]+"...") if len(desc)>280 else desc).replace("<","&lt;").replace(">","&gt;")
                st.markdown(f"""
                <div class="jcard anim" style="animation-delay:{i*0.04}s">
                  <div class="jcard-rank"><div class="jcard-rank-n">{i+1:02d}</div></div>
                  <div class="jcard-body"><div class="jcard-t">{row['Job_Title']}</div><div class="jcard-c">{row.get('Category','—')}</div></div>
                  <div class="jcard-score"><span class="jbadge {bc}">{bl}</span></div>
                  <div class="jcard-desc">{short}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No job listings found. Add jobs.csv to your project directory.")

    else:
        st.markdown("""
        <div class="tl-empty anim">
          <div style="margin-bottom:1.8rem;">
            <svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
              <defs><linearGradient id="eg1" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#6366f1" stop-opacity=".4"/><stop offset="100%" stop-color="#22d3ee" stop-opacity=".2"/></linearGradient></defs>
              <circle cx="60" cy="60" r="55" fill="none" stroke="rgba(99,102,241,0.12)" stroke-width="1"><animateTransform attributeName="transform" type="rotate" from="0 60 60" to="360 60 60" dur="40s" repeatCount="indefinite"/></circle>
              <circle cx="60" cy="60" r="40" fill="none" stroke="rgba(99,102,241,0.2)" stroke-width="1" stroke-dasharray="8 5"><animateTransform attributeName="transform" type="rotate" from="360 60 60" to="0 60 60" dur="25s" repeatCount="indefinite"/></circle>
              <circle cx="60" cy="60" r="24" fill="none" stroke="rgba(34,211,238,0.25)" stroke-width="1.5"/>
              <circle cx="60" cy="60" r="8" fill="url(#eg1)"/>
              <circle cx="60" cy="60" r="8" fill="none" stroke="rgba(99,102,241,0.4)" stroke-width="1"><animate attributeName="r" values="8;18;8" dur="3s" repeatCount="indefinite"/><animate attributeName="opacity" values="1;0;1" dur="3s" repeatCount="indefinite"/></circle>
              <circle cx="60" cy="5" r="2.5" fill="#6366f1" opacity="0.7"><animateTransform attributeName="transform" type="rotate" from="0 60 60" to="360 60 60" dur="10s" repeatCount="indefinite"/></circle>
              <circle cx="60" cy="20" r="1.5" fill="#22d3ee" opacity="0.5"><animateTransform attributeName="transform" type="rotate" from="120 60 60" to="480 60 60" dur="15s" repeatCount="indefinite"/></circle>
            </svg>
          </div>
          <div class="tl-empty-h">Your next role awaits</div>
          <p class="tl-empty-p">Upload a PDF résumé above to activate the matching engine. Semantic analysis, zero sign-up.</p>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  RECRUITER PAGE
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown(f"""
    <div class="rec-hero">
      <div class="hero-badge" style="--p-dim:rgba(34,211,238,.08);--p-line:rgba(34,211,238,.18);">
        <div class="hero-badge-dot" style="background:var(--cyan);box-shadow:0 0 8px rgba(34,211,238,.5);"></div>
        <span class="hero-badge-txt" style="color:var(--cyan);">Recruiter Mode &middot; Bulk Screening</span>
      </div>
      <h1 class="rec-h1">Screen <em>thousands</em><br>in seconds</h1>
      <p class="hero-p" style="max-width:620px;">Paste a job description, upload up to 10,000 résumés via PDFs or a ZIP archive, and TalentLens ranks every candidate by TF-IDF cosine similarity.</p>
      <div class="hero-chips">
        <span class="hero-chip"><span class="hero-chip-dot"></span>Up to <b>10,000</b> résumés</span>
        <span class="hero-chip"><span class="hero-chip-dot"></span>ZIP archive support</span>
        <span class="hero-chip"><span class="hero-chip-dot"></span>CSV export</span>
        <span class="hero-chip"><span class="hero-chip-dot"></span>Keyword gap analysis</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(D("Job Description"), unsafe_allow_html=True)
    st.markdown(KK("Step 01","Describe the <em>Role</em>"), unsafe_allow_html=True)
    st.markdown('<div class="info-strip"><div class="info-strip-line"></div><div class="info-strip-txt">Paste the full job description including responsibilities, skills, and qualifications. <b>More detail = more accurate ranking.</b></div></div>', unsafe_allow_html=True)

    jd=st.text_area("JOB DESCRIPTION",height=200,placeholder="e.g. Senior Data Scientist with 5+ years Python, ML, SQL...",key="jd")
    rj1,rj2=st.columns(2)
    with rj1: rt=st.text_input("ROLE TITLE (OPTIONAL)",placeholder="Senior Data Scientist",key="rt")
    with rj2: tn=st.number_input("TOP N CANDIDATES",min_value=5,max_value=500,value=20,step=5,key="tn")

    st.markdown(D("Upload Résumés"), unsafe_allow_html=True)
    st.markdown(KK("Step 02","Upload <em>Candidates</em>"), unsafe_allow_html=True)
    st.markdown('<div class="info-strip"><div class="info-strip-line"></div><div class="info-strip-txt"><b>Two modes:</b> drag multiple PDFs (up to 1,000), or upload a single <b>ZIP archive</b> containing up to 10,000 PDFs.</div></div>', unsafe_allow_html=True)

    um=st.selectbox("UPLOAD MODE",["Multiple PDF Files (up to 1,000)","ZIP Archive (up to 10,000 PDFs)"],key="um")
    if "Multiple" in um:
        rfs=st.file_uploader("",type=["pdf"],accept_multiple_files=True,label_visibility="collapsed",key="rfs")
    else:
        zf=st.file_uploader("",type=["zip"],label_visibility="collapsed",key="zf"); rfs=None

    st.markdown(D("Filter Options"), unsafe_allow_html=True)
    st.markdown(KK("Step 03","Configure <em>Filters</em>"), unsafe_allow_html=True)
    rf1,rf2=st.columns(2)
    with rf1: fms=st.slider("MINIMUM MATCH SCORE (%)",0,50,0,1,key="fms")
    with rf2: so=st.selectbox("SORT ORDER",["Best Match First","A–Z","Z–A"],key="so")

    st.markdown('<div class="run-btn" style="margin:1.5rem 0 0;">', unsafe_allow_html=True)
    run=st.button("Analyse and Rank All Résumés",use_container_width=True,key="run")
    st.markdown('</div>', unsafe_allow_html=True)

    if run:
        if not jd.strip():
            st.error("Please paste a job description first.")
        else:
            parsed=[]
            if "Multiple" in um:
                fl=rfs or []
                if not fl: st.error("Upload at least one PDF."); st.stop()
                prog=st.progress(0,text="Parsing résumés...")
                for i,f in enumerate(fl):
                    raw=pdf_text(f); nm=f.name.replace(".pdf","").replace("_"," ").replace("-"," ")
                    parsed.append({"name":nm,"text":clean(raw),"raw":raw})
                    prog.progress((i+1)/len(fl),text=f"Parsing {i+1}/{len(fl)}: {f.name}")
                prog.empty()
            else:
                if 'zf' not in dir() or zf is None: st.error("Upload a ZIP file."); st.stop()
                with st.spinner("Extracting ZIP..."):
                    zb=io.BytesIO(zf.read())
                    with zipfile.ZipFile(zb,"r") as z: pns=[n for n in z.namelist() if n.lower().endswith(".pdf")]
                if not pns: st.error("No PDFs in ZIP."); st.stop()
                prog=st.progress(0,text=f"Processing {len(pns):,} résumés...")
                for i,pn in enumerate(pns):
                    try:
                        with zipfile.ZipFile(zb,"r") as z: pd_data=z.read(pn)
                        raw=pdf_text(io.BytesIO(pd_data)); nm=os.path.basename(pn).replace(".pdf","").replace("_"," ").replace("-"," ")
                        parsed.append({"name":nm,"text":clean(raw),"raw":raw})
                    except: pass
                    if i%50==0: prog.progress((i+1)/len(pns),text=f"Parsed {i+1:,}/{len(pns):,}...")
                prog.empty()

            tu=len(parsed)
            if tu==0: st.error("No readable résumés found."); st.stop()

            jdc=clean(jd)
            with st.spinner(f"Vectorising {tu:,} résumés..."): rdf=rank_resumes(jdc,parsed,int(tn))
            rdf=rdf[rdf["Match Score"]>=fms]
            if so=="A–Z": rdf=rdf.sort_values("File Name")
            elif so=="Z–A": rdf=rdf.sort_values("File Name",ascending=False)
            rdf=rdf.reset_index(drop=True); jkw=top_kw(jd,12)

            st.markdown(D("Results"), unsafe_allow_html=True)
            avg=rdf["Match Score"].mean() if not rdf.empty else 0
            top_s=rdf["Match Score"].max() if not rdf.empty else 0
            abv=len(rdf[rdf["Match Score"]>=5])
            st.markdown(f"""
            <div class="stat-grid">
              <div class="stat-card anim"><div class="stat-accent sa1"></div><div class="stat-tag">TOTAL</div><div class="stat-n">{tu:,}</div><div class="stat-l">Résumés Processed</div></div>
              <div class="stat-card anim a1"><div class="stat-accent sa2"></div><div class="stat-tag">BEST</div><div class="stat-n">{top_s:.1f}%</div><div class="stat-l">Top Match Score</div></div>
              <div class="stat-card anim a2"><div class="stat-accent sa3"></div><div class="stat-tag">AVG</div><div class="stat-n">{avg:.1f}%</div><div class="stat-l">Average Score</div></div>
              <div class="stat-card anim a3"><div class="stat-accent sa4"></div><div class="stat-tag">STRONG</div><div class="stat-n">{abv}</div><div class="stat-l">Strong Matches</div></div>
            </div>""", unsafe_allow_html=True)

            if jkw:
                tags="".join(f'<span class="kw-ctag">{k}</span>' for k in jkw)
                st.markdown(f'<div class="kw-panel anim a1"><div class="kw-panel-k">Extracted from Job Description</div><div class="kw-panel-t">Matching Signals</div><div class="kw-cloud">{tags}</div></div>', unsafe_allow_html=True)

            st.markdown(KK("Ranked Results",f"Top <em>{len(rdf)}</em> Candidates"), unsafe_allow_html=True)
            ddf=rdf[["Rank","File Name","Match Score"]].copy()
            ddf["Match Score"]=ddf["Match Score"].apply(lambda x:f"{x:.2f}%")
            st.dataframe(ddf,use_container_width=True,hide_index=True)
            csv=rdf[["Rank","File Name","Match Score"]].to_csv(index=False).encode()
            st.download_button("Export Rankings as CSV",data=csv,file_name=f"talentlens_{(rt or 'rankings').replace(' ','_')}.csv",mime="text/csv",use_container_width=True,key="dl")

            st.markdown(D("Candidate Profiles"), unsafe_allow_html=True)
            st.markdown(KK("Detailed View","Candidate <em>Profiles</em>"), unsafe_allow_html=True)
            mset=set(jkw)
            for _,row in rdf.head(50).iterrows():
                rank=int(row["Rank"]); sc=float(row["Match Score"]); nm=row["File Name"]
                med=medal(rank); raw_t=next((r["raw"] for r in parsed if r["name"]==nm),"")
                rc2="hi" if sc>=10 else ("mid" if sc>=4 else "")
                bc,bl=("jb-hi",f"Strong {sc:.1f}%") if sc>=10 else (("jb-mid",f"Moderate {sc:.1f}%") if sc>=4 else ("jb-lo",f"Weak {sc:.1f}%"))
                ckw=top_kw(raw_t,10)
                kwh="".join(f'<span class="rtag{"  match" if k in mset else ""}">{k}</span>' for k in ckw[:8])
                snip=re.sub(r'\s+',' ',raw_t[:180].strip()).replace("<","&lt;").replace(">","&gt;") if raw_t else "No text extracted."
                st.markdown(f"""
                <div class="rcard anim">
                  <div class="rcard-rank"><div class="rcard-rank-n">{rank:02d}</div><div class="rcard-rank-m">{med}</div></div>
                  <div class="rcard-body">
                    <div class="rcard-name">{nm}</div>
                    <div class="rcard-meta">{len(raw_t):,} chars &middot; <span class="jbadge {bc}" style="padding:2px 8px;font-size:.56rem;">{bl}</span></div>
                    <div class="tag-row">{kwh}</div>
                    <div style="margin-top:9px;font-size:.75rem;color:var(--t3);line-height:1.75;font-weight:300;">{snip}...</div>
                  </div>
                  <div class="rcard-score"><div class="score-ring {rc2}"><div class="score-ring-v">{sc:.0f}%</div><div class="score-ring-l">MATCH</div></div></div>
                </div>""", unsafe_allow_html=True)

            if len(rdf)>50:
                st.markdown(f'<div class="info-strip"><div class="info-strip-line"></div><div class="info-strip-txt">Showing top 50 profiles. All {len(rdf)} results are in the exported CSV.</div></div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="tl-empty anim">
          <div style="margin-bottom:1.8rem;">
            <svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg">
              <defs><linearGradient id="eg2" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#22d3ee" stop-opacity=".4"/><stop offset="100%" stop-color="#6366f1" stop-opacity=".2"/></linearGradient></defs>
              <circle cx="60" cy="60" r="55" fill="none" stroke="rgba(34,211,238,0.12)" stroke-width="1"><animateTransform attributeName="transform" type="rotate" from="0 60 60" to="360 60 60" dur="40s" repeatCount="indefinite"/></circle>
              <rect x="24" y="24" width="72" height="72" rx="10" fill="none" stroke="rgba(34,211,238,0.2)" stroke-width="1" stroke-dasharray="8 4"><animateTransform attributeName="transform" type="rotate" from="0 60 60" to="360 60 60" dur="30s" repeatCount="indefinite"/></rect>
              <circle cx="60" cy="60" r="16" fill="none" stroke="rgba(99,102,241,0.3)" stroke-width="1.5"/>
              <circle cx="60" cy="60" r="6" fill="url(#eg2)"/>
              <circle cx="60" cy="60" r="6" fill="none" stroke="rgba(34,211,238,0.4)" stroke-width="1"><animate attributeName="r" values="6;18;6" dur="3s" repeatCount="indefinite"/><animate attributeName="opacity" values="1;0;1" dur="3s" repeatCount="indefinite"/></circle>
              <circle cx="60" cy="5" r="2" fill="#22d3ee" opacity="0.6"><animateTransform attributeName="transform" type="rotate" from="0 60 60" to="360 60 60" dur="12s" repeatCount="indefinite"/></circle>
            </svg>
          </div>
          <div class="tl-empty-h">Ready to screen candidates</div>
          <p class="tl-empty-p">Paste a job description, upload your résumé batch, then click Analyse and Rank.</p>
        </div>""", unsafe_allow_html=True)