import re, io, os, warnings, zipfile
import pandas as pd
import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

st.set_page_config(page_title="TalentLens", page_icon="◎", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@300;400;500&display=swap');
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stSidebarNav"],[data-testid="collapsedControl"],[data-testid="stHeader"],[data-testid="stStatusWidget"],.stDeployButton{display:none !important;visibility:hidden !important;}
:root{--bg:#080810;--bg1:#0d0d18;--bg2:#111120;--bg3:#161628;--bg4:#1c1c32;--p:#6366f1;--p2:#818cf8;--p3:#a5b4fc;--p-glow:rgba(99,102,241,0.3);--p-dim:rgba(99,102,241,0.1);--p-line:rgba(99,102,241,0.2);--cyan:#22d3ee;--cyan-dim:rgba(34,211,238,0.08);--cyan-line:rgba(34,211,238,0.18);--emerald:#10b981;--em-dim:rgba(16,185,129,0.08);--em-line:rgba(16,185,129,0.2);--amber:#f59e0b;--am-dim:rgba(245,158,11,0.08);--am-line:rgba(245,158,11,0.2);--rose:#f43f5e;--ro-dim:rgba(244,63,94,0.08);--t0:#ffffff;--t1:#e2e2f0;--t2:#9090b0;--t3:#55556a;--t4:#2a2a42;--b1:rgba(255,255,255,0.08);--b2:rgba(255,255,255,0.04);--b3:rgba(255,255,255,0.02);--r1:6px;--r2:12px;--r3:18px;--r4:24px;--spring:cubic-bezier(0.34,1.56,0.64,1);--smooth:cubic-bezier(0.4,0,0.2,1);--out:cubic-bezier(0,0,0.2,1);--sha:0 1px 3px rgba(0,0,0,.5),0 4px 16px rgba(0,0,0,.3);--shb:0 4px 20px rgba(0,0,0,.6),0 2px 6px rgba(0,0,0,.4);--shc:0 16px 64px rgba(0,0,0,.7),0 4px 16px rgba(0,0,0,.5);--shg:0 0 40px rgba(99,102,241,.15),0 0 80px rgba(99,102,241,.08);}
html,body,[data-testid="stAppViewContainer"],[data-testid="stMain"],section.main,.stApp{background:var(--bg) !important;color:var(--t1) !important;font-family:'Geist',system-ui,sans-serif !important;-webkit-font-smoothing:antialiased;}
.main .block-container{padding:0 2.5rem 8rem !important;max-width:1280px !important;}
::-webkit-scrollbar{width:2px;}::-webkit-scrollbar-thumb{background:var(--p);border-radius:99px;}
#tl-dot{position:fixed;z-index:2147483647;pointer-events:none;width:10px;height:10px;border-radius:50%;background:#818cf8;box-shadow:0 0 0 2px rgba(129,140,248,0.35),0 0 16px rgba(99,102,241,0.9),0 0 32px rgba(99,102,241,0.35);transform:translate(-50%,-50%);left:-100px;top:-100px;transition:width .15s,height .15s,background .15s,box-shadow .15s,opacity .2s;opacity:0;}
#tl-ring{position:fixed;z-index:2147483646;pointer-events:none;width:36px;height:36px;border-radius:50%;border:1.5px solid rgba(129,140,248,0.55);transform:translate(-50%,-50%);left:-100px;top:-100px;transition:width .3s var(--spring),height .3s var(--spring),border-color .2s,background .2s,opacity .2s;background:transparent;opacity:0;}
.tl-mesh{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden;}
.tl-mesh-g{position:absolute;inset:0;background:radial-gradient(ellipse 60% 50% at 20% 20%,rgba(99,102,241,.07) 0%,transparent 60%),radial-gradient(ellipse 50% 40% at 80% 80%,rgba(34,211,238,.05) 0%,transparent 60%),radial-gradient(ellipse 40% 35% at 60% 10%,rgba(99,102,241,.04) 0%,transparent 50%);animation:mesh 20s ease-in-out infinite alternate;}
@keyframes mesh{0%{opacity:1;transform:scale(1)}50%{opacity:.7;transform:scale(1.05) translate(-1%,1%)}100%{opacity:1;transform:scale(1) translate(1%,-.5%)}}
.tl-mesh-grid{position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.018) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.018) 1px,transparent 1px);background-size:72px 72px;mask-image:radial-gradient(ellipse 100% 100% at 50% 0%,black 30%,transparent 80%);}
.tl-mesh-noise{position:absolute;inset:0;opacity:.022;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");background-size:200px 200px;}
.tl-nav{display:flex;align-items:center;justify-content:space-between;padding:1.4rem 0 1.2rem;border-bottom:1px solid var(--b1);position:relative;z-index:20;animation:slideDown .5s var(--out) both;}
@keyframes slideDown{from{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:none}}
.tl-nav::after{content:'';position:absolute;bottom:-1px;left:0;width:180px;height:1px;background:linear-gradient(90deg,var(--p),transparent);}
.tl-brand{display:flex;align-items:center;gap:12px;}.tl-logomark{width:36px;height:36px;flex-shrink:0;}.tl-logomark svg{width:36px;height:36px;display:block;}
.tl-brand-text{display:flex;flex-direction:column;gap:3px;}
.tl-brand-name{font-family:'Instrument Serif',serif;font-size:1.25rem;font-weight:400;color:var(--t0);letter-spacing:-.01em;line-height:1;}
.tl-brand-name em{color:var(--p2);font-style:italic;}
.tl-brand-tag{font-family:'Geist Mono',monospace;font-size:0.6rem;letter-spacing:.18em;text-transform:uppercase;color:var(--t2);line-height:1;}
.tl-nav-tabs{display:flex;align-items:center;gap:2px;background:var(--bg2);border:1px solid var(--b1);border-radius:var(--r2);padding:3px;}
.tl-nav-tab{padding:.45rem 1.2rem;border-radius:10px;font-family:'Geist',sans-serif;font-size:.82rem;font-weight:500;color:var(--t2);border:1px solid transparent;transition:all .2s var(--smooth);white-space:nowrap;display:flex;align-items:center;gap:7px;}
.tl-nav-tab:hover{color:var(--t1);background:var(--bg3);}.tl-nav-tab.on{background:var(--bg3);color:var(--t0);border-color:var(--b1);box-shadow:var(--sha);}
.tl-tab-dot{width:6px;height:6px;border-radius:50%;background:var(--p2);box-shadow:0 0 6px var(--p-glow);animation:pulse 2.5s ease-in-out infinite;}
@keyframes pulse{0%,100%{box-shadow:0 0 6px var(--p-glow)}50%{box-shadow:0 0 12px var(--p-glow),0 0 20px rgba(99,102,241,.1)}}
.tl-nav-stats{display:flex;align-items:center;}
.tl-stat{display:flex;flex-direction:column;align-items:flex-end;gap:2px;padding:0 1.4rem;border-right:1px solid var(--b1);}
.tl-stat:last-child{border-right:none;padding-right:0;}
.tl-stat-n{font-family:'Geist Mono',monospace;font-size:1rem;font-weight:500;color:var(--t0);letter-spacing:-.04em;line-height:1;}
.tl-stat-k{font-family:'Geist Mono',monospace;font-size:.48rem;letter-spacing:.18em;text-transform:uppercase;color:var(--t3);}
[data-testid="stButton"] button{background:transparent !important;border:1px solid var(--b1) !important;color:var(--t2) !important;font-family:'Geist',sans-serif !important;font-size:.82rem !important;font-weight:500 !important;padding:.55rem 1.2rem !important;border-radius:10px !important;transition:all .2s !important;width:100% !important;letter-spacing:-.01em !important;}
[data-testid="stButton"] button:hover{color:var(--t1) !important;background:var(--bg3) !important;}
[data-testid="stButton"] button[kind="primary"]{background:var(--bg3) !important;color:var(--t0) !important;border-color:var(--p-line) !important;box-shadow:inset 0 0 0 1px var(--p-line),0 0 12px rgba(99,102,241,.1) !important;}
.hero-main{background:var(--bg1);border:1px solid var(--b1);border-radius:var(--r4);padding:3rem 3.2rem 2.8rem;position:relative;overflow:hidden;margin-top:2rem;animation:heroIn .6s var(--out) .1s both;}
@keyframes heroIn{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}
.hero-main::before{content:'';position:absolute;top:-100px;right:-60px;width:360px;height:360px;background:radial-gradient(circle,rgba(99,102,241,.12) 0%,transparent 65%);pointer-events:none;}
.hero-main::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--p-line) 30%,var(--p) 55%,var(--cyan-line) 80%,transparent);}
.hero-badge{display:inline-flex;align-items:center;gap:8px;background:var(--p-dim);border:1px solid var(--p-line);border-radius:100px;padding:5px 14px 5px 8px;margin-bottom:1.6rem;}
.hero-badge-dot{width:6px;height:6px;border-radius:50%;background:var(--p2);box-shadow:0 0 8px var(--p-glow);animation:pulse 2.5s ease-in-out infinite;}
.hero-badge-txt{font-family:'Geist Mono',monospace;font-size:.6rem;letter-spacing:.12em;text-transform:uppercase;color:var(--p2);}
.hero-h1{font-family:'Instrument Serif',serif !important;font-size:clamp(2.8rem,4.2vw,4.6rem) !important;font-weight:400 !important;line-height:1.02 !important;color:var(--t0) !important;letter-spacing:-.02em !important;margin:0 0 1.4rem !important;}
.hero-h1 em{color:transparent !important;background:linear-gradient(135deg,var(--p2),var(--cyan));-webkit-background-clip:text;background-clip:text;font-style:italic;}
.hero-p{font-size:1rem;color:var(--t2);line-height:1.7;max-width:460px;margin-bottom:2rem;font-weight:300;}
.hero-chips{display:flex;flex-wrap:wrap;gap:6px;}
.hero-chip{display:inline-flex;align-items:center;gap:6px;background:var(--bg2);border:1px solid var(--b1);border-radius:100px;padding:5px 12px;font-family:'Geist Mono',monospace;font-size:.62rem;color:var(--t2);transition:all .2s;}
.hero-chip:hover{border-color:var(--p-line);color:var(--p2);background:var(--p-dim);}
.hero-chip-dot{width:4px;height:4px;border-radius:50%;background:var(--p2);}
.hero-chip b{color:var(--t0);font-weight:600;}
.upload-panel{background:var(--bg1);border:1px solid var(--b1);border-radius:var(--r4);padding:2rem;display:flex;flex-direction:column;gap:1.2rem;position:relative;overflow:hidden;margin-top:2rem;animation:heroIn .6s var(--out) .2s both;}
.upload-panel::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--cyan-line),transparent);}
.up-label{font-family:'Geist Mono',monospace;font-size:.55rem;letter-spacing:.2em;text-transform:uppercase;color:var(--t2);display:flex;align-items:center;gap:8px;}
.up-label::before{content:'';display:block;width:16px;height:1px;background:var(--p-line);}
.up-title{font-family:'Instrument Serif',serif;font-size:1.55rem;font-weight:400;color:var(--t0);line-height:1.15;}
.up-title span{color:var(--p2);font-style:italic;}
.up-meta{font-size:.76rem;color:var(--t3);line-height:1.6;}
.up-steps{display:flex;flex-direction:column;border:1px solid var(--b2);border-radius:var(--r2);overflow:hidden;}
.up-step{display:flex;align-items:center;gap:12px;padding:.7rem 1rem;border-bottom:1px solid var(--b2);}
.up-step:last-child{border-bottom:none;}
.up-step-n{width:22px;height:22px;border-radius:50%;flex-shrink:0;border:1px solid var(--p-line);background:var(--p-dim);display:flex;align-items:center;justify-content:center;font-family:'Geist Mono',monospace;font-size:.58rem;color:var(--p2);}
.up-step-t{font-size:.78rem;color:var(--t2);}
[data-testid="stFileUploader"]{background:linear-gradient(135deg,rgba(99,102,241,.06),rgba(34,211,238,.03)) !important;border:1px dashed var(--p-line) !important;border-radius:var(--r3) !important;transition:all .3s !important;}
[data-testid="stFileUploader"]:hover{border-color:var(--p) !important;background:var(--p-dim) !important;box-shadow:0 0 30px rgba(99,102,241,.08) !important;}
[data-testid="stFileUploaderDropzone"]{background:transparent !important;border:none !important;}
[data-testid="stFileUploader"] p,[data-testid="stFileUploader"] span,[data-testid="stFileUploader"] small,[data-testid="stFileUploader"] div{color:var(--t2) !important;font-family:'Geist',sans-serif !important;}
[data-testid="stFileUploader"] button{background:linear-gradient(135deg,var(--p),#4f46e5) !important;color:white !important;border:none !important;font-weight:600 !important;border-radius:var(--r2) !important;font-family:'Geist',sans-serif !important;font-size:.8rem !important;box-shadow:0 4px 16px rgba(99,102,241,.3) !important;}
.tl-divide{display:flex;align-items:center;gap:1.2rem;margin:2.5rem 0 1.8rem;position:relative;z-index:5;}
.tl-divide-line{flex:1;height:1px;background:var(--b1);}
.tl-divide-label{font-family:'Geist Mono',monospace;font-size:.52rem;letter-spacing:.22em;text-transform:uppercase;color:var(--t4);white-space:nowrap;display:flex;align-items:center;gap:8px;}
.tl-divide-gem{width:4px;height:4px;border-radius:1px;background:var(--p);transform:rotate(45deg);box-shadow:0 0 6px var(--p-glow);}
hr{display:none !important;}
.tl-kicker{font-family:'Geist Mono',monospace;font-size:.55rem;letter-spacing:.2em;text-transform:uppercase;color:var(--t3);display:flex;align-items:center;gap:8px;margin-bottom:.5rem;position:relative;z-index:5;}
.tl-kicker::before{content:'';width:3px;height:14px;border-radius:2px;background:linear-gradient(180deg,var(--p),var(--cyan));}
.tl-h{font-family:'Instrument Serif',serif !important;font-size:1.9rem !important;font-weight:400 !important;color:var(--t0) !important;letter-spacing:-.02em !important;line-height:1.1 !important;margin:0 0 1.2rem !important;position:relative;z-index:5;}
.tl-h em{color:var(--p2);font-style:italic;}
.tl-term{background:var(--bg);border:1px solid var(--b1);border-radius:var(--r3);overflow:hidden;position:relative;z-index:5;box-shadow:var(--shc);}
.tl-term::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--p-line) 25%,var(--p-line) 75%,transparent);}
.tl-chrome{display:flex;align-items:center;justify-content:space-between;padding:.75rem 1.2rem;background:var(--bg1);border-bottom:1px solid var(--b1);}
.tl-dots{display:flex;gap:6px;}.tl-dot{width:10px;height:10px;border-radius:50%;}
.d-r{background:#ff5f57}.d-y{background:#ffbd2e}.d-g{background:#28c941}
.tl-chrome-id{font-family:'Geist Mono',monospace;font-size:.58rem;color:var(--t3);}
.tl-chrome-ok{font-family:'Geist Mono',monospace;font-size:.54rem;padding:2px 8px;border-radius:var(--r1);background:var(--em-dim);border:1px solid var(--em-line);color:var(--emerald);}
.tl-term-body{padding:1.4rem 1.6rem;font-family:'Geist Mono',monospace;font-size:.72rem;color:var(--t2);line-height:1.85;max-height:200px;overflow-y:auto;white-space:pre-wrap;}
.tl-prompt{color:var(--p2);}
.tl-blink{display:inline-block;width:7px;height:13px;background:var(--p2);vertical-align:middle;margin-left:2px;animation:blink 1s step-end infinite;}
@keyframes blink{50%{opacity:0}}
.cls-panel{background:var(--bg1);border:1px solid var(--b1);border-radius:var(--r3);overflow:hidden;z-index:5;box-shadow:var(--shb);}
.cls-hdr{padding:1.6rem 1.8rem 1.2rem;background:linear-gradient(135deg,var(--bg2),var(--bg1));border-bottom:1px solid var(--b1);position:relative;}
.cls-hdr::after{content:'';position:absolute;bottom:-1px;left:1.8rem;width:40px;height:1px;background:var(--p);}
.cls-kicker{font-family:'Geist Mono',monospace;font-size:.52rem;letter-spacing:.18em;text-transform:uppercase;color:var(--p2);margin-bottom:.4rem;}
.cls-role{font-family:'Instrument Serif',serif;font-size:1.8rem;font-weight:400;color:var(--t0);}
.cls-body{padding:1.2rem 1.8rem 1.6rem;}
.kw-row{display:flex;align-items:center;gap:10px;padding:.6rem 0;border-bottom:1px solid var(--b2);}
.kw-row:last-child{border-bottom:none;}
.kw-i{font-family:'Geist Mono',monospace;font-size:.52rem;color:var(--t4);min-width:18px;}
.kw-n{font-size:.82rem;color:var(--t2);min-width:160px;font-weight:300;}
.kw-n.top{color:var(--p2);font-weight:500;}
.kw-track{flex:1;height:2px;background:var(--b1);border-radius:99px;overflow:hidden;}
.kw-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--p),var(--cyan));}
.kw-v{font-family:'Geist Mono',monospace;font-size:.62rem;color:var(--p2);}
.chart-panel{background:var(--bg1);border:1px solid var(--b1);border-radius:var(--r3);padding:1.6rem 1.6rem 1.2rem;z-index:5;box-shadow:var(--shb);}
.chart-k{font-family:'Geist Mono',monospace;font-size:.52rem;letter-spacing:.18em;text-transform:uppercase;color:var(--t3);margin-bottom:.3rem;}
.chart-t{font-family:'Instrument Serif',serif;font-size:1.1rem;color:var(--t0);margin-bottom:1rem;}
.jcard{display:grid;grid-template-columns:60px 1fr auto;grid-template-rows:auto auto;background:var(--bg1);border:1px solid var(--b1);border-radius:var(--r3);overflow:hidden;margin-bottom:8px;box-shadow:var(--sha);position:relative;transition:border-color .25s,box-shadow .25s,transform .25s var(--spring);}
.jcard::before{content:'';position:absolute;left:0;top:0;bottom:0;width:2px;background:linear-gradient(180deg,var(--p),var(--cyan));opacity:0;transition:opacity .25s;}
.jcard:hover{border-color:var(--p-line);box-shadow:var(--shg),var(--shb);transform:translateX(4px) translateY(-1px);}
.jcard:hover::before{opacity:1;}
.jcard-rank{grid-row:1/3;background:var(--bg2);border-right:1px solid var(--b2);display:flex;align-items:center;justify-content:center;}
.jcard-rank-n{font-family:'Instrument Serif',serif;font-size:1.5rem;color:var(--t4);}
.jcard-body{padding:1rem 1.2rem .35rem;}
.jcard-t{font-family:'Instrument Serif',serif;font-size:1.05rem;color:var(--t0);margin-bottom:2px;}
.jcard-c{font-family:'Geist Mono',monospace;font-size:.58rem;color:var(--t3);}
.jcard-score{padding:1rem 1.2rem .35rem;}
.jbadge{display:inline-flex;align-items:center;padding:4px 10px;border-radius:100px;font-family:'Geist Mono',monospace;font-size:.6rem;white-space:nowrap;}
.jb-hi{background:var(--em-dim);color:var(--emerald);border:1px solid var(--em-line);}
.jb-mid{background:var(--am-dim);color:var(--amber);border:1px solid var(--am-line);}
.jb-lo{background:var(--b2);color:var(--t3);border:1px solid var(--b1);}
.jcard-desc{grid-column:2/-1;padding:.6rem 1.2rem 1rem;font-size:.78rem;color:var(--t3);line-height:1.7;border-top:1px solid var(--b2);font-weight:300;}
[data-testid="stDataFrame"]>div{border:1px solid var(--b1) !important;border-radius:var(--r3) !important;overflow:hidden !important;box-shadow:var(--shb) !important;}
[data-testid="stDataFrame"] thead tr th{background:var(--bg2) !important;color:var(--t3) !important;font-family:'Geist Mono',monospace !important;font-size:.54rem !important;letter-spacing:.18em !important;text-transform:uppercase !important;padding:12px 18px !important;border-bottom:1px solid var(--b1) !important;border-right:none !important;font-weight:400 !important;}
[data-testid="stDataFrame"] tbody tr td{background:var(--bg1) !important;color:var(--t1) !important;font-family:'Geist',sans-serif !important;font-size:.84rem !important;padding:11px 18px !important;border-bottom:1px solid var(--b2) !important;border-right:none !important;font-weight:300 !important;}
[data-testid="stDataFrame"] tbody tr:hover td{background:var(--bg2) !important;}
.rec-hero{background:var(--bg1);border:1px solid var(--b1);border-radius:var(--r4);padding:3rem 3.2rem;margin-top:2rem;position:relative;overflow:hidden;z-index:5;animation:heroIn .6s var(--out) .1s both;}
.rec-hero::before{content:'';position:absolute;top:-80px;right:-60px;width:340px;height:340px;background:radial-gradient(circle,rgba(34,211,238,.08) 0%,transparent 60%);pointer-events:none;}
.rec-hero::after{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--cyan-line),transparent);}
.rec-h1{font-family:'Instrument Serif',serif !important;font-size:clamp(2.5rem,3.8vw,4rem) !important;font-weight:400 !important;line-height:1.02 !important;color:var(--t0) !important;letter-spacing:-.02em !important;margin:1.4rem 0 1rem !important;}
.rec-h1 em{color:transparent !important;background:linear-gradient(135deg,var(--cyan),var(--p2));-webkit-background-clip:text;background-clip:text;font-style:italic;}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:1.6rem 0;z-index:5;}
.stat-card{background:var(--bg1);border:1px solid var(--b1);border-radius:var(--r3);padding:1.3rem 1.5rem;position:relative;overflow:hidden;transition:border-color .25s,transform .25s var(--spring);}
.stat-card:hover{border-color:var(--p-line);transform:translateY(-3px);}
.stat-accent{position:absolute;top:0;left:0;right:0;height:2px;}
.sa1{background:linear-gradient(90deg,var(--p),var(--p2))}.sa2{background:linear-gradient(90deg,var(--emerald),var(--cyan))}.sa3{background:linear-gradient(90deg,var(--cyan),var(--p2))}.sa4{background:linear-gradient(90deg,var(--amber),var(--rose))}
.stat-n{font-family:'Instrument Serif',serif;font-size:2rem;color:var(--t0);line-height:1;margin-bottom:4px;letter-spacing:-.03em;}
.stat-l{font-family:'Geist Mono',monospace;font-size:.52rem;letter-spacing:.16em;text-transform:uppercase;color:var(--t3);}
.stat-tag{position:absolute;top:1rem;right:1.2rem;font-family:'Geist Mono',monospace;font-size:.5rem;color:var(--t4);}
.info-strip{display:flex;align-items:flex-start;gap:10px;background:rgba(34,211,238,.04);border:1px solid var(--cyan-line);border-radius:var(--r2);padding:.9rem 1.2rem;margin:1rem 0;z-index:5;}
.info-strip-line{width:2px;flex-shrink:0;border-radius:2px;background:linear-gradient(180deg,var(--cyan),transparent);align-self:stretch;}
.info-strip-txt{font-size:.79rem;color:var(--t2);line-height:1.6;font-weight:300;}
.info-strip-txt b{color:var(--cyan);font-weight:500;}
.warn-strip{display:flex;align-items:flex-start;gap:10px;background:rgba(245,158,11,.04);border:1px solid var(--am-line);border-radius:var(--r2);padding:.9rem 1.2rem;margin:1rem 0;z-index:5;}
.warn-strip-line{width:2px;flex-shrink:0;border-radius:2px;background:linear-gradient(180deg,var(--amber),transparent);align-self:stretch;}
.warn-strip-txt{font-size:.79rem;color:var(--t2);line-height:1.6;font-weight:300;}
.warn-strip-txt b{color:var(--amber);font-weight:500;}
[data-testid="stTextArea"] textarea{background:var(--bg1) !important;border:1px solid var(--b1) !important;color:var(--t1) !important;font-family:'Geist',sans-serif !important;border-radius:var(--r2) !important;font-size:.86rem !important;font-weight:300 !important;line-height:1.7 !important;transition:border-color .2s,box-shadow .2s !important;}
[data-testid="stTextArea"] textarea:focus{border-color:var(--p) !important;box-shadow:0 0 0 3px var(--p-dim) !important;}
[data-testid="stTextArea"] label,[data-testid="stTextInput"] label,[data-testid="stNumberInput"] label,[data-testid="stSlider"] label,[data-testid="stSelectbox"] label{color:var(--t2) !important;font-family:'Geist Mono',monospace !important;font-size:.56rem !important;letter-spacing:.16em !important;text-transform:uppercase !important;}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input{background:var(--bg1) !important;border:1px solid var(--b1) !important;color:var(--t1) !important;border-radius:var(--r2) !important;font-family:'Geist',sans-serif !important;}
[data-testid="stSelectbox"]>div>div{background:var(--bg1) !important;border:1px solid var(--b1) !important;color:var(--t1) !important;border-radius:var(--r2) !important;}
.run-btn [data-testid="stButton"] button{display:flex !important;align-items:center !important;justify-content:center !important;width:100% !important;padding:1rem 2rem !important;background:linear-gradient(135deg,var(--p) 0%,#4f46e5 100%) !important;color:white !important;border:none !important;border-radius:var(--r3) !important;font-family:'Geist',sans-serif !important;font-weight:600 !important;font-size:.95rem !important;letter-spacing:-.01em !important;box-shadow:0 8px 32px rgba(99,102,241,.35),0 2px 8px rgba(99,102,241,.2) !important;transition:all .25s var(--spring) !important;}
.run-btn [data-testid="stButton"] button:hover{box-shadow:0 12px 48px rgba(99,102,241,.5) !important;transform:translateY(-2px) !important;}
[data-testid="stDownloadButton"] button{background:transparent !important;border:1px solid var(--b1) !important;color:var(--t2) !important;font-family:'Geist Mono',monospace !important;font-size:.68rem !important;border-radius:var(--r2) !important;padding:.7rem 1.5rem !important;transition:all .2s !important;width:100% !important;}
[data-testid="stDownloadButton"] button:hover{border-color:var(--p-line) !important;color:var(--p2) !important;background:var(--p-dim) !important;}
.rcard{display:grid;grid-template-columns:72px 1fr 80px;background:var(--bg1);border:1px solid var(--b1);border-radius:var(--r3);overflow:hidden;margin-bottom:8px;box-shadow:var(--sha);position:relative;z-index:5;transition:border-color .25s,box-shadow .25s,transform .25s var(--spring);}
.rcard::before{content:'';position:absolute;left:0;top:0;bottom:0;width:2px;background:linear-gradient(180deg,var(--emerald),var(--cyan));opacity:0;transition:opacity .25s;}
.rcard:hover{border-color:var(--p-line);box-shadow:var(--shg),var(--shb);transform:translateX(4px) translateY(-1px);}
.rcard:hover::before{opacity:1;}
.rcard-rank{background:var(--bg2);border-right:1px solid var(--b2);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:4px;}
.rcard-rank-n{font-family:'Instrument Serif',serif;font-size:1.7rem;color:var(--t4);}
.rcard-rank-m{font-family:'Geist Mono',monospace;font-size:.48rem;color:var(--p2);}
.rcard-body{padding:1.1rem 1.4rem;}
.rcard-name{font-family:'Instrument Serif',serif;font-size:1.1rem;color:var(--t0);margin-bottom:3px;}
.rcard-meta{font-family:'Geist Mono',monospace;font-size:.56rem;color:var(--t3);margin-bottom:7px;}
.tag-row{display:flex;flex-wrap:wrap;gap:4px;margin-top:5px;}
.rtag{display:inline-block;padding:2px 8px;border-radius:var(--r1);font-family:'Geist Mono',monospace;font-size:.56rem;border:1px solid var(--b1);color:var(--t3);}
.rtag.match{border-color:var(--em-line);color:var(--emerald);background:var(--em-dim);}
.rcard-score{border-left:1px solid var(--b2);display:flex;align-items:center;justify-content:center;}
.score-ring{width:54px;height:54px;border-radius:50%;border:1.5px solid var(--b1);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:1px;}
.score-ring.hi{border-color:var(--emerald);box-shadow:0 0 16px rgba(16,185,129,.2);}
.score-ring.mid{border-color:var(--amber);box-shadow:0 0 16px rgba(245,158,11,.2);}
.score-ring-v{font-family:'Instrument Serif',serif;font-size:1.05rem;color:var(--t0);line-height:1;}
.score-ring-l{font-family:'Geist Mono',monospace;font-size:.4rem;color:var(--t3);letter-spacing:.12em;text-transform:uppercase;}
.kw-panel{background:var(--bg1);border:1px solid var(--b1);border-radius:var(--r3);padding:1.5rem 1.6rem;margin-bottom:1.2rem;z-index:5;}
.kw-panel-k{font-family:'Geist Mono',monospace;font-size:.52rem;letter-spacing:.18em;text-transform:uppercase;color:var(--t3);margin-bottom:.4rem;}
.kw-panel-t{font-family:'Instrument Serif',serif;font-size:1.1rem;color:var(--t0);margin-bottom:.9rem;}
.kw-cloud{display:flex;flex-wrap:wrap;gap:8px;row-gap:8px;}
.kw-ctag{display:inline-flex;align-items:center;padding:5px 14px;border-radius:100px;font-family:'Geist Mono',monospace;font-size:.62rem;letter-spacing:.02em;white-space:nowrap;border:1px solid var(--em-line);color:var(--emerald);background:var(--em-dim);transition:background .2s,border-color .2s;}
.kw-ctag:hover{background:rgba(16,185,129,0.14);border-color:var(--emerald);}

.jda-hdr{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:1px;background:var(--b2);border-bottom:1px solid var(--b1);}
.jda-meta{background:var(--bg2);padding:1rem 1.4rem;display:flex;flex-direction:column;gap:3px;}
.jda-meta-k{font-family:"Geist Mono",monospace;font-size:.5rem;letter-spacing:.18em;text-transform:uppercase;color:var(--t3);}
.jda-meta-v{font-family:"Instrument Serif",serif;font-size:1.25rem;color:var(--t0);line-height:1.1;}
.jda-meta-v.cyan{color:var(--cyan);}.jda-meta-v.amber{color:var(--amber);}.jda-meta-v.emerald{color:var(--emerald);}
.jda-body{padding:1.4rem 1.6rem;}
.jda-section{margin-bottom:1.2rem;}.jda-section:last-child{margin-bottom:0;}
.jda-section-k{font-family:"Geist Mono",monospace;font-size:.52rem;letter-spacing:.18em;text-transform:uppercase;color:var(--t3);margin-bottom:.5rem;display:flex;align-items:center;gap:8px;}
.jda-section-k::before{content:"";display:block;width:12px;height:1px;background:var(--p-line);}
.skill-cloud{display:flex;flex-wrap:wrap;gap:6px;}
.skill-tag{display:inline-flex;align-items:center;padding:4px 12px;border-radius:100px;font-family:"Geist Mono",monospace;font-size:.6rem;white-space:nowrap;transition:all .2s;}
.skill-tag.must{border:1px solid var(--p-line);color:var(--p2);background:var(--p-dim);}
.skill-tag.must:hover{border-color:var(--p);background:rgba(99,102,241,.18);}
.skill-tag.nice{border:1px solid var(--am-line);color:var(--amber);background:var(--am-dim);}
.skill-tag.neutral{border:1px solid var(--b1);color:var(--t3);background:var(--b2);}
.quality-bar{height:6px;background:var(--b1);border-radius:99px;overflow:hidden;margin-top:6px;}
.quality-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--p),var(--cyan));transition:width .8s var(--out);}
.jda-cat-row{display:flex;align-items:center;gap:8px;padding:.4rem 0;border-bottom:1px solid var(--b2);}.jda-cat-row:last-child{border-bottom:none;}
.jda-cat-bar-track{flex:1;height:3px;background:var(--b1);border-radius:99px;overflow:hidden;}
.jda-cat-bar-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--cyan),var(--p2));}
.jda-cat-name{font-size:.78rem;color:var(--t2);min-width:170px;font-weight:300;}.jda-cat-name.top{color:var(--t0);font-weight:500;}
.jda-cat-score{font-family:"Geist Mono",monospace;font-size:.6rem;color:var(--p2);}
.tl-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:6rem 2rem 5rem;text-align:center;z-index:5;}
.tl-empty-h{font-family:'Instrument Serif',serif;font-size:1.9rem;color:var(--t0);margin-bottom:.6rem;letter-spacing:-.02em;}
.tl-empty-p{font-size:.86rem;color:var(--t3);line-height:1.75;font-weight:300;max-width:340px;}
.jd-analysis-panel{background:var(--bg1);border:1px solid var(--b1);border-radius:var(--r3);overflow:hidden;margin:1.2rem 0 0;box-shadow:var(--shb);position:relative;z-index:5;}
.jd-analysis-panel::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--p),var(--cyan),var(--emerald));}
.jda-hdr{display:flex;flex-wrap:wrap;align-items:center;gap:0;border-bottom:1px solid var(--b1);background:linear-gradient(135deg,var(--bg2),var(--bg1));}
.jda-meta{padding:1rem 1.6rem;border-right:1px solid var(--b2);flex:1;min-width:120px;}
.jda-meta:last-child{border-right:none;}
.jda-meta-k{font-family:'Geist Mono',monospace;font-size:.5rem;letter-spacing:.16em;text-transform:uppercase;color:var(--t3);margin-bottom:.3rem;}
.jda-meta-v{font-family:'Instrument Serif',serif;font-size:1.05rem;color:var(--t0);line-height:1.2;}
.jda-meta-v.cyan{color:var(--cyan);}.jda-meta-v.amber{color:var(--amber);}.jda-meta-v.emerald{color:var(--emerald);}
.quality-bar{height:2px;background:var(--bg3);border-radius:99px;margin-top:6px;overflow:hidden;}
.quality-fill{height:100%;background:linear-gradient(90deg,var(--emerald),var(--cyan));border-radius:99px;transition:width .6s var(--out);}
.jda-body{padding:1.2rem 1.6rem 1.4rem;display:flex;flex-direction:column;gap:1rem;}
.jda-section{display:flex;flex-direction:column;gap:.55rem;}
.jda-section-k{font-family:'Geist Mono',monospace;font-size:.52rem;letter-spacing:.16em;text-transform:uppercase;color:var(--t3);}
.skill-cloud{display:flex;flex-wrap:wrap;gap:6px;}
.skill-tag{display:inline-flex;align-items:center;padding:4px 12px;border-radius:100px;font-family:'Geist Mono',monospace;font-size:.62rem;white-space:nowrap;transition:background .2s;}
.skill-tag.must{background:var(--p-dim);border:1px solid var(--p-line);color:var(--p2);}
.skill-tag.must:hover{background:rgba(99,102,241,.18);}
.skill-tag.nice{background:var(--am-dim);border:1px solid var(--am-line);color:var(--amber);}
.skill-tag.nice:hover{background:rgba(245,158,11,.14);}
.jda-cat-row{display:flex;align-items:center;gap:10px;padding:.45rem 0;border-bottom:1px solid var(--b2);}
.jda-cat-row:last-child{border-bottom:none;}
.jda-cat-name{font-size:.8rem;color:var(--t2);min-width:180px;font-weight:300;}
.jda-cat-name.top{color:var(--p2);font-weight:500;}
.jda-cat-bar-track{flex:1;height:2px;background:var(--b1);border-radius:99px;overflow:hidden;}
.jda-cat-bar-fill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--p),var(--cyan));}
.jda-cat-score{font-family:'Geist Mono',monospace;font-size:.6rem;color:var(--p2);min-width:32px;text-align:right;}
[data-testid="stProgress"]>div>div{background:linear-gradient(90deg,var(--p),var(--cyan)) !important;border-radius:99px !important;}
[data-testid="stProgress"]>div{background:var(--bg2) !important;border-radius:99px !important;}
[data-testid="stSpinner"] p{color:var(--p2) !important;font-family:'Geist Mono',monospace !important;font-size:.72rem !important;}
[data-testid="stAlert"]{background:var(--bg1) !important;border:1px solid var(--b1) !important;border-radius:var(--r2) !important;color:var(--t1) !important;font-family:'Geist',sans-serif !important;}
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
.anim{animation:fadeUp .5s var(--out) both;}
.a1{animation-delay:.06s}.a2{animation-delay:.12s}.a3{animation-delay:.18s}
</style>
<div class="tl-mesh" aria-hidden="true"><div class="tl-mesh-g"></div><div class="tl-mesh-grid"></div><div class="tl-mesh-noise"></div></div>
<div id="tl-dot"></div><div id="tl-ring"></div>
<script>
(function(){var dot=document.getElementById('tl-dot'),ring=document.getElementById('tl-ring');if(!dot||!ring)return;var mx=-200,my=-200,rx=-200,ry=-200,visible=false,onHover=false;function hideCursorIn(doc){try{if(!doc||!doc.head)return;if(doc.querySelector('#__tl_hide'))return;var s=doc.createElement('style');s.id='__tl_hide';s.textContent='*,*::before,*::after{cursor:none!important}';doc.head.appendChild(s);}catch(e){}}function patchFrame(fr){try{var d=fr.contentDocument||fr.contentWindow.document;if(!d||d.__tlDone)return;d.__tlDone=true;hideCursorIn(d);d.addEventListener('mousemove',function(e){var r=fr.getBoundingClientRect();mx=r.left+e.clientX;my=r.top+e.clientY;show();moveDot();},{passive:true});d.addEventListener('mousedown',onDown);d.addEventListener('mouseup',onUp);Array.from(d.querySelectorAll('iframe')).forEach(patchFrame);}catch(e){}}function show(){if(visible)return;visible=true;dot.style.opacity='1';ring.style.opacity='1';}function hide(){visible=false;dot.style.opacity='0';ring.style.opacity='0';}function moveDot(){dot.style.left=mx+'px';dot.style.top=my+'px';}window.addEventListener('mousemove',function(e){mx=e.clientX;my=e.clientY;show();moveDot();},{passive:true});window.addEventListener('mouseleave',hide);window.addEventListener('mouseenter',show);function onDown(){dot.style.width='16px';dot.style.height='16px';dot.style.background='#c7d2fe';dot.style.boxShadow='0 0 0 3px rgba(129,140,248,.4),0 0 24px rgba(99,102,241,1)';ring.style.width='18px';ring.style.height='18px';ring.style.background='rgba(99,102,241,.15)';}function onUp(){var sz=onHover?'7px':'10px',rsz=onHover?'50px':'36px';dot.style.width=sz;dot.style.height=sz;dot.style.background='#818cf8';dot.style.boxShadow='0 0 0 2px rgba(129,140,248,.35),0 0 16px rgba(99,102,241,.9),0 0 32px rgba(99,102,241,.35)';ring.style.width=rsz;ring.style.height=rsz;ring.style.background='transparent';}window.addEventListener('mousedown',onDown);window.addEventListener('mouseup',onUp);(function raf(){rx+=(mx-rx)*0.1;ry+=(my-ry)*0.1;ring.style.left=rx+'px';ring.style.top=ry+'px';requestAnimationFrame(raf);})();function bindHover(doc){try{var sel='button,a,input,textarea,select,label,[role="button"],[data-testid="stFileUploader"]';doc.querySelectorAll(sel).forEach(function(el){if(el.__tlH)return;el.__tlH=true;el.addEventListener('mouseenter',function(){onHover=true;ring.style.width='50px';ring.style.height='50px';ring.style.borderColor='rgba(129,140,248,.8)';ring.style.background='rgba(99,102,241,.07)';dot.style.width='6px';dot.style.height='6px';dot.style.boxShadow='0 0 0 2px rgba(129,140,248,.5),0 0 20px rgba(99,102,241,1)';});el.addEventListener('mouseleave',function(){onHover=false;ring.style.width='36px';ring.style.height='36px';ring.style.borderColor='rgba(129,140,248,.55)';ring.style.background='transparent';dot.style.width='10px';dot.style.height='10px';dot.style.boxShadow='0 0 0 2px rgba(129,140,248,.35),0 0 16px rgba(99,102,241,.9),0 0 32px rgba(99,102,241,.35)';});});}catch(e){}}function poll(){hideCursorIn(document);Array.from(document.querySelectorAll('iframe')).forEach(patchFrame);bindHover(document);Array.from(document.querySelectorAll('iframe')).forEach(function(fr){try{bindHover(fr.contentDocument||fr.contentWindow.document);}catch(e){}});}poll();setInterval(poll,600);})();
</script>
""", unsafe_allow_html=True)


# ── DATA LOADERS ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_jobs():
    p = os.path.join(BASE, "jobs.csv")
    if not os.path.exists(p):
        return pd.DataFrame(columns=["Job_Title", "Category", "Job_Description"])
    df = pd.read_csv(p)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

@st.cache_data
def load_skill_profiles():
    p = os.path.join(BASE, "skill_profiles.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    return dict(zip(df["Job_Title"].str.strip(), df["Skills"].fillna("")))

@st.cache_data
def load_category_keywords():
    """
    Loads category_keywords.csv.
    Returns dict: {category: [keywords sorted longest-first]}.
    Longest keywords first so multi-word phrases match before their sub-terms.
    """
    p = os.path.join(BASE, "category_keywords.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    result = {}
    for _, r in df.iterrows():
        cat = r.get("Category", "")
        if pd.isna(cat) or str(cat).strip().lower() in ("", "nan", "none"):
            continue
        raw = r.get("Keywords", "")
        kws = [k.strip().lower() for k in str(raw if not pd.isna(raw) else "").split(",") if k.strip()]
        if kws:
            # sort longest first so multi-word phrases are checked before sub-words
            kws.sort(key=len, reverse=True)
            result[str(cat).strip()] = kws
    return result

@st.cache_data
def load_skill_signals():
    """
    Loads skill_signals.csv (domain, signal).
    Falls back to extracting all unique keywords from category_keywords.csv
    so the app always has signals even without a separate file.
    Returns list of signals sorted longest-first (multi-word phrases first).
    """
    # ── Try dedicated skill_signals.csv first ────────────────────────────────
    p = os.path.join(BASE, "skill_signals.csv")
    signals_set = set()
    if os.path.exists(p):
        try:
            df = pd.read_csv(p)
            if "signal" in df.columns:
                for s in df["signal"].dropna().str.strip().tolist():
                    if s:
                        signals_set.add(s.lower())
        except Exception:
            pass

    # ── Always supplement with every keyword from category_keywords.csv ───────
    # This guarantees signal coverage matches whatever keywords exist
    if CATK:
        for kws in CATK.values():
            for k in kws:
                if k and len(k) > 1:
                    signals_set.add(k.lower())

    # ── Sort: longest first so "react native" matches before "react" ──────────
    signals = sorted(signals_set, key=len, reverse=True)
    return signals

jobs_df = load_jobs()
SKP     = load_skill_profiles()
CATK    = load_category_keywords()
SIGNALS = load_skill_signals()

TJ = len(jobs_df)
TT = len(SKP)
TC = jobs_df["Category"].nunique() if "Category" in jobs_df.columns and len(jobs_df) > 0 else len(CATK)

# ── STOPWORDS for signal extraction (generic resume/JD words to filter out) ──
_SIGNAL_STOPWORDS = {
    "experience", "experiences", "using", "work", "working", "strong",
    "knowledge", "understanding", "ability", "skills", "skill", "good",
    "excellent", "proficient", "proficiency", "familiar", "familiarity",
    "develop", "developing", "developed", "developer", "build", "building",
    "built", "design", "designing", "designed", "manage", "managing",
    "managed", "create", "creating", "created", "implement", "implementing",
    "implemented", "maintain", "maintaining", "maintained", "support",
    "supporting", "supported", "team", "teams", "project", "projects",
    "environment", "application", "applications", "system", "systems",
    "solution", "solutions", "service", "services", "platform", "platforms",
    "tool", "tools", "technology", "technologies", "code", "coding",
    "software", "hardware", "data", "stack", "based", "driven", "oriented",
    "focused", "related", "various", "multiple", "different", "including",
    "responsible", "responsibilities", "requirement", "requirements",
    "following", "least", "years", "year", "month", "months",
    "bachelor", "master", "degree", "certification", "certified",
    "plus", "preferred", "required", "bonus", "nice",
}


# ── SIGNAL EXTRACTOR ──────────────────────────────────────────────────────────
def top_kw(txt, n=16):
    """
    Extracts the top-n skill/technology signals from text.
    - Searches for all known signals (from skill_signals + category_keywords)
    - Filters generic stopwords
    - Returns multi-word phrases before single tokens (longest-first matching)
    - Falls back to TF-IDF unigrams only if zero signals found
    """
    if not txt or not txt.strip():
        return []

    # Use soft-cleaned text so "node.js", "c++", "ci/cd" etc. survive
    tl = _clean_for_match(txt)
    matched, seen = [], set()

    for sig in SIGNALS:
        sl = sig.lower()
        # Skip generic words
        if sl in _SIGNAL_STOPWORDS:
            continue
        # Skip very short single chars
        if len(sl) < 2:
            continue
        if sl in tl and sl not in seen:
            matched.append(sig)
            seen.add(sl)
        if len(matched) >= n:
            break

    # Deduplicate: remove a signal if it's a sub-string of an already-matched one
    # e.g. if "react native" matched, skip bare "react" and "native"
    deduped = []
    for sig in matched:
        sl = sig.lower()
        if not any(sl != other.lower() and sl in other.lower() for other in matched):
            deduped.append(sig)

    return deduped[:n] if deduped else _fallback_kw(txt, n)


def _fallback_kw(txt, n=8):
    """Last-resort: TF-IDF unigrams filtered by stopwords."""
    try:
        v = CountVectorizer(stop_words="english", max_features=300,
                            token_pattern=r"[a-zA-Z0-9][a-zA-Z0-9\.\+\#\/\-]{1,}")
        v.fit([txt])
        w = v.get_feature_names_out()
        c = v.transform([txt]).toarray()[0]
        candidates = [(x, cnt) for x, cnt in sorted(zip(w, c), key=lambda x: -x[1])
                      if x.lower() not in _SIGNAL_STOPWORDS and len(x) > 2]
        return [x for x, _ in candidates[:n]]
    except:
        return []


# ── HELPERS ───────────────────────────────────────────────────────────────────
def clean(txt):
    t = str(txt).lower()
    t = re.sub(r'http\S+|\S+@\S+', ' ', t)
    t = re.sub(r'[^a-zA-Z ]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def pdf_text(f):
    try:
        with pdfplumber.open(f) as pdf:
            return ' '.join(pg.extract_text() or '' for pg in pdf.pages)
    except:
        return ""

# ── CLASSIFICATION ENGINE ────────────────────────────────────────────────────
def _kw_weight(kw: str) -> float:
    """
    Weight a keyword by specificity:
      - multi-word phrases  → 2.0  (e.g. "machine learning", "react native")
      - medium tokens 5-10 chars → 1.5  (e.g. "pytorch", "docker")
      - short tokens < 5 chars  → 1.0  (e.g. "sql", "aws", "r")
    """
    words = kw.split()
    if len(words) > 1:
        return 2.0
    if len(kw) >= 5:
        return 1.5
    return 1.0

def _clean_for_match(txt: str) -> str:
    """
    Softer clean: keeps hyphens, slashes, dots and digits so tokens like
    'c++', 'node.js', 'ci/cd', 'gpt-4', '3nf' still match.
    """
    t = str(txt).lower()
    t = re.sub(r'http\S+|\S+@\S+', ' ', t)          # strip urls/emails
    t = re.sub(r'[^a-z0-9 \.\-\/\+#]', ' ', t)      # keep useful punctuation
    return re.sub(r'\s+', ' ', t).strip()

def predict_cat(rc, raw_text=""):
    """
    3-pass hybrid classifier — returns (category, top5, method, debug).

    Pass 1 — Weighted keyword scoring on a 'soft-cleaned' resume string.
              Uses substring search (no \b) for short tokens and tech names
              that \b breaks (e.g. 'c++', 'node.js', '.net').
              Multi-word phrases score 2×, medium tokens 1.5×, short 1×.
              Normalised by sqrt(total_keywords) to avoid inflation in large
              categories.

    Pass 2 — TF-IDF cosine similarity against each category's keyword bag.
              Triggered when every weighted score == 0 (completely empty resume
              or zero vocabulary overlap).

    Pass 3 — Raw substring fallback with no normalisation as last resort.
    """
    if not CATK:
        return "Unknown", [], "no_data", {}

    # Use the softer-cleaned text for matching so punctuation survives
    rl = _clean_for_match(raw_text if raw_text else rc)
    # Also keep a plain alpha version for TF-IDF fallback
    rc_plain = rc  # already clean()'d

    # ── Pass 1: weighted keyword scoring ─────────────────────────────────────
    sc    = {}
    debug = {}
    for cat, kws in CATK.items():
        if not cat or str(cat).strip().lower() in ("nan", "none", ""):
            continue
        score = 0.0
        hits  = []
        # kws already sorted longest-first by load_category_keywords()
        seen_spans = []   # avoid double-counting overlapping matches
        for k in kws:
            kl = k.lower()
            # substring search — handles 'c++', 'node.js', 'ci/cd' etc.
            idx = rl.find(kl)
            if idx == -1:
                continue
            # simple overlap check: skip if this span is already covered
            end = idx + len(kl)
            overlaps = any(s <= idx < e or s < end <= e for s, e in seen_spans)
            if overlaps:
                continue
            seen_spans.append((idx, end))
            w = _kw_weight(k)
            score += w
            hits.append(f"{k} ({w:.0f})")
        # normalise to prevent large keyword lists from dominating
        n_kws  = max(len(kws), 1)
        sc[cat]    = round(score / (n_kws ** 0.5), 3)
        debug[cat] = hits

    ranked = sorted(sc.items(), key=lambda x: -x[1])

    if ranked and ranked[0][1] > 0:
        return ranked[0][0], ranked[:5], "keyword", debug

    # ── Pass 2: TF-IDF cosine on keyword bags ────────────────────────────────
    try:
        cat_names = list(CATK.keys())
        cat_docs  = [" ".join(CATK[c]) for c in cat_names]
        if rc_plain.strip():
            corpus = [rc_plain] + cat_docs
            vec    = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                     max_features=30000)
            mat    = vec.fit_transform(corpus)
            sims   = cosine_similarity(mat[0:1], mat[1:]).flatten()
            tfidf_ranked = sorted(zip(cat_names, (sims * 100).round(2)),
                                  key=lambda x: -x[1])
            if tfidf_ranked and tfidf_ranked[0][1] > 0:
                return tfidf_ranked[0][0], tfidf_ranked[:5], "tfidf", debug
    except Exception:
        pass

    # ── Pass 3: raw substring (no normalisation) ─────────────────────────────
    sc3 = {}
    for cat, kws in CATK.items():
        if not cat or str(cat).strip().lower() in ("nan", "none", ""):
            continue
        hits3 = [k for k in kws if k.lower() in rl]
        sc3[cat] = len(hits3)
    ranked3 = sorted(sc3.items(), key=lambda x: -x[1])
    if ranked3 and ranked3[0][1] > 0:
        return ranked3[0][0], ranked3[:5], "substring", debug

    return "No Strong Match", ranked[:5], "none", debug

def top_jobs(rc, df, n=10):
    if df.empty:
        return df
    df = df.copy()
    df["_r"] = df.apply(
        lambda r: f"{r.get('Job_Title','')} {r.get('Job_Title','')} "
                  f"{clean(r.get('Job_Description',''))} "
                  f"{SKP.get(r.get('Job_Title',''), '')}".lower(),
        axis=1
    )
    corp = [rc] + df["_r"].tolist()
    mat  = TfidfVectorizer(stop_words="english", max_features=15000,
                           ngram_range=(1, 2)).fit_transform(corp)
    sims = cosine_similarity(mat[0:1], mat[1:]).flatten()
    df["Match Score"] = (sims * 100).round(2)
    return df.sort_values("Match Score", ascending=False).head(n).reset_index(drop=True)

def rank_resumes(jd, res, n=50):
    if not res:
        return pd.DataFrame()
    corp = [jd] + [r["text"] for r in res]
    mat  = TfidfVectorizer(stop_words="english", max_features=20000,
                           ngram_range=(1, 2)).fit_transform(corp)
    sims = cosine_similarity(mat[0:1], mat[1:]).flatten()
    df   = pd.DataFrame({
        "File Name":   [r["name"] for r in res],
        "Match Score": (sims * 100).round(2),
        "_text":       [r["text"] for r in res]
    })
    df["Rank"] = df["Match Score"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("Match Score", ascending=False).head(n).reset_index(drop=True)

def medal(r):
    return {1: "GOLD", 2: "SILVER", 3: "BRONZE"}.get(r, "")


# ── JD ANALYSIS ENGINE ────────────────────────────────────────────────────────
def analyse_jd(jd_text: str) -> dict:
    """
    Full analysis of a job description. Returns:
      category, top5, method, required_skills, must_have, nice_to_have,
      seniority, word_count, char_count, quality_score
    """
    if not jd_text or not jd_text.strip():
        return {}

    raw = jd_text
    rc  = clean(jd_text)

    # Category classification
    category, top5, method, _ = predict_cat(rc, raw)

    # All signals in JD
    all_signals = top_kw(raw, n=30)

    # Split required vs preferred by scanning for section markers
    jdl = raw.lower()
    split_markers = [
        "nice to have", "nice-to-have", "preferred", "bonus", "plus",
        "desirable", "good to have", "advantageous", "ideally", "would be a plus",
    ]
    split_idx = len(jdl)
    for m in split_markers:
        idx = jdl.find(m)
        if 0 < idx < split_idx:
            split_idx = idx

    required_section  = jdl[:split_idx]
    preferred_section = jdl[split_idx:]

    must_have    = [s for s in all_signals if s.lower() in required_section]
    nice_to_have = [s for s in all_signals
                    if s.lower() in preferred_section and s not in must_have]

    # Seniority detection
    seniority = "Mid-Level"
    if any(x in jdl for x in ["10+ years", "10 years", "principal", "distinguished",
                                "fellow", "vp ", "vice president", "head of"]):
        seniority = "Principal / Executive"
    elif any(x in jdl for x in ["8+ years", "8 years", "senior", "sr.", "staff engineer",
                                  "lead engineer", "lead developer", "tech lead"]):
        seniority = "Senior"
    elif any(x in jdl for x in ["0-2 years", "0-1 year", "junior", "jr.", "entry level",
                                  "entry-level", "graduate", "intern", "fresh"]):
        seniority = "Junior / Entry"
    elif any(x in jdl for x in ["3+ years", "3 years", "4 years", "5 years",
                                  "mid-level", "mid level", "associate"]):
        seniority = "Mid-Level"

    # JD quality score (0-100)
    wc            = len(raw.split())
    length_score  = min(40, int(wc / 20))
    signal_score  = min(40, len(all_signals) * 2)
    section_score = sum(4 for kw in ["responsibilities", "requirements",
                                      "qualifications", "skills", "about"]
                        if kw in jdl)
    quality = min(100, length_score + signal_score + section_score)

    return {
        "category":        category,
        "top5":            top5,
        "method":          method,
        "required_skills": all_signals,
        "must_have":       must_have,
        "nice_to_have":    nice_to_have,
        "seniority":       seniority,
        "word_count":      wc,
        "char_count":      len(raw),
        "quality_score":   quality,
    }


D  = lambda lbl: (f'<div class="tl-divide"><div class="tl-divide-gem"></div>'
                  f'<div class="tl-divide-line"></div>'
                  f'<div class="tl-divide-label">{lbl}</div>'
                  f'<div class="tl-divide-line"></div>'
                  f'<div class="tl-divide-gem"></div></div>')
KK = lambda k, h: f'<div class="tl-kicker">{k}</div><div class="tl-h">{h}</div>'


def render_jd_analysis(jd_text: str):
    """Render a full JD analysis panel inline."""
    a = analyse_jd(jd_text)
    if not a:
        return

    q   = a["quality_score"]
    q_c = "emerald" if q >= 70 else ("amber" if q >= 40 else "")
    q_l = "Excellent" if q >= 70 else ("Good" if q >= 40 else "Needs Work")
    sen_c = "cyan" if a["seniority"] == "Senior" else ("amber" if a["seniority"] == "Principal / Executive" else "")

    # ── Header metrics ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="jd-analysis-panel anim">'
        '<div class="jda-hdr">'
        '<div class="jda-meta"><div class="jda-meta-k">Detected Role</div>'
        f'<div class="jda-meta-v">{a["category"]}</div></div>'
        '<div class="jda-meta"><div class="jda-meta-k">Seniority</div>'
        f'<div class="jda-meta-v {sen_c}">{a["seniority"]}</div></div>'
        '<div class="jda-meta"><div class="jda-meta-k">Word Count</div>'
        f'<div class="jda-meta-v cyan">{a["word_count"]:,}</div></div>'
        '<div class="jda-meta"><div class="jda-meta-k">JD Quality</div>'
        f'<div class="jda-meta-v {q_c}">{q}/100 '
        f'<span style="font-size:.72rem;font-family:Geist,sans-serif;color:var(--t3);">{q_l}</span></div>'
        f'<div class="quality-bar"><div class="quality-fill" style="width:{q}%"></div></div></div>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Skills body ────────────────────────────────────────────────────────────
    parts = []

    # Required skills section
    if a["must_have"]:
        tags = "".join('<span class="skill-tag must">' + s + '</span>' for s in a["must_have"])
        parts.append(
            '<div class="jda-section">'
            '<div class="jda-section-k">Required Skills &nbsp;&middot;&nbsp; '
            + str(len(a["must_have"])) + ' signals</div>'
            '<div class="skill-cloud">' + tags + '</div></div>'
        )

    # Nice-to-have section
    if a["nice_to_have"]:
        tags = "".join('<span class="skill-tag nice">' + s + '</span>' for s in a["nice_to_have"])
        parts.append(
            '<div class="jda-section">'
            '<div class="jda-section-k">Preferred / Nice to Have &nbsp;&middot;&nbsp; '
            + str(len(a["nice_to_have"])) + ' signals</div>'
            '<div class="skill-cloud">' + tags + '</div></div>'
        )

    # Flat JD — no split sections found
    if not a["must_have"] and not a["nice_to_have"] and a["required_skills"]:
        tags = "".join('<span class="skill-tag must">' + s + '</span>' for s in a["required_skills"])
        parts.append(
            '<div class="jda-section">'
            '<div class="jda-section-k">Detected Skills &nbsp;&middot;&nbsp; '
            + str(len(a["required_skills"])) + ' signals</div>'
            '<div class="skill-cloud">' + tags + '</div></div>'
        )

    # Category score breakdown
    if a["top5"]:
        ms   = max(s for _, s in a["top5"]) or 1
        rows = ""
        for i, (cat, sc) in enumerate(a["top5"]):
            pct = int(sc / ms * 100) if ms else 0
            tc  = "top" if i == 0 else ""
            pfx = "&#9658; " if i == 0 else ""
            scv = str(round(sc, 1)) if isinstance(sc, float) else str(sc)
            rows += (
                '<div class="jda-cat-row">'
                '<span class="jda-cat-name ' + tc + '">' + pfx + cat + '</span>'
                '<div class="jda-cat-bar-track">'
                '<div class="jda-cat-bar-fill" style="width:' + str(pct) + '%"></div></div>'
                '<span class="jda-cat-score">' + scv + '</span>'
                '</div>'
            )
        parts.append(
            '<div class="jda-section">'
            '<div class="jda-section-k">Role Category Match</div>' + rows + '</div>'
        )

    st.markdown(
        '<div class="jda-body">' + "".join(parts) + '</div></div>',
        unsafe_allow_html=True
    )

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "pg" not in st.session_state:
    st.session_state.pg = "cand"

# ── NAVBAR ────────────────────────────────────────────────────────────────────
pg    = st.session_state.pg
c_on  = "on" if pg == "cand" else ""
r_on  = "on" if pg == "rec"  else ""
c_dot = '<span class="tl-tab-dot"></span>' if pg == "cand" else ""
r_dot = '<span class="tl-tab-dot"></span>' if pg == "rec"  else ""
st.markdown(f"""
<div class="tl-nav">
  <div class="tl-brand">
    <div class="tl-logomark"><svg viewBox="0 0 36 36" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="nlg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#6366f1"/><stop offset="100%" stop-color="#22d3ee"/></linearGradient></defs><circle cx="18" cy="18" r="16" fill="rgba(99,102,241,0.08)" stroke="rgba(99,102,241,0.2)" stroke-width="1"/><circle cx="18" cy="18" r="10" fill="none" stroke="url(#nlg)" stroke-width="1.5" stroke-dasharray="4 3"><animateTransform attributeName="transform" type="rotate" from="0 18 18" to="360 18 18" dur="12s" repeatCount="indefinite"/></circle><circle cx="18" cy="18" r="4" fill="url(#nlg)"/><circle cx="18" cy="8" r="1.5" fill="#6366f1" opacity="0.7"/><circle cx="26" cy="24" r="1.5" fill="#22d3ee" opacity="0.7"/><circle cx="10" cy="24" r="1.5" fill="#818cf8" opacity="0.7"/></svg></div>
    <div class="tl-brand-text"><div class="tl-brand-name">Talent<em>Lens</em></div><div class="tl-brand-tag">AI Career Intelligence Platform</div></div>
  </div>
  <div class="tl-nav-tabs"><div class="tl-nav-tab {c_on}">{c_dot}Candidate Portal</div><div class="tl-nav-tab {r_on}">{r_dot}Recruiter Portal</div></div>
  <div class="tl-nav-stats">
    <div class="tl-stat"><div class="tl-stat-n">{TJ:,}</div><div class="tl-stat-k">Positions</div></div>
    <div class="tl-stat"><div class="tl-stat-n">{TT}</div><div class="tl-stat-k">Skill Tracks</div></div>
    <div class="tl-stat"><div class="tl-stat-n">{TC}</div><div class="tl-stat-k">Verticals</div></div>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown('<div style="margin:.75rem 0 0;">', unsafe_allow_html=True)
sc1, sc2 = st.columns(2)
with sc1:
    if st.button("Candidate Portal  —  Find My Role", use_container_width=True,
                 type="primary" if st.session_state.pg == "cand" else "secondary", key="sw_c"):
        st.session_state.pg = "cand"; st.rerun()
with sc2:
    if st.button("Recruiter Portal  —  Screen Resumes", use_container_width=True,
                 type="primary" if st.session_state.pg == "rec" else "secondary", key="sw_r"):
        st.session_state.pg = "rec"; st.rerun()
st.markdown('</div>', unsafe_allow_html=True)


# ── CANDIDATE PAGE ────────────────────────────────────────────────────────────
if st.session_state.pg == "cand":
    ch, cu = st.columns([1.4, 1], gap="small")
    with ch:
        st.markdown(f"""<div class="hero-main"><div class="hero-badge"><div class="hero-badge-dot"></div><span class="hero-badge-txt">Live &middot; TF-IDF Semantic Engine</span></div><h1 class="hero-h1">Find the role<br>built for <em>exactly</em><br>who you are</h1><p class="hero-p">Drop your resume. Our engine reads every skill signal, weights every keyword, and ranks {TJ:,} positions by true semantic fit in under three seconds. Private. Instant. Precise.</p><div class="hero-chips"><span class="hero-chip"><span class="hero-chip-dot"></span><b>{TJ:,}</b>&nbsp;positions</span><span class="hero-chip"><span class="hero-chip-dot"></span><b>{TT}</b>&nbsp;skill profiles</span><span class="hero-chip"><span class="hero-chip-dot"></span><b>{TC}</b>&nbsp;verticals</span><span class="hero-chip"><span class="hero-chip-dot"></span>Zero data stored</span><span class="hero-chip"><span class="hero-chip-dot"></span>No sign-up</span></div></div>""", unsafe_allow_html=True)
    with cu:
        st.markdown("""<div class="upload-panel"><div class="up-label">Step 01 of 01</div><div class="up-title">Upload your <span>resume</span></div><div class="up-meta">PDF format &middot; Processed locally &middot; Never stored</div><div class="up-steps"><div class="up-step"><div class="up-step-n">1</div><div class="up-step-t">Select or drag your PDF resume</div></div><div class="up-step"><div class="up-step-n">2</div><div class="up-step-t">Engine extracts and vectorises all text</div></div><div class="up-step"><div class="up-step-n">3</div><div class="up-step-t">Top roles surface instantly</div></div></div></div>""", unsafe_allow_html=True)
        uf = st.file_uploader("", type=["pdf"], label_visibility="collapsed", key="c_resume")

    if uf:
        with st.spinner("Extracting resume text..."):
            raw = pdf_text(uf)
            rc  = clean(raw)

        # ── Guard: empty extraction ───────────────────────────────────────────
        if not rc.strip() or len(rc.strip()) < 50:
            st.markdown(
                '<div class="warn-strip"><div class="warn-strip-line"></div>'
                '<div class="warn-strip-txt"><b>Warning:</b> Very little text was extracted from this PDF. '
                'The file may be scanned/image-based or password-protected. '
                'Try exporting your resume as a text-based PDF for best results.</div></div>',
                unsafe_allow_html=True
            )

        st.markdown(D("Document Analysis"), unsafe_allow_html=True)
        st.markdown(KK("01 · Parsed Document", "Resume <em>Preview</em>"), unsafe_allow_html=True)
        prev = (raw[:900] + "...") if len(raw) > 900 else raw
        prev = prev.replace("<", "&lt;").replace(">", "&gt;")
        st.markdown(
            f'<div class="tl-term anim"><div class="tl-chrome">'
            f'<div class="tl-dots"><div class="tl-dot d-r"></div><div class="tl-dot d-y"></div><div class="tl-dot d-g"></div></div>'
            f'<div class="tl-chrome-id">RESUME_PARSER &middot; {len(raw):,} chars</div>'
            f'<div class="tl-chrome-ok">PARSED</div></div>'
            f'<div class="tl-term-body"><span class="tl-prompt">&gt; </span>{prev}<span class="tl-blink"></span></div></div>',
            unsafe_allow_html=True
        )

        st.markdown(D("Profile Classification"), unsafe_allow_html=True)

        # ── Run improved classifier ───────────────────────────────────────────
        pc, top5, method, dbg = predict_cat(rc, raw)
        ms = max((s for _, s in top5), default=1) or 1

        # Surface a helpful hint when keyword matching found nothing
        if method in ("tfidf", "none", "substring"):
            hint_msg = (
                "<b>Note:</b> No exact keyword matches were found in your resume against "
                "<code>category_keywords.csv</code>. The category shown is based on semantic "
                "similarity instead. To improve accuracy, ensure your resume contains explicit "
                "skill/tool names that match your target category keywords."
            )
            if method == "none":
                hint_msg = (
                    "<b>No Strong Match:</b> Neither keyword matching nor semantic similarity "
                    "could confidently classify this resume. This usually means the resume text "
                    "extracted is too short, or its vocabulary doesn't overlap with any category "
                    "in <code>category_keywords.csv</code>. Check that keywords in that file "
                    "match real resume language."
                )
            st.markdown(
                f'<div class="warn-strip"><div class="warn-strip-line"></div>'
                f'<div class="warn-strip-txt">{hint_msg}</div></div>',
                unsafe_allow_html=True
            )

        st.markdown(KK("02 · Classification", "Career <em>Profile</em>"), unsafe_allow_html=True)

        # Method badge label
        method_label = {"keyword": "KEYWORD MATCH", "tfidf": "SEMANTIC MATCH",
                        "substring": "PARTIAL MATCH", "none": "NO MATCH"}.get(method, method.upper())

        cm, cc = st.columns([1, 1.05], gap="small")
        with cm:
            rows = ""
            for i, (c, s) in enumerate(top5):
                pct = int(s / ms * 100) if ms > 0 else 0
                ist = c == pc
                nl  = (f'<span class="kw-n top">&#9658; {c}</span>' if ist
                       else f'<span class="kw-n">{c}</span>')
                rows += (f'<div class="kw-row"><span class="kw-i">0{i+1}</span>'
                         f'{nl}<div class="kw-track"><div class="kw-fill" style="width:{pct}%"></div></div>'
                         f'<span class="kw-v">{round(s, 1) if isinstance(s, float) else s}</span></div>')
            st.markdown(
                f'<div class="cls-panel anim a1">'
                f'<div class="cls-hdr">'
                f'<div class="cls-kicker">Predicted Category &middot; {method_label}</div>'
                f'<div class="cls-role">{pc}</div>'
                f'</div><div class="cls-body">{rows}</div></div>',
                unsafe_allow_html=True
            )

        with cc:
            cdf = pd.DataFrame(top5, columns=["Category", "Score"])
            st.markdown(
                '<div class="chart-panel anim a2">'
                '<div class="chart-k">Keyword Hit Rate</div>'
                '<div class="chart-t">Score Distribution</div>',
                unsafe_allow_html=True
            )
            st.bar_chart(cdf.set_index("Category"), use_container_width=True, height=250)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(D("Role Matching"), unsafe_allow_html=True)
        st.markdown(KK("03 · Top Matches", "Top <em>Recommendations</em>"), unsafe_allow_html=True)

        with st.spinner("Ranking all positions..."):
            tj = top_jobs(rc, jobs_df, 10)

        if not tj.empty:
            dp = tj[["Job_Title", "Category", "Match Score"]].copy()
            dp.index = range(1, len(dp) + 1)
            st.dataframe(dp, use_container_width=True)

            st.markdown(D("Role Details"), unsafe_allow_html=True)
            st.markdown(KK("04 · Breakdown", "Job <em>Breakdown</em>"), unsafe_allow_html=True)
            for i, row in tj.iterrows():
                sc2  = row["Match Score"]
                bc, bl = (("jb-hi",  f"Strong {sc2}%")   if sc2 >= 6 else
                          (("jb-mid", f"Moderate {sc2}%") if sc2 >= 3 else
                           ("jb-lo",  f"Weak {sc2}%")))
                desc  = str(row.get("Job_Description", "")).strip() or "No description available."
                short = ((desc[:280] + "...") if len(desc) > 280 else desc)
                short = short.replace("<", "&lt;").replace(">", "&gt;")
                st.markdown(
                    f'<div class="jcard anim" style="animation-delay:{i * 0.04}s">'
                    f'<div class="jcard-rank"><div class="jcard-rank-n">{i + 1:02d}</div></div>'
                    f'<div class="jcard-body"><div class="jcard-t">{row["Job_Title"]}</div>'
                    f'<div class="jcard-c">{row.get("Category", "—")}</div></div>'
                    f'<div class="jcard-score"><span class="jbadge {bc}">{bl}</span></div>'
                    f'<div class="jcard-desc">{short}</div></div>',
                    unsafe_allow_html=True
                )
        else:
            st.info("No job listings found. Add jobs.csv to your project directory.")
    else:
        st.markdown('<div class="tl-empty anim"><div style="margin-bottom:1.8rem;"><svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="eg1" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#6366f1" stop-opacity=".4"/><stop offset="100%" stop-color="#22d3ee" stop-opacity=".2"/></linearGradient></defs><circle cx="60" cy="60" r="55" fill="none" stroke="rgba(99,102,241,0.12)" stroke-width="1"><animateTransform attributeName="transform" type="rotate" from="0 60 60" to="360 60 60" dur="40s" repeatCount="indefinite"/></circle><circle cx="60" cy="60" r="40" fill="none" stroke="rgba(99,102,241,0.2)" stroke-width="1" stroke-dasharray="8 5"><animateTransform attributeName="transform" type="rotate" from="360 60 60" to="0 60 60" dur="25s" repeatCount="indefinite"/></circle><circle cx="60" cy="60" r="24" fill="none" stroke="rgba(34,211,238,0.25)" stroke-width="1.5"/><circle cx="60" cy="60" r="8" fill="url(#eg1)"/><circle cx="60" cy="60" r="8" fill="none" stroke="rgba(99,102,241,0.4)" stroke-width="1"><animate attributeName="r" values="8;18;8" dur="3s" repeatCount="indefinite"/><animate attributeName="opacity" values="1;0;1" dur="3s" repeatCount="indefinite"/></circle><circle cx="60" cy="5" r="2.5" fill="#6366f1" opacity="0.7"><animateTransform attributeName="transform" type="rotate" from="0 60 60" to="360 60 60" dur="10s" repeatCount="indefinite"/></circle></svg></div><div class="tl-empty-h">Your next role awaits</div><p class="tl-empty-p">Upload a PDF resume above to activate the matching engine. Semantic analysis, zero sign-up.</p></div>', unsafe_allow_html=True)


# ── RECRUITER PAGE ────────────────────────────────────────────────────────────
else:
    st.markdown(f'<div class="rec-hero"><div class="hero-badge" style="--p-dim:rgba(34,211,238,.08);--p-line:rgba(34,211,238,.18);"><div class="hero-badge-dot" style="background:var(--cyan);box-shadow:0 0 8px rgba(34,211,238,.5);"></div><span class="hero-badge-txt" style="color:var(--cyan);">Recruiter Mode &middot; Bulk Screening</span></div><h1 class="rec-h1">Screen <em>thousands</em><br>in seconds</h1><p class="hero-p" style="max-width:620px;">Paste a job description, upload up to 10,000 resumes via PDFs or a ZIP archive, and TalentLens ranks every candidate by TF-IDF cosine similarity.</p><div class="hero-chips"><span class="hero-chip"><span class="hero-chip-dot"></span>Up to <b>10,000</b> resumes</span><span class="hero-chip"><span class="hero-chip-dot"></span>ZIP archive support</span><span class="hero-chip"><span class="hero-chip-dot"></span>CSV export</span><span class="hero-chip"><span class="hero-chip-dot"></span>Keyword gap analysis</span></div></div>', unsafe_allow_html=True)
    st.markdown(D("Job Description"), unsafe_allow_html=True)
    st.markdown(KK("Step 01", "Describe the <em>Role</em>"), unsafe_allow_html=True)
    st.markdown('<div class="info-strip"><div class="info-strip-line"></div><div class="info-strip-txt">Paste the full job description including responsibilities, skills, and qualifications. <b>More detail = more accurate ranking.</b></div></div>', unsafe_allow_html=True)
    jd = st.text_area("JOB DESCRIPTION", height=200, placeholder="e.g. Senior Data Scientist with 5+ years Python, ML, SQL...", key="jd")

    # ── Live JD Analysis ── fires as soon as the JD has enough text ──────────
    if jd and len(jd.strip()) >= 60:
        st.markdown(D("JD Intelligence"), unsafe_allow_html=True)
        st.markdown(KK("Step 01b", "Job Description <em>Analysis</em>"), unsafe_allow_html=True)
        render_jd_analysis(jd)

    rj1, rj2 = st.columns(2)
    with rj1: rt = st.text_input("ROLE TITLE (OPTIONAL)", placeholder="Senior Data Scientist", key="rt")
    with rj2: tn = st.number_input("TOP N CANDIDATES", min_value=5, max_value=500, value=20, step=5, key="tn")
    st.markdown(D("Upload Resumes"), unsafe_allow_html=True)
    st.markdown(KK("Step 02", "Upload <em>Candidates</em>"), unsafe_allow_html=True)
    st.markdown('<div class="info-strip"><div class="info-strip-line"></div><div class="info-strip-txt"><b>Two modes:</b> drag multiple PDFs (up to 1,000), or upload a single <b>ZIP archive</b> containing up to 10,000 PDFs.</div></div>', unsafe_allow_html=True)
    um = st.selectbox("UPLOAD MODE", ["Multiple PDF Files (up to 1,000)", "ZIP Archive (up to 10,000 PDFs)"], key="um")
    if "Multiple" in um:
        rfs = st.file_uploader("", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed", key="rfs")
    else:
        zf = st.file_uploader("", type=["zip"], label_visibility="collapsed", key="zf"); rfs = None
    st.markdown(D("Filter Options"), unsafe_allow_html=True)
    st.markdown(KK("Step 03", "Configure <em>Filters</em>"), unsafe_allow_html=True)
    rf1, rf2 = st.columns(2)
    with rf1: fms = st.slider("MINIMUM MATCH SCORE (%)", 0, 50, 0, 1, key="fms")
    with rf2: so  = st.selectbox("SORT ORDER", ["Best Match First", "A–Z", "Z–A"], key="so")
    st.markdown('<div class="run-btn" style="margin:1.5rem 0 0;">', unsafe_allow_html=True)
    run = st.button("Analyse and Rank All Resume", use_container_width=True, key="run")
    st.markdown('</div>', unsafe_allow_html=True)

    if run:
        if not jd.strip():
            st.error("Please paste a job description first.")
        else:
            parsed = []
            if "Multiple" in um:
                fl = rfs or []
                if not fl: st.error("Upload at least one PDF."); st.stop()
                prog = st.progress(0, text="Parsing resumes...")
                for i, f in enumerate(fl):
                    raw = pdf_text(f)
                    nm  = f.name.replace(".pdf", "").replace("_", " ").replace("-", " ")
                    parsed.append({"name": nm, "text": clean(raw), "raw": raw})
                    prog.progress((i + 1) / len(fl), text=f"Parsing {i + 1}/{len(fl)}: {f.name}")
                prog.empty()
            else:
                if 'zf' not in dir() or zf is None: st.error("Upload a ZIP file."); st.stop()
                with st.spinner("Extracting ZIP..."):
                    zb  = io.BytesIO(zf.read())
                    with zipfile.ZipFile(zb, "r") as z:
                        pns = [n for n in z.namelist() if n.lower().endswith(".pdf")]
                if not pns: st.error("No PDFs in ZIP."); st.stop()
                prog = st.progress(0, text=f"Processing {len(pns):,} resumes...")
                for i, pn in enumerate(pns):
                    try:
                        with zipfile.ZipFile(zb, "r") as z: pd_data = z.read(pn)
                        raw = pdf_text(io.BytesIO(pd_data))
                        nm  = os.path.basename(pn).replace(".pdf", "").replace("_", " ").replace("-", " ")
                        parsed.append({"name": nm, "text": clean(raw), "raw": raw})
                    except: pass
                    if i % 50 == 0:
                        prog.progress((i + 1) / len(pns), text=f"Parsed {i + 1:,}/{len(pns):,}...")
                prog.empty()

            tu = len(parsed)
            if tu == 0: st.error("No readable resumes found."); st.stop()
            jdc = clean(jd)
            with st.spinner(f"Vectorising {tu:,} resumes..."):
                rdf = rank_resumes(jdc, parsed, int(tn))
            rdf = rdf[rdf["Match Score"] >= fms]
            if so == "A–Z":   rdf = rdf.sort_values("File Name")
            elif so == "Z–A": rdf = rdf.sort_values("File Name", ascending=False)
            rdf = rdf.reset_index(drop=True)
            jkw = top_kw(jd, 12)
            st.markdown(D("Results"), unsafe_allow_html=True)
            avg   = rdf["Match Score"].mean() if not rdf.empty else 0
            top_s = rdf["Match Score"].max()  if not rdf.empty else 0
            abv   = len(rdf[rdf["Match Score"] >= 5])
            st.markdown(f'<div class="stat-grid"><div class="stat-card anim"><div class="stat-accent sa1"></div><div class="stat-tag">TOTAL</div><div class="stat-n">{tu:,}</div><div class="stat-l">Resumes Processed</div></div><div class="stat-card anim a1"><div class="stat-accent sa2"></div><div class="stat-tag">BEST</div><div class="stat-n">{top_s:.1f}%</div><div class="stat-l">Top Match Score</div></div><div class="stat-card anim a2"><div class="stat-accent sa3"></div><div class="stat-tag">AVG</div><div class="stat-n">{avg:.1f}%</div><div class="stat-l">Average Score</div></div><div class="stat-card anim a3"><div class="stat-accent sa4"></div><div class="stat-tag">STRONG</div><div class="stat-n">{abv}</div><div class="stat-l">Strong Matches</div></div></div>', unsafe_allow_html=True)
            if jkw:
                tags = "".join(f'<span class="kw-ctag">{k}</span>' for k in jkw)
                st.markdown(f'<div class="kw-panel anim a1"><div class="kw-panel-k">Extracted from Job Description</div><div class="kw-panel-t">Matching Signals</div><div class="kw-cloud">{tags}</div></div>', unsafe_allow_html=True)
            st.markdown(KK("Ranked Results", f"Top <em>{len(rdf)}</em> Candidates"), unsafe_allow_html=True)
            ddf = rdf[["Rank", "File Name", "Match Score"]].copy()
            ddf["Match Score"] = ddf["Match Score"].apply(lambda x: f"{x:.2f}%")
            st.dataframe(ddf, use_container_width=True, hide_index=True)
            csv = rdf[["Rank", "File Name", "Match Score"]].to_csv(index=False).encode()
            st.download_button("Export Rankings as CSV", data=csv,
                               file_name=f"talentlens_{(rt or 'rankings').replace(' ', '_')}.csv",
                               mime="text/csv", use_container_width=True, key="dl")
            st.markdown(D("Candidate Profiles"), unsafe_allow_html=True)
            st.markdown(KK("Detailed View", "Candidate <em>Profiles</em>"), unsafe_allow_html=True)
            mset = set(k.lower() for k in jkw)
            for _, row in rdf.head(50).iterrows():
                rank  = int(row["Rank"]); sc2 = float(row["Match Score"]); nm = row["File Name"]
                med   = medal(rank)
                raw_t = next((r["raw"] for r in parsed if r["name"] == nm), "")
                rc2   = "hi" if sc2 >= 10 else ("mid" if sc2 >= 4 else "")
                bc, bl = (("jb-hi",  f"Strong {sc2:.1f}%")   if sc2 >= 10 else
                          (("jb-mid", f"Moderate {sc2:.1f}%") if sc2 >= 4  else
                           ("jb-lo",  f"Weak {sc2:.1f}%")))
                ckw = top_kw(raw_t, 10)
                kwh = "".join(
                    f'<span class="rtag{"  match" if k.lower() in mset else ""}">{k}</span>'
                    for k in ckw[:8]
                )
                snip = re.sub(r'\s+', ' ', raw_t[:180].strip()).replace("<", "&lt;").replace(">", "&gt;") if raw_t else "No text extracted."
                st.markdown(
                    f'<div class="rcard anim"><div class="rcard-rank">'
                    f'<div class="rcard-rank-n">{rank:02d}</div>'
                    f'<div class="rcard-rank-m">{med}</div></div>'
                    f'<div class="rcard-body"><div class="rcard-name">{nm}</div>'
                    f'<div class="rcard-meta">{len(raw_t):,} chars &middot; '
                    f'<span class="jbadge {bc}" style="padding:2px 8px;font-size:.56rem;">{bl}</span></div>'
                    f'<div class="tag-row">{kwh}</div>'
                    f'<div style="margin-top:9px;font-size:.75rem;color:var(--t3);line-height:1.75;font-weight:300;">{snip}...</div>'
                    f'</div><div class="rcard-score">'
                    f'<div class="score-ring {rc2}"><div class="score-ring-v">{sc2:.0f}%</div>'
                    f'<div class="score-ring-l">MATCH</div></div></div></div>',
                    unsafe_allow_html=True
                )
            if len(rdf) > 50:
                st.markdown(
                    f'<div class="info-strip"><div class="info-strip-line"></div>'
                    f'<div class="info-strip-txt">Showing top 50 profiles. All {len(rdf)} results are in the exported CSV.</div></div>',
                    unsafe_allow_html=True
                )
    else:
        st.markdown('<div class="tl-empty anim"><div style="margin-bottom:1.8rem;"><svg width="120" height="120" viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="eg2" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#22d3ee" stop-opacity=".4"/><stop offset="100%" stop-color="#6366f1" stop-opacity=".2"/></linearGradient></defs><circle cx="60" cy="60" r="55" fill="none" stroke="rgba(34,211,238,0.12)" stroke-width="1"><animateTransform attributeName="transform" type="rotate" from="0 60 60" to="360 60 60" dur="40s" repeatCount="indefinite"/></circle><rect x="24" y="24" width="72" height="72" rx="10" fill="none" stroke="rgba(34,211,238,0.2)" stroke-width="1" stroke-dasharray="8 4"><animateTransform attributeName="transform" type="rotate" from="0 60 60" to="360 60 60" dur="30s" repeatCount="indefinite"/></rect><circle cx="60" cy="60" r="16" fill="none" stroke="rgba(99,102,241,0.3)" stroke-width="1.5"/><circle cx="60" cy="60" r="6" fill="url(#eg2)"/><circle cx="60" cy="60" r="6" fill="none" stroke="rgba(34,211,238,0.4)" stroke-width="1"><animate attributeName="r" values="6;18;6" dur="3s" repeatCount="indefinite"/><animate attributeName="opacity" values="1;0;1" dur="3s" repeatCount="indefinite"/></circle><circle cx="60" cy="5" r="2" fill="#22d3ee" opacity="0.6"><animateTransform attributeName="transform" type="rotate" from="0 60 60" to="360 60 60" dur="12s" repeatCount="indefinite"/></circle></svg></div><div class="tl-empty-h">Ready to screen candidates</div><p class="tl-empty-p">Paste a job description, upload your resume batch, then click Analyse and Rank.</p></div>', unsafe_allow_html=True)