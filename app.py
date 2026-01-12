import streamlit as st

st.set_page_config(
    page_title="Wine Quality • Analysis & Prediction",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded",
)

AVATAR_PATH = "asset/avatar.png"
LINKEDIN_URL = "https://www.linkedin.com/in/dwi-amalia-/"
GITHUB_URL = "https://github.com/dwiamalia123"
EMAIL = "dwiamalia228@gmail.com"

WINE_THEME_CSS = """
<style>
  /* ===== Base layout ===== */
  .block-container { padding-top: 1.2rem; padding-bottom: 2.2rem; max-width: 1250px; }

  html, body, [data-testid="stAppViewContainer"]{
    background:
      radial-gradient(900px 420px at 12% 0%, rgba(122,16,32,.10), transparent 60%),
      radial-gradient(760px 420px at 92% 8%, rgba(99,102,241,.08), transparent 55%),
      #FBF7F4 !important;
  }
  [data-testid="stHeader"] { background: transparent !important; }
  footer { visibility: hidden; }

  /* Keep Settings & Always rerun available => DO NOT hide MainMenu */
  /* #MainMenu { visibility: hidden; } */

  /* Sidebar cannot collapse */
  [data-testid="collapsedControl"] { display: none !important; }
  button[kind="headerNoPadding"] { display: none !important; }

  /* ===== Sidebar ===== */
  [data-testid="stSidebar"]{
    background:
      radial-gradient(420px 240px at 25% 0%, rgba(255,255,255,.14), transparent 60%),
      linear-gradient(180deg, rgba(122,16,32,0.98) 0%, rgba(70,6,15,0.98) 100%);
    border-right: 1px solid rgba(255,255,255,.14);
  }

  [data-testid="stSidebar"] .stMarkdown,
  [data-testid="stSidebar"] .stText,
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] span,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] a {
    color: rgba(255,255,255,.94) !important;
  }

  /* Hide sidebar nav */
  [data-testid="stSidebarNav"] { display: none !important; }

  .divider{ height:1px; background: rgba(255,255,255,.18); margin: 12px 0; }
  .side-name{ font-weight: 950; font-size: 18px; margin: 0 0 2px 0; letter-spacing:.2px; }
  .side-role{ opacity:.86; margin: 0; font-size: 12px; }

  .side-wrap{
    border: 1px solid rgba(255,255,255,.14);
    background: rgba(255,255,255,.08);
    border-radius: 18px;
    padding: 14px;
    backdrop-filter: blur(10px);
    box-shadow: 0 16px 42px rgba(0,0,0,.22);
  }

  .side-card{
    border: 1px solid rgba(255,255,255,.14);
    background: rgba(255,255,255,.10);
    border-radius: 16px;
    padding: 14px;
  }
  .side-card .title{
    font-weight: 950;
    margin: 0 0 10px 0;
    font-size: 14px;
    letter-spacing:.2px;
  }
  .side-card p{
    margin: 0 0 10px 0;
    line-height: 1.65;
    font-size: 13px;
  }
  .side-card .muted{
    opacity:.84;
    font-size: 12px;
    margin: 0;
  }

  /* Sidebar link buttons -> no white blanks */
  [data-testid="stSidebar"] .stLinkButton>button{
    width: 100% !important;
    background: rgba(255,255,255,.12) !important;
    border: 1px solid rgba(255,255,255,.22) !important;
    border-radius: 14px !important;
    color: rgba(255,255,255,.96) !important;
    font-weight: 900 !important;
    padding: 10px 12px !important;
    box-shadow: 0 10px 20px rgba(0,0,0,.18) !important;
  }
  [data-testid="stSidebar"] .stLinkButton>button:hover{
    background: rgba(255,255,255,.18) !important;
    border: 1px solid rgba(255,255,255,.34) !important;
    transform: translateY(-1px);
  }

  /* ===== Hero (more premium but still readable) ===== */
  .hero{
    border-radius: 22px;
    padding: 22px 22px;
    background:
      radial-gradient(900px 420px at 25% 0%, rgba(255,255,255,.18), transparent 58%),
      linear-gradient(90deg, rgba(122,16,32,0.98) 0%, rgba(70,6,15,0.98) 100%);
    color: white;
    border: 1px solid rgba(255,255,255,.16);
    box-shadow: 0 18px 48px rgba(70,6,15,.25);
    position: relative;
    overflow: hidden;
  }
  .hero:after{
    content:"";
    position:absolute;
    inset:-60px -80px auto auto;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, rgba(99,102,241,.20), transparent 60%);
    transform: rotate(18deg);
    filter: blur(2px);
  }

  .hero-title{
    font-weight: 950;
    font-size: 44px;
    line-height: 1.05;
    margin:0;
    letter-spacing: .3px;
  }
  .hero-sub{ opacity:.92; margin-top: 8px; font-size: 14px; }
  .hero-meta{ opacity:.78; font-size:12px; margin-top: 10px; }

  /* Chips */
  .chip{
    display:inline-flex;
    align-items:center;
    gap: 6px;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,.18);
    background: rgba(255,255,255,.10);
    font-size: 12px;
    font-weight: 900;
    opacity: .95;
    margin-right: 8px;
    margin-top: 10px;
  }
  .chip.green{
    background: rgba(34,197,94,.22);
    border: 1px solid rgba(34,197,94,.40);
  }
  .chip.purple{
    background: rgba(99,102,241,.22);
    border: 1px solid rgba(99,102,241,.40);
  }

  /* ===== Tabs -> modern pill (not flat) ===== */
  div[data-baseweb="tab-list"]{
    border-bottom: 0 !important;
    box-shadow: none !important;
    gap: 8px;
    padding: 6px 2px;
  }
  div[data-baseweb="tab-highlight"]{ display:none !important; }

  button[data-baseweb="tab"]{
    font-weight: 900 !important;
    padding: 10px 14px !important;
    border-radius: 999px !important;
    background: rgba(15,23,42,.05) !important;
    border: 1px solid rgba(15,23,42,.10) !important;
    color: rgba(15,23,42,.84) !important;
  }
  button[data-baseweb="tab"]:hover{
    background: rgba(15,23,42,.08) !important;
    transform: translateY(-1px);
  }
  button[data-baseweb="tab"][aria-selected="true"]{
    background: rgba(122,16,32,.10) !important;
    border: 1px solid rgba(122,16,32,.22) !important;
    color: #7A1020 !important;
  }

  /* Cards general look */
  [data-testid="stMetric"]{
    background: rgba(255,255,255,.72);
    border: 1px solid rgba(15,23,42,.08);
    border-radius: 16px;
    padding: 10px 12px;
    box-shadow: 0 12px 26px rgba(15,23,42,.06);
  }

  /* Buttons general */
  .stButton>button, .stDownloadButton>button{
    border-radius: 14px !important;
    font-weight: 900 !important;
    border: 1px solid rgba(122,16,32,.22) !important;
    background: rgba(255,255,255,.82) !important;
    box-shadow: 0 12px 22px rgba(122,16,32,.10) !important;
  }
  .stButton>button:hover, .stDownloadButton>button:hover{
    transform: translateY(-1px);
    box-shadow: 0 16px 26px rgba(122,16,32,.14) !important;
  }

  /* Metric accent */
  [data-testid="stMetricValue"]{
    color: #7A1020 !important;
    font-weight: 950 !important;
  }
</style>
"""
st.markdown(WINE_THEME_CSS, unsafe_allow_html=True)


# SIDEBAR
with st.sidebar:

    # Header
    c1, c2 = st.columns([1, 2.2], gap="small")
    with c1:
        try:
            st.image(AVATAR_PATH, width=86)
        except Exception:
            st.write("🧑‍💻")
    with c2:
        st.markdown("<div class='side-name'>Dwi Amalia</div>", unsafe_allow_html=True)
        st.markdown("<p class='side-role'>Data Enthusiast • Jakarta, Indonesia</p>", unsafe_allow_html=True)
        st.markdown("<span class='chip green'>🟢 Open to work</span>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="side-card">
          <div class="title">Quick Profile</div>
          <p>Halo! 👋 Aku <b>Dwi Amalia</b>, data enthusiast.</p>
          <p>
            Ini adalah <b>Final Project</b> aku: <b>Wine Quality Analysis & Prediction</b>.
            Aku ngerjain proses end-to-end mulai dari <b>data cleaning</b>,
            <b>EDA & dashboard</b>, sampai <b>machine learning</b> untuk memprediksi skor <b>quality</b>.
          </p>
          <p>
            Kamu bisa lihat alurnya di tab <b>Machine Learning</b>,
            lalu coba prediksi sendiri di tab <b>Prediction App</b>. 🍷
          </p>
          <p class="muted">Built with Python • Pandas • Scikit-learn • Plotly • Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# HERO
# =========================
st.markdown(
    """
    <div class="hero">
      <div>
        <div class="hero-title">🍷 Wine Quality</div>
        <div class="hero-sub">Analysis • Dashboard • Machine Learning • Prediction App</div>
        <div class="hero-meta">Final Project – Data Analytics & Data Science • January 2026</div>
        <div>
          <span class="chip"> EDA & Insights</span>
          <span class="chip purple"> ML Modeling</span>
          <span class="chip"> Quality Prediction</span>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "About Dataset",
    "Dashboard",
    "Machine Learning",
    "Prediction App",
    "Contact Me"
])

with tab1:
    import about
    about.show()

with tab2:
    import dashboard
    dashboard.show()

with tab3:
    import machinelearning
    machinelearning.show()

with tab4:
    import prediction
    prediction.prediction_app()

with tab5:
    import contactme
    contactme.show()
