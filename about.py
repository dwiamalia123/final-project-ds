import streamlit as st
from pathlib import Path

WINE_IMG_PATH = Path('asset/Wine.jpg.png')

def _inject_local_css():
    st.markdown(
        """
        <style>
          .card{
            border: 1px solid rgba(0,0,0,.10);
            border-radius: 18px;
            padding: 16px 16px;
            background: rgba(255,255,255,.70);
            backdrop-filter: blur(10px);
            margin-bottom: 14px;
          }
          .title{
            font-weight: 900;
            margin: 0 0 6px 0;
          }
          .subtle{
            opacity: .78;
            font-size: 13px;
            margin-top: 0;
          }
          .pill{
            display:inline-flex;
            align-items:center;
            gap: 6px;
            padding: 7px 10px;
            border-radius: 999px;
            border: 1px solid rgba(0,0,0,.10);
            background: rgba(255,255,255,.80);
            font-size: 12px;
            font-weight: 800;
            opacity: .9;
          }
          .callout{
            padding: 12px 14px;
            border-radius: 14px;
            border: 1px solid rgba(59,130,246,.25);
            background: rgba(59,130,246,.08);
          }
          .dd-grid{
            display:grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px 18px;
          }
          .dd-item b { font-weight: 900; }
          .dd-item{ font-size: 13px; line-height: 1.4; }
          @media (max-width: 900px){
            .dd-grid{ grid-template-columns: 1fr; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _render_image():
    if WINE_IMG_PATH.exists():
        st.image(str(WINE_IMG_PATH), use_container_width=True)
    else:
        st.info("Tambahkan gambar")

def show():
    _inject_local_css()

    # Header
    st.markdown("## Tentang Dataset")

    left, right = st.columns([1, 2], gap="large")
    with left :
      _render_image()
    with right:
        st.markdown(
            """
            Kualitas wine berperan penting dalam membentuk kepuasan konsumen, memengaruhi keputusan pembelian, serta persepsi terhadap merek. Namun, penilaian kualitas wine sering kali bersifat subjektif karena dipengaruhi oleh preferensi individu, pengalaman, dan standar penilaian yang berbeda-beda. Kondisi ini membuat proses evaluasi kualitas menjadi tidak selalu konsisten.
            Untuk membantu memberikan penilaian yang lebih objektif dan terukur, dataset Wine Quality menyediakan data karakteristik fisikokimia wine—seperti tingkat keasaman, kandungan gula, pH, dan kadar alkohol—yang diukur melalui pengujian laboratorium. Data ini dapat digunakan untuk menganalisis faktor-faktor yang memengaruhi kualitas wine serta membangun model machine learning guna memprediksi skor quality berdasarkan fitur-fitur tersebut.
            """
        )

        st.markdown(
            """
            <div class="callout">
              <b>Goal:</b> Memahami faktor yang paling memengaruhi <b>quality</b> dan membangun model regresi
              untuk prediksi skor <b>quality</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown("<p class='title'>Ringkasan Dataset</p>", unsafe_allow_html=True)
    c1, c2, c3= st.columns(3)
    c1.metric("Domain", "Food & Beverage")
    c2.metric("Task", "Regression")
    c3.metric("Target", "Quality Score")

    st.markdown("</div>", unsafe_allow_html=True)

    # Data Dictionary
    st.markdown("<p class='title'>Data Dictionary</p>", unsafe_allow_html=True)

    with st.expander("Lihat daftar fitur", expanded=True):
        st.markdown(
            """
            <div class="dd-grid">
              <div class="dd-item"><b>fixed acidity</b><br/>tingkat keasaman tetap</div>
              <div class="dd-item"><b>volatile acidity</b><br/>keasaman volatil (nilai tinggi bisa menurunkan kualitas)</div>

              <div class="dd-item"><b>citric acid</b><br/>kandungan asam sitrat</div>
              <div class="dd-item"><b>residual sugar</b><br/>gula tersisa setelah fermentasi</div>

              <div class="dd-item"><b>chlorides</b><br/>kandungan garam</div>
              <div class="dd-item"><b>free sulfur dioxide</b><br/>SO₂ bebas (pengawet)</div>

              <div class="dd-item"><b>total sulfur dioxide</b><br/>total SO₂</div>
              <div class="dd-item"><b>density</b><br/>densitas wine</div>

              <div class="dd-item"><b>pH</b><br/>derajat keasaman</div>
              <div class="dd-item"><b>sulphates</b><br/>sulfat (berpengaruh ke aroma/rasa)</div>

              <div class="dd-item"><b>alcohol</b><br/>kadar alkohol</div>
              <div class="dd-item"><b>quality</b><br/>skor kualitas (target regresi)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

