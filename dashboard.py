import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def show():

    # CSS
    st.markdown(
        """
        <style>
          .article-wrap{
            max-width: 1050px;
            margin: 0 auto;
          }
          .card{
            padding: 12px 12px;
            border-radius: 14px;
            border: 1px solid rgba(0,0,0,.10);
            background: rgba(255,255,255,.75);
          }
          .muted{ opacity:.75; font-size: 13px; }
          .title{ font-weight: 900; margin: 0 0 6px 0; }

          /* KPI cards */
          .kpi-grid{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
          }
          .kpi{
            padding: 12px 12px;
            border-radius: 14px;
            border: 1px solid rgba(0,0,0,.10);
            background: rgba(255,255,255,.75);
          }
          .kpi-label{
            font-size: 12px;
            opacity: .75;
            margin-bottom: 2px;
          }
          .kpi-value{
            font-size: 22px;
            font-weight: 900;
            line-height: 1.1;
          }
          .kpi-sub{
            font-size: 12px;
            opacity: .75;
            margin-top: 4px;
          }

          /* Academic insight (seragam) */
          .insight-academic{
            padding: 12px 14px;
            border-radius: 14px;
            border: 1px solid rgba(30,64,175,.25);
            background: rgba(30,64,175,.08);
          }
          .pill-academic{
            display:inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            color: #1e40af;
            border: 1px solid rgba(30,64,175,.30);
            background: rgba(30,64,175,.12);
            margin-bottom: 6px;
            margin-right: 6px;
          }

          /* Small screens: KPI jadi 2 kolom */
          @media (max-width: 900px){
            .kpi-grid{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
          }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='article-wrap'>", unsafe_allow_html=True)

    st.markdown("## Wine Quality Dashboard")

    # Load data & drop duplicates
    df_raw = pd.read_csv("9. Wine Quality.csv").drop_duplicates()
    df = df_raw.select_dtypes(include=[np.number]).copy()

    # Handling Missing Values
    df = df.dropna(subset=["quality"])
    df = df.fillna(df.median(numeric_only=True))

    # KPI cards
    features = [c for c in df.columns if c != "quality"]
    total_rows = df.shape[0]
    total_features = len(features)
    avg_quality = df["quality"].mean()
    q_min, q_max = df["quality"].min(), df["quality"].max()

    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi">
            <div class="kpi-label">Total Data</div>
            <div class="kpi-value">{total_rows:,}</div>
            <div class="kpi-sub">setelah drop duplicate </div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Total Features</div>
            <div class="kpi-value">{total_features}</div>
            <div class="kpi-sub">kolom fitur numerik </div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Average Quality</div>
            <div class="kpi-value">{avg_quality:.2f}</div>
            <div class="kpi-sub">Rata-Rata Kulitas</div>
          </div>
          <div class="kpi">
            <div class="kpi-label">Quality Range</div>
            <div class="kpi-value">{int(q_min)}–{int(q_max)}</div>
            <div class="kpi-sub">Min sampai Max</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    with st.expander("Lihat data", expanded=False):
        st.dataframe(df.head(5), use_container_width=True)

    st.write("")

    def two_col():
        return st.columns([1.35, 1], gap="large")

    def tighten(fig, height):
        fig.update_layout(height=height, margin=dict(l=12, r=12, t=45, b=12))
        return fig

    # 1) Distribusi Quality
    st.markdown("### 1) Distribusi Target (Quality)")
    gcol, icol = two_col()

    with gcol:
        fig = px.histogram(df, x="quality", nbins=10, title="Distribusi Quality Wine")
        fig = tighten(fig, 240)
        fig.update_layout(yaxis_title="Jumlah Data", xaxis_title="Quality")
        st.plotly_chart(fig, use_container_width=True)

    with icol:
        mode_quality = df["quality"].mode().iloc[0]
        st.markdown("<p class='title'>Insight</p>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="insight-academic">
              <div class="pill-academic">Modus: {mode_quality}</div>
              <div class="pill-academic">Mean: {avg_quality:.2f}</div>
              <div class="pill-academic">Range: {int(q_min)}–{int(q_max)}</div>
              <ul>
                <li>
                  Distribusi skor kualitas cenderung <b>terpusat pada rentang menengah</b>
                  (dominan pada quality 5), yang mengindikasikan mayoritas sampel berada
                  pada kualitas <b>moderat</b>.
                </li>
                <li>
                  Nilai ekstrem relatif lebih jarang, sehingga distribusi
                  <b>tidak didominasi oleh outlier pada target</b>.
                </li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # 2) Alcohol vs Quality
    st.markdown("### 2) Hubungan Quality dengan Fitur Utama")

    gcol, icol = two_col()
    with gcol:
        if "alcohol" in df.columns:
            fig = px.scatter(df, x="alcohol", y="quality", trendline="ols", title="Alcohol vs Quality")
            fig.update_traces(marker=dict(size=3, opacity=0.45))
            fig = tighten(fig, 250)
            st.plotly_chart(fig, use_container_width=True)
            corr_alcohol = df[["alcohol", "quality"]].corr().iloc[0, 1]
        else:
            corr_alcohol = None
            st.info("Kolom `alcohol` tidak tersedia.")

    with icol:
        st.markdown("<p class='title'>Insight: Alcohol</p>", unsafe_allow_html=True)
        if corr_alcohol is not None:
            st.markdown(
                f"""
                <div class="insight-academic">
                  <div class="pill-academic">Correlation: {corr_alcohol:.2f}</div>
                  <ul>
                    <li>
                      Terlihat <b>tren positif</b> antara kadar alcohol dan quality (garis tren menanjak).
                    </li>
                    <li>
                      Wine dengan alcohol lebih rendah lebih sering berada pada quality <b>4–5</b>,
                      sedangkan wine dengan alcohol lebih tinggi lebih banyak muncul pada quality <b>6–7</b>.
                    </li>
                    <li>
                      Meskipun sebaran data cukup lebar dan hubungan tidak sepenuhnya linear kuat,
                      alcohol tetap <b>berkontribusi positif</b> terhadap peningkatan kualitas wine.
                    </li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.write("Tidak ada insight karena kolom tidak tersedia.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # 3) Volatile Acidity vs Quality
    gcol, icol = two_col()
    with gcol:
        if "volatile acidity" in df.columns:
            fig = px.scatter(df, x="volatile acidity", y="quality", trendline="ols", title="Volatile Acidity vs Quality")
            fig.update_traces(marker=dict(size=3, opacity=0.45))
            fig = tighten(fig, 250)
            st.plotly_chart(fig, use_container_width=True)
            corr_va = df[["volatile acidity", "quality"]].corr().iloc[0, 1]
        else:
            corr_va = None
            st.info("Kolom `volatile acidity` tidak tersedia.")

    with icol:
        st.markdown("<p class='title'>Insight: Volatile Acidity</p>", unsafe_allow_html=True)
        if corr_va is not None:
            st.markdown(
                f"""
                <div class="insight-academic">
                  <div class="pill-academic">Correlation: {corr_va:.2f}</div>
                  <ul>
                    <li>
                      Terlihat <b>hubungan negatif</b> antara volatile acidity dan quality (garis tren menurun).
                    </li>
                    <li>
                      Wine dengan volatile acidity rendah lebih sering memiliki quality di kisaran <b>6–7</b>,
                      sedangkan wine dengan volatile acidity tinggi cenderung berada pada quality <b>4–5</b>.
                    </li>
                    <li>
                      Walau hubungan tidak sepenuhnya linear, kenaikan volatile acidity
                      cenderung diikuti oleh <b>penurunan kualitas wine</b>.
                    </li>
                  </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.write("Tidak ada insight karena kolom tidak tersedia.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # 4) Distribusi fitur numerik
    st.markdown("### 3) Distribusi Fitur Numerik")
    st.caption("Ringkasan distribusi beberapa fitur utama.")

    preferred = ["alcohol", "volatile acidity", "sulphates", "citric acid"]
    show_feats = [f for f in preferred if f in df.columns]

    if len(show_feats) < 4:
        for f in features:
            if f not in show_feats:
                show_feats.append(f)
            if len(show_feats) == 4:
                break

    row1 = st.columns(2, gap="large")
    row2 = st.columns(2, gap="large")
    grid = [row1[0], row1[1], row2[0], row2[1]]

    for i, feat in enumerate(show_feats):
        mean_val = df[feat].mean()

        fig = px.histogram(df, x=feat, nbins=30, title=f"Distribusi {feat}")
        fig.add_vline(x=mean_val, line_dash="dash")
        fig = tighten(fig, 260)

        with grid[i]:
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"""
                  <p class="title">Insight</p>
                  <div class="insight-academic">
                    <div class="pill-academic">Mean: {mean_val:.2f}</div>
                    <ul>
                      <li>
                        Distribusi <b>{feat}</b> menunjukkan variasi karakteristik kimia wine.
                        Pola sebaran yang tidak simetris/ekor panjang dapat mengindikasikan adanya nilai ekstrem.
                      </li>
                      <li>
                        Temuan distribusi menjadi dasar pertimbangan <i>pre-processing</i>,
                        khususnya normalisasi dan penanganan outlier sebelum modeling.
                      </li>
                    </ul>
                  </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")

    # 5) Boxplot outlier
    st.markdown("### 4) Boxplot Fitur Numerik")

    gcol, icol = two_col()
    with gcol:
        long_df = df[features].melt(var_name="feature", value_name="value")
        fig = px.box(long_df, x="feature", y="value", points="outliers", title="Boxplot Seluruh Fitur")
        fig.update_layout(xaxis_tickangle=-60)
        fig = tighten(fig, 500)
        st.plotly_chart(fig, use_container_width=True)

    with icol:
        st.markdown("<p class='title'>Insight</p>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="insight-academic">
              <ul>
                <li>
                  Terdapat outlier pada beberapa variabel, terutama <b>total sulfur dioxide</b>,
                  <b>free sulfur dioxide</b>, dan <b>residual sugar</b>, yang menunjukkan sebaran data tidak simetris.
                </li>
                <li>
                  Variabel <b>alcohol</b> memiliki sebaran lebih luas dan beberapa outlier,
                  yang berpotensi berkontribusi terhadap variasi nilai quality.
                </li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # 6) Heatmap
    st.markdown("### 5) Korelasi Antar Fitur (Heatmap)")

    corr = df.corr(numeric_only=True)
    z = corr.values

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=corr.columns,
            y=corr.index,
            zmin=-1,
            zmax=1,
            text=np.round(z, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        )
    )
    fig = tighten(fig, 460)
    fig.update_layout(title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

    top_corr = (
        corr["quality"]
        .drop("quality")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(6)
    )

    st.markdown("<p class='title'>Insight: Feature vs Target</p>", unsafe_allow_html=True)

    for f, v in top_corr.items():
        st.write(f"- **{f}**: {v:.2f}")

    st.markdown(
        """
        <div class="insight-academic">
          <ul>
            <li>
              <b>Alcohol</b> berkorelasi positif (~0.48) dengan quality, menunjukkan wine dengan kadar alkohol lebih tinggi
              cenderung memiliki kualitas yang lebih baik.
            </li>
            <li>
              <b>Volatile acidity</b> berkorelasi negatif (~-0.40) dengan quality, menunjukkan peningkatan volatile acidity
              cenderung menurunkan kualitas wine.
            </li>
            <li>
              Terdapat indikasi <b>multikolinearitas</b> antar beberapa fitur, sehingga pada tahap modeling
              digunakan pendekatan regularisasi seperti <b>Ridge</b> dan <b>Lasso</b>.
            </li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
