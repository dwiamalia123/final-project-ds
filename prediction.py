import streamlit as st
import pandas as pd
import numpy as np
import joblib


def prediction_app():
    st.markdown(
        """
        <style>
          body { background-color: #f8fafc; color: #0f172a; }
          .wrap{ max-width: 1050px; margin: 0 auto; }
          .card{
            padding: 14px;
            border-radius: 16px;
            border: 1px solid rgba(15,23,42,.12);
            background: #ffffff;
            margin-bottom: 12px;
          }
          .title{ font-weight: 900; margin: 0 0 6px 0; color: #0f172a; }
          .muted{ opacity: .85; font-size: 13px; color: #475569; margin-top: 0; }
          .pill{
            display:inline-block;
            padding: 5px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 800;
            color: #1e3a8a;
            border: 1px solid rgba(30,64,175,.25);
            background: #e0e7ff;
            margin-right: 6px;
            margin-bottom: 8px;
          }
          .pill-soft{
            display:inline-block;
            padding: 5px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 800;
            color: #0f172a;
            border: 1px solid rgba(15,23,42,.12);
            background: rgba(15,23,42,.04);
            margin-right: 6px;
            margin-bottom: 8px;
          }
          .pill-low{
            display:inline-block;
            padding: 5px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 900;
            color: #7c2d12;
            border: 1px solid rgba(245,158,11,.35);
            background: rgba(245,158,11,.18);
            margin-right: 6px;
            margin-bottom: 8px;
          }
          .pill-mid{
            display:inline-block;
            padding: 5px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 900;
            color: #1e40af;
            border: 1px solid rgba(30,64,175,.25);
            background: rgba(30,64,175,.10);
            margin-right: 6px;
            margin-bottom: 8px;
          }
          .pill-high{
            display:inline-block;
            padding: 5px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 900;
            color: #065f46;
            border: 1px solid rgba(16,185,129,.28);
            background: rgba(16,185,129,.12);
            margin-right: 6px;
            margin-bottom: 8px;
          }
          .insight{
            padding: 14px 16px;
            border-radius: 14px;
            border: 1px solid rgba(30,64,175,.18);
            background: #f1f5ff;
            color: #020617;
          }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='wrap'>", unsafe_allow_html=True)
    st.subheader("Prediction : Wine Quality Score")

    try:
        model = joblib.load("wine_best_model.pkl")
        scaler = joblib.load("wine_scaler.pkl")
        feature_order = joblib.load("wine_model_features.pkl")
        metadata = joblib.load("wine_model_metadata.pkl")
    except Exception as e:
        st.error("Gagal load artifact. Pastikan semua .pkl satu folder dengan prediction.py")
        st.exception(e)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    best_alpha = float(metadata.get("best_ridge_alpha", 0.0))

    st.markdown(
        f"""
        <div class="card">
          <p class="title">Model Info</p>
          <div class="insight">
            <div class="pill">Model: Ridge</div>
            <div class="pill">Alpha: {best_alpha:.2f}</div>
            <div class="pill">Intercept: {float(model.intercept_):.4f}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    def load_clean_df_for_defaults():
        df = pd.read_csv("9. Wine Quality.csv")
        df = df.drop_duplicates()
        df = df.dropna()
        df = df.select_dtypes(include=[np.number]).copy()

        numbers = df.select_dtypes(include="number").columns.tolist()
        Q1 = df[numbers].quantile(0.25)
        Q3 = df[numbers].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[numbers] < (Q1 - 1.5 * IQR)) | (df[numbers] > (Q3 + 1.5 * IQR))).any(axis=1)]

        drop_cols = [c for c in ["density", "pH"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        return df

    default_map = {}
    try:
        df_ref = load_clean_df_for_defaults()
        med = df_ref.median(numeric_only=True).to_dict()
        for f in feature_order:
            default_map[f] = float(med.get(f, 0.0))
    except Exception:
        for f in feature_order:
            default_map[f] = 0.0

    st.write("### Input Parameter")
    cols = st.columns(2)
    user_input = {}

    for i, feat in enumerate(feature_order):
        with cols[i % 2]:
            val = float(default_map.get(feat, 0.0))
            step = 0.01
            fmt = "%.2f"
            if feat == "chlorides":
                step, fmt = 0.001, "%.3f"
            user_input[feat] = st.number_input(feat, value=val, step=step, format=fmt)

    st.write("")

    def categorize_quality(q_int: int):
        if q_int <= 4:
            return "Low Quality", "3–4", "pill-low"
        elif q_int <= 6:
            return "Medium Quality", "5–6", "pill-mid"
        else:
            return "High Quality", "7–8", "pill-high"

    if st.button("Prediksi Quality"):
        X_user = pd.DataFrame([user_input], columns=feature_order)
        X_user_scaled = scaler.transform(X_user)

        pred = float(model.predict(X_user_scaled)[0])
        pred_round = int(np.rint(pred))
        pred_round = max(0, min(10, pred_round))

        label, rng, pill_class = categorize_quality(pred_round)

        st.markdown("---")
        st.write("## Hasil Prediksi")

        c1, c2 = st.columns([1, 1.1], gap="large")
        with c1:
            st.metric("Prediksi Skor Quality", f"{pred_round}")
            st.caption(f"Nilai mentah (sebelum pembulatan): {pred:.3f}")

        with c2:
            st.markdown(
                f"""
                <div class="card">
                  <p class="title">Kategori</p>
                  <div class="insight">
                    <div class="{pill_class}">{label}</div>
                    <div class="pill-soft">Range: {rng}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            f"""
            <div class="card">
              <p class="title">Interpretasi</p>
              <div class="insight">
                Model memprediksi skor quality sebesar <b>{pred_round}</b> (kategori: <b>{label}</b>).
                Prediksi dihasilkan menggunakan <b>Ridge Regression</b> dengan fitur yang sudah distandarisasi,
                sehingga kontribusi fitur dapat dibandingkan pada skala yang sama dan regularisasi membantu menjaga
                stabilitas estimasi saat fitur saling berkorelasi.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander("Lihat input", expanded=False):
            st.dataframe(X_user, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    prediction_app()
