import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score

from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def show():
    # =========================
    # CSS
    # =========================
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
            font-weight: 700;
            color: #1e3a8a;
            border: 1px solid rgba(30,64,175,.25);
            background: #e0e7ff;
            margin-right: 6px;
            margin-bottom: 8px;
          }
          .pill-warn{
            display:inline-block;
            padding: 5px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 800;
            color: #7c2d12;
            border: 1px solid rgba(245,158,11,.35);
            background: rgba(245,158,11,.18);
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
          .list-tight ul{ margin: 0.4rem 0 0.4rem 1.2rem; }
          .list-tight li{ margin: 0.3rem 0; line-height: 1.55; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='wrap'>", unsafe_allow_html=True)
    st.subheader("Machine Learning")

  
    # Load & Cleaning
  
    df = pd.read_csv("9. Wine Quality.csv")

    before_dup = df.shape[0]
    df = df.drop_duplicates()
    after_dup = df.shape[0]
    dup_removed = before_dup - after_dup

    before_na = df.shape[0]
    df = df.dropna()
    after_na = df.shape[0]
    na_removed = before_na - after_na

    df = df.select_dtypes(include=[np.number]).copy()

    st.markdown(
        f"""
        <div class="card list-tight">
          <p class="title">Pre-processing Summary</p>
          <div class="insight">
            <div class="pill">Duplicate removed: {dup_removed:,}</div>
            <div class="pill">Missing removed: {na_removed:,}</div>
            <ul>
              <li>Duplikasi dan missing value dihapus untuk memastikan kualitas data stabil sebelum modeling.</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 1) IQR outlier removal
    st.write("### 1. Handling Outlier dengan IQR")

    numbers = df.select_dtypes(include="number").columns.tolist()
    before_rows = df.shape[0]

    Q1 = df[numbers].quantile(0.25)
    Q3 = df[numbers].quantile(0.75)
    IQR = Q3 - Q1

    df = df[~((df[numbers] < (Q1 - 1.5 * IQR)) | (df[numbers] > (Q3 + 1.5 * IQR))).any(axis=1)]
    after_rows = df.shape[0]
    outlier_removed = before_rows - after_rows

    st.markdown(
        f"""
        <div class="card list-tight">
          <p class="title">Ringkasan IQR</p>
          <div class="insight">
            <div class="pill">Sebelum: {before_rows:,}</div>
            <div class="pill">Sesudah: {after_rows:,}</div>
            <div class="pill-warn">Terhapus: {outlier_removed:,}</div>
            <ul>
              <li>Outlier dihapus agar model lebih robust dan tidak terlalu dipengaruhi nilai ekstrem.</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2) VIF
    st.write("### 2. Cek Multikolinearitas (VIF) & Pemilihan Fitur")

    X_full = df.drop("quality", axis=1)
    y = df["quality"]

    # VIF table (tampilkan dulu)
    vif_values = [variance_inflation_factor(X_full.values, i) for i in range(X_full.shape[1])]
    vif_df = pd.DataFrame({"feature": X_full.columns, "VIF": vif_values}).sort_values("VIF", ascending=False)

    st.markdown(
        """
        <div class="card list-tight">
          <p class="title">Nilai VIF</p>
          <div class="insight">
            <ul>
              <li>VIF tinggi mengindikasikan multikolinearitas sehingga koefisien model linear bisa kurang stabil.</li>
              <li>Dalam project ini, <b>density</b> dan <b>pH</b dihapus karena VIF sangat tinggi untuk meningkatkan stabilitas & interpretabilitas.</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.dataframe(vif_df.style.format({"VIF": "{:.2f}"}), use_container_width=True)

    # Drop columns
    X = X_full.copy()
    dropped_cols = [c for c in ["density", "pH"] if c in X.columns]
    if dropped_cols:
        X = X.drop(columns=dropped_cols)

    # Feature summary (setelah VIF)
    st.markdown(
        f"""
        <div class="card list-tight">
          <p class="title">Fitur yang digunakan untuk modeling</p>
          <div class="insight">
            <div class="pill">Jumlah fitur awal: {X_full.shape[1]}</div>
            <div class="pill">Jumlah fitur akhir: {X.shape[1]}</div>
            <div class="pill-warn">Dropped: {", ".join(dropped_cols) if dropped_cols else "-"}</div>
            <div class="muted">Urutan fitur ini akan disimpan dan dipakai kembali pada Prediction App.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3) Train-test split
    st.write("### 3. Train–Test Split (80/20)")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    st.markdown(
        f"""
        <div class="card">
          <p class="title">Ringkasan Split</p>
          <div class="insight">
            <div class="pill">Train: {len(X_train_raw):,}</div>
            <div class="pill">Test: {len(X_test_raw):,}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 4) StandardScaler
    st.write("### 4. Normalisasi dengan StandardScaler")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    st.markdown(
        """
        <div class="card list-tight">
          <p class="title">Normalisasi StandardScaler</p>
          <div class="insight">
            <ul>
              <li>Pada <b>Ridge</b> & <b>Lasso</b> fitur dinormalisasi menggunakan Standar Scaler agar penalti regularisasi bekerja adil.</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 5) Tuning alpha
    st.write("### 5. Tuning Hyperparameter (Alpha) untuk Ridge & Lasso")

    alphas = np.logspace(-3, 3, 20)

    ridge_grid = GridSearchCV(
        Ridge(),
        {"alpha": alphas},
        cv=5,
        scoring="neg_mean_squared_error"
    )
    ridge_grid.fit(X_train_scaled, y_train)

    lasso_grid = GridSearchCV(
        Lasso(max_iter=30000),
        {"alpha": alphas},
        cv=5,
        scoring="neg_mean_squared_error"
    )
    lasso_grid.fit(X_train_scaled, y_train)

    best_ridge_alpha = float(ridge_grid.best_params_["alpha"])
    best_lasso_alpha = float(lasso_grid.best_params_["alpha"])

    st.markdown(
        f"""
        <div class="card list-tight">
          <p class="title">Hasil Tuning</p>
          <div class="insight">
            <div class="pill">Best α Ridge: {best_ridge_alpha:.3f}</div>
            <div class="pill">Best α Lasso: {best_lasso_alpha:.3f}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 6) Train models
    linear = LinearRegression()
    linear.fit(X_train_raw, y_train)

    ridge = Ridge(alpha=best_ridge_alpha)
    ridge.fit(X_train_scaled, y_train)

    lasso = Lasso(alpha=best_lasso_alpha, max_iter=30000)
    lasso.fit(X_train_scaled, y_train)

    # 6b) Intercept & Coef
    st.write("### 6. Intercept & Koefisien Model")

    def coef_df(model, cols):
        return pd.DataFrame({"feature": cols, "coefficient": model.coef_}).sort_values(
            "coefficient", key=lambda s: s.abs(), ascending=False
        )

    coef_linear = coef_df(linear, X_train_raw.columns)
    coef_ridge = coef_df(ridge, X_train_raw.columns)
    coef_lasso = coef_df(lasso, X_train_raw.columns)

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown("**Linear Regression**")
        st.markdown(f"<div class='pill'>Intercept: {float(linear.intercept_):.4f}</div>", unsafe_allow_html=True)
        st.dataframe(coef_linear.style.format({"coefficient": "{:.4f}"}), use_container_width=True, height=360)

    with c2:
        st.markdown("**Ridge Regression**")
        st.markdown(f"<div class='pill'>Intercept: {float(ridge.intercept_):.4f}</div>", unsafe_allow_html=True)
        st.dataframe(coef_ridge.style.format({"coefficient": "{:.4f}"}), use_container_width=True, height=360)

    with c3:
        st.markdown("**Lasso Regression**")
        st.markdown(f"<div class='pill'>Intercept: {float(lasso.intercept_):.4f}</div>", unsafe_allow_html=True)
        st.dataframe(coef_lasso.style.format({"coefficient": "{:.4f}"}), use_container_width=True, height=360)

    st.markdown(
        """
        <div class="card list-tight">
          <p class="title">Interpretasi Koefisien & Intercept</p>
          <div class="insight">
            <ul>
              <li><b>Koefisien positif</b> → peningkatan fitur cenderung <b>meningkatkan</b> nilai <i>quality</i>, sedangkan <b>koefisien negatif</b> cenderung <b>menurunkan</b> nilai <i>quality</i>.</li>
              <li><b>Linear (intercept = 2,09)</b> → baseline prediksi lebih rendah dan lebih sensitif terhadap variasi/noise data.</li>
              <li><b>Ridge & Lasso (intercept = 5,61)</b> → baseline lebih mendekati rata-rata <i>quality</i> dan lebih stabil setelah standardisasi/regularisasi.</li>
              <li><b>Ridge dipilih</b> karena lebih kuat menghadapi multikolinearitas, menjaga semua fitur tetap berkontribusi, dan prediksinya lebih stabil dibanding Linear & Lasso.</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 7) Evaluation
    st.write("### 7. Evaluasi Model")

    def eval_reg(y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred),
            "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "R2": float(r2_score(y_true, y_pred)),
        }

    y_pred_linear = linear.predict(X_test_raw)
    y_pred_ridge = ridge.predict(X_test_scaled)
    y_pred_lasso = lasso.predict(X_test_scaled)

    results = {
        "Linear Regression": eval_reg(y_test, y_pred_linear),
        "Ridge Regression": eval_reg(y_test, y_pred_ridge),
        "Lasso Regression": eval_reg(y_test, y_pred_lasso),
    }

    results_df = pd.DataFrame(results).T.sort_values("RMSE")

    st.dataframe(
        results_df.style.format({
            "MAE": "{:.4f}",
            "MAPE (%)": "{:.2f}",
            "RMSE": "{:.4f}",
            "R2": "{:.4f}"
        }),
        use_container_width=True
    )

    st.markdown(
        """
        <div class="card list-tight">
          <p class="title">Makna Metrik Evaluasi</p>
          <div class="insight">
            <ul>
              <li><b>MAE</b> (Mean Absolute Error): rata-rata selisih absolut prediksi vs aktual (lebih kecil = lebih baik).</li>
              <li><b>MAPE</b> (Mean Absolute Percentage Error): rata-rata error dalam persen terhadap nilai aktual (lebih kecil = lebih baik).</li>
              <li><b>RMSE</b> (Root Mean Squared Error): mirip MAE tapi memberi penalti lebih besar untuk error besar (lebih kecil = lebih baik).</li>
              <li><b>R²</b> (Coefficient of Determination): proporsi variasi data yang bisa dijelaskan model (lebih besar = lebih baik).</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card list-tight">
          <p class="title">Insight Evaluasi Model</p>
          <div class="insight">
            <ul>
              <li>
                <b>Linear Regression</b> menghasilkan <b>MAE 0.4607</b> dan <b>R² 0.3549</b>, artinya rata-rata kesalahan prediksi sekitar <b>0.46</b>
                dan model hanya mampu menjelaskan <b>35.5%</b> variasi kualitas wine. Sebagai baseline, performanya cukup baik, tetapi cenderung kurang stabil
                untuk pola data yang lebih kompleks.
              </li>
              <li>
                <b>Ridge Regression</b> memberikan performa terbaik dengan <b>MAE 0.4610</b>, <b>RMSE terendah 0.5792</b>, dan <b>R² tertinggi 0.3580</b>.
                Ini menunjukkan keseimbangan terbaik antara error prediksi dan kemampuan menjelaskan variasi kualitas wine, serta lebih stabil pada kondisi multikolinearitas.
              </li>
              <li>
                <b>Lasso Regression</b> menghasilkan <b>MAE 0.4608</b>, <b>RMSE 0.5802</b>, dan <b>R² 0.3556</b>, performanya hampir setara dengan Linear Regression,
                namun regularisasi L1 pada dataset ini belum memberikan peningkatan signifikan dibanding Ridge Regression.
              </li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    best_model_name = results_df.index[0]
    st.markdown(
        f"""
        <div class="card list-tight">
          <p class="title">Kesimpulan Evaluasi</p>
          <div class="insight">
            <div class="pill-warn">Best Model (by RMSE): {best_model_name}</div>
            <ul>
              <li>Ridge Regression dipilih sebagai model utama karena paling stabil untuk fitur yang saling berkorelasi (multikolinearitas).</li>
              <li class="muted">Catatan: error prediksi masih mungkin terjadi karena <b>R²</b> masih sekitar <b>0.35</b> (model baru menjelaskan ~35% variasi data).</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # 8. Kesimpulan & Rekomendasi
    st.write("### 8. Kesimpulan & Rekomendasi")

    st.markdown(
        """
        <div class="card list-tight">
          <p class="title">Kesimpulan</p>
          <div class="insight">
            <ul>
              <li>
                Multikolinearitas terdeteksi pada beberapa fitur. Nilai VIF menunjukkan multikolinearitas tinggi terutama pada
                <b>density</b> dan <b>pH</b>, sehingga keduanya dihapus agar model lebih stabil dan mudah diinterpretasikan.
              </li>
              <li>
                Analisis model menunjukkan fitur seperti <b>alcohol</b>, <b>sulphates</b>, dan <b>volatile acidity</b>
                memiliki pengaruh signifikan terhadap kualitas wine.
              </li>
              <li>
                <b>Ridge Regression</b> dengan alpha terbaik hasil tuning memberikan performa terbaik berdasarkan
                RMSE, MAE, MAPE, dan R².
              </li>
            </ul>
          </div>
        </div>

        <div class="card list-tight">
          <p class="title">Rekomendasi</p>
          <div class="insight">
            <ul>
              <li>Gunakan <b>Ridge Regression</b> sebagai model utama karena paling seimbang antara akurasi dan stabilitas pada fitur yang saling berkorelasi.</li>
              <li>Fokus pada faktor kimia kunci seperti <b>alcohol</b>, <b>volatile acidity</b>, dan <b>sulphates</b>.</li>
              <li>Pengembangan lanjut: coba model non-linear (mis. <i>Random Forest</i>) untuk menangkap pola yang tidak linear.</li>
              <li>Validasi eksternal: uji pada dataset wine lain sebelum dipakai secara operasional.</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # 9) Save artifacts (Ridge as best))
    joblib.dump(scaler, "wine_scaler.pkl")
    joblib.dump(ridge, "wine_best_model.pkl")
    joblib.dump(X_train_raw.columns.tolist(), "wine_model_features.pkl")

    metadata = {
        "best_ridge_alpha": float(best_ridge_alpha),
        "best_lasso_alpha": float(best_lasso_alpha),
        "best_model_name_rmse": str(best_model_name),
        "dropped_cols": dropped_cols
    }
    joblib.dump(metadata, "wine_model_metadata.pkl")

