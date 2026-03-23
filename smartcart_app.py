"""
SmartCart Clustering System — Streamlit Dashboard
Run: streamlit run smartcart_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCart Clustering",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* -------- BACKGROUND -------- */
.stApp {
    background-color: #f8fafc;
}

/* -------- SIDEBAR -------- */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

/* -------- TEXT -------- */
html, body, [class*="css"] {
    color: #111827;
    font-family: 'Inter', sans-serif;
}

/* -------- HEADINGS -------- */
h1 { color: #111827 !important; font-weight: 700 !important; }
h2 { color: #2563eb !important; font-weight: 600 !important; }
h3 { color: #374151 !important; }

/* -------- METRIC CARDS -------- */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 14px;
}
[data-testid="metric-container"] label {
    color: #6b7280 !important;
}
[data-testid="stMetricValue"] {
    color: #2563eb !important;
    font-weight: 700;
}

/* -------- TABS -------- */
[data-testid="stTabs"] button {
    color: #6b7280 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #2563eb !important;
    border-bottom: 2px solid #2563eb !important;
}

/* -------- TABLE -------- */
[data-testid="stDataFrame"] {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
}

/* -------- REMOVE BRANDING -------- */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Cluster Styling ───────────────────────────────────────────────────────────
CLUSTER_META = {
    0: {"name": "Premium Shoppers", "emoji": "💎", "color": "#00e5ff",  "strategy": "VIP Loyalty Program"},
    1: {"name": "Deal Hunters",     "emoji": "🏷️", "color": "#7c3aed",  "strategy": "Targeted Promotions"},
    2: {"name": "Budget Families",  "emoji": "🌱", "color": "#10b981",  "strategy": "Family Bundle Offers"},
    3: {"name": "Churn Risk",       "emoji": "⚠️", "color": "#f59e0b",  "strategy": "Re-Engagement Campaign"},
}
COLORS = [m["color"] for m in CLUSTER_META.values()]

# ── Data Loading & Pipeline ───────────────────────────────────────────────────
@st.cache_data
def load_and_process(path: str):
    df = pd.read_csv(path)

    # ── 1. Missing values
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # ── 2. Feature Engineering
    df["Age"] = 2026 - df["Year_Birth"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    ref = df["Dt_Customer"].max()
    df["Customer_Tenure_Days"] = (ref - df["Dt_Customer"]).dt.days
    df["Total_Spending"] = (df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
                            + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"])
    df["Total_Children"] = df["Kidhome"] + df["Teenhome"]
    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "Master": "Postgraduate", "PhD": "Postgraduate",
    })
    df["Living_With"] = df["Marital_Status"].replace({
        "Married": "Partner", "Together": "Partner",
        "Single": "Alone", "Divorced": "Alone",
        "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone",
    })

    # ── 3. Remove outliers
    df = df[(df["Age"] < 90) & (df["Income"] < 600_000)]

    # ── 4. Drop raw cols
    drop_cols = ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer",
                 "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
                 "MntSweetProducts", "MntGoldProds"]
    df_clean = df.drop(columns=drop_cols)

    # ── 5. Encode
    ohe = OneHotEncoder(sparse_output=False)
    cat_cols = ["Education", "Living_With"]
    enc = ohe.fit_transform(df_clean[cat_cols])
    enc_df = pd.DataFrame(enc, columns=ohe.get_feature_names_out(cat_cols), index=df_clean.index)
    df_enc = pd.concat([df_clean.drop(columns=cat_cols), enc_df], axis=1)

    # ── 6. Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_enc)

    # ── 7. PCA
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    return df, df_clean, df_enc, X_pca, pca

@st.cache_data
def run_clustering(X_pca_list, n_clusters: int, algo: str):
    X_pca = np.array(X_pca_list)
    if algo == "Agglomerative (Ward)":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    else:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_pca)
    return labels

@st.cache_data
def compute_elbow_silhouette(X_pca_list):
    X_pca = np.array(X_pca_list)
    wcss, sil = [], []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_pca)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(X_pca, lbl))
    return wcss, sil

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛒 SmartCart")
    st.markdown("---")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Upload smartcart_customers.csv")
    st.markdown("---")

    algo = st.selectbox("Clustering Algorithm", ["Agglomerative (Ward)", "KMeans"])
    n_clusters = st.slider("Number of Clusters (k)", 2, 8, 4)
    st.markdown("---")

    st.markdown("**Pipeline Steps**")
    steps = ["Data Loading", "Feature Engineering", "Outlier Removal",
             "Encoding", "Scaling", "PCA", "Clustering"]
    for i, s in enumerate(steps, 1):
        st.markdown(f"`{i:02d}` {s}")


# ── Load Data ─────────────────────────────────────────────────────────────────
if uploaded:
    data_path = uploaded
else:
    data_path = "smartcart_customers.csv"

try:
    df_raw, df_clean, df_enc, X_pca, pca = load_and_process(data_path)
    labels = run_clustering(X_pca.tolist(), n_clusters, algo)
    df_enc = df_enc.copy()
    df_enc["Cluster"] = labels
    df_raw_full = df_raw.copy()
    df_raw_full = df_raw_full.loc[df_enc.index]
    df_raw_full["Cluster"] = labels
    DATA_OK = True
except FileNotFoundError:
    DATA_OK = False

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("# 🛒 SmartCart Clustering System")
st.markdown("Customer segmentation using unsupervised machine learning · KMeans & Agglomerative")
st.markdown("---")

if not DATA_OK:
    st.warning("⚠️ `smartcart_customers.csv` not found. Upload it in the sidebar or place it in the same folder.")
    st.stop()

# ── KPI Metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Records",    f"{len(df_enc):,}")
col2.metric("Features (clean)", f"{df_enc.shape[1] - 1}")
col3.metric("Clusters",         n_clusters)
col4.metric("PCA Components",   "3")
col5.metric("Avg Silhouette",   f"{silhouette_score(X_pca, labels):.3f}")

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Cluster Overview",
    "🔬 EDA & Features",
    "📉 K Selection",
    "🌐 3D Projection",
    "🗂️ Data Explorer",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — CLUSTER OVERVIEW
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Cluster Profiles")

    # Cluster summary
    summary = df_enc.groupby("Cluster").mean(numeric_only=True)

    # Cluster cards (4 per row)
    cols = st.columns(min(n_clusters, 4))
    for i in range(n_clusters):
        meta = CLUSTER_META.get(i, {"name": f"Cluster {i}", "emoji": "●",
                                    "color": COLORS[i % len(COLORS)], "strategy": "—"})
        count = (df_enc["Cluster"] == i).sum()
        pct   = count / len(df_enc) * 100
        with cols[i % 4]:
            st.markdown(f"""
            <div style="background:#0e1420;border:1px solid {meta['color']}33;
                        border-top:3px solid {meta['color']};border-radius:10px;padding:18px;margin-bottom:16px">
                <div style="font-size:28px;margin-bottom:8px">{meta['emoji']}</div>
                <div style="color:{meta['color']};font-size:11px;letter-spacing:2px;
                            text-transform:uppercase;margin-bottom:6px">Cluster {i}</div>
                <div style="font-size:16px;font-weight:800;margin-bottom:12px">{meta['name']}</div>
                <div style="font-size:24px;font-weight:800;color:{meta['color']}">{count:,}</div>
                <div style="font-size:11px;color:#64748b;margin-bottom:10px">{pct:.1f}% of customers</div>
                <div style="background:{meta['color']}18;border-radius:4px;padding:6px 10px;
                            font-size:11px;color:{meta['color']}">🎯 {meta['strategy']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Spending & Income by Cluster")

    c1, c2 = st.columns(2)

    with c1:
        if "Total_Spending" in summary.columns:
            fig = px.bar(
                x=[CLUSTER_META.get(i, {"name": f"C{i}"})["name"] for i in summary.index],
                y=summary["Total_Spending"].values,
                color=[CLUSTER_META.get(i, {"name": f"C{i}"})["name"] for i in summary.index],
                color_discrete_sequence=COLORS[:len(summary)],
                title="Avg Total Spending",
                labels={"x": "Cluster", "y": "Amount (₹)"},
            )
            fig.update_layout(
                plot_bgcolor="#080c14", paper_bgcolor="#080c14",
                font_color="#e2e8f0", showlegend=False,
                title_font_color="#00e5ff",
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(gridcolor="#1e2a40")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "Income" in summary.columns:
            fig = px.bar(
                x=[CLUSTER_META.get(i, {"name": f"C{i}"})["name"] for i in summary.index],
                y=summary["Income"].values,
                color=[CLUSTER_META.get(i, {"name": f"C{i}"})["name"] for i in summary.index],
                color_discrete_sequence=COLORS[:len(summary)],
                title="Avg Income",
                labels={"x": "Cluster", "y": "Income (₹)"},
            )
            fig.update_layout(
                plot_bgcolor="#080c14", paper_bgcolor="#080c14",
                font_color="#e2e8f0", showlegend=False,
                title_font_color="#00e5ff",
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(gridcolor="#1e2a40")
            st.plotly_chart(fig, use_container_width=True)

    # Scatter: Income vs Spending
    st.markdown("## Income vs Total Spending")
    if "Total_Spending" in df_enc.columns and "Income" in df_enc.columns:
        scatter_df = df_enc.copy()
        scatter_df["Cluster Name"] = scatter_df["Cluster"].map(
            lambda x: CLUSTER_META.get(x, {"name": f"C{x}"})["name"])
        fig = px.scatter(
            scatter_df, x="Total_Spending", y="Income",
            color="Cluster Name",
            color_discrete_sequence=COLORS[:n_clusters],
            opacity=0.65,
            title="Income vs Spending — coloured by Cluster",
            labels={"Total_Spending": "Total Spending (₹)", "Income": "Annual Income (₹)"},
        )
        fig.update_layout(
            plot_bgcolor="#080c14", paper_bgcolor="#080c14",
            font_color="#e2e8f0", title_font_color="#00e5ff",
            legend_title_text="Cluster",
        )
        fig.update_xaxes(gridcolor="#1e2a40")
        fig.update_yaxes(gridcolor="#1e2a40")
        st.plotly_chart(fig, use_container_width=True)

    # Cluster Summary Table
    st.markdown("## Cluster Feature Averages")
    show_cols = [c for c in ["Income", "Total_Spending", "Age", "Total_Children",
                              "Recency", "NumWebPurchases", "NumStorePurchases",
                              "Customer_Tenure_Days"] if c in summary.columns]
    st.dataframe(
        summary[show_cols].round(1).style
            .background_gradient(cmap="Blues", axis=0)
            .format("{:.1f}"),
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════════
# TAB 2 — EDA & FEATURES
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## Exploratory Data Analysis")

    num_cols = df_enc.select_dtypes(include=np.number).columns.drop("Cluster").tolist()
    sel_feat = st.selectbox("Select Feature to Explore", num_cols)

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            df_enc, x=sel_feat, color="Cluster",
            color_discrete_sequence=COLORS[:n_clusters],
            barmode="overlay", opacity=0.7,
            title=f"Distribution of {sel_feat}",
            nbins=40,
        )
        fig.update_layout(
            plot_bgcolor="#080c14", paper_bgcolor="#080c14",
            font_color="#e2e8f0", title_font_color="#00e5ff",
        )
        fig.update_xaxes(gridcolor="#1e2a40")
        fig.update_yaxes(gridcolor="#1e2a40")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(
            df_enc, x="Cluster", y=sel_feat,
            color="Cluster",
            color_discrete_sequence=COLORS[:n_clusters],
            title=f"{sel_feat} per Cluster",
        )
        fig.update_layout(
            plot_bgcolor="#080c14", paper_bgcolor="#080c14",
            font_color="#e2e8f0", title_font_color="#00e5ff",
            showlegend=False,
        )
        fig.update_xaxes(gridcolor="#1e2a40")
        fig.update_yaxes(gridcolor="#1e2a40")
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.markdown("## Correlation Heatmap")
    heat_cols = [c for c in ["Income", "Total_Spending", "Age", "Recency",
                              "Total_Children", "NumWebPurchases",
                              "NumStorePurchases", "NumDealsPurchases",
                              "Customer_Tenure_Days"] if c in df_enc.columns]
    corr = df_enc[heat_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title="Feature Correlation Matrix",
        aspect="auto",
    )
    fig.update_layout(
        paper_bgcolor="#080c14", font_color="#e2e8f0",
        title_font_color="#00e5ff",
        coloraxis_colorbar=dict(tickfont_color="#e2e8f0"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # PCA variance
    st.markdown("## PCA — Explained Variance")
    pca_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(3)],
        "Explained Variance": pca.explained_variance_ratio_,
        "Cumulative": np.cumsum(pca.explained_variance_ratio_),
    })
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=pca_df["Component"], y=pca_df["Explained Variance"],
        name="Per Component", marker_color="#00e5ff",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=pca_df["Component"], y=pca_df["Cumulative"],
        name="Cumulative", marker_color="#7c3aed", mode="lines+markers",
    ), secondary_y=True)
    fig.update_layout(
        title="PCA Explained Variance", plot_bgcolor="#080c14",
        paper_bgcolor="#080c14", font_color="#e2e8f0",
        title_font_color="#00e5ff",
    )
    fig.update_yaxes(gridcolor="#1e2a40", secondary_y=False)
    fig.update_yaxes(gridcolor="#1e2a40", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — K SELECTION
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Optimal K — Elbow & Silhouette")

    with st.spinner("Computing scores for k = 2…10 …"):
        wcss, sil = compute_elbow_silhouette(X_pca.tolist())

    k_range = list(range(2, 11))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=k_range, y=wcss,
        name="WCSS (Elbow)", mode="lines+markers",
        marker=dict(color="#00e5ff", size=8),
        line=dict(color="#00e5ff", width=2),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=k_range, y=sil,
        name="Silhouette Score", mode="lines+markers",
        marker=dict(color="#f59e0b", size=8),
        line=dict(color="#f59e0b", width=2, dash="dash"),
    ), secondary_y=True)

    # Highlight k=4
    fig.add_vline(x=4, line_dash="dot", line_color="#10b981",
                  annotation_text="k=4 selected", annotation_font_color="#10b981")

    fig.update_layout(
        title="Elbow Method & Silhouette Score",
        plot_bgcolor="#080c14", paper_bgcolor="#080c14",
        font_color="#e2e8f0", title_font_color="#00e5ff",
        legend=dict(bgcolor="#0e1420", bordercolor="#1e2a40"),
    )
    fig.update_xaxes(gridcolor="#1e2a40", title="Number of Clusters (k)")
    fig.update_yaxes(gridcolor="#1e2a40", title="WCSS", secondary_y=False)
    fig.update_yaxes(title="Silhouette Score", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    # Table
    k_df = pd.DataFrame({
        "k": k_range,
        "WCSS": [round(w, 1) for w in wcss],
        "Silhouette": [round(s, 4) for s in sil],
    })
    k_df["Selected"] = k_df["k"].apply(lambda x: "✅" if x == n_clusters else "")
    st.dataframe(k_df.set_index("k"), use_container_width=True)

    st.markdown("""
    <div class="info-box">
        <b>📌 Why k = 4?</b><br>
        The elbow in the WCSS curve appears around k=4, and the Silhouette Score
        peaks or remains competitive at k=4, making it the optimal balance between
        cluster compactness and separation.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — 3D PROJECTION
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 3D PCA Projection")

    pca_plot_df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
    pca_plot_df["Cluster"] = labels
    pca_plot_df["Cluster Name"] = pca_plot_df["Cluster"].map(
        lambda x: f"{CLUSTER_META.get(x,{'name':f'C{x}'})['emoji']} {CLUSTER_META.get(x,{'name':f'C{x}'})['name']}"
    )

    if "Income" in df_enc.columns:
        pca_plot_df["Income"] = df_enc["Income"].values
    if "Total_Spending" in df_enc.columns:
        pca_plot_df["Total_Spending"] = df_enc["Total_Spending"].values

    fig = px.scatter_3d(
        pca_plot_df, x="PC1", y="PC2", z="PC3",
        color="Cluster Name",
        color_discrete_sequence=COLORS[:n_clusters],
        opacity=0.7,
        hover_data={k: True for k in ["Income", "Total_Spending"] if k in pca_plot_df.columns},
        title=f"3D PCA — {algo} · k={n_clusters}",
    )
    fig.update_traces(marker_size=3)
    fig.update_layout(
        paper_bgcolor="#080c14", font_color="#e2e8f0",
        title_font_color="#00e5ff",
        scene=dict(
            bgcolor="#0e1420",
            xaxis=dict(gridcolor="#1e2a40", color="#64748b"),
            yaxis=dict(gridcolor="#1e2a40", color="#64748b"),
            zaxis=dict(gridcolor="#1e2a40", color="#64748b"),
        ),
        legend_title_text="Cluster",
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cluster size donut
    st.markdown("## Cluster Size Distribution")
    size_df = pca_plot_df["Cluster Name"].value_counts().reset_index()
    size_df.columns = ["Cluster", "Count"]
    fig = px.pie(
        size_df, names="Cluster", values="Count",
        color="Cluster", hole=0.55,
        color_discrete_sequence=COLORS[:n_clusters],
        title="Customer Distribution across Clusters",
    )
    fig.update_layout(
        paper_bgcolor="#080c14", font_color="#e2e8f0",
        title_font_color="#00e5ff",
        legend=dict(bgcolor="#0e1420"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 5 — DATA EXPLORER
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## Data Explorer")

    sel_cluster = st.selectbox(
        "Filter by Cluster",
        ["All"] + [f"Cluster {i} — {CLUSTER_META.get(i,{'name':f'C{i}'})['name']}"
                   for i in range(n_clusters)]
    )

    view_df = df_enc.copy()
    view_df["Cluster Name"] = view_df["Cluster"].map(
        lambda x: CLUSTER_META.get(x, {"name": f"C{x}"})["name"])

    if sel_cluster != "All":
        c_idx = int(sel_cluster.split()[1])
        view_df = view_df[view_df["Cluster"] == c_idx]

    st.markdown(f"**{len(view_df):,} rows**")
    st.dataframe(view_df.reset_index(drop=True), use_container_width=True, height=420)

    # Download
    csv_bytes = view_df.to_csv(index=False).encode()
    st.download_button(
        label="⬇️ Download filtered CSV",
        data=csv_bytes,
        file_name="smartcart_clustered.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("## Raw Dataset Preview")
    st.dataframe(df_raw.head(50), use_container_width=True, height=300)