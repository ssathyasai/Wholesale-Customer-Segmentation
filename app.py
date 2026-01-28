import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# App Title & Description
# --------------------------------------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🟢 Customer Segmentation Dashboard")
st.write(
    "This system uses **K-Means Clustering** to group customers based on their "
    "purchasing behavior and similarities."
)

st.markdown(
    "👉 **Goal:** Discover hidden customer groups without predefined labels."
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("wholesale_customers.csv")

df = load_data()

numeric_features = [
    'Fresh', 'Milk', 'Grocery',
    'Frozen', 'Detergents_Paper', 'Delicassen'
]

# --------------------------------------------------
# Sidebar – Input Section (Mandatory)
# --------------------------------------------------
st.sidebar.header("Clustering Controls")

feature_1 = st.sidebar.selectbox(
    "Select Feature 1",
    numeric_features,
    index=0
)

feature_2 = st.sidebar.selectbox(
    "Select Feature 2",
    numeric_features,
    index=1
)

k = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=3
)

random_state = st.sidebar.number_input(
    "Random State (Optional)",
    min_value=0,
    value=42
)

run_button = st.sidebar.button("🟦 Run Clustering")

# --------------------------------------------------
# Run Clustering (Only when button is clicked)
# --------------------------------------------------
if run_button:

    # Feature selection
    X = df[[feature_1, feature_2]]

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    clusters = kmeans.fit_predict(X_scaled)

    df['Cluster'] = clusters

    # --------------------------------------------------
    # Visualization Section
    # --------------------------------------------------
    st.subheader("📊 Cluster Visualization")

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        df[feature_1],
        df[feature_2],
        c=df['Cluster'],
        cmap='viridis'
    )

    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c='red',
        s=250,
        marker='X',
        label='Centroids'
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title("Customer Clusters")
    ax.legend()

    st.pyplot(fig)

    # --------------------------------------------------
    # Cluster Summary Section
    # --------------------------------------------------
    st.subheader("📋 Cluster Summary")

    summary = df.groupby('Cluster').agg(
        Count=('Cluster', 'count'),
        Avg_Feature_1=(feature_1, 'mean'),
        Avg_Feature_2=(feature_2, 'mean')
    )

    st.dataframe(summary)

    # --------------------------------------------------
    # Business Interpretation Section
    # --------------------------------------------------
    st.subheader("💡 Business Interpretation")

    for cluster_id in summary.index:
        avg1 = summary.loc[cluster_id, 'Avg_Feature_1']
        avg2 = summary.loc[cluster_id, 'Avg_Feature_2']

        if avg1 > summary[["Avg_Feature_1"]].mean().values[0] and \
           avg2 > summary[["Avg_Feature_2"]].mean().values[0]:
            insight = "High-spending customers across selected categories"
        elif avg1 < summary[["Avg_Feature_1"]].mean().values[0] and \
             avg2 < summary[["Avg_Feature_2"]].mean().values[0]:
            insight = "Budget-conscious customers with lower annual spending"
        else:
            insight = "Moderate spenders with selective purchasing behavior"

        st.write(f"🟢 **Cluster {cluster_id}:** {insight}")

    # --------------------------------------------------
    # User Guidance / Insight Box
    # --------------------------------------------------
    st.info(
        "Customers in the same cluster exhibit similar purchasing behaviour "
        "and can be targeted with similar business strategies."
    )

else:
    st.warning("⬅️ Please select features and click **Run Clustering** to begin.")
