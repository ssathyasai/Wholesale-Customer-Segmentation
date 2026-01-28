import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Wholesale Customer Segmentation", layout="wide")

st.title("📦 Wholesale Customer Segmentation using K-Means")

# ---------------------------------------
# Load Dataset
# ---------------------------------------
df = pd.read_csv("Wholesale customers data.csv")

st.subheader("🔹 Task 1: Data Exploration")
st.write("Dataset Preview:")
st.dataframe(df.head())

st.write("Dataset Information:")
st.write(df.describe())

# ---------------------------------------
# Feature Selection
# ---------------------------------------
st.subheader("🔹 Task 2: Feature Selection")

features = ['Fresh', 'Milk', 'Grocery', 'Frozen',
            'Detergents_Paper', 'Delicassen']

st.write("Selected features represent customer purchasing behavior:")
st.write(features)

X = df[features]

# ---------------------------------------
# Data Scaling
# ---------------------------------------
st.subheader("🔹 Task 3: Data Preparation (Scaling)")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.success("Data standardized successfully.")

# ---------------------------------------
# Elbow Method
# ---------------------------------------
st.subheader("🔹 Task 4 & 5: Optimal Cluster Identification (Elbow Method)")

wcss = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

fig1, ax1 = plt.subplots()
ax1.plot(K, wcss, marker='o')
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("WCSS")
ax1.set_title("Elbow Method")
st.pyplot(fig1)

st.info("Optimal number of clusters chosen as K = 3")

# ---------------------------------------
# KMeans Model
# ---------------------------------------
st.subheader("🔹 Task 6: Cluster Assignment")

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

st.write("Dataset with Cluster Labels:")
st.dataframe(df.head())

# ---------------------------------------
# Visualization
# ---------------------------------------
st.subheader("🔹 Task 7: Cluster Visualization")

fig2, ax2 = plt.subplots()

ax2.scatter(df[df['Cluster']==0]['Grocery'],
            df[df['Cluster']==0]['Detergents_Paper'],
            label='Cluster 0')

ax2.scatter(df[df['Cluster']==1]['Grocery'],
            df[df['Cluster']==1]['Detergents_Paper'],
            label='Cluster 1')

ax2.scatter(df[df['Cluster']==2]['Grocery'],
            df[df['Cluster']==2]['Detergents_Paper'],
            label='Cluster 2')

ax2.set_xlabel("Grocery Spending")
ax2.set_ylabel("Detergents Paper Spending")
ax2.set_title("Customer Segments")
ax2.legend()

st.pyplot(fig2)

# ---------------------------------------
# Cluster Profiling
# ---------------------------------------
st.subheader("🔹 Task 8: Cluster Profiling")

profile = df.groupby('Cluster')[features].mean()
st.write("Average Spending per Cluster:")
st.dataframe(profile)

st.markdown("""
**Cluster Interpretation:**
- Cluster 0: High grocery & detergents → Retail Stores  
- Cluster 1: Balanced spending → Hotels & Restaurants  
- Cluster 2: Low overall spending → Cafés / Small buyers  
""")

# ---------------------------------------
# Business Insights
# ---------------------------------------
st.subheader("🔹 Task 9: Business Insights")

st.markdown("""
- **Cluster 0:** Bulk discounts & priority inventory  
- **Cluster 1:** Customized bundles & loyalty programs  
- **Cluster 2:** Entry-level pricing & upselling strategies  
""")

# ---------------------------------------
# Stability Check
# ---------------------------------------
st.subheader("🔹 Task 10: Stability & Limitations")

kmeans_test = KMeans(n_clusters=3, init='k-means++', random_state=99)
df['Cluster_Test'] = kmeans_test.fit_predict(X_scaled)

st.write("Cluster comparison with different random state:")
st.dataframe(df[['Cluster', 'Cluster_Test']].head())

st.warning("""
Limitation:
K-Means is sensitive to outliers and assumes clusters are spherical in shape.
""")