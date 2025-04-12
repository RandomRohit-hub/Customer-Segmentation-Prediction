import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.set_page_config(page_title="🛍️ Mall Customer Clustering", layout="centered")
st.title("🛍️ Mall Customer Clustering")

st.write("This app uses KMeans clustering to segment customers based on annual income and spending score. 📊")

# Read CSV file directly from code
csv_file_path = "Mall_Customers.csv"
try:
    customer_data = pd.read_csv(csv_file_path)
    st.success("✅ Data loaded from 'Mall_Customers.csv'")

    st.subheader("🔍 Data Preview")
    st.dataframe(customer_data.head())

    st.subheader("🚫 Missing Values")
    st.write(customer_data.isnull().sum())

    X = customer_data.iloc[:, [3, 4]].values

    st.subheader("📈 Elbow Method")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots()
    sns.set()
    ax1.plot(range(1, 11), wcss, marker='o')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('WCSS')
    st.pyplot(fig1)

    st.subheader("🧠 Clustering Results")
    optimal_clusters = st.slider("Choose number of clusters", 2, 10, 5)

    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=0)
    Y = kmeans.fit_predict(X)

    fig2, ax2 = plt.subplots()
    colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']
    for i in range(optimal_clusters):
        ax2.scatter(X[Y == i, 0], X[Y == i, 1], s=50, c=colors[i], label=f"Cluster {i+1}")
    ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='gold', label='Centroids', marker='X')
    ax2.set_title('Customer Groups')
    ax2.set_xlabel('Annual Income (k$)')
    ax2.set_ylabel('Spending Score (1-100)')
    ax2.legend()
    st.pyplot(fig2)

except FileNotFoundError:
    st.error("❌ 'Mall_Customers.csv' not found. Make sure the file is in the same directory as this app.")
