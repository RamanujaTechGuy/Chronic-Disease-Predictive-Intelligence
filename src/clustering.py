
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

def Clustering(df,n_clusters =4):
    """
    This function is used to cluster the states based on the attributes and number of clusters.
    :param df: Input dataframe
    :param n_clusters: Number of clusters 
    """
    print("Clustering states.....")
    pivot = df.pivot_table(index='LocationAbbr',columns='Topic',values='DataValue',aggfunc='mean').fillna(0)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot)
    kmeans = KMeans(n_clusters = n_clusters,init='k-means++', random_state=42, n_init=10)
    pivot['HealthCluster'] = kmeans.fit_predict(scaled_data)
    # 5. VISUALIZATION (Using PCA to see 10D data in 2D)
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(scaled_data)
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
    x=pca_results[:, 0], y=pca_results[:, 1], 
    hue=pivot['HealthCluster'], 
    palette='viridis', s=100, style=pivot['HealthCluster'])

    # Annotate the points with State Abbreviations
    for i, state in enumerate(pivot.index):
        plt.annotate(state, (pca_results[i, 0], pca_results[i, 1]), fontsize=9, alpha=0.7)

    plt.title('US State Health Clusters (Based on Chronic Disease Prevalence)', fontsize=15)
    plt.xlabel('Principal Component 1 (General Health Burden)')
    plt.ylabel('Principal Component 2 (Disease Variance)')
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.savefig("Clustered_states.png")
    
    
    pivot=pivot.reset_index()

    df = pd.merge(df,pivot[['LocationAbbr','HealthCluster']],on='LocationAbbr',how='left')
    return df
