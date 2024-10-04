import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
import pandas as pd

# Download NLTK stopwords if not already done
nltk.download('stopwords')

# Database connection parameters
db_params = {
    'host': "localhost",
    'dbname': "ml",
    'user': "ml_user",
    'password': "ml_10925"
}


def fetch_articles():
    try:
        # Establish connection to PostgreSQL
        connection = psycopg2.connect(**db_params)
        cursor = connection.cursor()

        # Query to fetch all articles
        cursor.execute("SELECT id, content, category FROM articles")
        articles = cursor.fetchall()

        return articles

    except Exception as error:
        print(f"Error fetching articles: {error}")
    finally:
        if connection:
            cursor.close()
            connection.close()


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))  # Get stopwords
    # Simple text preprocessing (you can extend this as needed)
    words = [word for word in text.lower().split() if word not in stop_words]
    return ' '.join(words)


def calculate_tfidf(articles):
    # Extract the article texts
    article_texts = [preprocess_text(a[1]) for a in articles]

    # Initialize the TfidfVectorizer
    # Adjust df values as needed
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=2)

    # Fit and transform the articles to get the TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(article_texts)

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    return tfidf_matrix, feature_names


def find_optimal_clusters(tfidf_matrix, max_k=10):
    inertia_values = []
    k_values = list(range(1, max_k + 1))

    # Fit KMeans with different values of k and store the inertia
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(tfidf_matrix)
        inertia_values.append(kmeans.inertia_)

    # Plot the Elbow graph
    plt.plot(k_values, inertia_values, 'bx-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.show()


def apply_pca(tfidf_matrix, variance_threshold=0.95):
    pca = PCA(random_state=0)
    pca.fit(tfidf_matrix.toarray())  # Fit PCA on the dense TF-IDF matrix

    # Determine the number of components to retain based on the explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    # +1 to include the component that reaches the threshold
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

    # Fit PCA again with the determined number of components
    pca = PCA(n_components=n_components, random_state=0)
    reduced_data = pca.fit_transform(tfidf_matrix.toarray())

    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"Number of components retained: {n_components}")
    print(f"Explained variance with {
          n_components} components: {explained_variance:.2f}")

    return reduced_data


def cluster_articles(tfidf_matrix, num_clusters=5):
    # Initialize K-Means with a specified number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)

    # Fit the K-Means model to the TF-IDF matrix
    kmeans.fit(tfidf_matrix)

    # Return the cluster labels (which cluster each article belongs to)
    return kmeans.labels_


def save_clustered_data_to_excel(articles, tfidf_matrix, feature_names, cluster_labels, cluster_top_terms):
    dense_matrix = tfidf_matrix.todense()
    df = pd.DataFrame(dense_matrix, columns=feature_names)

    article_ids = [a[0] for a in articles]
    df.insert(0, 'Article_ID', article_ids)
    df['Cluster'] = cluster_labels

    # Add cluster top terms as an additional sheet in the Excel file
    writer = pd.ExcelWriter(
        "clustered_tfidf_results.xlsx", engine='xlsxwriter')
    df.to_excel(writer, sheet_name='TF-IDF Clusters', index=False)

    # Create a DataFrame for the top terms of each cluster
    top_terms_data = []
    for cluster_num, terms in cluster_top_terms.items():
        for rank, (term, score) in enumerate(terms, 1):
            top_terms_data.append([cluster_num, rank, term, score])

    # Create DataFrame for top terms, including their rank and score
    cluster_top_terms_df = pd.DataFrame(
        top_terms_data, columns=['Cluster', 'Rank', 'Term', 'TF-IDF Score'])
    cluster_top_terms_df.to_excel(
        writer, sheet_name='Cluster_Top_Terms', index=False)

    writer.close()
    print("Clustered TF-IDF results and sorted top terms saved to clustered_tfidf_results.xlsx")


def find_top_terms_per_cluster(tfidf_matrix, feature_names, labels, num_clusters, top_n=5):
    cluster_top_terms = {}

    for cluster_num in range(num_clusters):
        # Get indices of documents in the current cluster
        cluster_indices = np.where(labels == cluster_num)

        # Extract the TF-IDF values of documents in the current cluster
        cluster_matrix = tfidf_matrix[cluster_indices]

        # Calculate the average TF-IDF score for each term in the cluster
        avg_tfidf = np.mean(cluster_matrix, axis=0).A1  # Flatten to 1D array

        # Get the indices of the top N terms, sorted by average TF-IDF in descending order
        top_term_indices = np.argsort(avg_tfidf)[-top_n:][::-1]

        # Get the corresponding feature (term) names and their scores for the top terms
        top_terms = [(feature_names[i], avg_tfidf[i])
                     for i in top_term_indices]

        cluster_top_terms[cluster_num] = top_terms

    return cluster_top_terms


def evaluate_clustering_with_silhouette(reduced_data, cluster_labels):
    # Calculate the average silhouette score for the entire clustering
    silhouette_avg = silhouette_score(reduced_data, cluster_labels)
    print(f"Average Silhouette Score: {silhouette_avg:.3f}")

    # Calculate the silhouette score for each sample
    sample_silhouette_values = silhouette_samples(reduced_data, cluster_labels)

    # Visualize the silhouette scores for each cluster
    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10
    for i in range(len(np.unique(cluster_labels))):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / len(np.unique(cluster_labels)))
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10  # 10 for the next cluster

    ax.set_title("Silhouette plot for the various clusters")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_yticks([])  # Clear the y-axis labels
    ax.set_xlim([-0.1, 1])
    plt.show()


def compare_clusters_with_labels(cluster_labels, real_labels):
    # Adjusted Rand Index (ARI)
    ari_score = adjusted_rand_score(real_labels, cluster_labels)
    print(f"Adjusted Rand Index (ARI): {ari_score:.3f}")

    # Normalized Mutual Information (NMI)
    nmi_score = normalized_mutual_info_score(real_labels, cluster_labels)
    print(f"Normalized Mutual Information (NMI): {nmi_score:.3f}")

    # Homogeneity, Completeness, V-measure
    homogeneity = homogeneity_score(real_labels, cluster_labels)
    completeness = completeness_score(real_labels, cluster_labels)
    v_measure = v_measure_score(real_labels, cluster_labels)
    print(f"Homogeneity: {homogeneity:.3f}")
    print(f"Completeness: {completeness:.3f}")
    print(f"V-measure: {v_measure:.3f}")


if __name__ == "__main__":
    # Fetch articles from the database
    articles = fetch_articles()

    # Calculate TF-IDF
    tfidf_matrix, feature_names = calculate_tfidf(articles)

    # Print the number of dimensions (features) before PCA
    num_dimensions_before_pca = tfidf_matrix.shape[1]
    print(f"Number of dimensions (features) before PCA: {
          num_dimensions_before_pca}")

    # Use the Elbow Method to find the optimal number of clusters
    find_optimal_clusters(tfidf_matrix, max_k=10)

    # Perform clustering
    # Apply PCA to reduce the dimensionality
    # Adjust the variance threshold as needed
    reduced_data = apply_pca(tfidf_matrix, variance_threshold=0.95)

    optimal_k = 2  # Replace this with the actual optimal number of clusters
    cluster_labels = cluster_articles(reduced_data, num_clusters=optimal_k)

    # Find the top terms for each cluster
    top_terms_per_cluster = find_top_terms_per_cluster(
        tfidf_matrix, feature_names, cluster_labels, optimal_k, top_n=10)

    # Evaluate the clustering quality using the Silhouette method
    evaluate_clustering_with_silhouette(reduced_data, cluster_labels)

    # Compare the clusters with real article categories
    # Assuming `cluster_labels` are from your K-means model and `real_labels` are the true categories
    real_labels = [a[-1] for a in articles]
    compare_clusters_with_labels(cluster_labels, real_labels)

    # Save the clustered data along with the top terms to an Excel file
    save_clustered_data_to_excel(
        articles, tfidf_matrix, feature_names, cluster_labels, top_terms_per_cluster)
