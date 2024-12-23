import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


# Load a sample dataset (Iris dataset)
def load_data():
    iris = load_iris()
    X = iris.data[:, :2]  # Use only the first two features for simplicity
    return X


# Standardize the data (zero mean and unit variance)
def standardize_data(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_standardized = (X - mean) / std
    return X_standardized, mean, std


# Compute the covariance matrix
def compute_covariance_matrix(X):
    return np.cov(X, rowvar=False)


# Compute eigenvalues and eigenvectors
def compute_eigenvalues_eigenvectors(cov_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors


# Transform data to the new principal component space
def project_data(X, eigenvectors, num_components):
    W = eigenvectors[:, :num_components]  # Projection matrix
    return X @ W


# Visualize the original data and projected data
def visualize_pca(X, X_pca):
    plt.figure(figsize=(10, 5))

    # Original data
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], color='blue', alpha=0.7)
    plt.title("Original Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)

    # Data after PCA
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), color='red', alpha=0.7)
    plt.title("Data After PCA")
    plt.xlabel("Principal Component 1")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Main program
if __name__ == "__main__":
    # Load and preprocess the data
    X = load_data()
    X_standardized, mean, std = standardize_data(X)

    # Compute the covariance matrix
    cov_matrix = compute_covariance_matrix(X_standardized)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(cov_matrix)

    # Display eigenvalues and eigenvectors
    print("Eigenvalues:\n", eigenvalues)
    print("Eigenvectors:\n", eigenvectors)

    # Project data onto the first principal component
    X_pca = project_data(X_standardized, eigenvectors, num_components=1)

    # Visualize the results
    visualize_pca(X_standardized, X_pca)
