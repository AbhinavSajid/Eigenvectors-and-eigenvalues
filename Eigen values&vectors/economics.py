import numpy as np

# Define the input-output matrix
def create_input_output_matrix():
    # Example input-output matrix for a three-sector economy
    A = np.array([
        [0.2, 0.3, 0.1],  # Sector 1: Consumes from itself, Sector 2, and Sector 3
        [0.4, 0.1, 0.2],  # Sector 2: Consumes from Sector 1, itself, and Sector 3
        [0.3, 0.2, 0.2]   # Sector 3: Consumes from Sector 1, Sector 2, and itself
    ])
    return A

# Compute eigenvalues and eigenvectors
def compute_eigenvalues_eigenvectors(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Find the dominant eigenvalue (largest magnitude)
    dominant_index = np.argmax(np.abs(eigenvalues))
    dominant_eigenvalue = eigenvalues[dominant_index]
    dominant_eigenvector = eigenvectors[:, dominant_index]

    return eigenvalues, eigenvectors, dominant_eigenvalue, dominant_eigenvector

# Normalize the dominant eigenvector
def normalize_eigenvector(eigenvector):
    normalized_vector = eigenvector / np.sum(eigenvector)
    return normalized_vector.real  # Ensure values are real numbers

# Main program
if __name__ == "__main__":
    # Create the input-output matrix
    A = create_input_output_matrix()
    print("Input-Output Matrix:\n", A)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors, dominant_eigenvalue, dominant_eigenvector = compute_eigenvalues_eigenvectors(A)

    # Display results
    print("\nEigenvalues:\n", eigenvalues)
    print("\nEigenvectors:\n", eigenvectors)
    print("\nDominant Eigenvalue:", dominant_eigenvalue.real)
    print("\nDominant Eigenvector (Unnormalized):\n", dominant_eigenvector)

    # Normalize the dominant eigenvector
    normalized_vector = normalize_eigenvector(dominant_eigenvector)
    print("\nNormalized Dominant Eigenvector:\n", normalized_vector)

    # Interpretation
    print("\nInterpretation:")
    print("The dominant eigenvalue represents the growth factor of the economy.")
    print("The normalized dominant eigenvector represents the relative importance of each sector.")
