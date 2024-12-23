import numpy as np
import matplotlib.pyplot as plt


# Define the mass and stiffness matrices
def create_matrices():
    # Mass matrix (assume masses of 1 kg each)
    M = np.array([[1, 0],
                  [0, 1]])

    # Stiffness matrix (springs with stiffness constants k1 and k2)
    k1 = 50  # N/m
    k2 = 30  # N/m
    K = np.array([[k1 + k2, -k2],
                  [-k2, k2]])

    return M, K


# Compute eigenvalues and eigenvectors
def compute_eigenvalues_eigenvectors(M, K):
    # Generalized eigenvalue problem: K * x = lambda * M * x
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(M) @ K)

    # Sort eigenvalues and eigenvectors in ascending order
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


# Visualize modes of vibration
def plot_vibration_modes(eigenvectors):
    modes = eigenvectors.T
    for i, mode in enumerate(modes):
        plt.figure()
        plt.bar(["Mass 1", "Mass 2"], mode, color=["blue", "orange"])
        plt.title(f"Mode {i + 1}: Vibration Shape")
        plt.ylabel("Amplitude")
        plt.xlabel("Masses")
        plt.grid(True)
        plt.show()


# Main program
if __name__ == "__main__":
    # Create matrices
    M, K = create_matrices()

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(M, K)

    # Display results
    print("Natural Frequencies (rad/s):", np.sqrt(eigenvalues))
    print("Modes of Vibration (eigenvectors):\n", eigenvectors)

    # Plot vibration modes
    plot_vibration_modes(eigenvectors)
