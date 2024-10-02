import cv2
import numpy as np
import matplotlib.pyplot as plt

def calcMatrix(M, opc):
    # Case of V Matrix
    if opc == 1:
        newM = np.dot(M.T, M)
    # Case of U Matrix
    if opc == 2:
        newM = np.dot(M, M.T)

    eigenvalues, eigenvectors = np.linalg.eig(newM)
    eigenvectors = np.real(eigenvectors)

    # Case of V Matrix, return transpose
    if opc == 1:
        return eigenvectors[:,].T
    # Case of U, return normally
    else:
        return eigenvectors[:,]

# Function that calculates Eigenvalues corresponding to the Sigma Matrix
def calcDiagonal(M):
    if (np.size(np.dot(M, M.T)) > np.size(np.dot(M.T, M))):
        newM = np.dot(M.T, M)
    else:
        newM = np.dot(M, M.T)

    eigenvalues, eigenvectors = np.linalg.eig(newM)
    eigenvalues = np.sqrt(np.abs(eigenvalues))
    return eigenvalues

img = cv2.imread('turp.bmp')
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_image = gray_image.astype(np.float64)

U, s, V = np.linalg.svd(gray_image, full_matrices=False)
Vt_manuel = calcMatrix(gray_image, 1)
U_manuel = calcMatrix(gray_image, 2)
Sigma_manuel = calcDiagonal(gray_image)


k_values = [10, 50, 100]

plt.figure(figsize=(12,6))

for i in range(len(k_values)):
    low_rank = U[:, :k_values[i]] @ np.diag(s[:k_values[i]]) @ V[:k_values[i], :]
    plt.subplot(2,3,i+1),
    plt.imshow(low_rank, cmap='gray'),
    plt.title(f"For K value = {k_values[i]}")
    plt.savefig("Reconstruction_with_k_values.png")

for i in range(len(k_values)):
    low_rank_manuel = U_manuel[:, :k_values[i]] @ np.diag(Sigma_manuel[:k_values[i]]) @ Vt_manuel[:k_values[i], :]
    #low_rank_manuel = np.abs(low_rank_manuel)
    plt.subplot(2,3,i+1),
    plt.imshow(low_rank_manuel, cmap='gray'),
    plt.title(f"For K value = {k_values[i]}")
    plt.savefig("Manuel_reconstruction_with_k_values.png")
