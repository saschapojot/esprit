import numpy as np
import matplotlib.pyplot as plt

# --- 1. Simulation Parameters ---
M = 4              # Number of antenna elements
K = 2              # Number of signal sources
N = 1000           # Number of snapshots (time samples)
wavelength = 1.0   # Lambda
d = wavelength / 2 # Element spacing (half-wavelength)
snr_db = 10        # Signal-to-Noise Ratio in dB
# Ground truth angles (in degrees)
theta_true_deg = np.array([15.0, -30.0])
theta_true_rad = np.deg2rad(theta_true_deg)
print(f"Ground Truth Angles: {theta_true_deg}")
# Construct Steering Matrix A (Eq. 90)
# Shape: (M, K)
A = np.zeros((M, K), dtype=complex)
for k in range(K):
    # The phase shift term from Eq. 90: exp(-i * 2*pi*d/lambda * m * sin(theta))
    phase_shift = -1j * (2 * np.pi * d / wavelength) * np.sin(theta_true_rad[k])
    A[:, k] = np.exp(phase_shift * np.arange(M))

# Generate Signal Matrix S (Eq. 87)
# Shape: (K, N). Modeled as random complex Gaussian.
S = (np.random.randn(K, N) + 1j * np.random.randn(K, N)) / np.sqrt(2)
# Generate Noise N_noise (Eq. 88)
# Shape: (M, N)
noise = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) / np.sqrt(2)
# Scale noise based on SNR
signal_power = np.mean(np.abs(A @ S)**2)
noise_power = signal_power / (10**(snr_db/10))
noise = noise * np.sqrt(noise_power)

# Received Signal X (Eq. 86)
X = A @ S + noise
# --- 3. ESPRIT Algorithm Implementation ---
# Step A: Covariance Matrix (Eq. 92)
# R_x = (1/N) * X * X^H
R_x = (X @ X.conj().T) / N
# Step B: Eigendecomposition (Eq. 98 - 101)
# We use eigh because R_x is Hermitian
eigenvalues, eigenvectors = np.linalg.eigh(R_x)

# Sort eigenvalues in descending order (Eq. 100)
idx = eigenvalues.argsort()[::-1]
eigenvectors = eigenvectors[:, idx]
# Step C: Signal Subspace Extraction (Eq. 102)
# Take the first K eigenvectors (columns)
E_s = eigenvectors[:, :K]
# Step D: Subarray Formation (Eq. 104 - 105)
# E_s1: First M-1 rows
E_s1 = E_s[:-1, :]
# E_s2: Last M-1 rows
E_s2 = E_s[1:, :]

# Step E: TLS Solution (Eq. 116 - 125)
# Form E_xy = [E_s1, E_s2] (Eq. 116)
E_xy = np.hstack((E_s1, E_s2))
# SVD of E_xy (Eq. 117)
# Note: numpy returns V^H (V hermitian), so we take the transpose conjugate to get V
U_tls, S_tls, VH_tls = np.linalg.svd(E_xy)
V_tls = VH_tls.conj().T

# Partition V into 4 KxK blocks (Eq. 121)
# The matrix is 2K x 2K.
# We need V01 (top right) and V11 (bottom right)
# V = [[V00, V01],
#      [V10, V11]]
V01 = V_tls[:K, K:]
V11 = V_tls[K:, K:]
# Calculate Psi_TLS (Eq. 125)
Psi_tls = -V01 @ np.linalg.inv(V11)
# Step F: Eigenvalues of Psi (Eq. 126)
phi_eigenvalues = np.linalg.eigvals(Psi_tls)
# Step G: Angle Estimation (Eq. 127)
# arg(lambda) = - (2*pi*d / wavelength) * sin(theta)
# theta = arcsin( arg(lambda) / (-2*pi*d/wavelength) )

args = np.angle(phi_eigenvalues)
estimated_sin = args / (-2 * np.pi * d / wavelength)

# Clip to [-1, 1] to avoid numerical errors in arcsin
estimated_sin = np.clip(estimated_sin, -1, 1)

estimated_thetas_rad = np.arcsin(estimated_sin)
estimated_thetas_deg = np.rad2deg(estimated_thetas_rad)

# Sort for comparison
estimated_thetas_deg.sort()
theta_true_deg_sorted = np.sort(theta_true_deg)

print(f"Estimated Angles:  {estimated_thetas_deg}")
# Calculate Error
error = np.abs(estimated_thetas_deg - theta_true_deg_sorted)
print(f"Absolute Error:    {error}")
