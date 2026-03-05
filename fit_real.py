import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def fit_exponential_quadrature(f_data, dt, K, L=None):
    """
    Implements the algorithm from Appendix C to fit f(t) = sum(a_j * exp(lambda_j * t)).

    Parameters:
    -----------
    f_data : array_like
        The measured time series data f(0), f(1), ..., f(M-1).
    dt : float
        Time step delta t.
    K : int
        Number of exponential components to fit.
    L : int, optional
        Pencil parameter (window length). Constraint: K < L < M - K.
        If None, defaults to M // 2.

    Returns:
    --------
    lambdas : ndarray
        The estimated exponents lambda_j.
    amplitudes : ndarray
        The estimated coefficients a_j.
    f_reconstructed : ndarray
        The reconstructed signal.
    """
    f = np.array(f_data)
    M = len(f)

    # 1. Define Matrix Dimensions (Eq. 136-137)
    if L is None:
        L = M // 2  # A standard choice is roughly half the data length

    P = M - L + 1

    if not (K < L < M - K):
        print(f"Warning: Constraint K < L < M - K might be violated. K={K}, L={L}, M={M}")

    # 2. Construct Hankel Matrix H (Eq. 138)
    # We use scipy.linalg.hankel. The first column is f[0]...f[L-1].
    # The last row is f[L-1]...f[M-1].
    first_col = f[:L]
    last_row = f[L-1:]
    H = la.hankel(first_col, last_row)

    # Verify dimensions of H
    assert H.shape == (L, P), f"Hankel matrix shape mismatch. Expected ({L}, {P}), got {H.shape}"

    # 3. Compute Covariance Matrix Ry (Eq. 149-150)
    # Note: The text uses 1/P sum(y(s)y(s)^dag). In matrix form this is (1/P) * H * H^H.
    Ry = (1.0 / P) * (H @ H.conj().T)

    # 4. Signal Subspace Decomposition (Eq. 153-154)
    # Perform SVD on Ry to get Signal Subspace Us
    U, S, Vh = la.svd(Ry)

    # Keep only the first K columns corresponding to the largest eigenvalues (signal subspace)
    Us = U[:, :K]

    # 5. Rotational Invariance / ESPRIT (Eq. 160-165)
    # Partition Us into Us1 (first L-1 rows) and Us2 (last L-1 rows)
    Us1 = Us[:-1, :]
    Us2 = Us[1:, :]

    # Form Uxy = [Us1, Us2] (Eq. 165)
    Uxy = np.hstack((Us1, Us2))

    # 6. Total Least Squares (TLS) via SVD (Eq. 167-174)
    # We perform SVD on Uxy.
    # Note: numpy/scipy svd returns Vh (Hermitian transpose).
    # The text refers to V^R, which are the right singular vectors (columns of V).
    U_tls, S_tls, Vh_tls = la.svd(Uxy)
    # === ADD THIS BLOCK ===
    print("\n--- Diagnostic: Singular Values of Uxy (S_tls) ---")
    print(S_tls)
    print("--------------------------------------------------\n")
    # ======================
    V_R = Vh_tls.conj().T  # Get the V matrix

    # The matrix V_R is 2K x 2K. We split it into 4 blocks of K x K (Eq. 171).
    # V_R = [[V00, V01],
    #        [V10, V11]]
    K_dim = K
    V01 = V_R[:K_dim, K_dim:]  # Top-right block
    V11 = V_R[K_dim:, K_dim:]  # Bottom-right block

    # Calculate Psi_TLS (Eq. 174)
    # Psi_TLS = -V01 * inv(V11)
    Psi_TLS = -V01 @ la.inv(V11)

    # 7. Eigenvalues and Exponents (Eq. 175-176)
    mu = la.eigvals(Psi_TLS)
    print(f"mu={mu}")
    # lambda_j = log(mu_j) / dt
    lambdas = np.log(mu) / dt
    print(f"lambdas={lambdas}")
    # 8. Recover Amplitudes (Least Squares)
    # Now that we have lambdas, we fit the linear equation: f(t) = Sum(a_j * exp(lambda_j * t))
    # We construct a Vandermonde-like matrix A for the full time series
    t_full = np.arange(M) * dt
    A_matrix = np.zeros((M, K), dtype=complex)

    for i in range(K):
        A_matrix[:, i] = np.exp(lambdas[i] * t_full)

    # Solve A * amplitudes = f
    amplitudes, residuals, rank, s = la.lstsq(A_matrix, f)

    # Reconstruct the signal for verification
    f_reconstructed = A_matrix @ amplitudes

    return lambdas, amplitudes, f_reconstructed

# ==========================================
# Main Verification Script
# ==========================================


# 1. Setup Ground Truth Parameters
# We use the model from Eq. 128: f(t) = sum(a_j * exp(lambda_j * t))
# The text assumes lambda_j > 0 (growing exponentials), though the math works for decaying too.
# Let's test with growing exponentials as per the text's assumption.

dt = 0.05
M = 100  # Number of samples
t = np.arange(M) * dt

# Ground Truth Values
true_lambdas = np.array([0.5, 1.2])
true_amplitudes = np.array([1.0, 0.5])
K = len(true_lambdas)

# Generate Signal
signal_clean = np.zeros(M, dtype=complex)
for a, lam in zip(true_amplitudes, true_lambdas):
    signal_clean += a * np.exp(lam * t)
# print(signal_clean)
# Add Gaussian Noise (Eq. 130)
# np.random.seed(42)
print(f"np.max(np.abs(signal_clean))={np.max(np.abs(signal_clean))}")
noise_level = 0.01 * np.min(np.abs(signal_clean)) # Small noise relative to signal magnitude
noise = noise_level * (np.random.randn(M) + 1j * np.random.randn(M))
f_measured = signal_clean + noise

print(f"Generating data with M={M}, dt={dt}")
print(f"True Lambdas: {true_lambdas}")
print(f"True Amplitudes: {true_amplitudes}")
print("-" * 40)

# 2. Run the Algorithm
est_lambdas, est_amplitudes, f_recon = fit_exponential_quadrature(f_measured, dt, K)

# Sort results for comparison (order of eigenvalues is arbitrary)
# We sort by the real part of lambda
idx = np.argsort(est_lambdas.real)
est_lambdas = est_lambdas[idx]
est_amplitudes = est_amplitudes[idx]

# 3. Display Results
print("Estimated Results:")
for i in range(K):
    print(f"Component {i+1}:")
    print(f"  Lambda: {est_lambdas[i].real:.4f} + {est_lambdas[i].imag:.4f}j  (True: {true_lambdas[i]})")
    print(f"  Amp:    {est_amplitudes[i].real:.4f} + {est_amplitudes[i].imag:.4f}j  (True: {true_amplitudes[i]})")

# 4. Visualization
plt.figure(figsize=(10, 6))

# Plot Real parts
plt.subplot(2, 1, 1)
plt.plot(t, f_measured.real, 'k.', label='Measured (Noisy)', markersize=4, alpha=0.5)
plt.plot(t, f_recon.real, 'r-', label='Reconstructed via Alg C', linewidth=1.5)
plt.plot(t, signal_clean.real, 'g--', label='Ground Truth', linewidth=1)
plt.title("Real Part of Signal")
plt.legend()
plt.grid(True)

# Plot Imaginary parts (should be near zero for real inputs, but algorithm handles complex)
plt.subplot(2, 1, 2)
plt.plot(t, f_measured.imag, 'k.', label='Measured (Noisy)', markersize=4, alpha=0.5)
plt.plot(t, f_recon.imag, 'r-', label='Reconstructed', linewidth=1.5)
plt.title("Imaginary Part of Signal")
plt.xlabel("Time (t)")
plt.grid(True)

plt.tight_layout()
plt.savefig("real.png")
plt.close()