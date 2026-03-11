import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.signal import savgol_filter

def generate_data(x, xc, gamma, c, A=1.0, sigma=0.001):
    """Generate power-law singularity data with noise."""
    y_exact = A * (xc - x)**(-gamma) + c
    noise = np.random.normal(0, sigma, size=len(x))
    return y_exact + noise

def esprit(y, K, dx):
    """
    Standard ESPRIT algorithm to fit y as a sum of K exponentials.
    Returns the exponential rates (t) and complex weights (w).
    """
    N = len(y)
    L = N // 2  # Pencil parameter (half the data length is usually optimal)

    # 1. Build Hankel matrix
    H = np.zeros((L, N - L + 1), dtype=complex)
    for i in range(L):
        H[i, :] = y[i:i + N - L + 1]

    # 2. SVD
    U, S, Vh = scipy.linalg.svd(H, full_matrices=False)
    Us = U[:, :K]

    # 3. Shift matrices
    U1 = Us[:-1, :]
    U2 = Us[1:, :]

    # 4. Solve for rotation matrix Psi (Total Least Squares / Pseudo-inverse)
    Psi = np.linalg.pinv(U1) @ U2

    # 5. Eigenvalues give the poles
    Z = np.linalg.eigvals(Psi)

    # 6. Convert poles to continuous rates
    t = np.log(Z) / dx

    # 7. Solve for weights using linear least squares
    x_idx = np.arange(N) * dx
    V = np.exp(np.outer(x_idx, t))
    w = np.linalg.lstsq(V, y, rcond=None)[0]

    return t, w

def fit_singularity(x_data, y_data, K):
    """
    Fits f(x) = c + A*(xc - x)^(-gamma) using ESPRIT + Linear Regression.
    """
    dx = x_data[1] - x_data[0]

    # --- FIX 1: Pre-smooth the data to handle higher noise ---
    # A Savitzky-Golay filter removes high-frequency noise that would otherwise
    # cause wild oscillations when calculating the analytical derivative later.
    window_length = len(y_data) // 4
    if window_length % 2 == 0:
        window_length += 1 # Window length must be odd
    y_smooth = savgol_filter(y_data, window_length=window_length, polyorder=3)

    # --- FIX 2: Shift X-axis to start at 0 to prevent exp() overflow ---
    x0 = x_data[0]
    x_shifted = x_data - x0

    # 1. Run ESPRIT on the SMOOTHED data
    t, w = esprit(y_smooth, K, dx)

    # 2. Reconstruct the smoothed function and its analytical derivative
    V_recon = np.exp(np.outer(x_shifted, t))
    f_recon = np.real(V_recon @ w)
    df_recon = np.real(V_recon @ (w * t))

    # 3. Extract parameters using the differential equation:
    # f(x) = c + (xc_shifted / gamma) * f'(x) - (1 / gamma) * x * f'(x)
    X1 = df_recon
    X2 = x_shifted * df_recon

    A_mat = np.column_stack((np.ones_like(x_shifted), X1, X2))
    beta, _, _, _ = np.linalg.lstsq(A_mat, f_recon, rcond=None)

    c_est = beta[0]
    gamma_est = -1.0 / beta[2]
    xc_shifted_est = beta[1] * gamma_est

    # Unshift xc to get the true critical point
    xc_est = xc_shifted_est + x0

    # Calculate RMS error of the fit against the original noisy data
    rms = np.sqrt(np.mean((y_data - f_recon)**2))

    return xc_est, gamma_est, c_est, f_recon, df_recon, rms


# --- Configuration ---
xc_true = 1.0
gamma_true = 0.5
c_true = 2.0
sigma = 0.001  # Increased noise to 0.01
M = 400

x = np.linspace(0.8, 0.9, M)
np.random.seed(42) # For reproducibility
y = generate_data(x, xc_true, gamma_true, c_true, A=1.0, sigma=sigma)

print("============================================================")
print(f"True Parameters: xc = {xc_true}, gamma = {gamma_true}, c = {c_true}")
print(f"Noise Level (sigma) = {sigma}")
print("============================================================\n")

print("--- Sweep over K ---")
print(f"{'K':<5} {'xc_est':<12} {'gamma_est':<12} {'c_est':<12} {'RMS':<12}")

best_K = 3
best_f_recon = None
best_df_recon = None
best_c_est = None
best_xc_est = None
best_gamma_est = None

# Sweep K from 2 to 6
for K in range(2, 7):
    try:
        xc_est, gamma_est, c_est, f_recon, df_recon, rms = fit_singularity(x, y, K)
        print(f"{K:<5} {xc_est:<12.6f} {gamma_est:<12.6f} {c_est:<12.6f} {rms:<12.2e}")
        if K == best_K:
            best_f_recon = f_recon
            best_df_recon = df_recon
            best_c_est = c_est
            best_xc_est = xc_est
            best_gamma_est = gamma_est
    except Exception as e:
        print(f"{K:<5} FAILED: {e}")

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: The function fit
ax1.plot(x, y, '.', label='Noisy Data ($\\sigma=0.01$)', alpha=0.3)
y_exact = generate_data(x, xc_true, gamma_true, c_true, sigma=0)
ax1.plot(x, y_exact, 'k--', label='True Function')

if best_f_recon is not None:
    ax1.plot(x, best_f_recon, 'r-', linewidth=2, label=f'ESPRIT Fit (K={best_K})')

ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.set_title('Singularity Fitting using ESPRIT (with Pre-smoothing)')
ax1.legend()
ax1.grid(True)

# Plot 2: The ratio (f - c) / f'
# True ratio: (xc - x) / gamma
true_ratio = (xc_true - x) / gamma_true
ax2.plot(x, true_ratio, 'k--', label=f'True Ratio: $(x_c - x) / \\gamma$\n($x_c={xc_true}$, $\\gamma={gamma_true}$)')

if best_f_recon is not None and best_df_recon is not None:
    # Reconstructed ratio: (f_recon - c_est) / df_recon
    recon_ratio = (best_f_recon - best_c_est) / best_df_recon

    # Add the estimated parameters to the label
    label_str = f'Reconstructed Ratio (K={best_K})\nEst: $x_c={best_xc_est:.4f}$, $\\gamma={best_gamma_est:.4f}$'
    ax2.plot(x, recon_ratio, 'r-', linewidth=2, label=label_str)

ax2.set_xlabel('x')
ax2.set_ylabel('$(f(x) - c) / f\'(x)$')
ax2.set_title('Linearization of the Singularity')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("critical_exponent_fit_robust.png")
print("\nFigure saved to critical_exponent_fit_robust.png")
# plt.show()