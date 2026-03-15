import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.signal import savgol_filter


def generate_f(x, xc, gamma, c, A=1.0, sigma=0.001):
    y_exact = A * (xc - x)**(-gamma) + c
    noise = np.random.normal(0, sigma, size=len(x))
    return y_exact + noise


def generate_fp(x, xc, gamma, c, A=1.0, sigma=0.001):
    yp_exact=gamma*A*(xc - x)**(-gamma-1)
    noise = np.random.normal(0, sigma, size=len(x))
    return yp_exact+noise


def make_tridiag_C0(n):
    """
    Build the n×n tridiagonal noise-covariance kernel C_0 (Eq. 285):
        C_0 = tridiag(-1, 2, -1)
    arising from the differenced noise  η_m = ε(x_{m+1}) - ε(x_m).
    """
    return 2.0 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)


def esprit(y, K, dx):
    """
    Noise-whitened TLS-ESPRIT following Section E.

    The input y contains the differenced data
        g_m = f(x_{m+1}) - f(x_m),   m = 0, …, N-1       (Eq. 276)
    whose signal part is  Σ_j  W_j exp(t_j m Δx)  and whose noise
    η_m = ε(x_{m+1}) - ε(x_m) has covariance  σ² C_0  with
    C_0 = tridiag(-1, 2, -1).

    Algorithm
    ---------
    1.  Cholesky-factorise C_0 = A A^T  (Eq. 286)
    2.  Build Hankel matrix H from g_m  (Eq. 279)
    3.  Whiten each column  h̃_m = A^{-1} h_m  (Eq. 295)
    4.  Form whitened covariance  R̃ = (1/P) Σ h̃_m h̃_m^T  (Eq. 297)
    5.  Eigen-decompose R̃ = Ũ Λ̃ Ũ†  (Eq. 298)
    6.  Un-whiten  U = A Ũ  (Eq. 302)
    7.  Signal subspace U_s (first K columns of U)  (Eq. 304)
    8.  Partition U_s into U_{s1}, U_{s2}  (Eqs. 310-311)
    9.  TLS-ESPRIT: SVD of [U_{s1}, U_{s2}], extract Ψ_TLS  (Eqs. 315-324)
    10. Eigenvalues of Ψ_TLS → rates  t_j = log(μ_j)/Δx  (Eq. 326)
    11. Whitened OLS for weights W_j  (Eqs. 331-341)

    Parameters
    ----------
    y  : real array, shape (N,)
         Differenced data g_m.  N = M-1  where M = number of data points.
    K  : int
         Number of exponential components to extract.
    dx : float
         Uniform spacing  Δx  between original sample points.

    Returns
    -------
    t : complex array, shape (K,)
        Exponential rates.
    W : complex array, shape (K,)
        Weights in  g_m = Σ_j W_j exp(t_j m Δx).
    """
    N = len(y)
    L = N // 2                                      # Hankel row dimension
    P = N + 1 - L                                   # Hankel col dimension (Eq. 278: P = M - L)

    assert K < L and L < N - K, (
        f"Need K < L < N-K, got K={K}, L={L}, N-K={N-K}")

    # ==================================================================
    # Step 1  –  Cholesky of L×L noise covariance  (Eq. 286)
    #            C_0^{(L)} = A_L  A_L^T
    # ==================================================================
    C0_L = make_tridiag_C0(L)
    A_L = scipy.linalg.cholesky(C0_L, lower=True)

    # ==================================================================
    # Step 2  –  Hankel matrix  H ∈ R^{L×P}  (Eq. 279)
    #            H[i, s] = g_{i+s}
    # ==================================================================
    H = np.zeros((L, P))
    for s in range(P):
        H[:, s] = y[s:s + L]

    # ==================================================================
    # Step 3  –  Whiten each column:  h̃_s = A_L^{-1} h_s  (Eq. 295)
    # ==================================================================
    H_white = scipy.linalg.solve_triangular(A_L, H, lower=True)

    # ==================================================================
    # Step 4  –  Whitened covariance  R̃ = (1/P) Σ_s h̃_s h̃_s^T  (Eq. 297)
    # ==================================================================
    R_tilde = (1.0 / P) * (H_white @ H_white.T)

    # ==================================================================
    # Step 5  –  Eigen-decompose  R̃ = Ũ Λ̃ Ũ†  (Eq. 298)
    #            eigenvalues in *decreasing* order  (Eq. 300)
    # ==================================================================
    eigvals, U_tilde = np.linalg.eigh(R_tilde)
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    U_tilde = U_tilde[:, idx]
    print(f"eigvals[:K]={eigvals[:K]}")
    # ==================================================================
    # Step 6  –  Un-whiten:  U = A_L  Ũ  (Eq. 302)
    # ==================================================================
    U_mat = A_L @ U_tilde

    # ==================================================================
    # Step 7  –  Signal subspace  U_s  (first K columns)  (Eq. 304)
    # ==================================================================
    Us = U_mat[:, :K]

    # ==================================================================
    # Step 8  –  Partition U_s  (Eqs. 310-311)
    #            U_{s1} = first L-1 rows,   U_{s2} = last L-1 rows
    #            shift-invariance:  U_{s2} = U_{s1} Ψ   (Eq. 313)
    # ==================================================================
    Us1 = Us[:-1, :]                                # (L-1) × K
    Us2 = Us[1:, :]                                 # (L-1) × K

    # ==================================================================
    # Step 9  –  TLS-ESPRIT  (Eqs. 315-324)
    #   U_xy = [U_{s1}, U_{s2}]  ∈ C^{(L-1)×2K}
    #   SVD:  U_xy = V^L  Σ  (V^R)†              (Eq. 317)
    #   Partition V^R into 4  K×K  blocks         (Eq. 321)
    #   Ψ_TLS = -V^R_{01} (V^R_{11})^{-1}        (Eq. 324)
    # ==================================================================
    Uxy = np.hstack([Us1, Us2])                     # (L-1) × 2K

    _, _, Vh = scipy.linalg.svd(Uxy, full_matrices=False)
    # numpy convention:  Uxy = U_svd  diag(s)  Vh
    # document:          Uxy = V^L    Σ        (V^R)†
    # so  V^R = Vh^H
    VR = Vh.conj().T                                # 2K × 2K

    V01 = VR[:K, K:]                                # top-right    K × K
    V11 = VR[K:, K:]                                # bottom-right K × K

    Psi_TLS = -V01 @ np.linalg.inv(V11)             # (Eq. 324)

    # ==================================================================
    # Step 10  –  Eigenvalues → rates  (Eqs. 325-326)
    #             t_j = log(μ_j) / Δx
    # ==================================================================
    mu = np.linalg.eigvals(Psi_TLS)
    t = np.log(np.abs(mu)) / dx

    mu_sorted = mu[np.argsort(-np.abs(mu))]
    print(f"  [ESPRIT] K={K} Eigenvalues (sorted by magnitude):")
    for i, val in enumerate(mu_sorted):
        print(f"    {i+1}: {val.real:+.6f} {val.imag:+.6f}j  "
              f"(mag: {np.abs(val):.6f})")

    # ==================================================================
    # Step 11  –  Whitened OLS for weights W  (Eqs. 331-341)
    #   G[m, j] = exp(t_j · m · Δx),  m = 0, …, N-1        (Eq. 332)
    #   C_0^{(N)} of size N  →  Cholesky A_f
    #   G̃ = A_f^{-1} G,   g̃ = A_f^{-1} g                  (Eqs. 338-339)
    #   W = (G̃^T G̃)^{-1} G̃^T g̃                           (Eq. 341)
    # ==================================================================
    m_idx = np.arange(N)
    G = np.exp(np.outer(m_idx, t * dx))             # N × K  (complex)

    C0_N = make_tridiag_C0(N)
    A_f = scipy.linalg.cholesky(C0_N, lower=True)

    g_tilde = scipy.linalg.solve_triangular(
        A_f, y.astype(complex), lower=True)
    G_tilde = scipy.linalg.solve_triangular(
        A_f, G, lower=True)

    W = np.linalg.lstsq(G_tilde, g_tilde, rcond=None)[0]

    return t, W


def reconstruct_f(x_data, y_data, K):
    """
    Reconstruct f(x) = sum_j a_j exp(t_j (x - x0)) + c  using ESPRIT
    on the differenced, smoothed data.

    Parameters
    ----------
    x_data : array, shape (M,)
        Uniformly spaced sample points.
    y_data : array, shape (M,)
        Noisy observations f(x_m) + epsilon.
    K : int
        Number of exponential components.

    Returns
    -------
    f_recon : array, shape (M,)
        Reconstructed function values at x_data.
    c_est : float
        Estimated additive constant.
    """
    dx = x_data[1] - x_data[0]
    M = len(y_data)

    # Pre-smooth
    window_length = M // 4
    if window_length % 2 == 0:
        window_length += 1
    y_smooth = savgol_filter(y_data, window_length=window_length, polyorder=3)

    # Shift x-axis to start at 0
    x0 = x_data[0]
    x_shifted = x_data - x0

    # Differences to remove constant c
    g = np.diff(y_smooth)

    # ESPRIT on differences
    t, W = esprit(g, K, dx)
    print(f"K={K}, t={t}")
    print(f"W={W}")
    # Recover weights a_j from differenced weights W_j
    w_shifted = W / (np.exp(t * dx) - 1)

    # Reconstruct f(x') - c
    V_recon = np.exp(np.outer(x_shifted, t))
    f_minus_c = np.real(V_recon @ w_shifted)

    # Estimate c from residuals
    c_est = np.mean(y_data - f_minus_c)

    f_recon = f_minus_c + c_est

    return f_recon, c_est


def reconstruct_fp(x_data, yp_data, K):
    """
    Reconstruct f'(x) = sum_j a_j exp(t_j (x - x0)) + c  using ESPRIT
    on the differenced, smoothed derivative data.

    Parameters
    ----------
    x_data : array, shape (M,)
        Uniformly spaced sample points.
    yp_data : array, shape (M,)
        Noisy observations f'(x_m) + epsilon.
    K : int
        Number of exponential components.

    Returns
    -------
    fp_recon : array, shape (M,)
        Reconstructed derivative values at x_data.
    cp_est : float
        Estimated additive constant.
    """
    dx = x_data[1] - x_data[0]
    M = len(yp_data)

    # Pre-smooth
    window_length = M // 4
    if window_length % 2 == 0:
        window_length += 1
    yp_smooth = savgol_filter(yp_data, window_length=window_length, polyorder=3)

    # Shift x-axis to start at 0
    x0 = x_data[0]
    x_shifted = x_data - x0

    # Differences to remove constant
    g = np.diff(yp_smooth)

    # ESPRIT on differences
    t, W = esprit(g, K, dx)
    print(f"K={K}, t={t}")

    # Recover weights a_j from differenced weights W_j
    w_shifted = W / (np.exp(t * dx) - 1)

    # Reconstruct f'(x') - c
    V_recon = np.exp(np.outer(x_shifted, t))
    fp_minus_c = np.real(V_recon @ w_shifted)

    # Estimate c from residuals
    cp_est = np.mean(yp_data - fp_minus_c)
    cp_var=np.var(yp_data - fp_minus_c)

    fp_recon = fp_minus_c + cp_est

    return fp_recon, cp_est

# ================================================================
# Configuration
# ================================================================
xc_true = 1.0
gamma_true = 0.5
c_true = 2.0
A_true = 1.0
sigma0 = 0.001
sigma1 = 0.1
M = 400
K0 = 2
K1=2
np.random.seed(42)
x = np.linspace(0.8, 0.9, M)

y_noisy = generate_f(x, xc_true, gamma_true, c_true, A=A_true, sigma=sigma0)
yp_noisy=generate_fp(x, xc_true, gamma_true, c_true, A=A_true, sigma=sigma1)
y_exact = A_true * (xc_true - x)**(-gamma_true) + c_true
yp_exact = gamma_true * A_true * (xc_true - x)**(-gamma_true - 1)

# ================================================================
# Reconstruct f and f'
# ================================================================
f_recon, c_est = reconstruct_f(x, y_noisy, K0)
fp_recon, cp_est = reconstruct_fp(x, yp_noisy, K1)
print(f"c_est={c_est}, cp_est={cp_est}")
rms_f = np.sqrt(np.mean((y_exact - f_recon)**2))
rms_fp = np.sqrt(np.mean((yp_exact - fp_recon)**2))

print(f"True c = {c_true},  Estimated c = {c_est:.6f}")
print(f"RMS error  f : {rms_f:.2e}")
print(f"RMS error  f': {rms_fp:.2e}")

# ================================================================
# Plot
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Top-left: f(x) fit ---
ax = axes[0, 0]
ax.plot(x, y_noisy, '.', color='grey', alpha=0.3, markersize=2,
        label=f'Noisy data ($\\sigma={sigma0}$)')
ax.plot(x, y_exact, 'k--', lw=2.5, label='True $f(x)$')
ax.plot(x, f_recon, 'r-', lw=2, label=f'ESPRIT reconstruction (K={K0})')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Function Reconstruction')
ax.legend()
ax.grid(True)

# --- Top-right: f(x) residual ---
ax = axes[0, 1]
ax.plot(x, y_exact - f_recon, '.', color='steelblue', alpha=0.4, markersize=2)
ax.axhline(0, color='k', ls='--', lw=0.8)
ax.set_xlabel('x')
ax.set_ylabel('$f_{\\mathrm{true}} - f_{\\mathrm{recon}}$')
ax.set_title(f'$f$ Residuals  (RMS = {rms_f:.2e})')
ax.grid(True)

# --- Bottom-left: f'(x) fit ---
ax = axes[1, 0]
ax.plot(x, yp_exact, 'k--', lw=2.5, label="True $f'(x)$")
ax.plot(x, fp_recon, 'r-', lw=2, label=f'ESPRIT reconstruction (K={K0})')
ax.set_xlabel('x')
ax.set_ylabel("f'(x)")
ax.set_title('Derivative Reconstruction')
ax.legend()
ax.grid(True)

# --- Bottom-right: f'(x) residual ---
ax = axes[1, 1]
ax.plot(x, yp_exact - fp_recon, '.', color='darkorange', alpha=0.4, markersize=2)
ax.axhline(0, color='k', ls='--', lw=0.8)
ax.set_xlabel('x')
ax.set_ylabel("$f'_{\\mathrm{true}} - f'_{\\mathrm{recon}}$")
ax.set_title(f"$f'$ Residuals  (RMS = {rms_fp:.2e})")
ax.grid(True)

plt.tight_layout()
plt.savefig("reconstruct_f_and_fp.png", dpi=150)
print("\nFigure saved to reconstruct_f_and_fp.png")

# ================================================================
# Third plot: ratio (f - c) / (f' - cp)
# ================================================================
# Exact:  (f - c) / f' = (xc - x) / gamma   (linear in x)
ratio_exact = (xc_true - x) / gamma_true
ratio_recon = (f_recon - c_est) / (fp_recon - cp_est)

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# --- Left: ratio ---
ax = axes2[0]
ax.plot(x, ratio_exact, 'k--', lw=2.5, label=r'Exact $(x_c - x)/\gamma$')
ax.plot(x, np.real(ratio_recon), 'r-', lw=2, label=r'$(f - c_{\rm est})/(f^\prime - c^\prime_{\rm est})$')
ax.set_xlabel('x')
ax.set_ylabel(r'$(f - c)\;/\;f^\prime$')
ax.set_title('Ratio  $(f - c) / (f^\\prime - c^\\prime)$')
ax.legend()
ax.grid(True)

# --- Right: residual ---
ax = axes2[1]
ratio_residual = np.real(ratio_recon) - ratio_exact
rms_ratio = np.sqrt(np.mean(ratio_residual**2))
ax.plot(x, ratio_residual, '.', color='green', alpha=0.4, markersize=2)
ax.axhline(0, color='k', ls='--', lw=0.8)
ax.set_xlabel('x')
ax.set_ylabel('Ratio residual')
ax.set_title(f'Ratio Residuals  (RMS = {rms_ratio:.2e})')
ax.grid(True)

plt.tight_layout()
plt.savefig("ratio_f_over_fp.png", dpi=150)
print(f"Ratio figure saved to ratio_f_over_fp.png")

# --- Extract xc and gamma from a linear fit to the ratio ---
# (f - c) / (f' - cp) = (xc - x) / gamma = -x/gamma + xc/gamma
#   slope = -1/gamma,   intercept = x
# --- Extract xc and gamma from a linear fit to the ratio ---
# (f - c_est) / (f' - cp_est) ≈ (xc - x) / gamma = -x/gamma + xc/gamma
#   slope = -1/gamma,   intercept = xc/gamma
coeffs = np.polyfit(x, np.real(ratio_recon), 1)
slope, intercept = coeffs
gamma_est = -1.0 / slope
xc_est = intercept * gamma_est

print(f"\n{'='*50}")
print(f"Parameter extraction from ratio (f-c)/(f'-cp):")
print(f"  Linear fit:  slope = {slope:.6f},  intercept = {intercept:.6f}")
print(f"  gamma_est = {gamma_est:.6f}   (true: {gamma_true})")
print(f"  xc_est    = {xc_est:.6f}   (true: {xc_true})")
print(f"  gamma error = {abs(gamma_est - gamma_true):.2e}")
print(f"  xc error    = {abs(xc_est - xc_true):.2e}")

# --- Back out A from f - c = A * (xc - x)^(-gamma) ---
A_est = np.mean((f_recon - c_est) * (xc_est - x)**gamma_est)
print(f"  A_est     = {np.real(A_est):.6f}   (true: {A_true})")
print(f"  A error   = {abs(np.real(A_est) - A_true):.2e}")
print(f"  c_est     = {np.real(c_est):.6f}   (true: {c_true})")
print(f"{'='*50}")

# --- Overlay the linear fit on the ratio plot ---
fig3, ax3 = plt.subplots(1, 1, figsize=(7, 5))
ax3.plot(x, ratio_exact, 'k--', lw=2.5, label=r'Exact $(x_c - x)/\gamma$')
ax3.plot(x, np.real(ratio_recon), 'r-', lw=2, label='Reconstructed ratio')
ax3.plot(x, np.polyval(coeffs, x), 'b:', lw=2,
         label=f'Linear fit ($\\gamma={gamma_est:.4f}$, $x_c={xc_est:.4f}$)')
ax3.set_xlabel('x')
ax3.set_ylabel(r'$(f - c_{\rm est})\;/\;(f^\prime - c^\prime_{\rm est})$')
ax3.set_title('Ratio with linear fit for $x_c$ and $\\gamma$')
ax3.legend()
ax3.grid(True)
plt.tight_layout()
plt.savefig("ratio_linear_fit.png", dpi=150)
print("Linear fit figure saved to ratio_linear_fit.png")