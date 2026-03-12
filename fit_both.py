import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.signal import savgol_filter


def generate_f(x, xc, a, beta, c, sigma=0.001):
    """
    Generate f(x) = a*(xc - x)^beta + c + epsilon(x)
    """
    assert np.all(x < xc), "Need x < xc for (xc - x)^beta to be real"
    f_clean = a * (xc - x) ** beta + c
    noise = np.random.normal(0.0, sigma, size=x.shape)
    f_noisy = f_clean + noise
    return f_noisy, f_clean


def generate_f_prime(x, xc, a, beta, sigma=0.001):
    """
    Generate f'(x) = -a*beta*(xc - x)^(beta-1) + epsilon(x)
    """
    assert np.all(x < xc), "Need x < xc for (xc - x)^(beta-1) to be real"
    fp_clean = -a * beta * (xc - x) ** (beta - 1)
    noise = np.random.normal(0.0, sigma, size=x.shape)
    fp_noisy = fp_clean + noise
    return fp_noisy, fp_clean


def esprit(y, K, dx):
    """
    Standard ESPRIT algorithm to fit y as a sum of K exponentials.
    Returns the exponential rates (t) and complex weights (w).
    """
    N = len(y)
    L = N // 2

    H = np.zeros((L, N - L + 1), dtype=complex)
    for i in range(L):
        H[i, :] = y[i:i + N - L + 1]

    U, S, Vh = scipy.linalg.svd(H, full_matrices=False)
    Us = U[:, :K]

    U1 = Us[:-1, :]
    U2 = Us[1:, :]

    Psi = np.linalg.pinv(U1) @ U2
    Z = np.linalg.eigvals(Psi)

    Z_sorted = Z[np.argsort(-np.abs(Z))]
    print(f"  [ESPRIT] K={K} Eigenvalues (sorted by magnitude):")
    for i, val in enumerate(Z_sorted):
        print(f"    {i+1}: {val.real:+.6f} {val.imag:+.6f}j  (mag: {np.abs(val):.6f})")

    t = np.log(Z) / dx

    x_idx = np.arange(N) * dx
    V = np.exp(np.outer(x_idx, t))
    w = np.linalg.lstsq(V, y, rcond=None)[0]

    return t, w


def fit_esprit(x_data, y_data, K):
    """
    Fit y_data using ESPRIT exponential sum.
    Returns c_est, f_recon, df_recon, rms.

    The constant c is extracted via the differential-equation identity:
        f = c + (f'/f) integrated  =>  f = c + b1*f' + b2*(x*f')
    where b0 = c from linear regression.

    The analytical derivative df/dx is computed from the exponential sum
    as V @ (w * t).
    """
    dx = x_data[1] - x_data[0]

    # Pre-smooth
    window_length = len(y_data) // 4
    if window_length % 2 == 0:
        window_length += 1
    y_smooth = savgol_filter(y_data, window_length=window_length, polyorder=3)

    # Shift x to start at 0
    x0 = x_data[0]
    x_s = x_data - x0

    # ESPRIT
    t, w = esprit(y_smooth, K, dx)

    # Reconstruct f and analytical derivative f'
    V = np.exp(np.outer(x_s, t))
    f_recon = np.real(V @ w)
    df_recon = np.real(V @ (w * t))

    # Linear regression: f = b0 + b1*f' + b2*(x_s * f')
    # b0 = c (the constant offset)
    X1 = df_recon
    X2 = x_s * df_recon
    A_mat = np.column_stack([np.ones_like(x_s), X1, X2])
    b, _, _, _ = np.linalg.lstsq(A_mat, f_recon, rcond=None)

    c_est = b[0]

    rms = np.sqrt(np.mean((y_data - f_recon) ** 2))
    return c_est, f_recon, df_recon, rms


# ================================================================
# Configuration
# ================================================================
xc_true = 1.0
beta_true = 0.5
c_true = 2.0
a_true = 1.0
sigma0 = 0.001
sigma1 = 0.001
M = 400
K_end = 10
best_K_f = 4
best_K_fp = 4

x = np.linspace(0.8, 0.9, M)
np.random.seed(42)

f_noisy, f_clean = generate_f(x, xc_true, a_true, beta_true, c_true, sigma0)
fp_noisy, fp_clean = generate_f_prime(x, xc_true, a_true, beta_true, sigma1)

print("=" * 60)
print(f"True: xc={xc_true}, beta={beta_true}, a={a_true}, c={c_true}")
print(f"Noise: sigma_f={sigma0}, sigma_fp={sigma1}, M={M}")
print("=" * 60)

# ================================================================
# Sweep K for f(x)
# ================================================================
print("\n--- Fitting f(x): sweep over K ---")
print(f"{'K':<5} {'c_est':<12} {'RMS':<12}")

best_f_results = None
for K in range(2, K_end):
    try:
        res = fit_esprit(x, f_noisy, K)
        c_e, _, _, rms = res
        print(f"{K:<5} {c_e:<12.6f} {rms:<12.2e}\n")
        if K == best_K_f:
            best_f_results = res
    except Exception as e:
        print(f"{K:<5} FAILED: {e}\n")

# ================================================================
# Sweep K for f'(x)
# ================================================================
print("\n--- Fitting f'(x): sweep over K ---")
print(f"{'K':<5} {'cp_est':<14} {'RMS':<12}")

best_fp_results = None
for K in range(2, K_end):
    try:
        res = fit_esprit(x, fp_noisy, K)
        cp_e, _, _, rms = res
        print(f"{K:<5} {cp_e:<14.6e} {rms:<12.2e}\n")
        if K == best_K_fp:
            best_fp_results = res
    except Exception as e:
        print(f"{K:<5} FAILED: {e}\n")

# ================================================================
# Report best results
# ================================================================
print("\n" + "=" * 60)
print("  ESPRIT Fit Results")
print("=" * 60)

c_f, f_recon, df_recon, rms_f = best_f_results
print(f"\n  From f(x)  [K={best_K_f}]:")
print(f"    c    = {c_f:.6f}  (true: {c_true})")
print(f"    RMS  = {rms_f:.2e}")
f_recond_minus_c=f_recon-c_f
cp_fp, fp_recon, dfp_recon, rms_fp = best_fp_results
print(f"\n  From f'(x) [K={best_K_fp}]:")
print(f"    c'   = {cp_fp:.6e}  (true: 0.0)")
print(f"    RMS  = {rms_fp:.2e}")
fp_recon_minus_c=fp_recon-cp_fp

# ================================================================
# Plotting: 2x2 grid — f and f' fitting only
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- Top Left: f(x) comparison ---
ax = axes[0, 0]
ax.plot(x, f_clean, 'b-', linewidth=2, label='Exact $f(x)$')
ax.plot(x, f_noisy, color='gray', alpha=0.3, linewidth=0.5, label='Noisy data')
ax.plot(x, f_recon, 'r--', linewidth=2, label=f'ESPRIT (K={best_K_f})')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('$f(x)$: ESPRIT Fit vs Exact')
ax.legend()
ax.grid(True, alpha=0.3)
# print(f"f_recon={f_recon}")
# --- Top Right: f(x) absolute error ---
ax = axes[0, 1]
ax.plot(x, np.abs(f_recon - f_clean), 'r-', linewidth=1.5, label='|ESPRIT $-$ Exact|')
ax.set_xlabel('x')
ax.set_ylabel('Absolute Error')
ax.set_title(f'$f(x)$: Absolute Error  (RMS={rms_f:.2e})')
ax.legend()
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# --- Bottom Left: f'(x) comparison ---
ax = axes[1, 0]
ax.plot(x, fp_clean, 'b-', linewidth=2, label="Exact $f'(x)$")
ax.plot(x, fp_noisy, color='gray', alpha=0.3, linewidth=0.5, label='Noisy data')
ax.plot(x, fp_recon, 'r--', linewidth=2, label=f"ESPRIT (K={best_K_fp})")
ax.set_xlabel('x')
ax.set_ylabel("f'(x)")
ax.set_title("$f'(x)$: ESPRIT Fit vs Exact")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Bottom Right: f'(x) absolute error ---
ax = axes[1, 1]
ax.plot(x, np.abs(fp_recon - fp_clean), 'r-', linewidth=1.5,
        label="|ESPRIT $-$ Exact|")
ax.set_xlabel('x')
ax.set_ylabel('Absolute Error')
ax.set_title(f"$f'(x)$: Absolute Error  (RMS={rms_fp:.2e})")
ax.legend(fontsize=8)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

fig.suptitle(
    f'ESPRIT Fitting:  $c$={c_f:.4f} (true {c_true}),  '
    f"$c'$={cp_fp:.2e} (true 0.0)",
    fontsize=13, y=1.01
)

plt.tight_layout()
plt.savefig('fit_both_results.png', dpi=150, bbox_inches='tight')
print("\nFigure saved to fit_both_results.png")