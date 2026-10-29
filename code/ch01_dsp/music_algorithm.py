"""
1.8 矩阵分解应用 - 代码实验

本实验演示：
1. SVD分解
2. 特征值分解（EVD）
3. MUSIC算法（频率估计）
4. 子空间方法
5. 低秩近似和去噪
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, signal

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)

print("=" * 60)
print("1.8 Matrix Decomposition Demo")
print("=" * 60)

# ============================================================================
# 1. Generate multi-signal observations
# ============================================================================
print("\n1. Generate multi-signal observations")
print("-" * 60)

N = 256
signal_count = 2
subspace_dim = 4
f1, f2 = 0.100, 0.145
A1, A2 = 1.0, 0.75
SNR_dB = 12
SNR = 10 ** (SNR_dB / 10)

t = np.arange(N)
s1 = A1 * np.exp(1j * 2 * np.pi * f1 * t)
s2 = A2 * np.exp(1j * 2 * np.pi * f2 * t + 1j * 0.35)
s_true_complex = s1 + s2
s_true = np.real(s_true_complex)

sigma2 = np.mean(s_true**2) / SNR
sigma = np.sqrt(sigma2)
w = sigma * np.random.randn(N)
y = s_true + w

print(f"Signal 1 frequency: {f1:.4f}, amplitude: {A1:.4f}")
print(f"Signal 2 frequency: {f2:.4f}, amplitude: {A2:.4f}")
print(f"Observation length: {N}")
print(f"SNR (dB): {SNR_dB:.2f}")

# ============================================================================
# 2. SVD decomposition
# ============================================================================
print("\n2. SVD decomposition")
print("-" * 60)

M = 80
L = N - M + 1
Hankel = linalg.hankel(y[:M], y[M - 1:])
U, s, Vh = linalg.svd(Hankel, full_matrices=False)

print(f"Hankel matrix shape: {Hankel.shape}")
print(f"Number of singular values: {len(s)}")
print(f"First 5 singular values: {s[:5]}")
print(f"Singular value decay ratio: {s[0] / s[-1]:.2f}")

# ============================================================================
# 3. EVD decomposition
# ============================================================================
print("\n3. EVD decomposition")
print("-" * 60)

R = Hankel @ Hankel.T / L
eigenvalues, eigenvectors = linalg.eigh(R)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"Covariance matrix shape: {R.shape}")
print(f"Number of eigenvalues: {len(eigenvalues)}")
print(f"First 5 eigenvalues: {eigenvalues[:5]}")
print(f"Eigenvalue decay ratio: {eigenvalues[0] / eigenvalues[-1]:.2f}")

# ============================================================================
# 4. MUSIC spectrum
# ============================================================================
print("\n4. MUSIC frequency estimation")
print("-" * 60)

En = eigenvectors[:, subspace_dim:]
freq_range = np.linspace(0.01, 0.25, 2400)
music_spectrum = np.zeros_like(freq_range)

for i, f in enumerate(freq_range):
    steering = np.exp(1j * 2 * np.pi * f * np.arange(M))
    denom = np.linalg.norm(En.T.conj() @ steering) ** 2
    music_spectrum[i] = 1.0 / max(denom, 1e-12)

music_spectrum_db = 10 * np.log10(music_spectrum / np.max(music_spectrum))
peak_indices, _ = signal.find_peaks(music_spectrum, distance=80)
peak_indices = peak_indices[np.argsort(music_spectrum[peak_indices])[-signal_count:]]
peak_indices = peak_indices[np.argsort(freq_range[peak_indices])]
estimated_freqs = freq_range[peak_indices]

print(f"True frequencies: {f1:.4f}, {f2:.4f}")
print(f"Estimated frequencies: {estimated_freqs[0]:.4f}, {estimated_freqs[1]:.4f}")
print(f"Frequency errors: {abs(estimated_freqs[0] - f1):.6f}, {abs(estimated_freqs[1] - f2):.6f}")

# ============================================================================
# 5. Low-rank denoising
# ============================================================================
print("\n5. Low-rank denoising")
print("-" * 60)

rank_for_denoising = 4
Hankel_approx = U[:, :rank_for_denoising] @ np.diag(s[:rank_for_denoising]) @ Vh[:rank_for_denoising, :]

def diagonal_averaging(hankel_matrix):
    rows, cols = hankel_matrix.shape
    output = np.zeros(rows + cols - 1)
    counts = np.zeros(rows + cols - 1)
    for i in range(rows):
        for j in range(cols):
            output[i + j] += hankel_matrix[i, j]
            counts[i + j] += 1
    return output / counts

y_denoised = diagonal_averaging(Hankel_approx)

mse_noisy = np.mean((y - s_true) ** 2)
mse_denoised = np.mean((y_denoised - s_true) ** 2)
snr_improvement = 10 * np.log10(mse_noisy / mse_denoised)

print(f"Noisy-signal MSE: {mse_noisy:.6f}")
print(f"Denoised-signal MSE: {mse_denoised:.6f}")
print(f"SNR improvement: {snr_improvement:.2f} dB")

# ============================================================================
# 6. PCA-like feature extraction
# ============================================================================
print("\n6. Subspace feature extraction")
print("-" * 60)

features = eigenvectors[:, :subspace_dim].T @ Hankel
print(f"Feature shape: {features.shape}")
print(f"Feature variances: {np.var(features, axis=1)}")

# ============================================================================
# 7. Visualization
# ============================================================================
print("\n7. Generate visualizations")
print("-" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

axes[0, 0].plot(t, s_true, 'g-', linewidth=2, label='True Signal')
axes[0, 0].plot(t, y, 'b-', linewidth=0.7, alpha=0.7, label='Noisy Observation')
axes[0, 0].set_title('Observed Signal')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].semilogy(range(1, len(s) + 1), s, 'b-o', linewidth=2, markersize=3)
axes[0, 1].axvline(x=rank_for_denoising, color='r', linestyle='--', linewidth=2, label=f'Rank={rank_for_denoising}')
axes[0, 1].set_title('SVD Singular Values')
axes[0, 1].set_xlabel('Index')
axes[0, 1].set_ylabel('Singular Value')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, which='both')

axes[0, 2].semilogy(range(1, len(eigenvalues) + 1), eigenvalues, 'r-s', linewidth=2, markersize=3)
axes[0, 2].axvline(x=subspace_dim, color='b', linestyle='--', linewidth=2, label=f'Signal subspace={subspace_dim}')
axes[0, 2].set_title('EVD Eigenvalues')
axes[0, 2].set_xlabel('Index')
axes[0, 2].set_ylabel('Eigenvalue')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3, which='both')

axes[1, 0].plot(freq_range, music_spectrum_db, 'b-', linewidth=1.2)
axes[1, 0].plot(estimated_freqs, music_spectrum_db[peak_indices], 'ro', markersize=7, label='Estimated Peaks')
axes[1, 0].axvline(x=f1, color='g', linestyle='--', linewidth=1.5, label='True Frequencies')
axes[1, 0].axvline(x=f2, color='g', linestyle='--', linewidth=1.5)
axes[1, 0].set_title('MUSIC Spectrum')
axes[1, 0].set_xlabel('Frequency')
axes[1, 0].set_ylabel('Relative Power (dB)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t, s_true, 'g-', linewidth=2, label='True Signal')
axes[1, 1].plot(t, y, 'b-', linewidth=0.5, alpha=0.45, label='Noisy')
axes[1, 1].plot(t, y_denoised, 'r-', linewidth=1.2, alpha=0.9, label='Denoised')
axes[1, 1].set_title('Denoising Effect')
axes[1, 1].set_xlabel('Time')
axes[1, 1].set_ylabel('Amplitude')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

freq_axis = np.fft.rfftfreq(N, d=1.0)
Y_fft = np.abs(np.fft.rfft(y))
Y_denoised_fft = np.abs(np.fft.rfft(y_denoised))
axes[1, 2].plot(freq_axis, Y_fft, 'b-', linewidth=1, label='Noisy')
axes[1, 2].plot(freq_axis, Y_denoised_fft, 'r-', linewidth=1, label='Denoised')
axes[1, 2].axvline(x=f1, color='g', linestyle='--', linewidth=1.5, label='True Frequencies')
axes[1, 2].axvline(x=f2, color='g', linestyle='--', linewidth=1.5)
axes[1, 2].set_title('Frequency-Domain Comparison')
axes[1, 2].set_xlabel('Frequency')
axes[1, 2].set_ylabel('Amplitude')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/ch01_matrix_decomposition.png', dpi=150, bbox_inches='tight')
print('Figure saved to assets/ch01_matrix_decomposition.png')

# ============================================================================
# 8. Subspace diagnostics
# ============================================================================
print("\n8. Subspace diagnostics")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

normalized_eigenvalues = eigenvalues / eigenvalues[0]
axes[0].semilogy(
    range(1, 13), normalized_eigenvalues[:12], "o-", color="#2563eb", linewidth=2
)
axes[0].axvspan(0.5, subspace_dim + 0.5, color="#16a34a", alpha=0.12, label="Signal subspace")
axes[0].axvspan(subspace_dim + 0.5, 12.5, color="#64748b", alpha=0.12, label="Noise subspace")
axes[0].axvline(subspace_dim + 0.5, color="#dc2626", linestyle="--", linewidth=1.5)
axes[0].set_xlabel("Eigenvalue index")
axes[0].set_ylabel("Normalized eigenvalue")
axes[0].set_title("Step 1: Split Signal and Noise Subspaces")
axes[0].set_xlim(0.5, 12.5)
axes[0].legend()
axes[0].grid(True, alpha=0.3, which="both")

axes[1].plot(freq_range, music_spectrum_db, color="#7c3aed", linewidth=1.5)
axes[1].plot(
    estimated_freqs,
    music_spectrum_db[peak_indices],
    "o",
    color="#dc2626",
    markersize=7,
    label="Estimated peaks",
)
axes[1].axvline(f1, color="#16a34a", linestyle="--", linewidth=1.5, label="True frequencies")
axes[1].axvline(f2, color="#16a34a", linestyle="--", linewidth=1.5)
axes[1].set_xlabel("Normalized frequency")
axes[1].set_ylabel("Relative pseudospectrum (dB)")
axes[1].set_title("Step 2: Search Steering Vectors")
axes[1].set_ylim(-55, 3)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/ch01_matrix_decomposition_performance.png', dpi=150, bbox_inches='tight')
print('Diagnostic figure saved to assets/ch01_matrix_decomposition_performance.png')

print("\n" + "=" * 60)
print("Experiment completed!")
print("=" * 60)
