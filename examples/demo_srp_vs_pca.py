import numpy as np
import time
import sys, os


from glassboxml.models.sparse_random_projection import SparseRandomProjection
from glassboxml.models.pca import PCA 

def main():
    print("==================================================")
    print(" GlassBox-ML Benchmark: SRP vs. PCA               ")
    print("==================================================\n")

    # 1. Generate a massive, high-dimensional dataset
    n_samples = 2000
    n_features = 1000
    n_components = 50 

    print(f"Generating heavy synthetic dataset: {n_samples} samples, {n_features} features...")
    # Pure random noise is perfect for a raw speed test
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features))

    print(f"Target dimensions to compress down to: {n_components}\n")

    # --- 2. Benchmark Principal Component Analysis (PCA) ---
    print("--- Testing PCA (Calculating Covariance & Eigendecomposition) ---")
    pca = PCA(n_components=n_components)
    
    start_time = time.time()
    # Note: If your PCA doesn't have fit_transform, change this to pca.fit(X) then pca.transform(X)
    X_pca = pca.fit_transform(X) 
    pca_time = time.time() - start_time
    
    print(f"-> PCA Execution Time: {pca_time:.4f} seconds")
    print(f"-> Output Shape: {X_pca.shape}\n")

    # --- 3. Benchmark Sparse Random Projection (SRP) ---
    print("--- Testing SRP (Achlioptas Sparse Matrix Multiplication) ---")
    srp = SparseRandomProjection(n_components=n_components, density='auto', random_state=42)
    
    start_time = time.time()
    X_srp = srp.fit_transform(X)
    srp_time = time.time() - start_time
    
    print(f"-> SRP Execution Time: {srp_time:.4f} seconds")
    print(f"-> Output Shape: {X_srp.shape}")
    print(srp.explain())

    # --- 4. The Verdict ---
    print("\n==================================================")
    if srp_time < pca_time:
        speedup = pca_time / (srp_time + 1e-9)
        print(f"🏆 Winner: Sparse Random Projection was {speedup:.1f}x faster!")
    else:
        print("Wait, PCA won? (Try increasing n_features to 2000+ to really see SRP shine)")
    print("==================================================")

if __name__ == "__main__":
    main()