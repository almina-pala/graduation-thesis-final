import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.express as px
import psutil
import tracemalloc

from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, ttest_rel

# Scikit-learn — Datasets & Preprocessing
from sklearn.datasets import fetch_openml, load_breast_cancer, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Scikit-learn — Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness

# Scikit-learn — Clustering & Classification
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Scikit-learn — Metric Engine
from sklearn.metrics import (
    accuracy_score, confusion_matrix, silhouette_score,
    davies_bouldin_score, adjusted_rand_score,
    normalized_mutual_info_score, calinski_harabasz_score
)

import umap

# FIX 1 — Keras import: TF >= 2.16'da tensorflow.keras çalışmaz
# try/except ile her iki yolu da dene
try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.optimizers import Adam
except ImportError:
    from keras.models import Model
    from keras.layers import Input, Dense
    from keras.optimizers import Adam

# --- REPRODUCIBILITY & ENVIRONMENT CONFIGURATION ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore")

# FIX 2 — Publication-Quality Matplotlib Configuration
# "font.family": "serif" Kaggle'da render sorunu çıkarabilir, DejaVu Serif daha güvenli
matplotlib.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Serif",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.bbox": "tight"
})

# FIX 3 — OUTPUT_DIR: Kaggle'da mutlak path kullanmak daha güvenli
OUTPUT_DIR = "/kaggle/working/output_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
for sub in ["interactive", "classification", "stability", "noise", "hyperparameter", "latex"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

print("[System] Environment fully configured for academic reporting.")

# SECTION 1 — SYSTEM PROFILING & UTILITY FUNCTIONS

def profile_execution(func, *args, **kwargs):
    """
    Profiles CPU, Wall-Clock Time, and Peak Memory allocations.
    Returns: (func_result, runtime_s, peak_mem_mb, cpu_pct)
    """
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    _ = process.cpu_percent(interval=None)  # Flush initial state
    t_start = time.perf_counter()

    result = func(*args, **kwargs)

    t_end = time.perf_counter()
    cpu_end = process.cpu_percent(interval=None)
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtime = t_end - t_start
    peak_mem_mb = peak_mem / (1024 ** 2)
    return result, runtime, peak_mem_mb, cpu_end

def add_gaussian_noise(X, noise_factor=2.0, seed=RANDOM_SEED):
    """Corrupts high-dimensional matrices with additive Gaussian white noise."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size=X.shape)
    return X + noise_factor * noise

# SECTION 2 — ACADEMIC VISUALIZATION RADAR

def plot_2d_embedding(embedding, labels, title, filename, subdirectory="", caption=""):
    save_dir = os.path.join(OUTPUT_DIR, subdirectory) if subdirectory else OUTPUT_DIR
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=6, alpha=0.7, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Target Class Labels")
    ax.set_title(title)
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    if caption:
        fig.text(0.5, -0.05, f"Figure: {caption}", ha="center", wrap=True, fontsize=9, fontstyle="italic")

    fig.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_3d_interactive(embedding, labels, title, filename):
    """Generates rotatable interactive 3D embeddings via Plotly."""
    df = pd.DataFrame(embedding, columns=["Dim 1", "Dim 2", "Dim 3"])
    df["label"] = labels.astype(str)
    fig = px.scatter_3d(df, x="Dim 1", y="Dim 2", z="Dim 3", color="label", title=title, opacity=0.6)
    fig.update_traces(marker=dict(size=2.0))
    fig.update_layout(legend_title_text="Class", font=dict(family="Times New Roman", size=11))
    fig.write_html(os.path.join(OUTPUT_DIR, "interactive", f"{filename}.html"))

# SECTION 3 — MULTI-DATASET DYNAMIC LOADING

def load_and_preprocess(dataset_name="mnist", sample_size=2000):
    print(f"[Data] Fetching '{dataset_name}' pipeline...")
    if dataset_name == "mnist":
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        y = y.astype(int)
    elif dataset_name == "fashion_mnist":
        X, y = fetch_openml("Fashion-MNIST", version=1, return_X_y=True, as_frame=False)
        y = y.astype(int)
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        sample_size = len(y)
    else:
        raise ValueError("Invalid target dataset selection.")

    if sample_size and sample_size < len(X):
        rng = np.random.default_rng(RANDOM_SEED)
        indices = rng.choice(len(X), size=sample_size, replace=False)
        X, y = X[indices], y[indices]

    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, y

# Execute data assembly lines
X_mnist, y_mnist = load_and_preprocess("mnist", sample_size=10000)
X_fmnist, y_fmnist = load_and_preprocess("fashion_mnist", sample_size=2000)
X_bc, y_bc = load_and_preprocess("breast_cancer")
X_mnist_noisy = add_gaussian_noise(X_mnist, noise_factor=2.0)

# SECTION 4 — DIMENSIONALITY REDUCTION SCHEDULER

def run_reduction(X, method="pca", n_components=2, **kwargs):
    if method == "pca":
        model = PCA(n_components=n_components, **kwargs)
    elif method == "tsne":
        # FIX 4 — TSNE: max_iter sklearn 1.5'te n_iter'ın yerini aldı
        # Her iki versiyonu da desteklemek için güvenli parametre adı kullan
        import sklearn
        sklearn_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        if sklearn_version >= (1, 5):
            model = TSNE(n_components=n_components, max_iter=300,
                         random_state=RANDOM_SEED, n_jobs=-1, **kwargs)
        else:
            model = TSNE(n_components=n_components, n_iter=300,
                         random_state=RANDOM_SEED, n_jobs=-1, **kwargs)
    elif method == "umap":
        # FIX 5 — UMAP: n_jobs=-1 bazı ortamlarda fork/multiprocessing sorunu çıkarır
        # Kaggle'da güvenli çalışması için n_jobs=1 olarak bırakıyoruz
        model = umap.UMAP(n_components=n_components, random_state=RANDOM_SEED,
                          n_jobs=1, **kwargs)
    else:
        raise ValueError("Unknown configuration protocol.")

    embedding = model.fit_transform(X)
    return model, embedding

# Latent Mapping Executions
print("[Execution] Launching standard low-dimensional embedding computations...")
(_, pca_emb_2d), *perf_pca_2d = profile_execution(run_reduction, X_mnist, "pca", 2)
(_, tsne_emb_2d), *perf_tsne_2d = profile_execution(run_reduction, X_mnist, "tsne", 2)
(_, umap_emb_2d), *perf_umap_2d = profile_execution(run_reduction, X_mnist, "umap", 2)

(_, pca_emb_3d), *perf_pca_3d = profile_execution(run_reduction, X_mnist, "pca", 3)
(_, tsne_emb_3d), *perf_tsne_3d = profile_execution(run_reduction, X_mnist, "tsne", 3)
(_, umap_emb_3d), *perf_umap_3d = profile_execution(run_reduction, X_mnist, "umap", 3)

# Save 2D & 3D Profiles
plot_2d_embedding(pca_emb_2d, y_mnist, "PCA Projection (2D) - MNIST", "pca_2d",
                  caption="2D Spectral map based on linear eigenvalue decomposition.")
plot_2d_embedding(tsne_emb_2d, y_mnist, "t-SNE Manifold (2D) - MNIST", "tsne_2d",
                  caption="Nonlinear t-SNE projection optimizing local probability divergence.")
plot_2d_embedding(umap_emb_2d, y_mnist, "UMAP Manifold (2D) - MNIST", "umap_2d",
                  caption="Uniform Manifold Projection mapping topological relationships.")

plot_3d_interactive(pca_emb_3d, y_mnist, "PCA 3D Representation", "pca_3d")
plot_3d_interactive(tsne_emb_3d, y_mnist, "t-SNE 3D Representation", "tsne_3d")
plot_3d_interactive(umap_emb_3d, y_mnist, "UMAP 3D Representation", "umap_3d")

# SECTION 5 — EIGEN-ANALYSIS, ACCURACY VARIANCE & SCREE PLOTS

pca_full = PCA().fit(X_mnist)
cum_variance = np.cumsum(pca_full.explained_variance_ratio_)

# FIX 6 — Scree plot: bar ve line x-ekseni uyumsuzluğu düzeltildi
# Önceden: line 0-783 arası, bar 0-99 arası çiziliyordu → yanıltıcı görünüm
# Düzeltme: İkincil eksen (twin axis) kullanarak doğru görselleştirme
N_COMPONENTS_PLOT = 100
fig, ax1 = plt.subplots(figsize=(8, 4))

ax1.bar(range(N_COMPONENTS_PLOT),
        pca_full.explained_variance_ratio_[:N_COMPONENTS_PLOT],
        alpha=0.5, label="Individual Variance", color="steelblue")
ax1.set_xlabel("Principal Component Spectrum Index")
ax1.set_ylabel("Individual Variance Ratio", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")

ax2 = ax1.twinx()
ax2.plot(range(N_COMPONENTS_PLOT),
         cum_variance[:N_COMPONENTS_PLOT],
         color="darkblue", linewidth=2, label="Cumulative Explained Variance")
ax2.axhline(y=0.90, color="crimson", linestyle="--", alpha=0.8, label="90% Threshold")
ax2.set_ylabel("Cumulative Variance Ratio", color="darkblue")
ax2.tick_params(axis="y", labelcolor="darkblue")
ax2.set_ylim(0, 1.05)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
ax1.set_title("PCA Scree & Variance Information Extraction Profile (First 100 Components)")
fig.savefig(os.path.join(OUTPUT_DIR, "pca_scree_plot.png"))
plt.close(fig)

# Component Size vs Downstream Classifier Accuracy Analysis
component_steps = [2, 5, 10, 30, 50, 100]
comp_accuracy = []
for c in component_steps:
    X_reduced = PCA(n_components=c, random_state=RANDOM_SEED).fit_transform(X_mnist)
    X_tr, X_te, y_tr, y_te = train_test_split(X_reduced, y_mnist, test_size=0.2, random_state=RANDOM_SEED)
    clf = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
    comp_accuracy.append(accuracy_score(y_te, clf.predict(X_te)))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(component_steps, comp_accuracy, marker="s", color="darkgreen", linestyle="-.", linewidth=1.5)
ax.set_xlabel("Replicated Eigen-components Range")
ax.set_ylabel("KNN Test Set Prediction Accuracy")
ax.set_title("Downstream Accuracy Profile vs. Preserved Component Scale")
ax.grid(True)
fig.savefig(os.path.join(OUTPUT_DIR, "accuracy_vs_components.png"))
plt.close(fig)

# SECTION 6 — SCALABILITY, PERFORMANCE BENCHMARKS & COMPLEXITY

sample_sizes = [1000, 2500, 5000, 10000]
bench_records = []

for size in sample_sizes:
    X_sub = X_mnist[:size]
    for method in ["pca", "tsne", "umap"]:
        (model_obj, emb_obj), run_t, peak_m, cpu_u = profile_execution(run_reduction, X_sub, method, 2)
        bench_records.append({
            "Method": method.upper(), "Dataset Size": size,
            "Runtime (s)": run_t, "Peak RAM (MB)": peak_m, "CPU (%)": cpu_u
        })

df_bench = pd.DataFrame(bench_records)
df_bench.to_csv(os.path.join(OUTPUT_DIR, "latex", "benchmark_metrics.csv"), index=False)

# Scalability Visualizations
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.lineplot(data=df_bench, x="Dataset Size", y="Runtime (s)", hue="Method", marker="o", ax=ax)
ax.set_title("Computational Runtime Scalability Across High-Dimensional Sample Boundaries")
fig.savefig(os.path.join(OUTPUT_DIR, "runtime_scalability.png"))
plt.close(fig)

# SECTION 7 — HYBRID COMPRESSION LINE SYSTEM (PCA → UMAP)

print("[Pipeline System] Initializing Hybrid Pipeline Protocol: PCA(50) -> UMAP(2)")
(_, hybrid_pca_step), t_hpca, m_hpca, _ = profile_execution(run_reduction, X_mnist, "pca", 50)
(hybrid_model, hybrid_final_emb), t_humap, m_humap, _ = profile_execution(run_reduction, hybrid_pca_step, "umap", 2)

total_hybrid_time = t_hpca + t_humap
total_hybrid_mem = max(m_hpca, m_humap)

plot_2d_embedding(hybrid_final_emb, y_mnist, "Hybrid Low-Dim Mapping (PCA+UMAP)", "hybrid_pca50_umap",
                  caption="Hybrid processing line compressing initial boundaries through linear mapping prior to non-linear topology matching.")

# SECTION 8 — ACADEMIC CLASSIFICATION SYSTEM VALIDATION

def evaluate_downstream_classifiers(X_emb, labels, tag):
    X_train, X_test, y_train, y_test = train_test_split(X_emb, labels, test_size=0.2, random_state=RANDOM_SEED)
    classifiers = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED),
        "SVM": SVC(kernel="rbf", random_state=RANDOM_SEED)
    }

    local_accuracies = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        local_accuracies[name] = acc

        # Generation of Structural Confusion Matrices
        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False, ax=ax)
        ax.set_title(f"Confusion Matrix - {name} Over {tag}")
        ax.set_xlabel("Predicted Digit Class")
        ax.set_ylabel("True Target Digit Class")
        fig.savefig(os.path.join(OUTPUT_DIR, "classification", f"cm_{tag}_{name}.png"))
        plt.close(fig)

    return local_accuracies

print("[Classifier Validation] Training supervised model variants across spatial manifolds...")
acc_pca_dict = evaluate_downstream_classifiers(pca_emb_2d, y_mnist, "PCA_2D")
acc_tsne_dict = evaluate_downstream_classifiers(tsne_emb_2d, y_mnist, "t-SNE_2D")
acc_umap_dict = evaluate_downstream_classifiers(umap_emb_2d, y_mnist, "UMAP_2D")
acc_hybrid_dict = evaluate_downstream_classifiers(hybrid_final_emb, y_mnist, "Hybrid_2D")

# Assemble Reporting Metrics Frame
df_clf_perf = pd.DataFrame([acc_pca_dict, acc_tsne_dict, acc_umap_dict, acc_hybrid_dict],
                           index=["PCA", "t-SNE", "UMAP", "Hybrid (PCA+UMAP)"])

fig, ax = plt.subplots(figsize=(8, 4.5))
df_clf_perf.plot(kind="bar", edgecolor="black", alpha=0.85, ax=ax)
ax.set_ylabel("Classification Hit Metric (Accuracy)")
ax.set_title("Downstream Classifier Benchmark Comparison Across Extracted Manifolds")
ax.set_ylim(0, 1.1)
plt.xticks(rotation=0)
fig.savefig(os.path.join(OUTPUT_DIR, "classification_comparison.png"))
plt.close(fig)

# SECTION 9 — CLUSTER SEPARATION & INTERNAL STABILITY CRITERIA

def structural_clustering_assessment(embedding, labels):
    km = KMeans(n_clusters=10, random_state=RANDOM_SEED, n_init="auto").fit(embedding)
    agglo = AgglomerativeClustering(n_clusters=10).fit(embedding)

    return {
        "Silhouette": silhouette_score(embedding, km.labels_),
        "DaviesBouldin": davies_bouldin_score(embedding, km.labels_),
        "CalinskiHarabasz": calinski_harabasz_score(embedding, km.labels_),
        "ARI_KMeans": adjusted_rand_score(labels, km.labels_),
        "ARI_Agglo": adjusted_rand_score(labels, agglo.labels_),
        "NMI_KMeans": normalized_mutual_info_score(labels, km.labels_)
    }

print("[Clustering Engine] Extracting spatial separation quality scores...")
cluster_metrics_pca = structural_clustering_assessment(pca_emb_2d, y_mnist)
# FIX 7 — t-SNE clustering metrics eksikti, eklendi (karşılaştırma tutarlılığı için)
cluster_metrics_tsne = structural_clustering_assessment(tsne_emb_2d, y_mnist)
cluster_metrics_umap = structural_clustering_assessment(umap_emb_2d, y_mnist)

# Combine Topology Data — FIX 7 devamı: t-SNE satırı eklendi
df_cluster_report = pd.DataFrame(
    [cluster_metrics_pca, cluster_metrics_tsne, cluster_metrics_umap],
    index=["PCA", "t-SNE", "UMAP"]
)
df_cluster_report.to_latex(os.path.join(OUTPUT_DIR, "latex", "clustering_metrics_table.tex"))

# SECTION 10 — LOCAL/GLOBAL TOPOLOGY PRESERVATION ENGINES

def evaluate_continuity_fractional(X_orig, X_emb, neighbors=5):
    nn_o = NearestNeighbors(n_neighbors=neighbors).fit(X_orig)
    nn_e = NearestNeighbors(n_neighbors=neighbors).fit(X_emb)
    nbrs_o = nn_o.kneighbors(return_distance=False)
    nbrs_e = nn_e.kneighbors(return_distance=False)

    overlaps = [len(set(nbrs_o[i]) & set(nbrs_e[i])) / neighbors for i in range(len(X_orig))]
    return float(np.mean(overlaps))

print("[Topology Analyzer] Computing cross-space neighborhood stability projections...")
sub_sample_limit = 2000
tw_pca_metric = trustworthiness(X_mnist[:sub_sample_limit], pca_emb_2d[:sub_sample_limit], n_neighbors=5)
tw_umap_metric = trustworthiness(X_mnist[:sub_sample_limit], umap_emb_2d[:sub_sample_limit], n_neighbors=5)

cont_pca_metric = evaluate_continuity_fractional(X_mnist[:sub_sample_limit], pca_emb_2d[:sub_sample_limit])
cont_umap_metric = evaluate_continuity_fractional(X_mnist[:sub_sample_limit], umap_emb_2d[:sub_sample_limit])

# Global Distance Preservation Check
d_matrix_sample = X_mnist[:1000]
flat_orig_distances = pdist(d_matrix_sample)
spearman_pca, _ = spearmanr(flat_orig_distances, pdist(pca_emb_2d[:1000]))
spearman_umap, _ = spearmanr(flat_orig_distances, pdist(umap_emb_2d[:1000]))

# SECTION 11 — HYPERPARAMETER SENSITIVITY GRID HEATMAPS

print("[Grid Optimization Search] Running hyperparameter sensitivity matrix explorations...")
sub_grid_data = X_mnist[:1500]
sub_grid_labels = y_mnist[:1500]

n_neighbors_opt = [5, 15, 30]
min_dist_opt = [0.1, 0.5, 0.8]
umap_heatmap_matrix = np.zeros((len(n_neighbors_opt), len(min_dist_opt)))

for i, nn in enumerate(n_neighbors_opt):
    for j, md in enumerate(min_dist_opt):
        test_umap = umap.UMAP(n_neighbors=nn, min_dist=md, n_components=2,
                               random_state=RANDOM_SEED,
                               n_jobs=1).fit_transform(sub_grid_data)  # FIX 5 burada da uygulandı
        umap_heatmap_matrix[i, j] = silhouette_score(test_umap, sub_grid_labels)

fig, ax = plt.subplots(figsize=(6, 4.5))
sns.heatmap(umap_heatmap_matrix, annot=True, fmt=".3f", xticklabels=min_dist_opt,
            yticklabels=n_neighbors_opt, cmap="viridis", ax=ax)
ax.set_title("UMAP Operational Parameter Space (Silhouette Index Mapping)")
ax.set_xlabel("Minimum Distance Variable (min_dist)")
ax.set_ylabel("Nearest Neighbor Scale Boundary (n_neighbors)")
fig.savefig(os.path.join(OUTPUT_DIR, "hyperparameter", "umap_sensitivity_heatmap.png"))
plt.close(fig)

# SECTION 12 — ENVIRONMENTAL ROBUSTNESS CRITERIA

print("[Noise Isolation Testing] Benchmarking robustness limits under Gaussian corruption...")
(model_npca, noisy_pca_emb), r_time_pca, p_mem_pca, cpu_pca = profile_execution(run_reduction, X_mnist_noisy, "pca", 2)
(model_ntsne, noisy_tsne_emb), r_time_tsne, p_mem_tsne, cpu_tsne = profile_execution(run_reduction, X_mnist_noisy, "tsne", 2)
(model_numap, noisy_umap_emb), r_time_umap, p_mem_umap, cpu_umap = profile_execution(run_reduction, X_mnist_noisy, "umap", 2)

plot_2d_embedding(noisy_pca_emb, y_mnist, "Noisy Gaussian Map Structure - PCA", "noisy_pca", subdirectory="noise")
plot_2d_embedding(noisy_tsne_emb, y_mnist, "Noisy Gaussian Map Structure - t-SNE", "noisy_tsne", subdirectory="noise")
plot_2d_embedding(noisy_umap_emb, y_mnist, "Noisy Gaussian Map Structure - UMAP", "noisy_umap", subdirectory="noise")

# SECTION 13 — CROSS-DATASET GENERALIZATION (BREAST CANCER VALIDATION)

print("[Generalization Engine] Processing medical benchmarking matrices (Breast Cancer Diagnostic)...")
(_, bc_pca_emb), *bc_perf_pca = profile_execution(run_reduction, X_bc, "pca", 2)
(_, bc_umap_emb), *bc_perf_umap = profile_execution(run_reduction, X_bc, "umap", 2)

plot_2d_embedding(bc_pca_emb, y_bc, "PCA Diagnostics Space - Wisconsin Breast Cancer", "bc_pca", subdirectory="stability")
plot_2d_embedding(bc_umap_emb, y_bc, "UMAP Diagnostics Space - Wisconsin Breast Cancer", "bc_umap", subdirectory="stability")

# SECTION 14 — INDUSTRIAL AUTOMATED DEEP AUTOENCODER INFRASTRUCTURE

print("[Neural Processing] Launching deep parametric reconstruction autoencoder network...")
input_features_count = X_mnist.shape[1]

input_layer = Input(shape=(input_features_count,), name="Pipeline_Input")
encoded = Dense(128, activation="relu", name="Encoder_Hidden_1")(input_layer)
encoded = Dense(64, activation="relu", name="Encoder_Hidden_2")(encoded)
latent_bottleneck = Dense(2, activation="linear", name="Latent_Space_Bottleneck")(encoded)

decoded = Dense(64, activation="relu", name="Decoder_Hidden_1")(latent_bottleneck)
decoded = Dense(128, activation="relu", name="Decoder_Hidden_2")(decoded)
output_reconstruction = Dense(input_features_count, activation="linear", name="Network_Output_Reconstruction")(decoded)

autoencoder_model = Model(input_layer, output_reconstruction, name="Symmetric_Deep_Autoencoder")
encoder_model = Model(input_layer, latent_bottleneck, name="Target_Encoder_Extraction")

autoencoder_model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
history_log = autoencoder_model.fit(X_mnist, X_mnist, epochs=15, batch_size=256, shuffle=True, verbose=0)

ae_latent_coordinates = encoder_model.predict(X_mnist, verbose=0)
plot_2d_embedding(ae_latent_coordinates, y_mnist, "Parametric Compression Mapping - Deep Autoencoder", "autoencoder_2d",
                  caption="Parametric nonlinear projection path trained minimizing structural mean squared errors.")

print("\n" + "=" * 80)
print("EXPERIMENTAL THESIS PIPELINE SUCCESSFUL")
print(f"Comprehensive empirical validation files compiled completely in: {os.path.abspath(OUTPUT_DIR)}")
print("=" * 80)
