import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from ICKDC import ICKDC

# 1. Download the "Aggregation" dataset (used in the paper)
url = "http://cs.joensuu.fi/sipu/datasets/Aggregation.txt"
print("Downloading Aggregation dataset...")
response = urllib.request.urlopen(url)
data = np.loadtxt(response)

# The first two columns are the X, Y coordinates. The last column is the true label.
X = data[:, :2]
true_labels = data[:, 2]

# 2. Apply Min-Max Normalization (CRITICAL step from the paper)
# The paper states all datasets are adjusted by min-max normalization before tests
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# 3. Initialize YOUR algorithm with the exact parameter from the paper
# Table 2 specifies gamma = 2.1 for the Aggregation dataset
gamma_val = 2.1
clusterer = ICKDC(gamma=gamma_val)

# 4. Fit and predict
print(f"Running ICKDC with gamma={gamma_val}...")
predicted_labels = clusterer.fit_predict(X_normalized)

# 5. Plot the results to compare with Figure 3 in the paper!
plt.figure(figsize=(12, 5))

# Plot A: Ground Truth
plt.subplot(1, 2, 1)
plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=true_labels, cmap='tab10', s=15)
plt.title("Aggregation - Ground Truth")

# Plot B: Your ICKDC Results
plt.subplot(1, 2, 2)
plt.scatter(X_normalized[:, 0], X_normalized[:, 1], c=predicted_labels, cmap='tab10', s=15)

# Highlight the core points identified by your algorithm
if hasattr(clusterer, 'core_points_'):
    core_coords = X_normalized[clusterer.core_points_]
    plt.scatter(core_coords[:, 0], core_coords[:, 1], c='red', marker='x', s=60, label='Core Points')
    plt.legend()

plt.title(f"ICKDC Result (gamma={gamma_val})")
plt.tight_layout()
plt.show()