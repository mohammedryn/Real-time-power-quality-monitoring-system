import numpy as np
from src.dsp.features import extract_features

# Generate dummy 500-sample voltage and current arrays
v_dummy = np.random.randn(500) * 325.0
i_dummy = np.random.randn(500) * 10.0

print("Extracting features...")
features = extract_features(v_dummy, i_dummy)

print("SUCCESS!")
print(f"Shape: {features.shape}")
assert len(features) == 282, f"Failed! Length was {len(features)}"
print(f"Contains NaNs: {np.isnan(features).any()}")
