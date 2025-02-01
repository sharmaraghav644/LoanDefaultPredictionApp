import os

model_path = "dl_model.h5"
size_in_bytes = os.path.getsize(model_path)
size_in_mb = size_in_bytes / (1024 * 1024)  # Convert to MB

print(f"Model size DL: {size_in_mb:.2f} MB")

model_path = "rf_model_compressed.pkl.bz2"
size_in_bytes = os.path.getsize(model_path)
size_in_mb = size_in_bytes / (1024 * 1024)  # Convert to MB

print(f"Model size RF: {size_in_mb:.2f} MB")

model_path = "xgb_model_compressed.pkl.bz2"
size_in_bytes = os.path.getsize(model_path)
size_in_mb = size_in_bytes / (1024 * 1024)  # Convert to MB

print(f"Model size XGB: {size_in_mb:.2f} MB")