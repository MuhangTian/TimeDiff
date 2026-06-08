import pandas as pd
import numpy as np
import torch
import os

print("=" * 60)
print("Preprocessing Exchange Rate Data")
print("=" * 60)

# Load the exchange rate data
data = pd.read_csv('data/exchange_rate/exchange_rate.csv')
print(f"Loaded data shape: {data.shape}")

# Remove date column and OT column if present
if 'date' in data.columns:
    data = data.drop('date', axis=1)
if 'OT' in data.columns:
    data = data.drop('OT', axis=1)

print(f"After dropping columns: {data.shape}")
print(f"Columns: {data.columns.tolist()}")

# Convert to numpy
data_np = data.values

# Create sliding windows for time series
sequence_length = 96  # 96 time steps per sequence (divisible by 8)
stride = 1

sequences = []
for i in range(0, len(data_np) - sequence_length + 1, stride):
    seq = data_np[i:i + sequence_length, :]
    sequences.append(seq)

sequences = np.array(sequences)
print(f"Number of sequences: {len(sequences)}")
print(f"Sequence shape: {sequences.shape}")

# Transpose to (num_samples, num_channels, seq_length)
# From (N, seq_length, channels) to (N, channels, seq_length)
sequences = np.transpose(sequences, (0, 2, 1))
print(f"Final shape: {sequences.shape}")

# Split into train/test (80/20)
train_size = int(0.8 * len(sequences))
train_data = sequences[:train_size]
test_data = sequences[train_size:]

print(f"Train: {train_data.shape}, Test: {test_data.shape}")
print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# Convert to torch tensors
train_tensor = torch.from_numpy(train_data).float()
test_tensor = torch.from_numpy(test_data).float()

# Create output directory
os.makedirs('data/exchange_rate_processed', exist_ok=True)

# Save
torch.save(train_tensor, 'data/exchange_rate_processed/TRAIN_exchange_rate.pt')
torch.save(test_tensor, 'data/exchange_rate_processed/TEST_exchange_rate.pt')

print("\n" + "=" * 60)
print("✓ Data preprocessing complete!")
print("=" * 60)
print(f"Saved to:")
print(f"  - data/exchange_rate_processed/TRAIN_exchange_rate.pt")
print(f"  - data/exchange_rate_processed/TEST_exchange_rate.pt")
print("=" * 60)

