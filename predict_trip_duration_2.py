import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import neural_network_library as nnl
from scipy import stats

# Load data
dataset = np.load('nyc_taxi_data.npy', allow_pickle=True).item()
X_train, y_train = dataset['X_train'], dataset['y_train']
X_test, y_test = dataset['X_test'], dataset['y_test']

# Enhanced Feature Engineering Functions ============================================
NYC_BOUNDS = {
    'min_lat': 40.4774, 'max_lat': 40.9176,
    'min_lon': -74.2591, 'max_lon': -73.7004
}

AIRPORTS = {
    'JFK': (40.6413, -73.7781),
    'LGA': (40.7769, -73.8740),
    'EWR': (40.6895, -74.1745)
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c
    
def enhanced_preprocessing(df, y=None):
    """Enhanced feature engineering pipeline"""
    # Temporal features
    dt = pd.to_datetime(df['pickup_datetime'])
    df['pickup_day_of_year'] = dt.dt.dayofyear
    df['pickup_day_of_week'] = dt.dt.dayofweek
    df['pickup_hour'] = dt.dt.hour
    df['is_weekend'] = dt.dt.dayofweek.isin([5,6]).astype(int)
    df['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == 'Y').astype(int)
    df['vendor_id'] = df['vendor_id'].astype(int)

    print('done with temporal features')
    
    # Anomaly detection
    daily_counts = dt.dt.date.value_counts()
    z_scores = np.abs(stats.zscore(daily_counts))
    df['is_anomaly'] = dt.dt.date.isin(daily_counts.index[z_scores > 3]).astype(int)

    print('done with anomaly detection')
    
    # Geospatial features
    df['distance_km'] = haversine(df['pickup_latitude'], df['pickup_longitude'],
                                 df['dropoff_latitude'], df['dropoff_longitude'])

    print('done with geo features')
    
    # Airport proximity
    for airport, (lat, lon) in AIRPORTS.items():
        dist = haversine(df['pickup_latitude'], df['pickup_longitude'], lat, lon)
        df[f'near_{airport}'] = (dist < 3).astype(int)
    print('Done with airport proximity')
            
    # NYC boundary check
    df['valid_nyc'] = (
        df['pickup_latitude'].between(NYC_BOUNDS['min_lat'], NYC_BOUNDS['max_lat']) &
        df['pickup_longitude'].between(NYC_BOUNDS['min_lon'], NYC_BOUNDS['max_lon'])
    ).astype(int)

    print('done with nyc boundary')
    
    # Clean passengers
    passenger_mask = df['passenger_count'].between(1, 6)
    df = df[passenger_mask]
    if y is not None:
        y = y.loc[passenger_mask.values]

    print('done with clean passengers')
    
    # Cyclical encoding
    df['pickup_hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24)
    df['pickup_hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24)
    df['pickup_dow_sin'] = np.sin(2 * np.pi * df['pickup_day_of_week'] / 7)
    df['pickup_dow_cos'] = np.cos(2 * np.pi * df['pickup_day_of_week'] / 7)

    print('done with cyclical encoding')
    
    # Clean trip durations
    if y is not None:
        Q1 = y.quantile(0.05)
        Q3 = y.quantile(0.95)
        duration_mask = y.between(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))
        df = df.loc[duration_mask.values]
        y = y.loc[duration_mask.values]

    print('done with clean trip durations')
    
    # Final feature selection
    features = [
        'vendor_id', 'passenger_count', 'distance_km',
        'pickup_hour_sin', 'pickup_hour_cos', 'pickup_dow_sin', 'pickup_dow_cos',
        'is_anomaly', 'valid_nyc', 'near_JFK', 'near_LGA', 'near_EWR',
        'store_and_fwd_flag', 'is_weekend', 'pickup_latitude', 'pickup_longitude'
    ]
    
    return df[features], y

# Preprocessing Pipeline ==========================================================
# Apply enhanced preprocessing
X_train, y_train = enhanced_preprocessing(X_train, y_train)
X_test, _ = enhanced_preprocessing(X_test)

print(f'X_train type: {type(X_train)}')
print(f'y_train type: {type(y_train)}')

# Convert to numpy arrays
X_train = X_train.astype(np.float32)
y_train = np.log1p(y_train.values).astype(np.float32).reshape(-1, 1)  # Convert y to numpy
X_test = X_test.astype(np.float32)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f'training data shape: {X_train.shape}, {y_train.shape}')
print(f'validation data shape: {X_val.shape}, {y_val.shape}')

# Model Configuration =============================================================
model = nnl.Sequential([
    nnl.Linear(16, 256),  # Updated input size
    nnl.ReLU(),
    nnl.Linear(256, 128),
    nnl.ReLU(),
    nnl.Linear(128, 64),
    nnl.ReLU(),
    nnl.Linear(64, 1)
])

loss = nnl.MseLoss()
learning_rate = 0.001
batch_size = 512
n_epochs = 200
patience = 5
best_val_loss = np.inf
patience_counter = 0
train_losses, val_losses = [], []

# Enhanced Training Loop ==========================================================
for epoch in range(n_epochs):
    # Shuffle data
    indices = np.random.permutation(len(X_train))
    epoch_loss = []
    
    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        batch_idx = indices[i:i+batch_size]
        X_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]
        
        # Forward pass
        y_pred = model.forward(X_batch)
        batch_loss = loss.forward(y_pred, y_batch)
        epoch_loss.append(batch_loss)
        
        # Backward pass
        grad = loss.backward()
        model.backward(grad)
        
        # Update weights with gradient clipping
        for layer in model.layers:
            if isinstance(layer, nnl.Linear):
                layer.weights -= learning_rate * np.clip(layer.grad_weights, -1, 1)
                layer.bias -= learning_rate * np.clip(layer.grad_bias, -1, 1)
    
    # Validation
    train_loss = np.mean(epoch_loss)
    val_pred = model.forward(X_val)
    val_loss = loss.forward(val_pred, y_val)
    
    # Early stopping with patience
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best weights
        model.save('best_model.npz')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            model.load('best_model.npz')  # Restore best weights
            break
    
    # Record metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Patience: {patience_counter}/{patience}")

# Final Evaluation ================================================================
test_pred = np.expm1(model.forward(X_test).flatten())
results = {
    'RMSLE': np.sqrt(np.mean((np.log1p(y_test) - np.log1p(test_pred))**2)),
    'RMSE': np.sqrt(np.mean((y_test - test_pred)**2)),
    'MAE': np.mean(np.abs(y_test - test_pred))
}

print("\nFinal Test Metrics:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training Progress')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()