import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Feature engineering functions
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

# Feature engineering
def create_features(df):
    df['distance_km'] = haversine(df['pickup_latitude'], df['pickup_longitude'],
                                 df['dropoff_latitude'], df['dropoff_longitude'])
    dt = pd.to_datetime(df['pickup_datetime'])
    df['pickup_hour_sin'] = np.sin(2*np.pi*dt.dt.hour/24)
    df['pickup_hour_cos'] = np.cos(2*np.pi*dt.dt.hour/24)
    df['pickup_dow_sin'] = np.sin(2*np.pi*dt.dt.dayofweek/7)
    df['pickup_dow_cos'] = np.cos(2*np.pi*dt.dt.dayofweek/7)
    df['is_weekend'] = dt.dt.dayofweek.isin([5,6]).astype(int)
    df.drop(['id', 'dropoff_datetime', 'pickup_datetime'], axis=1, inplace=True)
    df['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == 'Y').astype(int)
    df['vendor_id'] = df['vendor_id'].astype(int)
    return df


def preprocess_data():

    dataset = np.load(r"Data\nyc_taxi_data.npy", allow_pickle=True).item()
    X_train, y_train, X_test, y_test = dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"]
    final_features = [
        "vendor_id", "passenger_count", "store_and_fwd_flag", "distance_km",
        "pickup_hour_sin", "pickup_hour_cos", "pickup_dow_sin", "pickup_dow_cos",
        "is_weekend", "pickup_latitude", "pickup_longitude"
    ]

    X_train = create_features(X_train)
    X_test = create_features(X_test)

    scaler = StandardScaler()
    X_train[final_features] = scaler.fit_transform(X_train[final_features])
    X_test[final_features] = scaler.transform(X_test[final_features])

    # Convert to numpy arrays
    X_train = X_train[final_features].values.astype(np.float32)
    y_train = np.log1p(y_train.values).reshape(-1, 1).astype(np.float32) # Log-transform the target, helps with large values and outliers w/o it I get a lot of NaNs caused gradients to explode
    X_test = X_test[final_features].values.astype(np.float32)
    y_test = y_test.values.astype(np.float32)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val, X_test, y_test