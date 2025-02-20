import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

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
    df.drop(['id', 'dropoff_datetime', 'pickup_datetime', 'dropoff_latitude', 'dropoff_longitude'], axis=1, inplace=True)
    df['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == 'Y').astype(int)
    df['vendor_id'] = df['vendor_id'].astype(int)
    return df


def preprocess_data():

    dataset = np.load(r"Data\nyc_taxi_data.npy", allow_pickle=True).item()
    X_train, y_train, X_test, y_test = dataset["X_train"], dataset["y_train"], dataset["X_test"], dataset["y_test"]

    X_train = create_features(X_train)
    X_test = create_features(X_test)

    numerical_features = ['distance_km', 'passenger_count']
    categorical_features = ['vendor_id', 'store_and_fwd_flag']
    geospacial_features = ['pickup_latitude', 'pickup_longitude']
    
    cyclic_features = ['pickup_hour_sin', 'pickup_hour_cos', 'pickup_dow_sin', 'pickup_dow_cos']
    binary_features = ['is_weekend']

    # Apply transformations
    num_scaler = StandardScaler().fit(X_train[numerical_features])
    geo_scaler = MinMaxScaler().fit(X_train[geospacial_features])
    one_hot = OneHotEncoder(sparse_output= False, drop='first').fit(X_train[categorical_features])

    # Transform features
    X_train_num = num_scaler.transform(X_train[numerical_features])
    X_test_num = num_scaler.transform(X_test[numerical_features])

    X_train_geo = geo_scaler.transform(X_train[geospacial_features])
    X_test_geo = geo_scaler.transform(X_test[geospacial_features])

    X_train_cat = one_hot.transform(X_train[categorical_features])
    X_test_cat = one_hot.transform(X_test[categorical_features])

    X_train_cyclic = X_train[cyclic_features].values
    X_test_cyclic = X_test[cyclic_features].values

    X_train_binary = X_train[binary_features].values
    X_test_binary = X_test[binary_features].values
    
    X_train_processed = np.concatenate([X_train_num, X_train_geo, X_train_cat, X_train_cyclic, X_train_binary], axis=1).astype(np.float32)
    X_test_processed = np.concatenate([X_test_num, X_test_geo, X_test_cat, X_test_cyclic, X_test_binary], axis=1).astype(np.float32)

    y_train = np.log1p(y_train.values).reshape(-1, 1).astype(np.float32)
    y_test = y_test.values.astype(np.float32)

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_processed, y_train, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val, X_test_processed, y_test