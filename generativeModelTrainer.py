import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense


def train_autoencoder(X_scaled, encoding_dim=16, epochs=50, batch_size=8):
    input_dim = X_scaled.shape[1]

    # Build autoencoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, verbose=0)

    # Extract encoder model
    encoder = Model(inputs=input_layer, outputs=encoded)

    return autoencoder, encoder


def sample_synthetic_features(autoencoder, encoder, X_scaled, num_samples=10, noise_std=0.5):
    synthetic_features = []

    for _ in range(num_samples):
        # Pick a random sample from training data
        base_vector = encoder.predict(X_scaled[random.randint(0, len(X_scaled)-1)].reshape(1, -1))
        noise = np.random.normal(0, noise_std, size=base_vector.shape)
        z = base_vector + noise
        generated = autoencoder.predict(z)
        synthetic_features.append(generated.flatten())

    return np.array(synthetic_features)


def generate_synthetic_lofi_dataset(csv_path='chillhop_dataset.csv', num_samples=10, output_csv='synthetic_dataset.csv'):
    # Load and preprocess dataset
    df = pd.read_csv(csv_path)
    df_features = df.drop(columns=['file', 'label'], errors='ignore')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    # Train model
    autoencoder, encoder = train_autoencoder(X_scaled)

    # Sample synthetic data
    synthetic_scaled = sample_synthetic_features(autoencoder, encoder, X_scaled, num_samples=num_samples)

    # Inverse scale
    synthetic_original = scaler.inverse_transform(synthetic_scaled)
    synthetic_df = pd.DataFrame(synthetic_original, columns=df_features.columns)

    if output_csv:
        synthetic_df.to_csv(output_csv, index=False)
        print(f"✅ Saved {num_samples} synthetic examples to {output_csv}")

    return synthetic_df

generate_synthetic_lofi_dataset()