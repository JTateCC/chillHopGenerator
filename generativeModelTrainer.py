import pandas as pd
import ast
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense




def train_autoencoder(X_scaled, encoding_dim=16, epochs=50, batch_size=8):
    input_dim = X_scaled.shape[1]

    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded_output = Dense(encoding_dim, activation='relu')(encoded)

    # Decoder
    decoder_input = Input(shape=(encoding_dim,))
    decoder_layer = Dense(64, activation='relu')(decoder_input)
    decoder_output = Dense(input_dim, activation='linear')(decoder_layer)

    encoder = Model(inputs=input_layer, outputs=encoded_output)
    decoder = Model(inputs=decoder_input, outputs=decoder_output)

    # Autoencoder
    autoencoder_output = decoder(encoder(input_layer))
    autoencoder = Model(inputs=input_layer, outputs=autoencoder_output)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, verbose=0)

    return autoencoder, encoder, decoder


def sample_synthetic_features(encoder, decoder, X_scaled, num_samples=10, noise_std=0.5):
    synthetic_features = []

    for _ in range(num_samples):
        base = encoder.predict(X_scaled[random.randint(0, len(X_scaled)-1)].reshape(1, -1))
        noise = np.random.normal(0, noise_std, size=base.shape)
        z = base + noise
        generated = decoder.predict(z)
        synthetic_features.append(generated.flatten())

    return np.array(synthetic_features)



def generate_synthetic_lofi_dataset(csv_path='chillhop_dataset.csv', num_samples=10, output_csv='synthetic_dataset.csv'):
    # Load and preprocess dataset
    df = pd.read_csv(csv_path)
    df['tempo'] = df['tempo'].apply(lambda x: ast.literal_eval(x)[0] if isinstance(x, str) else x)
    df_features = df.drop(columns=['file', 'label'], errors='ignore')

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    # Train model and get encoder/decoder
    autoencoder, encoder, decoder = train_autoencoder(X_scaled)

    # Sample synthetic data
    synthetic_scaled = sample_synthetic_features(encoder, decoder, X_scaled, num_samples=num_samples)

    # Inverse scale to original feature values
    synthetic_original = scaler.inverse_transform(synthetic_scaled)
    synthetic_df = pd.DataFrame(synthetic_original, columns=df_features.columns)

    if output_csv:
        synthetic_df.to_csv(output_csv, index=False)
        print(f"✅ Saved {num_samples} synthetic examples to {output_csv}")

    return synthetic_df


generate_synthetic_lofi_dataset()