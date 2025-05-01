import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib
print(matplotlib.matplotlib_fname())

print(matplotlib.get_backend())



# 1. Загрузка данных
iris = load_iris()
X = iris.data

# 2. Нормализация
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Создаём autoencoder
input_dim = X.shape[1]
encoding_dim = 2  # для визуализации

input_layer = keras.Input(shape=(input_dim,))
encoded = layers.Dense(8, activation='relu')(input_layer)
encoded = layers.Dense(encoding_dim, activation='tanh')(encoded)

decoded = layers.Dense(4, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='linear')(decoded)

autoencoder = keras.Model(inputs=input_layer, outputs=decoded)

# 4. Компиляция и обучение
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=16, verbose=0)

# 5. Извлекаем энкодер
encoder = keras.Model(inputs=input_layer, outputs=encoded)
X_encoded = encoder.predict(X_scaled)

# 6. Кластеризация скрытых признаков
kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=32)
cluster_labels = kmeans.fit_predict(X_encoded)  # исправлено с 'encoded' на 'X_encoded'

# 7. Визуализация
plt.scatter(X_encoded[:, 0], X_encoded[:, 1], c=cluster_labels, cmap='viridis')  # исправлено
plt.colorbar()
plt.title("MiniBatchKMeans кластеры")
plt.xlabel("Фича 1")
plt.ylabel("Фича 2")
plt.show()

