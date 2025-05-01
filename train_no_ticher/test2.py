import matplotlib

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Загрузка данных
iris = load_iris()
X = iris.data

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Список для хранения ошибок
sse = []

# Пробуем кластеризацию для разных значений K (от 1 до 10)
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)  # SSE (сумма квадратов ошибок)

# Строим график
plt.plot(range(1, 11), sse, marker='o')
plt.title('Метод локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('Сумма квадратов ошибок (SSE)')
plt.show()
