import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# PEDIR DATOS DE LA CASA
metraje = int(input("Ingresa cuantos metros cuadrados tiene construida la casa: "))
baños = int(input("Ingresa número de baños: "))
cuartos = int(input("Ingresa número de habitaciones: "))
if_piscina = int(input("Ingresa cantidad de piscinas: "))
num_estrato = int(input("Ingresa el estrato: "))

# DATOS DE LOS ESTRATOS
estratos_precio_por_m2 = {
    'Estrato_1': (0.1, 1.25),
    'Estrato_2': (1.25, 2.5),
    'Estrato_3': (2.5, 3.75),
    'Estrato_4': (3.75, 5),
    'Estrato_5': (5, 8),
    'Estrato_6': (7, 9)
}

# CASA SANTA MARIA
datos_casa_SantaMaria = {
    'm2': 120,
    'num_baños': 4,
    'num_cuartos': 5,
    'piscina': 0,
    'estrato': 4,
    'precio_inicial': 110
}

# OTRAS CASAS SAN JOSÉ
datos_otrasCasas_SanJoseBavaria = {
    'm2': 360,
    'num_baños': 5,
    'num_cuartos': 4,
    'piscina': 1,
    'estrato': 5,
    'precio_inicial': 1600
}

datos_casa_nueva = {
    'm2': metraje,
    'num_baños': baños,
    'num_cuartos': cuartos,
    'piscina': if_piscina,
    'estrato': num_estrato,
}

# Obtener características de las casas
features_casa_SantaMaria = np.array(list(datos_casa_SantaMaria.values())[:-1], dtype=float).reshape((1, -1))
features_otrasCasas_SanJoseBavaria = np.array(list(datos_otrasCasas_SanJoseBavaria.values())[:-1], dtype=float).reshape((1, -1))

# features_casa_SanJoseBavaria = np.array(list(datos_casa_SanJoseBavaria.values())[:-1], dtype=float).reshape((1, -1))
features_casa_nueva = np.array(list(datos_casa_nueva.values()), dtype=float).reshape((1, -1))

# Crear el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=3, input_shape=(features_casa_SantaMaria.shape[1],)),
    tf.keras.layers.Dense(units=3),
    tf.keras.layers.Dense(units=1)
])

# Compilar el modelo
modelo.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

# Obtener precios de las casas iniciales
precio_SantaMaria = datos_casa_SantaMaria['precio_inicial']
precio_otrasCasas_SanJoseBavaria = datos_otrasCasas_SanJoseBavaria['precio_inicial']

# Ajustar la forma de las características para que sea una matriz
features_SantaMaria = features_casa_SantaMaria.reshape((1, -1))
features_otrasCasas_SanJoseBavaria = features_otrasCasas_SanJoseBavaria.reshape((1, -1))

# Convertir precios a arrays NumPy
precio_SantaMaria = np.array([precio_SantaMaria], dtype=float)
precio_otrasCasas_SanJoseBavaria = np.array([precio_otrasCasas_SanJoseBavaria], dtype=float)

print("Comienzo entrenamiento...")

# Entrenar el modelo con los datos de las dos casas iniciales
historial = modelo.fit(
    features_SantaMaria,
    precio_SantaMaria,
    epochs=1000,
    verbose=False
)

historial_otras_casas = modelo.fit(
    features_otrasCasas_SanJoseBavaria,
    precio_otrasCasas_SanJoseBavaria,
    epochs=1000,
    verbose=False
)

# Hacer la predicción utilizando el modelo entrenado
precio_estimado_casa_nueva = modelo.predict(features_casa_nueva)
print(f"Precio estimado para casa_nueva: ${round(float(precio_estimado_casa_nueva[0]), 2)} millones")
