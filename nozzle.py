import numpy as np
import matplotlib.pyplot as plt

# Constantes
gamma = 1.4  # Razão de capacidades caloríficas (ar ideal)

# Funções auxiliares
def prandtl_meyer(M):
    """Calcula o ângulo de Prandtl-Meyer em radianos."""
    term1 = np.sqrt((gamma + 1) / (gamma - 1))
    term2 = np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1)))
    term3 = np.arctan(np.sqrt(M**2 - 1))
    return term1 * term2 - term3

def mach_angle(M):
    """Calcula o ângulo de Mach em radianos."""
    return np.arcsin(1 / M)

# Parâmetros iniciais
M_exit = 3.0  # Número de Mach desejado na saída
theta_max = prandtl_meyer(M_exit) / 2  # Ângulo máximo de expansão
num_points = 50  # Número de pontos para discretização

# Criando a malha inicial
theta = np.linspace(0, theta_max, num_points)
nu = theta  # Ângulo de Prandtl-Meyer no ponto inicial
M = np.zeros(num_points)

# Iteração para encontrar M a partir de nu
for i in range(num_points):
    func = lambda M: prandtl_meyer(M) - nu[i]
    M[i] = np.linspace(1, 10, 1000)[np.argmin(np.abs(func(np.linspace(1, 10, 1000))))]

# Coordenadas da malha
x = np.zeros(num_points)
y = np.zeros(num_points)

# Gerar o contorno da tubeira
x_wall = [0]  # Coordenada x da parede
y_wall = [0]  # Coordenada y da parede

for i in range(1, num_points):
    dx = np.cos(theta[i])
    dy = np.sin(theta[i])
    x_wall.append(x_wall[-1] + dx)
    y_wall.append(y_wall[-1] + dy)

# Plotando os resultados
plt.figure(figsize=(10, 6))
plt.plot(x_wall, y_wall, label="Parede da Tubeira", color="blue")
plt.scatter(x, y, color="red", s=10, label="Malha de Características")
plt.xlabel("x (Comprimento)")
plt.ylabel("y (Altura)")
plt.title("Contorno da Tubeira pelo Método das Características")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
