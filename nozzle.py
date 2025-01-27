import numpy as np
import matplotlib.pyplot as plt

# Constantes
gamma = 1.4  # Razão de capacidades caloríficas (ar ideal)
M_exit = 3

# Funções auxiliares
def prandtl_meyer(M):
    """Calcula o ângulo de Prandtl-Meyer em radianos. v"""
    term1 = np.sqrt((gamma + 1) / (gamma - 1))
    term2 = np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1)))
    term3 = np.arctan(np.sqrt(M**2 - 1))
    return term1 * term2 - term3

def mach_angle(M):
    """Calcula o ângulo de Mach em radianos. Mi"""
    return np.arcsin(1 / M)

def prandtl_meyer_optimize(M, v_real):
    """Calcula o ângulo de Prandtl-Meyer em radianos. v"""
    term1 = np.sqrt((gamma + 1) / (gamma - 1))
    term2 = np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1)))
    term3 = np.arctan(np.sqrt(M**2 - 1))
    v_new = term1 * term2 - term3
    return v_new - v_real

def calcular_intersecao(x1, y1, a1, x2, y2, a2):
    """
    Calcula as coordenadas de interseção (x, y) de duas retas definidas por:
    - Ponto 1 (x1, y1) e inclinação a1 (em radianos)
    - Ponto 2 (x2, y2) e inclinação a2 (em radianos)

    Retorna:
        (x, y): Coordenadas do ponto de interseção ou None se as retas forem paralelas.
    """
    # Calcula os coeficientes angulares (direções das retas)
    a1 = np.rad2deg(a1)
    a2 = np.rad2deg(a2)

    v1_x, v1_y = np.cos(a1), np.sin(a1)
    v2_x, v2_y = np.cos(a2), np.sin(a2)
    
    # Determinante da matriz
    det = v1_x * (-v2_y) - v1_y * (-v2_x)
    
    # Verifica se o determinante é zero (retas paralelas ou coincidentes)
    if abs(det) < 1e-9:  # Tolerância para considerar paralelas
        return None  # Retas paralelas ou coincidentes, sem interseção única
    
    # Calcula t e s (parâmetros para interseção)
    t = ((x2 - x1) * (-v2_y) - (y2 - y1) * (-v2_x)) / det
    
    # Coordenadas do ponto de interseção
    x = x1 + t * v1_x
    y = y1 + t * v1_y
    
    return x, y

def bissecao(f, a, b, v3, rtol=1e-5, iter_max=1000):
    """
    Método da Bisseção para encontrar uma raiz da função f no intervalo [a, b].

    Parâmetros:
        f (callable): Função para a qual se busca a raiz.
        a (float): Limite inferior do intervalo.
        b (float): Limite superior do intervalo.
        rtol (float): Tolerância relativa para o critério de parada. Default é 1e-5.
        iter_max (int): Número máximo de iterações. Default é 100.

    Retorna:
        raiz (float): Aproximação da raiz encontrada.
        iteracoes (int): Número de iterações realizadas.

    Lança:
        ValueError: Se f(a) e f(b) tiverem o mesmo sinal.
    """
    if f(a, v3) * f(b, v3) > 0:
        raise ValueError("A função deve ter sinais opostos em a e b.")

    iteracoes = 0
    while iteracoes < iter_max:
        c = (a + b) / 2  # Ponto médio
        fc = f(c, v3)

        # Critério de parada: f(c) próximo de zero ou intervalo pequeno o suficiente
        if abs(fc) < rtol or abs(b - a) / 2 < rtol:
            return c, iteracoes

        # Atualizar os limites do intervalo
        if f(a, v3) * fc < 0:
            b = c
        else:
            a = c

        iteracoes += 1

    # Se o número máximo de iterações for atingido, retornar a última estimativa
    return (a + b) / 2, iteracoes

def node3_calculator(node1, node2):
    # Dados M, theta, x e y em 1 e 2, calcula 3
    # Pra 1:
    v1 = node1["nu"]
    mu1 = node1["mu"]
    theta1 = node1["theta"]
    Kminus1 = theta1 + v1

    # Pra 2:
    v2 = node2["nu"]
    mu2 = node2["mu"]
    theta2 = node2["theta"]
    Kplus2 = theta2 - v2

    # Pra 3:
    theta3 = (Kminus1 + Kplus2)/2
    v3 = (Kminus1 - Kplus2)/2
    M3, steps = bissecao(prandtl_meyer_optimize, 1, M_exit+1, v3)
    mu3 = mach_angle(M3)

    Cminus = np.rad2deg(((theta1-mu1) + (theta3-mu3))/2)
    Cplus =  np.rad2deg(((theta2+mu2) + (theta3+mu3))/2)

    x3 = -  (node1["coordenadas"]["x"]*np.tan(Cminus) - node2["coordenadas"]["x"]*np.tan(Cplus) + (node2["coordenadas"]["y"] - node1["coordenadas"]["y"])) / (np.tan(Cminus)-np.tan(Cplus))
    y3 = (np.tan(Cminus)*np.tan(Cplus)*(node1["coordenadas"]["x"]-node2["coordenadas"]["x"]) + np.tan(Cminus)*node2["coordenadas"]["y"] - np.tan(Cplus)*node1["coordenadas"]["y"])    /      (np.tan(Cminus)-np.tan(Cplus))

    node3 = {
    "Point": None,
    "coordenadas": {"x": x3, "y": y3},       # Coordenadas do nó
    "theta": theta3,                         # Ângulo de inclinação (graus)
    "mu": mu3,                               # Ângulo de Mach (graus)
    "nu": v3,                                # Função de Prandtl-Meyer (graus)
    "mach": M3,                              # Número de Mach
    }
    return node3


def calcular_xb(xa, ya, thetaA, yb):
    """
    Calcula a coordenada xB do ponto B que passa por uma reta com inclinação thetaA
    e que também passa pelo ponto A (xa, ya).
    
    Args:
        xa (float): Coordenada x do ponto A.
        ya (float): Coordenada y do ponto A.
        thetaA (float): Inclinação da reta em graus.
        yb (float): Coordenada y do ponto B.
    
    Returns:
        float: Coordenada x do ponto B (xB).
    """
    # Converte thetaA para radianos
    thetaA_rad = np.rad2deg(thetaA)
    
    # Verifica se a reta é vertical
    #if np.isclose(np.cos(thetaA_rad), 0, abs_tol=1e-9):
    #    raise ValueError("A reta é vertical, xB será igual a xA.")
    
    # Calcula xB usando a fórmula
    xb = xa + (yb - ya) / np.tan(thetaA_rad)
    return xb

def wall_point(node1, wallPrev):
    theta1 = node1["theta"]
    mu1 = node1["mu"]
    v1 = node1["nu"]
    M1 = node1["mach"]
    Kplus1 = theta1 - v1

    thetaPrev = wallPrev["theta"]
    muPrev = wallPrev["mu"]
    vPrev = wallPrev["nu"]
    MPrev = wallPrev["mach"]

    x, y = calcular_intersecao(node1["coordenadas"]["x"], node1["coordenadas"]["y"], Kplus1, wallPrev["coordenadas"]["x"], wallPrev["coordenadas"]["y"], (theta1+thetaPrev)/2)


    wall_node = {
    "Point":None,
    "coordenadas": {"x": x, "y": y},      # Coordenadas do nó
    "theta": theta1,                             # Ângulo de inclinação (graus)
    "mu": mu1,                                # Ângulo de Mach (graus)
    "nu": v1,                                # Função de Prandtl-Meyer (graus)
    "mach": M1,                              # Número de Mach
    }
    return wall_node

def primeiro_wall_point(node1, angle, nozzle_radius):
    theta1 = node1["theta"]
    mu1 = node1["mu"]
    v1 = node1["nu"]
    M1 = node1["mach"]
    Kplus1 = theta1 - v1
    
    x, y = calcular_intersecao(node1["coordenadas"]["x"], node1["coordenadas"]["y"], Kplus1, 0, nozzle_radius, angle)


    wall_node = {
    "Point":None,
    "coordenadas": {"x": x, "y": y},      # Coordenadas do nó
    "theta": theta1,                             # Ângulo de inclinação (graus)
    "mu": mu1,                                # Ângulo de Mach (graus)
    "nu": v1,                                # Função de Prandtl-Meyer (graus)
    "mach": M1,                              # Número de Mach
    }
    return wall_node



def montar_node(x, y, theta, mu, v, M):
    node = {
    "Point":None,
    "coordenadas": {"x": x, "y": y},      # Coordenadas do nó
    "theta": theta,                             # Ângulo de inclinação (graus)
    "mu": mu,                                # Ângulo de Mach (graus)
    "nu": v,                                # Função de Prandtl-Meyer (graus)
    "mach": M,                              # Número de Mach
    }
    return node

def node2_calculator(node1, angle, nozzle_radius):
    M, steps = bissecao(prandtl_meyer_optimize, 1, M_exit, angle)
    mu = mach_angle(M)
    Kplus = node1["theta"] - node1["mu"]
    x, y = calcular_intersecao(node1["coordenadas"]['x'], node1["coordenadas"]['y'], Kplus, 0, nozzle_radius, angle)
    node = {
    "Point":None,
    "coordenadas": {"x": x, "y": y},      # Coordenadas do nó
    "theta": angle,                             # Ângulo de inclinação (graus)
    "mu": mu,                                # Ângulo de Mach (graus)
    "nu": angle,                                # Função de Prandtl-Meyer (graus)
    "mach": M,                              # Número de Mach
    }
    return node
    
def node_center(node1):
    theta = 0
    Kminus = node1["theta"] + node1["nu"]
    Kplus = -Kminus
    v = Kminus
    M, steps = bissecao(prandtl_meyer_optimize, 1, M_exit+1, v)
    mu = mach_angle(M)
    x = calcular_xb(node1["coordenadas"]['x'], node1["coordenadas"]['y'], Kminus, 0)

    node = {
    "Point":None,
    "coordenadas": {"x": x, "y": 0},      # Coordenadas do nó
    "theta": theta,                             # Ângulo de inclinação (graus)
    "mu": mu,                                # Ângulo de Mach (graus)
    "nu": v,                                # Função de Prandtl-Meyer (graus)
    "mach": M,                              # Número de Mach
    }
    return node

def printar(node, point):
    Kminus = np.rad2deg(node['theta'] + node["nu"])
    Kplus = np.rad2deg(node['theta'] - node["nu"])
    theta = np.rad2deg(node['theta'])
    v = np.rad2deg(node["nu"])
    M = node["mach"]
    mu = np.rad2deg(node["mu"])
    print(f"{point:3.0f} - {Kminus:.2f} - {Kplus:.2f} - {theta:.2f} - {v:.2f} - {M:.2f} - {mu:.2f}")

def main(n, M_exit, nozzle_radius):
    alpha = 1
    nodes_count = 0
    contador = 1
    for i in range(n):
        nodes_count += n + alpha
        alpha -= 1
    print(f"Serão analisados {nodes_count} nós")
    nodes = []

    theta_max = prandtl_meyer(M_exit)
    thetas = np.linspace(0, theta_max/2, n)
    
    
    #Primeira linha
    for i in range(n):
        theta = thetas[i]
        if i == 0:
            node = montar_node(np.sin(theta)*nozzle_radius, 0, theta, mach_angle(1), theta, 1)
        else:
            node = node2_calculator(nodes[0], theta, nozzle_radius)
        node['Point'] = contador
        contador +=1
        nodes.append(node)
    
    #Primeiro da wall
    node = primeiro_wall_point(nodes[-1], theta_max, nozzle_radius)
    node['Point'] = contador
    contador +=1
    nodes.append(node)

    last_wall = nodes[-1]

    # Próximos
    beta = 1
    for i in range(n, 1, -1):
        #Calcula center line
        node = node_center(nodes[-i])
        node['Point'] = contador
        contador +=1
        nodes.append(node)
        center_node = nodes[-1]
        
        #Calcula intermediários
        beta += 1
        for alpha in range(0, n-beta, 1):
            node = node3_calculator(nodes[-(i)], center_node)
            node['Point'] = contador
            contador +=1
            nodes.append(node)
        
        #Calcula Wall
        node = wall_point(nodes[-1], last_wall)
        node['Point'] = contador
        contador +=1
        nodes.append(node)
        last_wall = nodes[-1]

    a = 1
    for node in nodes:
        printar(node, a)
        a+=1

    xs = []
    ys = []
    for node in nodes:
        xs.append(node["coordenadas"]["x"])
        ys.append(node["coordenadas"]["y"])

    plt.scatter(xs, ys)
    plt.show()
main(7, 2.4, 0.1)