import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Constantes
gamma = 1.4  # Razão de capacidades caloríficas (ar ideal)
M_exit = 5

# Funções auxiliares
def prandtl_meyer(M):
    """Calcula o ângulo de Prandtl-Meyer em radianos. v"""
    term1 = np.sqrt((gamma + 1) / (gamma - 1))
    term2 = np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1)))
    term3 = np.arctan(np.sqrt(M**2 - 1))
    return term1 * term2 - term3


def find_zero(ponto, angulo, y_raiz):
    # Ponto por onde a reta passa
    x0, y0 = ponto
    
    # Converter o ângulo de graus para radianos
    angulo_rad = np.deg2rad(angulo)
    
    # Calcular a inclinação m da reta
    a = np.tan(angulo_rad)
    b = y0 - a*x0

    x_raiz = (y_raiz-b)/1#a
    
    return a, b, x_raiz

def find_zero1(ponto, angulo, y_raiz):
    # Ponto por onde a reta passa
    x0, y0 = ponto
    
    # Converter o ângulo de graus para radianos
    angulo_rad = np.deg2rad(angulo)
    
    # Calcular a inclinação m da reta
    a = np.tan(angulo_rad)
    b = y0 - a*x0

    x_raiz = (y_raiz-b)/a
    
    return a, b, x_raiz

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


def find_intersection2(x1, y1, v1, theta1, x2, y2, v2, theta2):
    """
    Calcula as coordenadas (x, y) do nó no método das características.

    Parâmetros:
    - x1, y1: coordenadas do nó na linha C+
    - v1: ângulo de Prandtl-Meyer no nó na linha C+
    - theta1: ângulo do fluxo no nó na linha C+
    - x2, y2: coordenadas do nó na linha C-
    - v2: ângulo de Prandtl-Meyer no nó na linha C-
    - theta2: ângulo do fluxo no nó na linha C-

    Retorna:
    - x, y: coordenadas do nó calculadas
    """
    # Inclinações das linhas características
    m1 = np.tan(np.deg2rad(theta1 + v1))  # Inclinação da linha C+
    m2 = np.tan(np.deg2rad(theta2 - v2))  # Inclinação da linha C-

    # Sistema de equações:
    # y - y1 = m1 * (x - x1)  -->  y = m1 * (x - x1) + y1
    # y - y2 = m2 * (x - x2)  -->  y = m2 * (x - x2) + y2

    # Igualando as duas expressões para y:
    # m1 * (x - x1) + y1 = m2 * (x - x2) + y2

    # Resolvendo para x:
    x = ((m1 * x1 - m2 * x2) + (y2 - y1)) / (m1 - m2)

    # Substituindo x em uma das equações para encontrar y:
    y = m1 * (x - x1) + y1

    return x, y

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
    return xa + (yb - ya)/np.tan(thetaA)
    

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


def xyFind(xa, ya, alpha, xb, yb, beta):
    a1, b1, x1 = find_zero((xa, ya), alpha, 0)
    a2, b2, x2 = find_zero((xb, yb), beta, 0)

    #y = ax + b iguais
    x  = (b2 - b1)/(a1-a2)
    y = a1*x + b1

    return x, y


    '''# Converter os ângulos de graus para radianos
    alpha_rad = np.deg2rad(alpha)
    beta_rad = np.deg2rad(beta)
    
    # Calcular as tangentes dos ângulos
    tan_alpha = np.tan(alpha_rad)
    tan_beta = np.tan(beta_rad)
    
    # Verificar se as retas são paralelas
    if np.isclose(tan_alpha, tan_beta, rtol=1e-9):
        return None  # Retorna None se as retas forem paralelas (sem interseção ou infinitas interseções)
    
    # Calcular X
    X = ((yb - ya) + xb * tan_beta - xa * tan_alpha) / (tan_alpha - tan_beta)
    
    # Calcular Y usando a equação da reta que passa por A
    Y = ya + tan_alpha * (X - xa)
    
    return X, Y'''





def printar(node, point):
    point = node['Point']
    Kminus = np.rad2deg(node['theta'] + node["nu"])
    Kplus = np.rad2deg(node['theta'] - node["nu"])
    theta = np.rad2deg(node['theta'])
    v = np.rad2deg(node["nu"])
    M = node["mach"]
    mu = np.rad2deg(node["mu"])

    node = {
    "Point": point,
    "coordenadas": {"x": None, "y": None},       # Coordenadas do nó
    "theta": theta,                         # Ângulo de inclinação (graus)
    "mu": mu,                               # Ângulo de Mach (graus)
    "nu": v,                                # Função de Prandtl-Meyer (graus)
    "mach": M,                              # Número de Mach
    "Kminus": Kminus,
    "Kplus": Kplus
    }
    return node

def populate(n, M_exit, nozzle_radius):
    alpha = 1
    nodes_count = 0
    contador = 1
    for i in range(n):
        nodes_count += n + alpha
        alpha -= 1
    print(f"Serão analisados {nodes_count} nós")
    nodes = []

    theta_max = prandtl_meyer(M_exit)
    thetas = np.linspace(np.deg2rad(0.00375), theta_max/2, n)
    
    
    #Primeira linha
    for i in range(n):
        theta = thetas[i]
        if i == 0:
            diegoMach, diegoSteps = bissecao(prandtl_meyer_optimize, 1, M_exit, theta)
            node = montar_node(np.sin(theta)*nozzle_radius, 0, theta, np.deg2rad(mach_angle(diegoMach)), theta, diegoMach)
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
        node = printar(node, a)
        nodes[a-1] = node
        a+=1

    return theta_max/2, nodes, thetas


def positions(n, theta_max, thetas, nodes, nozzle_raidus):
    contador = 0
    for i in range(n+1, 1, -1):
        for j in range(0, i, 1):
            node = nodes[contador]
            if j == 0:
                #Centro
                if i == n+1:
                    xa = 0
                    ya = nozzle_raidus
                    alpha = -90 + np.rad2deg(thetas[j])

                    xb = 0
                    yb = 0
                    beta = 0

                    x, y = xyFind(xa, ya, alpha, xb, yb, beta)

                    node["coordenadas"]["x"] = x
                    node["coordenadas"]["y"] = y

                else:
                    node_p = nodes[contador-i]

                    xa = 0
                    ya = 0
                    alpha = 0

                    xb = node_p["coordenadas"]["x"]
                    yb = node_p["coordenadas"]["y"]
                    beta = node_p["theta"] - node_p["mu"]

                    x, y = xyFind(xa, ya, alpha, xb, yb, beta)
                    print(y)
                    node["coordenadas"]["x"] = x
                    node["coordenadas"]["y"] = y
        
            elif j == i-1:
                #Diego Certo
                #Wall
                if i == n+1:
                    xa = 0
                    ya = nozzle_raidus
                    alpha = (theta_max + node["theta"])/2

                    xb = nodes[contador-1]["coordenadas"]["x"]
                    yb = nodes[contador-1]["coordenadas"]["y"]
                    beta = (nodes[contador-1]["theta"] + nodes[contador-1]["mu"])

                    x, y = xyFind(xa, ya, alpha, xb, yb, beta)
                    node["coordenadas"]["x"] = x
                    node["coordenadas"]["y"] = y
                    
                else:
                    node_p = nodes[contador-i]
                    xa = node_p["coordenadas"]["x"] 
                    ya = node_p["coordenadas"]["y"]
                    alpha = (node_p["theta"] + node["theta"])/2
               
                    xb = nodes[contador-1]["coordenadas"]["x"]
                    yb = nodes[contador-1]["coordenadas"]["y"]
                    beta = (nodes[contador-1]["theta"] + nodes[contador-1]["mu"])

                    x, y = xyFind(xa, ya, alpha, xb, yb, beta)
                    node["coordenadas"]["x"] = x
                    node["coordenadas"]["y"] = y

            else:
                #nodes
                if i == n+1:
                    #Diego CErto
                    node_p = nodes[contador-1]

                    xa = 0
                    ya = nozzle_raidus
                    alpha = - (90-np.rad2deg(thetas[j]))

                    xb = node_p["coordenadas"]["x"]
                    yb = node_p["coordenadas"]["y"]
                    beta = (node_p["theta"] + node_p["mu"])

                    x, y = xyFind(xa, ya, alpha, xb, yb, beta)
                    node["coordenadas"]["x"] = x
                    node["coordenadas"]["y"] = y

                else:
                    #Diego Certo
                    node_c = nodes[contador-i]
                    node_p = nodes[contador-1]

                    xa = node_c["coordenadas"]["x"]
                    ya = node_c["coordenadas"]["y"]
                    alpha = node_p["theta"] - node_p["mu"]

                    xb = node_p["coordenadas"]["x"]
                    yb = node_p["coordenadas"]["y"]
                    beta = (node_p["theta"] + node_p["mu"])

                    x, y = xyFind(xa, ya, alpha, xb, yb, beta)
                    node["coordenadas"]["x"] = x
                    node["coordenadas"]["y"] = y
            
            contador += 1
    
    return nodes


def pressures(nodes, m_exit, p_opt):
    P0 = p_opt * (1 + (gamma - 1) / 2 * m_exit**2) ** (gamma / (gamma - 1))
    contador = len(nodes)
    for i in range(contador):
        node = nodes[i]
        node['P'] = P0 * (1 + (gamma - 1) / 2 * node['mach']**2) ** (-gamma / (gamma - 1))

    return nodes


'''
def pressuresout(nodes, p_opt):
    contador = len(nodes)
    for i in range(n+1, 1, -1):
        for j in range(0, i, 1):
            node = nodes[contador]
            if j == 0:
                #Centro
                if i == n+1:
                else:
            elif j == i-1:
                #Wall
                if i == n+1:
                else:
            else:
                #nodes
                if i == n+1:
                else:
            contador -= 1
    return nodes
'''


node_quantity = 7
mach_exit = 2.4
the = []
tam = []
area = []
def goal():
    theta_max, nodes, thetas = populate(node_quantity, mach_exit, 0.1)
    the.append(np.rad2deg(theta_max))
    for node in nodes:
        point = node['Point']
        Kminus = node['Kminus']
        Kplus = node['Kplus']
        theta = node['theta']
        v = node["nu"]
        M = node["mach"]
        mu = node["mu"]
        print(f"{point:7.3f} - {Kminus:7.3f} - {Kplus:7.3f} - {theta:7.3f} - {v:7.3f} - {M:7.3f} - {mu:7.3f}")

    nodes = positions(node_quantity, theta_max, thetas, nodes, 0.1)
    nodes = pressures(nodes, mach_exit, 101300)
    xs = [0]
    ys = [0.1]
    ts = [0]
    Ms = [1]
    ps = [0]


    xs_wall = [0]
    ys_wall = [0.1]
    ts_wall = [0]

    xs_node = []
    ys_node = []
    ts_node = []

    xs_center = []
    ys_center = []
    ts_center = []
    contador = 0
    for i in range(node_quantity+1, 1, -1):
        for j in range(0, i, 1):
            node = nodes[contador]
            if j == i-1:
                #wall
                xs_wall.append(node["coordenadas"]["x"])
                ys_wall.append(node["coordenadas"]["y"])
                ts_wall.append(node["Point"])
            elif j == 0:
                xs_center.append(node["coordenadas"]["x"])
                ys_center.append(node["coordenadas"]["y"])
                ts_center.append(node["Point"])
            else:
                xs_node.append(node["coordenadas"]["x"])
                ys_node.append(node["coordenadas"]["y"])
                ts_node.append(node["Point"])
            xs.append(node["coordenadas"]["x"])
            ys.append(node["coordenadas"]["y"])
            ts.append(node["Point"])
            Ms.append(node["mach"])
            ps.append(node['P'])
            contador +=1



plt.plot(machs, the)
plt.title("Theta_max em função do mach de saída")
plt.xlabel("Mach de saída")
plt.ylabel("Theta_max = v(M_e)/2")
plt.grid()
plt.show()


'''

for i in range(len(xs)):
    if ys[i] != None:
        plt.text(xs[i], ys[i]*1.03, str(ts[i]), color='black', fontsize=10, ha='center')
plt.scatter(xs_wall, ys_wall, color='black')
plt.scatter(xs_center, ys_center, color='g')
plt.scatter(xs_node, ys_node, color='b')

plt.plot(xs_wall, ys_wall, 'k', label="Wall")
plt.legend()
#plt.plot([0, 0.0007], [0.1, -152.786*0.0007+0.1])


a, b, x0 = find_zero1((0, 0.1), -90+0.375, 0)
plt.plot([0, x0], [0.1, a*x0+b])
a1, b1, x1 = find_zero1((x0, a*x0+b), 74.1+0.375, 0.1)
x1 = x1/5
plt.plot([x0, x1], [a*x0+b, a1*x1+b1])
a, b, x0 = find_zero1((0, 0.1), -90+3.375, 0)
plt.plot([0, x0], [0.1, a*x0+b])
plt.show()


triang = tri.Triangulation(xs, ys)

# Criando o gráfico de contornos preenchidos
plt.figure(figsize=(6, 6))
plt.plot(xs_wall, ys_wall, "k")
plt.tricontourf(triang, Ms, cmap='viridis')
plt.colorbar(label='Mach')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mach ao longo da tubeira')
plt.show()










plt.figure(figsize=(6, 6))

for i in range(len(xs)):
    if ys[i] != None:
        plt.text(xs[i], ys[i]*1.03, str(ts[i]), color='black', fontsize=10, ha='center')
plt.scatter(xs_wall, ys_wall, color='black')
plt.scatter(xs_center, ys_center, color='g')
plt.scatter(xs_node, ys_node, color='b')
plt.plot(xs_wall, ys_wall, 'k', label="Wall")
triang = tri.Triangulation(xs, -np.array(ys))
plt.plot(xs_wall, -np.array(ys_wall), "k")
plt.tricontourf(triang, Ms, cmap='viridis')
plt.colorbar(label='Mach')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mach ao longo da tubeira')
plt.show()

'''

triang = tri.Triangulation(xs, np.array(ys))
# Criando o gráfico de contornos preenchidos
plt.figure(figsize=(6, 6))
plt.plot(xs_wall, ys_wall, "k")
plt.tricontourf(triang, ps, cmap='viridis')
plt.colorbar(label='Pressure')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mach ao longo da tubeira')
plt.show()
