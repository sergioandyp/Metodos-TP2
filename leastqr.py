import matplotlib as mpl
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

# Resuelve el problema de cuadrados minimos lineal utilizando descomposicion QR
def leastq(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    Q, R = qr(A)
    return triangsup(R, Q.T @ b)

# Calcula la descomposicion QR por Gram-Schmidt
def qr(A: np.ndarray):
    Q = np.ndarray(A.shape)
    R = np.zeros([A.shape[1], A.shape[1]])

    R[0][0] = np.linalg.norm(A[:, 0])
    Q[:, 0] = A[:, 0] / R[0][0]
    for i in range(1, A.shape[1]):
        Q[:, i] = A[:, i]

        for j in range(0, i):  # Resta de las proyecciones
            R[j][i] = np.dot(Q[:, j], A[:, i])
            Q[:, i] -= R[j][i] * Q[:, j]

        R[i][i] = np.linalg.norm(Q[:, i])       # Normalizacion
        Q[:, i] /= R[i][i]

    return Q, R


def triangsup(A: np.ndarray, b: np.ndarray):  #Ax = b ,  A triangular superior nxn
    x = np.zeros([A.shape[1], b.shape[1]])

    for j in range (0, x.shape[1]):
        x[-1][j] = b[-1][-1]/A[-1][-1]

        for i in range(x.shape[j] - 2, -1, -1):
            x[i][j] = (b[i][j] - np.dot(A[i, i+1: A.shape[1]], x[i + 1: x.shape[0], 0]))/A[i][i]

    return x

def test():

    A = []
    b = []

    # Ejemplo sencillo
    A.append(np.array([[1, 2], [0, 5]]))
    b.append(np.array([[5], [4]]))
    
    # Ejemplo de la clase
    A.append(np.array([[1.02,1],[1.01,1],[0.94,1], [0.99, 1]]))
    b.append(np.array([[2.05],[1.99],[2.02],[1.93]]))


    for i in range(len(A)):
        print("Matriz A:")
        print(A[i])
        print("Matriz b:")
        print(b[i])
        # Q, R = qr(A)
        # print(Q)
        # print(R)
        x = leastq(A[i], b[i])
        print("Solucion:")
        print(x)
        x = np.linalg.lstsq(A[i], b[i], rcond=None)[0]
        print("Solucion por numpy:")
        print(x)


def sonido():

    df  = pd.read_csv('sound.txt', header=None, names=['ti','yi'], dtype={'ti':np.float64,'yi':np.float64}, sep=' ')
    t  = np.array(df['ti'])
    
    N = len(t)

    b = np.ndarray([N, 1])
    b[:, 0]  = np.array(df['yi'])


    A = np.ndarray([N, 6])  # [ a1, b1, a2, b2, a3, b3 ]
    
    for i in range(3):
        A[:, 2*i] = np.cos(1000*(i+1)*np.pi*t)
        A[:, 2*i+1] = np.sin(1000*(i+1)*np.pi*t)

    x = leastq(A, b)

    y = A @ x   # Solucion ajustada

    e = b - y # Error

    return y, e


if (__name__ == "__main__"):

    test()

# Ploteo de las curvas de sound original y ajustada
    # y, e = sonido()

    # df  = pd.read_csv('sound.txt', header=None, names=['ti','yi'], dtype={'ti':np.float64,'yi':np.float64}, sep=' ')
    # mpl.rcParams['font.size'] = 16

    # plt.plot(df['ti'], df['yi'], 'o', label='Original')
    # plt.plot(df['ti'], y, label='Ajustada')
    # # plt.plot(e, label='error')
    # plt.xlabel('$tiempo$', fontsize=28)
    # plt.ylabel('$y$', fontsize=28)
    # plt.xlim(0, 0.01)
    # plt.legend()
    # plt.grid()
    # plt.show()