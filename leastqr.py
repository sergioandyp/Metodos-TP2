import numpy as np

# AT*A*x = AT*b
# Resuelve el problema de cuadrados minimos lineal utilizando descomposicion QR
def leastq(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    Q, R = qr(A)
    return triangsup(R, Q.T @ b)

# Calcula la descomposicion QR por Gram-Schmidt
def qr(A: np.ndarray):
    Q = np.ndarray(A.shape)
    R = np.zeros([A.shape[1], A.shape[1]])

    R[0][0] = np.linalg.norm(A[:,0])
    Q[:,0] = A[:,0] / R[0][0]
    for i in range(1, A.shape[1]):
        Q[:,i] = A[:,i]
        
        for j in range(0, i):   # Resta de las proyecciones
            R[j][i] = np.dot(Q[:,j], A[:,i])
            Q[:,i] -=  R[j][i] * Q[:,j]
        
        R[i][i] = np.linalg.norm(Q[:,i])
        Q[:,i] /= R[i][i]

    return Q,R

def triangsup(A: np.ndarray, b: np.ndarray):  #Ax = b ,  A triangular superior nxn
    x = np.ndarray([A.shape[1], b.shape[1]])
    
    for j in range (0, x.shape[-1]):
        x[-1][j] = b[b.shape[1] - 1][j]/A[A.shape[1]-1][A[A.shape[1]-1]]
        for i in range ( x.shape[j] - 2, -1 , -1):
            x[i][j] = (b[i][j] - A[i ,i+1: -1 ] @ x[i + 1: -1, 0])/A[i][i]
        
    return x

def test():
    A = np.array([[12,-51,4],[6,167,-68],[-4,24,-41]])
    # A = np.array([[1.02,1],[1.01,1],[0.94,1], [0.99, 1]])
    # b = np.array([[2.05],[1.99],[2.02],[1.93]])
    print(A)
    Q, R = qr(A)
    print(Q)
    print(R)
    # print(leastq(A, b))

test()