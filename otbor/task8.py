import numpy as np
from numpy.linalg import matrix_power

def matrix_mod_power(matrix, exp, mod):
    result = np.eye(matrix.shape[0], dtype=np.ulonglong)

    matrix = np.copy(matrix)

    while exp > 0:

        if exp % 2 == 1:
            result = (result @ matrix) % mod

        matrix = (matrix @ matrix) % mod
        exp //= 2

    return result

a3 = np.array([[0,1,1], [0,1,0], [0,0,1]])
a2 = np.array([[1,0,0], [1,0,1], [0,0,1]])
a1 = np.array([[1,0,0], [0,1,0], [1,1,0]])

step = (a1 @ (a2 @ a3)).astype(np.ulonglong)

modfactor = 1000000007 

num_of_input_data = int(input())

for _ in range(num_of_input_data):
    n, k = input().split(' ')
    init_arr = np.array(input().split(' '), dtype=np.ulonglong) % modfactor
    operator = matrix_mod_power(step, int(k), modfactor)
    ans = (operator @ init_arr) % modfactor
    print(*ans, sep=' ')