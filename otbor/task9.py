# import numpy as np

# n = int(input())
# m = int(input())
# matrix = []
# for _ in range(n):
#     row = input().split(',')
#     row = [int(number.strip()) for number in row]
#     matrix.append(row)
# matrix = np.array(matrix)

# def R(i, j):

#     if (i == 0) and (j == 0):
#         return matrix[i,j]
    
#     if (i < 0) or (i>j):
#         return -100
    
#     return matrix[i, j] + max(R(i-1, j-1), R(i, j-1))

# print(R(n-1, m-1))


import numpy as np

n = int(input())
m = int(input())
matrix = []
for _ in range(n):
    row = input().split(',')
    row = [int(number.strip()) for number in row]
    matrix.append(row)
matrix = np.array(matrix)


memo = {}

def R(i, j):

    if (i, j) in memo:
        return memo[(i, j)]

    if (i == 0) and (j == 0):
        return matrix[i, j]

    if (i < 0) or (i > j):
        return -100

    memo[(i, j)] = matrix[i, j] + max(R(i - 1, j - 1), R(i, j - 1))
    return memo[(i, j)]

print(R(n - 1, m - 1))


