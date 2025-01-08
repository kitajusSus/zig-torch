import torch
import numpy as np 
import zigtorch as zgt
import time
import random

cuda = torch.device('cuda')
torch.set_num_threads(4)
# Generating random size of a 2D tensor (matrix)
I = random.randint(1000, 10000)
J = random.randint(1000, 10000)
K = random.randint(1000, 10000)
print(f"I={I}, J={J}, K={K}. Random size in range 1000-10000")

matrix_size1 = (I, J)
matrix_size2 = (J, K)

# Test for multiplying 
def pytorch_test(A, B, n):
    print("Number of matrices for pytorch:", n)
    results = []
    for i in range(n):
        start1 = time.time()
        C = torch.mm(A[i], B[i])  # Mnożenie macierzy
        end1 = time.time()
        print(f'{i} time : {end1-start1}')
        results.append(end1 - start1)
    return sum(results), results

def zig_mm(A, B, n):
    print("Number of matrices for zigtorch:", n)
    results = []
    for i in range(n):
        start2 = time.time()
        D = zgt.mm(A[i], B[i])
        end2 = time.time()
        print(f'{i} time : {end2-start2}')
        results.append(end2 - start2)
    return sum(results), results

def generate_matrix(matrix_size):
    return torch.rand(*matrix_size, dtype=torch.float32)

def generate_matrices(matrix_size1, matrix_size2, n):
    aa = [generate_matrix(matrix_size1) for _ in range(n)]
    ba = [generate_matrix(matrix_size2) for _ in range(n)]
    return aa, ba

def move_to_cuda(matrices):
    return [matrix.to(cuda) for matrix in matrices]

# Generate matrices
aa, ba = generate_matrices(matrix_size1, matrix_size2, 100)

# Move matrices to CUDA for PyTorch test
aa_cuda = move_to_cuda(aa)
ba_cuda = move_to_cuda(ba)

# Run tests
pytorch_total_time, pytorch_times = pytorch_test(aa, ba, 100)
zigtorch_total_time, zigtorch_times = zig_mm(aa, ba, 100)

print(f"Execution time for pytorch.mm (cuda): {pytorch_total_time:.9f} seconds")
print(f"Execution time for zigtorch.mm: {zigtorch_total_time:.9f} seconds")




