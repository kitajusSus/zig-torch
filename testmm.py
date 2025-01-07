import torch
import numpy as np 
import zigtorch as zgt
import time
import random
import os
number = 50
#num_cores = os.cpu_count()
#print(f"Number of CPU cores available: {num_cores}")


# Generating random size of a 2D tensor (matrix)
I = random.randint(1000, 5000)
J = random.randint(1000, 5000)
K = random.randint(1000, 5000)


matrix_size1 = (I, J)
matrix_size2 = (J, K)
print("works : 1")
# Test for multiplying 
def pytorch_test(A, B, n):
    print("Number of matrices for pytorch:", n)
    results = []
    for i in range(n):
        start1 = time.time()
        C = torch.mm(A[i], B[i])  # Mnożenie macierzy
        end1 = time.time()
        # print(f'{i} time : {end1-start1}')
        results.append(end1 - start1)
    return sum(results), results

def zig_mm(A, B, n):
    print("Number of matrices for zigtorch:", n)
    results = []
    for i in range(n):
        start2 = time.time()
        D = zgt.mm(A[i], B[i])
        end2 = time.time()
        #print(f'{i} time : {end2-start2}')
        results.append(end2 - start2)
    return sum(results), results

def generate_matrix(matrix_size):
    return torch.rand(*matrix_size, dtype=torch.float32)

aa = []
bb = []
for i in range(number): 
    aa.append(generate_matrix(matrix_size1))
    bb.append(generate_matrix(matrix_size2))
print("test generate: true")
def move_to(matrices):
    return [matrix.to(device) for matrix in matrices]

# Generate matrices
device = torch.device('cuda')
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please check your CUDA installation.")

aa_torch = move_to(aa)
bb_torch = move_to(bb)  
print("works:2")


# Run tests

pytorch_total_time, pytorch_times = pytorch_test(aa_torch, bb_torch, number)
zigtorch_total_time, zigtorch_times = zig_mm(aa, bb, number)
print(f"I={I}, J={J}, K={K}. Random size in range 1000-5000")
print(f"number of matrices to multiply: {number}")
print(f"Execution time for pytorch.mm cuda : {pytorch_total_time:.9f} seconds")
print(f"Execution time for zigtorch.mm: {zigtorch_total_time:.9f} seconds")




