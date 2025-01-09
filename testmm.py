import torch
import numpy as np 
import zigtorch #or add as zgt
import time
import random
import os
number = 1
#num_cores = os.cpu_count()
#print(f"Number of CPU cores available: {num_cores}")

# Generating random size of a 2D tensor (matrix)
I = random.randint(500,1000)
J = random.randint(500, 1000)
K = random.randint(500, 1000)

n = 1

# Funkcje generujące macierze
def gen_matrices():
    A_matrices = [torch.randn(I, J, dtype=torch.float32) for _ in range(n)]
    B_matrices = [torch.randn(J, K, dtype=torch.float32) for _ in range(n)]
    return A_matrices, B_matrices

# Generowanie macierzy
print("Generowanie macierzy...")
A_list, B_list = gen_matrices()

# Liczenie z zigtorch
print("\nLiczenie z zigtorch:")
zig_times = []
for i in range(n):
    start_time = time.time()
    C = zigtorch.mm(A_list[i], B_list[i])
    zig_time = time.time() - start_time
    zig_times.append(zig_time)
    print(f"Zigtorch.mm time (matrix {i+1}): {zig_time:.6f} seconds")

# Liczenie z pytorch
print("\nLiczenie z pytorch:")
torch_times = []
for i in range(n):
    start_time = time.time()
    torch_result = torch.mm(A_list[i], B_list[i])
    torch_time = time.time() - start_time
    torch_times.append(torch_time)
    print(f"PyTorch.mm time (matrix {i+1}): {torch_time:.6f} seconds")

# Podsumowanie
print("\n After all:")
print(f"Averge time for zigtorch.mm: {sum(zig_times) / n:.6f} seconds")
print(f"Average time for pytorch.mm: {sum(torch_times) / n:.6f} seconds")

print("zigtorch matrix: ",zigtorch.mm(A_list[0], B_list[0]))
print("pytorch matrix:", torch.mm(A_list[0], B_list[0]))
