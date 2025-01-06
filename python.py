import torch
import numpy as np 
import zigtorch as zgt
import time
import random

I = random.randint(100, 1000)
J = random.randint(100,1000)
K = random.randint(100,1000)
print(f"I={I}, J={J}, K={K}. Random in range 1000-10000")

A = torch.randn(1000, 1000, dtype=torch.float32)
B = torch.randn(1000, 1000, dtype=torch.float32)

def torch_mm(A,B):
    return np.matmul(A,B)

start1 = time.time()
C = torch_mm(A,B)
end1 = time.time()
print(f"Execution time for torch.mm: {end1-start1}, seconds")

def zig_mm(A,B):
    return zgt.mm(A,B)

start2 = time.time()
D = zig_mm(A,B)
end2 = time.time()
print(f"Execution time for zig.mm: {end2-start2}, seconds")





