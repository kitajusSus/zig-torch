import torch
import numpy as np 
import zigtorch as zgt
import time
import random

I = random.randint(1000, 10000)
J = random.randint(1000,10000)
K = random.randint(1000,10000)
print(f"I={I}, J={J}, K={K}. Random size in range 1000-10000")
matrix_size1 = (I,J)
matrix_size2 = (J,K)

A = torch.rand(*matrix_size1, dtype=torch.float32)
B = torch.rand(*matrix_size2, dtype=torch.float32)

def pytorch_test(A,B):
    n=10
    print("Number of matrices for pytorch:",n)
    results = []
    for i in range(n):
        start1 = time.time()
        C = torch.mm(A, B)  # Mnożenie macierzy
        end1 = time.time()
        print(f'{i} time : {end1-start1}')
        results.append(end1 - start1)
    return sum(results), results

def zig_mm(A,B):
    n=10
    print("Number of matrices for zigtorch:",n)
    results= []
    for i in range (n):
        start2 = time.time()
        D = zgt.mm(A,B)
        end2 = time.time()
        print(f'{i}time:{end2-start2} ')
        results.append(end2-start2)
    return sum(results), results



pytorch_total_time, pytorch_times = pytorch_test(A,B)
zigtorch_total_time, zigtorch_times = zig_mm(A,B)
print(f"Execution time for pytorch.mm: {pytorch_total_time:.9f}, seconds")
print(f"Execution time for zigtorch.mm: {zigtorch_total_time:.9f}, seconds!!!!!!!!!!1")




