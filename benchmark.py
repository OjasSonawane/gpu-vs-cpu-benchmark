import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# Detect device (Mac GPU via MPS or CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

sizes = [500, 1000, 1500, 2000]

cpu_times = []
gpu_times = []
speedups = []
transfer_times = []

for size in sizes:
    print(f"\nMatrix size: {size} x {size}")

    # Generate random matrices
    a_cpu = np.random.rand(size, size).astype(np.float32)
    b_cpu = np.random.rand(size, size).astype(np.float32)

    # ---------------- CPU ----------------
    start = time.time()
    result_cpu = np.dot(a_cpu, b_cpu)
    cpu_time = time.time() - start
    cpu_times.append(cpu_time)

    print(f"CPU Time: {cpu_time:.4f}s")

    # ---------------- TRANSFER ----------------
    start = time.time()
    a_gpu = torch.tensor(a_cpu, device=device)
    b_gpu = torch.tensor(b_cpu, device=device)
    transfer_time = time.time() - start
    transfer_times.append(transfer_time)

    print(f"Transfer Time: {transfer_time:.4f}s")

    # ---------------- GPU ----------------
    start = time.time()
    result_gpu = torch.matmul(a_gpu, b_gpu)

    if device.type != "cpu":
        torch.mps.synchronize()

    gpu_time = time.time() - start
    gpu_times.append(gpu_time)

    print(f"GPU Time: {gpu_time:.4f}s")

    # ---------------- SPEEDUP ----------------
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    speedups.append(speedup)

    print(f"Speedup: {speedup:.2f}x")

    # ---------------- MEMORY ----------------
    if device.type != "cpu":
        print(f"GPU Memory Used: {torch.mps.current_allocated_memory()} bytes")

# ---------------- VISUALIZATION ----------------

# CPU vs GPU Time
plt.figure()
plt.plot(sizes, cpu_times, marker='o', label='CPU')
plt.plot(sizes, gpu_times, marker='o', label='GPU (MPS)')
plt.xlabel("Matrix Size")
plt.ylabel("Execution Time (seconds)")
plt.title("CPU vs GPU Performance")
plt.legend()
plt.grid()
plt.show()

# Speedup
plt.figure()
plt.plot(sizes, speedups, marker='o')
plt.xlabel("Matrix Size")
plt.ylabel("Speedup (CPU/GPU)")
plt.title("GPU Speedup")
plt.grid()
plt.show()