# mpirun -n 8 --oversubscribe python summa.py
from mpi4py import MPI
import numpy as np

# Compute C = A @ B with numpy
M, K, N = 256, 1024, 512
A = np.random.rand(M, K)
B = np.random.rand(K, N)
C_expect = np.matmul(A, B)

# Compute C = A @ B with Summa algo
R, C = 4, 2
A_tile_row, A_tile_col = M // R, K // C
B_tile_row, B_tile_col = K // R, N // C
C_tile_row, C_tile_col = M // R, N // C

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
assert world_size == R * C

# Split worker procs into rows and cols
color_row, color_col = rank // C, rank % C
comm_row = comm.Split(color=color_row, key=rank)
comm_col = comm.Split(color=color_col, key=rank)

# Distribute A and B to procs
A_buf = np.ascontiguousarray(
    A.reshape([R, A_tile_row, C, A_tile_col]).transpose([0, 2, 1, 3])
)
B_buf = np.ascontiguousarray(
    B.reshape([R, B_tile_row, C, B_tile_col]).transpose([0, 2, 1, 3])
)
C_buf = np.empty([R, C, C_tile_row, C_tile_col])
A_tile = np.empty((A_tile_row, A_tile_col))
B_tile = np.empty((B_tile_row, B_tile_col))
C_tile = np.zeros((C_tile_row, C_tile_col))
comm.Scatter(A_buf, A_tile, root=0)
comm.Scatter(B_buf, B_tile, root=0)

# Summa iteration
A_recv = np.empty((A_tile_row, 1))
B_recv = np.empty((1, B_tile_col))
for k in range(K):
    # Broadcast A_tile that stores A[:, k] to all procs in the same row
    A_recv[:, 0] = A_tile[:, k % A_tile_col]
    comm_row.Bcast(A_recv, k // A_tile_col)

    # Broadcast B_tile that stores B[k, :] to all procs in the same col
    B_recv[0, :] = B_tile[k % B_tile_row, :]
    comm_col.Bcast(B_recv, k // B_tile_row)

    # Compute partial sum C += A[:, k] @ B[k , :]
    C_tile += np.matmul(A_recv, B_recv)

# Collect to main proc
comm.Gather(C_tile, C_buf, root=0)
if rank == 0:
    C_result = C_buf.transpose([0, 2, 1, 3]).reshape([M, N])
    print(f"Result equal: {np.allclose(C_expect, C_result)}")
