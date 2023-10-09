# mpirun -n 16 --oversubscribe python cannon.py
# Cannon algo: https://www3.nd.edu/~zxu2/acms60212-40212-S12/Lec-07-3.pdf
from mpi4py import MPI
import numpy as np

# Compute C = A @ B with numpy
M, K, N = 256, 1024, 512
A = np.random.rand(M, K)
B = np.random.rand(K, N)
C_expect = np.matmul(A, B)

# Compute C = A @ B with Cannon algo
R, C = 4, 4
A_tile_row, A_tile_col = M // R, K // C
B_tile_row, B_tile_col = K // R, N // C
C_tile_row, C_tile_col = M // R, N // C

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
world_size = comm.Get_size()
assert R == C
assert world_size == R * C

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

# Initial alignment
comm_cart = comm.Create_cart(dims=[R, C], periods=[True, True], reorder=False)
left, right = comm_cart.Shift(direction=1, disp=rank // C)
comm_cart.Sendrecv_replace(A_tile, dest=left, source=right)
up, down = comm_cart.Shift(direction=0, disp=rank % C)
comm_cart.Sendrecv_replace(B_tile, dest=up, source=down)

# Shifting
for p in range(C):
    # Compute partial sum C_tile += A_tile @ B_tile
    C_tile += np.matmul(A_tile, B_tile)

    # Shift A_tile
    left, right = comm_cart.Shift(direction=1, disp=1)
    comm_cart.Sendrecv_replace(A_tile, dest=left, source=right)

    # Shift B_tile
    up, down = comm_cart.Shift(direction=0, disp=1)
    comm_cart.Sendrecv_replace(B_tile, dest=up, source=down)

# Collect to main proc
comm.Gather(C_tile, C_buf, root=0)
if rank == 0:
    C_result = C_buf.transpose([0, 2, 1, 3]).reshape([M, N])
    print(f"Result equal: {np.allclose(C_expect, C_result)}")
