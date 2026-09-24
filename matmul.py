import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2)
]

@triton.autotune(configs=autotune_configs, key=['M','N','K'])
@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    A_M_stride, A_K_stride,
    B_K_stride, B_N_stride,
    C_M_stride, C_N_stride,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    group_cols = tl.cdiv(M, BLOCK_SIZE_M)
    group_rows = tl.cdiv(N, BLOCK_SIZE_N)
    p_per_group = GROUP_SIZE * group_rows
    gid = pid // p_per_group
    g_m_start = gid * GROUP_SIZE

    group_size = min(group_cols - g_m_start, GROUP_SIZE)
    p_m = g_m_start + pid % p_per_group % group_size
    p_n = pid % p_per_group // group_size

    M_offsets = p_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    N_offsets = p_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    K_offsets = tl.arange(0, BLOCK_SIZE_K)

    A_offsets = M_offsets[:, None] * A_M_stride + K_offsets[None, :] * A_K_stride
    B_offsets = N_offsets[None, :] * B_N_stride + K_offsets[:, None] * B_K_stride

    A_mask = M_offsets < M
    B_mask = N_offsets < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        K_mask = K_offsets < K - k * BLOCK_SIZE_K

        a = tl.load(A_ptr + A_offsets, mask=A_mask[:, None] & K_mask[None, :], other=0.0)
        b = tl.load(B_ptr + B_offsets, mask=K_mask[:, None] & B_mask[None, :], other=0.0)

        accumulator = tl.dot(a, b, accumulator)

        A_ptr += BLOCK_SIZE_K * A_K_stride
        B_ptr += BLOCK_SIZE_K * B_K_stride

    C_offsets = M_offsets[:, None] * C_M_stride + N_offsets[None, :] * C_N_stride
    C_mask = A_mask[:, None] & B_mask[None, :]
    tl.store(C_ptr + C_offsets, accumulator, C_mask)

def matmul(A, B):
    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]

    C = torch.empty((M, N), dtype=torch.float16, device=DEVICE)
    A, B = A.to(torch.float16), B.to(torch.float16)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    _matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C

def test(size):
    A = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    B = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    # run kernel & pytorch reference implementation
    c_tri = matmul(A, B)
    c_ref = torch.matmul(A, B)
    # compare
    torch.testing.assert_close(c_tri, c_ref, atol=1e-2, rtol=1e-1)
    print("PASSED")

if __name__ == "__main__":
    test((1024, 1024))