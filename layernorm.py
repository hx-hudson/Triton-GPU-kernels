import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')
eps = 1e-5

@triton.jit
def _layernorm_forward(
    x_ptr, y_ptr,
    weight_ptr, bias_ptr,
    mean_ptr, rstd_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    x_ptr += pid * N
    y_ptr += pid * N

    sum_for_mean = 0.0
    sum_for_std = 0.0

    for i in tl.range(0, N, BLOCK_SIZE): # type: ignore
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask, other=0.0).to(tl.float32)
        sum_for_mean += tl.sum(x, 0)

    mean = sum_for_mean / N

    for i in tl.range(0, N, BLOCK_SIZE): # type: ignore
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        sum_for_std += tl.sum(diff * diff, 0)

    var = sum_for_std / N
    rstd = tl.rsqrt(var + 1e-5)

    for i in tl.range(0, N, BLOCK_SIZE): # type: ignore
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask, other=0.0)
        bias = tl.load(bias_ptr + offsets, mask, other=0.0)
        y = weight * (x - mean) * rstd + bias
        tl.store(y_ptr + offsets, y, mask)

    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)

@triton.jit
def _layernorm_backward_dx(
    x_ptr, dy_ptr, dx_ptr,
    weight_ptr, mean_ptr, rstd_ptr,
    dw_intermediate_ptr, db_intermediate_ptr, key_ptr,
    N, BLOCK_SIZE: tl.constexpr, MEM_SZIE: tl.constexpr
):
    pid = tl.program_id(0)

    x_ptr += pid * N
    dx_ptr += pid * N
    dy_ptr += pid * N
    mean = tl.load(mean_ptr + pid)
    rstd = tl.load(rstd_ptr + pid)

    accumulator_dx_hat = 0.0
    accumulator_dot = 0.0

    for i in tl.range(0, N, BLOCK_SIZE): # type: ignore
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offsets, mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask, other=0.0).to(tl.float32)
        
        x_hat = (x - mean) * rstd
        dx_hat = dy * weight
        accumulator_dx_hat += tl.sum(dx_hat, 0)
        accumulator_dot += tl.sum(x_hat * dx_hat, 0)

    arg1 = accumulator_dx_hat / N
    arg2 = accumulator_dot / N

    for i in tl.range(0, N, BLOCK_SIZE): # type: ignore
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offsets, mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask, other=0.0).to(tl.float32)
        
        x_hat = (x - mean) * rstd
        dx_hat = dy * weight

        tl.store(dx_ptr + offsets, rstd * (dx_hat - arg1 - x_hat * arg2), mask)

    

    key_id = pid % MEM_SZIE
    dw_intermediate_ptr += key_id * N
    db_intermediate_ptr += key_id * N

    key_ptr += key_id
    count_ptr = key_ptr + MEM_SZIE

    while tl.atomic_cas(key_ptr, 0, 1) == 1:
        pass

    count = tl.load(count_ptr)
    for i in tl.range(0, N, BLOCK_SIZE): # type: ignore
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N
        x = tl.load(x_ptr + offsets, mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_ptr + offsets, mask, other=0.0).to(tl.float32)
        partial_dw = (x - mean) * rstd * dy
        partial_db = dy
        
        if(count == 1):
            partial_dw += tl.load(dw_intermediate_ptr + offsets, mask, other=0.0)
            partial_db += tl.load(db_intermediate_ptr + offsets, mask, 0.0)

        tl.store(dw_intermediate_ptr + offsets, partial_dw, mask)
        tl.store(db_intermediate_ptr + offsets, partial_db, mask)

    tl.atomic_xchg(count_ptr, 1)

    tl.debug_barrier()

    tl.atomic_xchg(key_ptr, 0)

@triton.jit
def _layernorm_backward_dwdb(
    dw_intermediate_ptr, db_intermediate_ptr,
    dw_ptr, db_ptr,
    N, MEM_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid = tl.program_id(0)

    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    cols_mask = cols < N

    dw = tl.zeros((BLOCK_SIZE_N, ), tl.float32)
    db = tl.zeros((BLOCK_SIZE_N, ), tl.float32)

    for i in tl.range(0, MEM_SIZE, BLOCK_SIZE_M): # type: ignore
        row = i + tl.arange(0, BLOCK_SIZE_M)
        row_mask = row < MEM_SIZE
        offsets = row[:, None] * N + cols[None, :]
        mask = row_mask[:, None] & cols_mask[None, :]
        acc_w = tl.load(dw_intermediate_ptr + offsets, mask, 0.0)
        acc_b = tl.load(db_intermediate_ptr + offsets, mask, 0.0)
        dw += tl.sum(acc_w, 0)
        db += tl.sum(acc_b, 0)

    tl.store(dw_ptr + cols, dw, cols_mask)
    tl.store(db_ptr + cols, db, cols_mask)

class Layernorm_wrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        y = torch.empty_like(x)

        M, N = x.reshape(-1, x.shape[-1]).shape
        mean = torch.empty(M, dtype=torch.float32, device=DEVICE)
        rstd = torch.empty(M, dtype=torch.float32, device=DEVICE)

        BLOCK_SIZE = min(2048, triton.next_power_of_2(N))

        _layernorm_forward[(M, )](
            x, y,
            weight, bias,
            mean, rstd,
            N,
            BLOCK_SIZE=BLOCK_SIZE # type: ignore
        )

        ctx.save_for_backward(x, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE

        return y

    @staticmethod
    def backward(ctx, dy):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        M, N = x.reshape(-1, x.shape[-1]).shape

        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        db = torch.empty_like(bias)

        MEM_SIZE = 256

        db_intermediate = torch.zeros((MEM_SIZE, N), device=DEVICE)
        dw_intermediate = torch.zeros((MEM_SIZE, N), device=DEVICE)

        key = torch.zeros((2 * MEM_SIZE, ),dtype=torch.int32, device=DEVICE)

        _layernorm_backward_dx[(M, )](
            x, dy, dx, weight, mean, rstd, dw_intermediate, db_intermediate,
            key, N,
            ctx.BLOCK_SIZE, MEM_SIZE # type: ignore
        )

        grid = lambda meta: [triton.cdiv(N, meta['BLOCK_SIZE_N'])]
        _layernorm_backward_dwdb[grid](
            dw_intermediate, db_intermediate,
            dw, db,
            N, MEM_SIZE, BLOCK_SIZE_M=32, BLOCK_SIZE_N=128 #type: ignore
        )

        return dx, dw, db

layernorm = Layernorm_wrapper.apply

def test(size):
    M, N = size
    x = -2.3 + 0.5 * torch.randn((M, N), device=DEVICE)
    weight = torch.rand((N, ), device=DEVICE, requires_grad=True)
    bias = torch.rand((N, ), device=DEVICE, requires_grad=True)
    dLdy = .1 * torch.randn_like(x)
    x.requires_grad_(True)

    y_tri = layernorm(x, weight, bias)
    y_ref = torch.nn.functional.layer_norm(x, (N,), weight, bias, eps)
    torch.testing.assert_close(y_tri, y_ref, atol=1e-2, rtol=0) 
    print("Passed fwd")

    y_tri.backward(dLdy, retain_graph=True)
    dLdx_tri, dLdw_tri, dLdb_tri = [_.grad.clone() for _ in [x, weight, bias]]

    x.grad, weight.grad, bias.grad = None, None, None

    y_ref.backward(dLdy, retain_graph=True)
    dLdx_ref, dLdw_ref, dLdb_ref = [_.grad.clone() for _ in [x, weight, bias]]

    torch.testing.assert_close(dLdx_tri, dLdx_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdb_tri, dLdb_ref, atol=1e-2, rtol=0)
    torch.testing.assert_close(dLdw_tri, dLdw_ref, atol=1e-2, rtol=0)

    print("Passed bwd")

if __name__ == "__main__":
    test((1024, 1024))
    test((1023, 2046))