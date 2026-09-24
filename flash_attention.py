import torch
import triton
import triton.language as tl
import math

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def _forward_inner_kernel(
    Q, o, K_T_offsets, V_offsets,
    max, sum,
    FLAG: tl.constexpr,
    T, D: tl.constexpr,
    BLOCK_SIZE_QO: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    scale,
    mask_D
):
    row = tl.program_id(0)
    if(FLAG == True):
        low = row * BLOCK_SIZE_QO
        high = low + BLOCK_SIZE_QO
    else:
        low = 0
        high = row * BLOCK_SIZE_QO

    K_T_offsets += low * D
    V_offsets += low * D
    kv_offsets = tl.arange(0, BLOCK_SIZE_KV) + low

    for i in tl.range(low, high, BLOCK_SIZE_KV): #type: ignore

        mask_T = kv_offsets < T
        mask_K = mask_T[None, :] & mask_D[:, None]
        K_tile = tl.load(K_T_offsets, mask_K, 0.0) * scale

        S = tl.dot(Q, K_tile)

        if FLAG:
            q_offsets = row * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO)
            mask_dia = q_offsets[: ,None] >= kv_offsets
            S = tl.where(mask_dia, S, -float("inf"))

        new_max = tl.maximum(max, tl.max(S, 1))

        S -= new_max[:, None]

        P = tl.exp2(S)

        alpha = tl.exp2(max - new_max)
        new_sum = tl.sum(P, 1)
        sum = sum * alpha + new_sum

        mask_V = mask_T[:, None] & mask_D[None, :]
        V = tl.load(V_offsets, mask_V, 0.0)

        o = o * alpha[:, None]
        o = tl.dot(P, V, acc=o)

        max = new_max
        K_T_offsets += D * BLOCK_SIZE_KV
        V_offsets += D * BLOCK_SIZE_KV
        kv_offsets += BLOCK_SIZE_KV

    return o, max, sum

@triton.jit
def _flash_attention_forward_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    B, H, T, D: tl.constexpr, scale,
    LSE_ptr,
    BLOCK_SIZE_QO: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    rln2: tl.constexpr = 1.4426950408889634
    scale *= rln2


    row_id = tl.program_id(0) * BLOCK_SIZE_QO
    BH_id = tl.program_id(1)
    B_id = BH_id // H
    H_id = BH_id % H

    Q_ptr += B_id * T * D * H + H_id * T * D
    K_ptr += B_id * T * D * H + H_id * T * D
    V_ptr += B_id * T * D * H + H_id * T * D
    O_ptr += B_id * T * D * H + H_id * T * D
    LSE_ptr += BH_id * T

    Q_offsets = row_id + tl.arange(0, BLOCK_SIZE_QO)
    cols_offsets = tl.arange(0, BLOCK_D)
    mask_T = Q_offsets < T
    mask_D = cols_offsets < D
    Q = tl.load(
        Q_ptr + Q_offsets[:, None] * D + cols_offsets[None, :],
        mask_T[:, None] & mask_D[None, :],
        0.0
        )

    offsets_KV = tl.arange(0, BLOCK_SIZE_KV)
    K_T_offsets = K_ptr + cols_offsets[:, None] + offsets_KV[None, :] * D
    V_offsets = V_ptr + offsets_KV[:, None] * D + cols_offsets[None, :]

    max = tl.full((BLOCK_SIZE_QO, ), value=float('-inf'), dtype=tl.float32)
    sum = tl.zeros((BLOCK_SIZE_QO, ), dtype=tl.float32)
    o = tl.zeros((BLOCK_SIZE_QO, BLOCK_D), dtype=tl.float32)

    o, max, sum = _forward_inner_kernel(
        Q, o, K_T_offsets, V_offsets,
        max, sum,
        True, 
        T, D,
        BLOCK_SIZE_QO,
        BLOCK_SIZE_KV,
        scale,
        mask_D
    )

    o, max, sum = _forward_inner_kernel(
        Q, o, K_T_offsets, V_offsets,
        max, sum,
        False,
        T, D,
        BLOCK_SIZE_QO,
        BLOCK_SIZE_KV,
        scale,
        mask_D
    )

    o = o / sum[:, None]

    lse = max + tl.math.log2(sum)

    tl.store(
        O_ptr + Q_offsets[:, None] * D + cols_offsets[None, :],
        o, mask_T[:, None] & mask_D[None, :]
        )
    tl.store(LSE_ptr + Q_offsets, lse, mask_T)

@triton.jit
def _flash_attention_backward_kernel_delta(
    delta_ptr, o_ptr, do_ptr,
    B, H, T, D: tl.constexpr,
    BLOCK_SIZE_ROW: tl.constexpr
):
    row_id = tl.program_id(0) * BLOCK_SIZE_ROW
    head_id = tl.program_id(1)

    delta_ptr += head_id * T + row_id
    o_ptr += head_id * T * D + row_id * D
    do_ptr += head_id * T * D + row_id * D

    offsets_row = tl.arange(0, BLOCK_SIZE_ROW)
    offsets = tl.arange(0, D)[None, :] + offsets_row[:, None] * D
    mask = offsets_row + row_id < T
    o = tl.load(o_ptr + offsets, mask[:, None], other=0.0)
    do = tl.load(do_ptr + offsets, mask[:, None], other=0.0)

    delta = tl.sum(o * do, 1)

    tl.store(delta_ptr + offsets_row, delta, mask)

@triton.jit
def _backward_inner_kernel_dkdv(
    q_offsets, k_tile, v_tile, dk, dv,
    q_ptr, do_ptr, lse_ptr, delta_ptr,
    B, H, T, D,
    BLOCK_SIZE_MACRO: tl.constexpr, BLOCK_SIZE_MICRO: tl.constexpr,
    FLAG: tl.constexpr
):
    kv_start = tl.program_id(0) * BLOCK_SIZE_MACRO
    if FLAG:
        start_row = kv_start
        end_row = kv_start + BLOCK_SIZE_MACRO
    else:
        start_row = kv_start + BLOCK_SIZE_MACRO
        end_row = T

    q_offsets += start_row * D
    k_offsets = tl.arange(0, BLOCK_SIZE_MACRO) + kv_start

    for i in tl.range(start_row, end_row, BLOCK_SIZE_MICRO): #type: ignore
        row_offsets = tl.arange(0, BLOCK_SIZE_MICRO) + i
        q_mask = row_offsets < T
        q_tile = tl.load(q_ptr + q_offsets, q_mask[:, None], 0.0)
        do_tile = tl.load(do_ptr + q_offsets, q_mask[:, None], 0.0)

        lse = tl.load(lse_ptr + row_offsets, q_mask, 0.0)
        delta = tl.load(delta_ptr + row_offsets, q_mask, 0.0)

        s_T = tl.dot(k_tile, tl.trans(q_tile))
        p_T = tl.exp2(s_T - lse[None, :])

        if FLAG:
            causal_mask = row_offsets[None, :] >= k_offsets[:, None]
            p_T = tl.where(causal_mask, p_T, 0.0)
        
        dv = tl.dot(p_T, do_tile, acc=dv)

        dp_T = tl.dot(v_tile, tl.trans(do_tile))
        ds_T = p_T * (dp_T - delta[None, :]) * 0.6931471824645996
        dk = tl.dot(ds_T, q_tile, acc=dk)

        q_offsets += BLOCK_SIZE_MICRO * D

    return dv, dk

@triton.jit
def _backward_inner_kernel_dq(
    kv_T_offsets, q_tile, do_tile, dq,
    k_ptr, v_ptr,
    lse_ptr, delta_ptr,
    T, D,
    BLOCK_SIZE_MACRO: tl.constexpr, BLOCK_SIZE_MICRO: tl.constexpr,
    FLAG: tl.constexpr
):
    q_start = tl.program_id(0) * BLOCK_SIZE_MACRO
    if FLAG:
        start_row = q_start
        end_row = q_start + BLOCK_SIZE_MACRO
    else:
        start_row = 0
        end_row = q_start

    kv_T_offsets += start_row * D
    q_row_offsets = q_start + tl.arange(0, BLOCK_SIZE_MACRO)

    lse = tl.load(lse_ptr + q_row_offsets, q_row_offsets < T, 0.0)
    delta = tl.load(delta_ptr + q_row_offsets, q_row_offsets < T, 0.0)

    for i in tl.range(start_row, end_row, BLOCK_SIZE_MICRO): # type: ignore
        row_offsets = tl.arange(0, BLOCK_SIZE_MICRO) + i
        mask = row_offsets < T
        k_T = tl.load(k_ptr + kv_T_offsets, mask[None, :], 0.0)
        v_T = tl.load(v_ptr + kv_T_offsets, mask[None, :], 0.0)

        dp = tl.dot(do_tile, v_T)
        s = tl.dot(q_tile, k_T)
        p = tl.exp2(s - lse[:, None])
        if FLAG:
            causal_mask = q_row_offsets[:, None] >= row_offsets[None, :]
            p = tl.where(causal_mask, p, 0.0)

        ds = p * (dp - delta[:, None]) * 0.6931471824645996
        dq = tl.dot(ds, tl.trans(k_T), acc=dq)

        kv_T_offsets += BLOCK_SIZE_MICRO * D

    return dq



@triton.jit
def _flash_attention_backward_kernel_dqdkqv(
    dq_ptr, dk_ptr, dv_ptr,
    q_ptr, k_ptr, v_ptr,
    o_ptr, do_ptr,
    lse_ptr, delta_ptr,
    B, H, T, D: tl.constexpr,
    scale,
    BLOCK_SIZE_MACRO: tl.constexpr, BLOCK_SIZE_MICRO: tl.constexpr,
):
    ln2: tl.constexpr = 0.6931471824645996
    rln2: tl.constexpr = 1.4426950408889634

    row_id = tl.program_id(0) * BLOCK_SIZE_MACRO
    head_id = tl.program_id(1)

    q_ptr += head_id * T * D
    k_ptr += head_id * T * D
    v_ptr += head_id * T * D
    o_ptr += head_id * T * D
    do_ptr += head_id * T * D
    lse_ptr += head_id * T
    delta_ptr += head_id * T

    dq_ptr += head_id * T * D
    dk_ptr += head_id * T * D
    dv_ptr += head_id * T * D

    # dk, dv
    k_col_offsets = tl.arange(0, D)
    k_row_offsets = tl.arange(0, BLOCK_SIZE_MACRO)
    k_mask = (row_id + k_row_offsets < T)[:, None]
    k_offsets = k_row_offsets[:, None] * D + k_col_offsets[None, :] + row_id * D
    
    k_tile = tl.load(k_ptr + k_offsets, k_mask, 0.0) * scale * rln2
    v_tile = tl.load(v_ptr + k_offsets, k_mask, 0.0)

    q_row_offsets = tl.arange(0, BLOCK_SIZE_MICRO) * D
    q_offsets = q_row_offsets[:, None] + k_col_offsets[None, :]

    dv = tl.zeros((BLOCK_SIZE_MACRO, D), dtype=tl.float32)
    dk = tl.zeros((BLOCK_SIZE_MACRO, D), dtype=tl.float32)

    dv, dk = _backward_inner_kernel_dkdv(
        q_offsets, k_tile, v_tile, dk, dv,
        q_ptr, do_ptr, lse_ptr, delta_ptr,
        B, H, T, D,
        BLOCK_SIZE_MACRO, BLOCK_SIZE_MICRO, FLAG=True # type: ignore
    )

    dv, dk = _backward_inner_kernel_dkdv(
        q_offsets, k_tile, v_tile, dk, dv,
        q_ptr, do_ptr, lse_ptr, delta_ptr,
        B, H, T, D,
        BLOCK_SIZE_MACRO, BLOCK_SIZE_MICRO, FLAG=False # type: ignore
    )

    tl.store(dk_ptr + k_offsets, dk * scale * rln2, k_mask)
    tl.store(dv_ptr + k_offsets, dv, k_mask)

    # dq
    q_tile = tl.load(q_ptr + k_offsets, k_mask, 0.0) * scale * rln2
    do_tile = tl.load(do_ptr + k_offsets, k_mask, 0.0)

    kv_T_offsets = q_row_offsets[None, :] + k_col_offsets[:, None]

    dq = tl.zeros((BLOCK_SIZE_MACRO, D), dtype=tl.float32)

    dq = _backward_inner_kernel_dq(
        kv_T_offsets, q_tile, do_tile, dq,
        k_ptr, v_ptr,
        lse_ptr, delta_ptr,
        T, D,
        BLOCK_SIZE_MACRO, BLOCK_SIZE_MICRO, FLAG=True # type: ignore
    )

    dq = _backward_inner_kernel_dq(
        kv_T_offsets, q_tile, do_tile, dq,
        k_ptr, v_ptr,
        lse_ptr, delta_ptr,
        T, D,
        BLOCK_SIZE_MACRO, BLOCK_SIZE_MICRO, FLAG=False # type: ignore
    )

    tl.store(dq_ptr + k_offsets, dq * scale * rln2, k_mask)
    

class _flash_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        B, H, T, D = q.shape
        assert k.shape == v.shape == (B, H, T, D)
        BH = B * H

        scale = math.sqrt(1 / D)

        o = torch.empty_like(q, device=DEVICE)
        LSE = torch.empty(BH * T, device=DEVICE)

        BLOCK_D = triton.next_power_of_2(D)

        grid = lambda meta: (triton.cdiv(T, meta['BLOCK_SIZE_QO']), BH)

        _flash_attention_forward_kernel[grid](
            q, k, v, o,
            B, H, T, D, scale,
            LSE,
            BLOCK_SIZE_QO=32, BLOCK_SIZE_KV=32, BLOCK_D=BLOCK_D  # type: ignore
        )

        ctx.save_for_backward(q, k, v, o, LSE)
        ctx.scale = scale

        return o

    @staticmethod
    def backward(ctx, do):
        Q, K, V, O, LSE = ctx.saved_tensors
        scale = ctx.scale

        B, H, T, D = Q.shape

        dq = torch.empty_like(Q, device=DEVICE)
        dk = torch.empty_like(K, device=DEVICE)
        dv = torch.empty_like(V, device=DEVICE)

        delta = torch.empty((B * H, T), device=DEVICE)

        grid = lambda meta: (triton.cdiv(T, meta['BLOCK_SIZE_ROW']), B * H)

        _flash_attention_backward_kernel_delta[grid](
            delta, O, do,
            B, H, T, D,
            BLOCK_SIZE_ROW=16 # type: ignore
        )

        grid = lambda meta: (triton.cdiv(T, meta['BLOCK_SIZE_MACRO']), B * H)

        _flash_attention_backward_kernel_dqdkqv[grid](
            dq, dk, dv,
            Q, K, V,
            O, do,
            LSE, delta,
            B, H, T, D,
            scale,
            BLOCK_SIZE_MACRO=32, BLOCK_SIZE_MICRO=16 #type: ignore
        )

        return dq, dk, dv



flash_attention = _flash_attention.apply

def test(shape):
    B, H, T, D = shape

    q = torch.randn((shape), dtype=torch.float32, device=DEVICE, requires_grad=True)
    k = torch.randn((shape), dtype=torch.float32, device=DEVICE, requires_grad=True)
    v = torch.randn((shape), dtype=torch.float32, device=DEVICE, requires_grad=True)

    o_triton = flash_attention(q, k, v)
    o_torch = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=math.sqrt(1 / D)
    )

    torch.testing.assert_close(o_triton, o_torch, atol=1e-2, rtol=1e-2)
    print("fwd passed")

    dLdout = 0.1 * torch.randn_like(q)
    o_triton.backward(dLdout, retain_graph=True)
    dLdq_tri, dLdk_tri, dLdv_tri = [_.grad.clone() for _ in [q, k, v]]
    q.grad, k.grad, v.grad = None, None, None

    o_torch.backward(dLdout, retain_graph=True)
    dLdq_ref, dLdk_ref, dLdv_ref = [_.grad.clone() for _ in [q, k, v]]
    q.grad, k.grad, v.grad = None, None, None

    torch.testing.assert_close(dLdq_tri, dLdq_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(dLdk_tri, dLdk_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(dLdv_tri, dLdv_ref, atol=1e-2, rtol=1e-2)
    print("Passed bwd")

if __name__ == "__main__":
    test((2, 12, 128, 64))