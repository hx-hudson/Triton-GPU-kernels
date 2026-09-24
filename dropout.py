import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def _dropout_kernel(
    x_ptr, y_ptr,
    n_elements,
    p, seed,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask)

    random = tl.rand(seed, offsets)
    keep = random > p

    y = tl.where(keep, x / (1 - p), 0.0)
    tl.store(y_ptr + offsets, y, mask)

def dropout(x, p, seed):
    y = torch.empty_like(x)

    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    _dropout_kernel[grid](
        x, y,
        n_elements,
        p, seed,
        BLOCK_SIZE=1024
    )

    return y

if __name__ == "__main__":
    x = torch.rand((4,4), device=DEVICE)
    print(dropout(x, 0.5, 42))