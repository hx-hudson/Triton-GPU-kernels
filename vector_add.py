import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)

    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask)
    y = tl.load(y_ptr + offsets, mask)

    output = x + y

    tl.store(output_ptr + offsets, output, mask)

def add(x: torch.Tensor, y):
    output = torch.empty_like(x)

    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=tl.constexpr(1024))

    return output

def test(size):

    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)

    torch_output = x + y
    triton_output = add(x, y)

    torch.testing.assert_close(triton_output, torch_output, rtol=1e-3, atol=1e-3)
    print('pass')

if __name__ == "__main__":
    test(1024)