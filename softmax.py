import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@triton.jit
def softmax_kernel(
    x_ptr, y_ptr,
    x_row_stride, y_row_stride,
    row, col,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAGE: tl.constexpr
):
    pid = tl.program_id(0)

    row_per_grid = tl.num_programs(0)

    for row_id in tl.range(pid, row, row_per_grid, NUM_STAGE): # type: ignore
        starts = row_id * x_row_stride + x_ptr
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < col

        x = tl.load(starts + offsets, mask, float('-inf'))

        row_max = tl.max(x, axis=0)
        x = x - row_max
        x = tl.exp(x)
        sum = tl.sum(x, axis=0)
        y = x / sum

        y_offsets = row_id * y_row_stride + offsets
        tl.store(y_ptr + y_offsets, y, mask)

properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index) # type: ignore

NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
TOTAL_SRAM_PER_SM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]

def softmax(x):
    y = torch.empty_like(x)

    row, col = x.shape
    BLOCK_SIZE =  triton.next_power_of_2(col)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    num_stages = 4 if TOTAL_SRAM_PER_SM > 200_000 else 2

    kernel = softmax_kernel.warmup(
        x, y,
        x.stride(0), y.stride(0),
        row, col,
        BLOCK_SIZE, num_stages,
        num_warps=num_warps,
        grid=(1, )
    )

    kernel._init_handles()
    n_regs = kernel.n_regs
    sram_needed_per_program = kernel.metadata.shared
    reg_occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program

    programs_per_sm = min(reg_occupancy, sram_occupancy)
    num_programs = min(NUM_SM * programs_per_sm, row)

    grid = (num_programs, 1, 1)

    kernel[grid](x, y, x.stride(0), y.stride(0), row, col)

    return y

def test(size):
    x = torch.rand(size, device=DEVICE)

    torch_y = torch.softmax(x, dim=1)
    triton_y = softmax(x)

    torch.testing.assert_close(triton_y, torch_y, atol=1e-3, rtol=1e-3)
    print("pass")

if __name__ == "__main__":
    test((1024, 1024))
    test((1024, 2047))