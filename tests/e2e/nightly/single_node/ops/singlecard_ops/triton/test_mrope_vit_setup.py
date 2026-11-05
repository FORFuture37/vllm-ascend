# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.patch.worker.patch_mrope_vit_setup import vit_mrope_setup


def _reference_positions(grids, merge):
    positions = []
    for t, h, w in grids:
        axes = torch.meshgrid(torch.arange(t), torch.arange(h), torch.arange(w), indexing="ij")
        positions.append(
            torch.stack(axes, dim=-1)
            .reshape(t, h // merge, merge, w // merge, merge, 3)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, 3)
        )
    return torch.cat(positions).float()


@pytest.mark.parametrize(
    "grids,merge",
    [
        ([[1, 1, 1]], 1),
        ([[1, 1, 17]], 1),
        ([[1, 1, 33]], 1),
        ([[1, 5, 7], [2, 3, 5]], 1),
        ([[1, 2, 2], [3, 6, 10]], 2),
        ([[0, 4, 4], [2, 4, 8], [0, 4, 4]], 2),
        ([[1, 48, 48]], 2),
        ([[4, 48, 64]], 2),
    ],
)
@pytest.mark.parametrize("sections", [(21, 21, 21), (16, 24, 24), (0, 16, 16), (32, 32, 32)])
@torch.inference_mode()
def test_vit_mrope_setup(grids, merge, sections):
    """Compile the patched entry point once and compare it with the reference.

    Do not also launch ``_vit_mrope_setup_kernel`` here. That second launch
    compiles another BLOCK_N=16 specialization in the same case, and the
    extra compile still overflows Ascend UB when BLOCK_D is larger than 64.
    """
    init_device_properties_triton()
    inv_cpu = [torch.linspace(0.001, 1.0, size) for size in sections]
    inv = [freq.to("npu") for freq in inv_cpu]
    positions = _reference_positions(grids, merge)
    cos, sin = vit_mrope_setup(*inv, grids, merge)
    cos = cos.cpu()
    sin = sin.cpu()
    start = 0
    for axis, size in enumerate(sections):
        end = start + size
        if size == 0:
            start = end
            continue
        freq = positions[:, axis, None] * inv_cpu[axis]
        torch.testing.assert_close(cos[axis, :, start:end], freq.cos(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(sin[axis, :, start:end], freq.sin(), atol=1e-5, rtol=1e-5)
        start = end
