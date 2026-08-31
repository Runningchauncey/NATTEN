#################################################################################################
# Copyright (c) 2022 - 2026 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################

import pytest
import torch

import natten
from natten._environment import _IS_CUDA_AVAILABLE, HAS_LIBNATTEN
from natten.sparse_na2d_reference import sparse_na2d_pytorch

pytestmark = pytest.mark.skipif(
    not _IS_CUDA_AVAILABLE or not HAS_LIBNATTEN,
    reason="CUDA or libnatten is not available.",
)


def _reference_sparse_na2d(
    query,
    key,
    value,
    coords,
    kernel_size,
    scale=None,
    sample_mode="indexed",
    apply_key_rope=False,
):
    return sparse_na2d_pytorch(
        query,
        key,
        value,
        coords,
        kernel_size,
        scale=scale,
        sample_mode=sample_mode,
        apply_key_rope=apply_key_rope,
    )


@pytest.mark.parametrize("sample_mode", ["indexed", "bilinear"])
@pytest.mark.parametrize("kernel_size", [(1, 1), (3, 3), (3, 5)])
def test_sparse_na2d_forward_matches_reference(kernel_size, sample_mode):
    torch.manual_seed(0)
    query = torch.randn(2, 7, 3, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    key = torch.randn(2, 4, 5, 3, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    value = torch.randn(2, 4, 5, 3, 6, device="cuda", dtype=torch.float32, requires_grad=True)
    coords = torch.tensor(
        [
            [[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-0.5, 0.75], [0.8, -0.8], [1.0, -1.0], [-1.0, 1.0]],
            [[0.5, -0.5], [0.25, 0.25], [-0.25, -0.25], [0.0, 1.0], [1.0, 0.0], [-1.0, 0.0], [0.0, -1.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    actual = natten.sparse_na2d(query, key, value, coords, kernel_size=kernel_size, sample_mode=sample_mode)
    expected = _reference_sparse_na2d(query, key, value, coords, kernel_size=kernel_size, sample_mode=sample_mode)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("sample_mode", ["indexed", "bilinear"])
def test_sparse_na2d_preserves_query_order_and_duplicate_boundary_coords(sample_mode):
    torch.manual_seed(2)
    query = torch.randn(1, 6, 2, 4, device="cuda", dtype=torch.float32)
    key = torch.randn(1, 3, 3, 2, 4, device="cuda", dtype=torch.float32)
    value = torch.randn(1, 3, 3, 2, 5, device="cuda", dtype=torch.float32)
    coords = torch.tensor(
        [
            [
                [1.0, 1.0],
                [-1.0, -1.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [-1.0, 1.0],
                [0.0, 0.0],
            ]
        ],
        device="cuda",
        dtype=torch.float32,
    )
    order = torch.tensor([3, 0, 5, 1, 4, 2], device="cuda")

    actual = natten.sparse_na2d(query, key, value, coords, kernel_size=(3, 3), sample_mode=sample_mode)
    expected = _reference_sparse_na2d(query, key, value, coords, kernel_size=(3, 3), sample_mode=sample_mode)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    actual_reordered = natten.sparse_na2d(
        query[:, order],
        key,
        value,
        coords[:, order],
        kernel_size=(3, 3),
        sample_mode=sample_mode,
    )
    torch.testing.assert_close(actual_reordered, actual[:, order], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("sample_mode", ["indexed", "bilinear"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sparse_na2d_shape_lse_and_dtype(dtype, sample_mode):
    torch.manual_seed(3)
    query = torch.randn(2, 4, 2, 8, device="cuda", dtype=dtype, requires_grad=True)
    key = torch.randn(2, 4, 5, 2, 8, device="cuda", dtype=dtype, requires_grad=True)
    value = torch.randn(2, 4, 5, 2, 6, device="cuda", dtype=dtype, requires_grad=True)
    coords = torch.rand(2, 4, 2, device="cuda", dtype=torch.float32) * 2.0 - 1.0

    output, lse = natten.sparse_na2d(
        query,
        key,
        value,
        coords,
        kernel_size=(3, 3),
        sample_mode=sample_mode,
        return_lse=True,
    )

    assert output.shape == (2, 4, 2, 6)
    assert output.dtype == dtype
    assert lse.shape == (2, 4, 2)
    assert lse.dtype == torch.float32

    output.float().sum().backward()
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None


def test_sparse_na2d_simple_matches_indexed_sparse_na2d_forward_and_lse():
    torch.manual_seed(7)
    query = torch.randn(2, 6, 3, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    key = torch.randn(2, 4, 5, 3, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    value = torch.randn(2, 4, 5, 3, 6, device="cuda", dtype=torch.float32, requires_grad=True)
    coords = torch.rand(2, 6, 2, device="cuda", dtype=torch.float32) * 2.0 - 1.0

    actual, actual_lse = natten.sparse_na2d_simple(
        query,
        key,
        value,
        coords,
        kernel_size=(3, 3),
        return_lse=True,
    )
    expected, expected_lse = natten.sparse_na2d(
        query,
        key,
        value,
        coords,
        kernel_size=(3, 3),
        sample_mode="indexed",
        return_lse=True,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-4, atol=1e-4)


def test_sparse_na2d_simple_matches_indexed_sparse_na2d_backward():
    torch.manual_seed(8)
    query = torch.randn(1, 5, 2, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    key = torch.randn(1, 4, 5, 2, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    value = torch.randn(1, 4, 5, 2, 7, device="cuda", dtype=torch.float32, requires_grad=True)
    coords = torch.rand(1, 5, 2, device="cuda", dtype=torch.float32) * 2.0 - 1.0
    grad = torch.randn(1, 5, 2, 7, device="cuda", dtype=torch.float32)

    actual = natten.sparse_na2d_simple(query, key, value, coords, kernel_size=(3, 3))
    actual.backward(grad)
    actual_grads = (query.grad.detach().clone(), key.grad.detach().clone(), value.grad.detach().clone())

    query_ref = query.detach().clone().requires_grad_(True)
    key_ref = key.detach().clone().requires_grad_(True)
    value_ref = value.detach().clone().requires_grad_(True)
    expected = natten.sparse_na2d(
        query_ref,
        key_ref,
        value_ref,
        coords,
        kernel_size=(3, 3),
        sample_mode="indexed",
    )
    expected.backward(grad)

    for actual_grad, expected_grad in zip(actual_grads, (query_ref.grad, key_ref.grad, value_ref.grad)):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-4)


def test_sparse_na2d_bilinear_matches_sparse_na2d_forward_and_lse():
    torch.manual_seed(9)
    query = torch.randn(2, 6, 3, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    key = torch.randn(2, 4, 5, 3, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    value = torch.randn(2, 4, 5, 3, 6, device="cuda", dtype=torch.float32, requires_grad=True)
    coords = torch.rand(2, 6, 2, device="cuda", dtype=torch.float32) * 2.0 - 1.0

    actual, actual_lse = natten.sparse_na2d_bilinear(
        query,
        key,
        value,
        coords,
        kernel_size=(3, 3),
        return_lse=True,
    )
    expected, expected_lse = natten.sparse_na2d(
        query,
        key,
        value,
        coords,
        kernel_size=(3, 3),
        sample_mode="bilinear",
        return_lse=True,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-4, atol=1e-4)


def test_sparse_na2d_bilinear_matches_sparse_na2d_backward():
    torch.manual_seed(10)
    query = torch.randn(1, 5, 2, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    key = torch.randn(1, 4, 5, 2, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    value = torch.randn(1, 4, 5, 2, 7, device="cuda", dtype=torch.float32, requires_grad=True)
    coords = torch.rand(1, 5, 2, device="cuda", dtype=torch.float32) * 2.0 - 1.0
    grad = torch.randn(1, 5, 2, 7, device="cuda", dtype=torch.float32)

    actual = natten.sparse_na2d_bilinear(query, key, value, coords, kernel_size=(3, 3))
    actual.backward(grad)
    actual_grads = (query.grad.detach().clone(), key.grad.detach().clone(), value.grad.detach().clone())

    query_ref = query.detach().clone().requires_grad_(True)
    key_ref = key.detach().clone().requires_grad_(True)
    value_ref = value.detach().clone().requires_grad_(True)
    expected = natten.sparse_na2d(
        query_ref,
        key_ref,
        value_ref,
        coords,
        kernel_size=(3, 3),
        sample_mode="bilinear",
    )
    expected.backward(grad)

    for actual_grad, expected_grad in zip(actual_grads, (query_ref.grad, key_ref.grad, value_ref.grad)):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-4)


def test_sparse_na2d_bilinear_matches_sparse_na2d_fp16_even_dim():
    torch.manual_seed(11)
    query = torch.randn(2, 7, 3, 32, device="cuda", dtype=torch.float16, requires_grad=True)
    key = torch.randn(2, 9, 10, 3, 32, device="cuda", dtype=torch.float16, requires_grad=True)
    value = torch.randn(2, 9, 10, 3, 32, device="cuda", dtype=torch.float16, requires_grad=True)
    coords = torch.rand(2, 7, 2, device="cuda", dtype=torch.float32) * 2.0 - 1.0
    grad = torch.randn(2, 7, 3, 32, device="cuda", dtype=torch.float16)

    actual, actual_lse = natten.sparse_na2d_bilinear(
        query,
        key,
        value,
        coords,
        kernel_size=(5, 5),
        return_lse=True,
    )
    actual.backward(grad)
    actual_grads = (query.grad.detach().clone(), key.grad.detach().clone(), value.grad.detach().clone())

    query_ref = query.detach().clone().requires_grad_(True)
    key_ref = key.detach().clone().requires_grad_(True)
    value_ref = value.detach().clone().requires_grad_(True)
    expected, expected_lse = natten.sparse_na2d(
        query_ref,
        key_ref,
        value_ref,
        coords,
        kernel_size=(5, 5),
        sample_mode="bilinear",
        return_lse=True,
    )
    expected.backward(grad)

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=2e-3, atol=2e-3)
    for actual_grad, expected_grad in zip(actual_grads, (query_ref.grad, key_ref.grad, value_ref.grad)):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize("sample_mode", ["indexed", "bilinear"])
def test_sparse_na2d_backward_matches_reference(sample_mode):
    torch.manual_seed(1)
    query = torch.randn(1, 5, 2, 4, device="cuda", dtype=torch.float32, requires_grad=True)
    key = torch.randn(1, 3, 4, 2, 4, device="cuda", dtype=torch.float32, requires_grad=True)
    value = torch.randn(1, 3, 4, 2, 5, device="cuda", dtype=torch.float32, requires_grad=True)
    coords = torch.tensor(
        [[[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [0.25, -0.75], [-0.25, 0.75]]],
        device="cuda",
        dtype=torch.float32,
    )
    grad = torch.randn(1, 5, 2, 5, device="cuda", dtype=torch.float32)

    actual = natten.sparse_na2d(query, key, value, coords, kernel_size=(3, 3), sample_mode=sample_mode)
    actual.backward(grad)
    actual_grads = (query.grad.detach().clone(), key.grad.detach().clone(), value.grad.detach().clone())

    query_ref = query.detach().clone().requires_grad_(True)
    key_ref = key.detach().clone().requires_grad_(True)
    value_ref = value.detach().clone().requires_grad_(True)
    expected = _reference_sparse_na2d(query_ref, key_ref, value_ref, coords, kernel_size=(3, 3), sample_mode=sample_mode)
    expected.backward(grad)

    for actual_grad, expected_grad in zip(actual_grads, (query_ref.grad, key_ref.grad, value_ref.grad)):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("sample_mode", ["indexed", "bilinear"])
def test_sparse_na2d_key_rope_matches_reference(sample_mode):
    torch.manual_seed(5)
    query = torch.randn(1, 6, 4, 4, device="cuda", dtype=torch.float32, requires_grad=True)
    key = torch.randn(1, 4, 5, 4, 4, device="cuda", dtype=torch.float32, requires_grad=True)
    value = torch.randn(1, 4, 5, 4, 7, device="cuda", dtype=torch.float32, requires_grad=True)
    coords = torch.tensor(
        [[[-0.9, -0.7], [0.95, 0.85], [0.0, 0.0], [0.35, -0.25], [-0.15, 0.55], [0.8, -0.9]]],
        device="cuda",
        dtype=torch.float32,
    )
    grad = torch.randn(1, 6, 4, 7, device="cuda", dtype=torch.float32)

    actual = natten.sparse_na2d(
        query,
        key,
        value,
        coords,
        kernel_size=(3, 3),
        sample_mode=sample_mode,
        apply_key_rope=True,
    )
    actual.backward(grad)
    actual_grads = (query.grad.detach().clone(), key.grad.detach().clone(), value.grad.detach().clone())

    query_ref = query.detach().clone().requires_grad_(True)
    key_ref = key.detach().clone().requires_grad_(True)
    value_ref = value.detach().clone().requires_grad_(True)
    expected = _reference_sparse_na2d(
        query_ref,
        key_ref,
        value_ref,
        coords,
        kernel_size=(3, 3),
        sample_mode=sample_mode,
        apply_key_rope=True,
    )
    expected.backward(grad)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    for actual_grad, expected_grad in zip(actual_grads, (query_ref.grad, key_ref.grad, value_ref.grad)):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("qk_norm_before_rope", [True, False])
def test_sparse_na2d_qk_norm_rope_order_matches_reference(qk_norm_before_rope):
    torch.manual_seed(6)
    query = torch.randn(1, 5, 4, 4, device="cuda", dtype=torch.float32, requires_grad=True)
    key = torch.randn(1, 4, 5, 4, 4, device="cuda", dtype=torch.float32, requires_grad=True)
    value = torch.randn(1, 4, 5, 4, 7, device="cuda", dtype=torch.float32, requires_grad=True)
    coords = torch.rand(1, 5, 2, device="cuda", dtype=torch.float32) * 2.0 - 1.0
    q_weight = torch.randn(16, device="cuda", dtype=torch.float32)
    k_weight = torch.randn(16, device="cuda", dtype=torch.float32)
    grad = torch.randn(1, 5, 4, 7, device="cuda", dtype=torch.float32)

    actual = natten.sparse_na2d(
        query,
        key,
        value,
        coords,
        kernel_size=(3, 3),
        sample_mode="bilinear",
        apply_query_rope=True,
        apply_key_rope=True,
        apply_qk_norm=True,
        q_norm_weight=q_weight,
        k_norm_weight=k_weight,
        qk_norm_before_rope=qk_norm_before_rope,
    )
    actual.backward(grad)
    actual_grads = (query.grad.detach().clone(), key.grad.detach().clone(), value.grad.detach().clone())

    query_ref = query.detach().clone().requires_grad_(True)
    key_ref = key.detach().clone().requires_grad_(True)
    value_ref = value.detach().clone().requires_grad_(True)
    expected = sparse_na2d_pytorch(
        query_ref,
        key_ref,
        value_ref,
        coords,
        kernel_size=(3, 3),
        sample_mode="bilinear",
        apply_query_rope=True,
        apply_key_rope=True,
        apply_qk_norm=True,
        q_norm_weight=q_weight,
        k_norm_weight=k_weight,
        qk_norm_before_rope=qk_norm_before_rope,
    )
    expected.backward(grad)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    for actual_grad, expected_grad in zip(actual_grads, (query_ref.grad, key_ref.grad, value_ref.grad)):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-4)


def test_sparse_na2d_does_not_break_dense_na2d_smoke():
    torch.manual_seed(4)
    query = torch.randn(1, 4, 4, 2, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    key = torch.randn(1, 4, 4, 2, 8, device="cuda", dtype=torch.float32, requires_grad=True)
    value = torch.randn(1, 4, 4, 2, 8, device="cuda", dtype=torch.float32, requires_grad=True)

    output = natten.na2d(query, key, value, kernel_size=(3, 3), dilation=(1, 1), stride=1)
    assert output.shape == query.shape
    output.sum().backward()
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None
