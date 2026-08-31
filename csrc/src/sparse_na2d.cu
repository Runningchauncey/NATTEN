/***************************************************************************************************
 * Copyright (c) 2022 - 2026 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 **************************************************************************************************/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <type_traits>

#include <natten/helpers.h>
#include <natten/fna.h>

namespace natten {
namespace {

constexpr int kThreads = 256;
constexpr int kSimpleThreads = 64;

template <typename scalar_t>
__device__ inline float load_as_float(const scalar_t* ptr) {
  return static_cast<float>(*ptr);
}

template <typename coord_t>
__device__ inline int coord_to_index(coord_t coord, int extent) {
  float x = static_cast<float>(coord);
  int idx = static_cast<int>(rintf(((x + 1.0f) * 0.5f * extent) - 0.5f));
  return max(0, min(idx, extent - 1));
}

template <typename coord_t>
__device__ inline float coord_to_position(coord_t coord, int extent) {
  float x = static_cast<float>(coord);
  return fminf(fmaxf(((x + 1.0f) * 0.5f * extent) - 0.5f, 0.0f), static_cast<float>(extent - 1));
}

__device__ inline void bilinear_indices_and_weights(
    float y,
    float x,
    int height,
    int width,
    int& y0,
    int& y1,
    int& x0,
    int& x1,
    float& wy0,
    float& wy1,
    float& wx0,
    float& wx1) {
  y = fminf(fmaxf(y, 0.0f), static_cast<float>(height - 1));
  x = fminf(fmaxf(x, 0.0f), static_cast<float>(width - 1));
  y0 = static_cast<int>(floorf(y));
  x0 = static_cast<int>(floorf(x));
  y1 = min(y0 + 1, height - 1);
  x1 = min(x0 + 1, width - 1);
  wy1 = y - static_cast<float>(y0);
  wx1 = x - static_cast<float>(x0);
  wy0 = 1.0f - wy1;
  wx0 = 1.0f - wx1;
}

template <typename scalar_t>
__device__ inline float bilinear_load(
    const scalar_t* __restrict__ tensor,
    int batch_idx,
    int height,
    int width,
    int heads,
    int head_idx,
    int dim,
    int channel,
    float y,
    float x) {
  int y0, y1, x0, x1;
  float wy0, wy1, wx0, wx1;
  bilinear_indices_and_weights(y, x, height, width, y0, y1, x0, x1, wy0, wy1, wx0, wx1);
  int base00 = ((((batch_idx * height + y0) * width + x0) * heads + head_idx) * dim) + channel;
  int base01 = ((((batch_idx * height + y0) * width + x1) * heads + head_idx) * dim) + channel;
  int base10 = ((((batch_idx * height + y1) * width + x0) * heads + head_idx) * dim) + channel;
  int base11 = ((((batch_idx * height + y1) * width + x1) * heads + head_idx) * dim) + channel;
  return wy0 * wx0 * load_as_float(tensor + base00) +
      wy0 * wx1 * load_as_float(tensor + base01) +
      wy1 * wx0 * load_as_float(tensor + base10) +
      wy1 * wx1 * load_as_float(tensor + base11);
}

template <typename scalar_t>
__device__ inline void bilinear_atomic_add(
    scalar_t* __restrict__ grad,
    int batch_idx,
    int height,
    int width,
    int heads,
    int head_idx,
    int dim,
    int channel,
    float y,
    float x,
    float grad_value,
    int numel) {
  int y0, y1, x0, x1;
  float wy0, wy1, wx0, wx1;
  bilinear_indices_and_weights(y, x, height, width, y0, y1, x0, x1, wy0, wy1, wx0, wx1);
  int idx00 = ((((batch_idx * height + y0) * width + x0) * heads + head_idx) * dim) + channel;
  int idx01 = ((((batch_idx * height + y0) * width + x1) * heads + head_idx) * dim) + channel;
  int idx10 = ((((batch_idx * height + y1) * width + x0) * heads + head_idx) * dim) + channel;
  int idx11 = ((((batch_idx * height + y1) * width + x1) * heads + head_idx) * dim) + channel;
  at::native::fastAtomicAdd(grad, idx00, numel, static_cast<scalar_t>(wy0 * wx0 * grad_value), true);
  at::native::fastAtomicAdd(grad, idx01, numel, static_cast<scalar_t>(wy0 * wx1 * grad_value), true);
  at::native::fastAtomicAdd(grad, idx10, numel, static_cast<scalar_t>(wy1 * wx0 * grad_value), true);
  at::native::fastAtomicAdd(grad, idx11, numel, static_cast<scalar_t>(wy1 * wx1 * grad_value), true);
}

__device__ inline int rope_frequency_index(int global_channel, int total_dim) {
  int quarter_dim = total_dim / 4;
  int half_dim = total_dim / 2;
  int local_channel = global_channel < half_dim ? global_channel : global_channel - half_dim;
  return local_channel % quarter_dim;
}

__device__ inline float rope_frequency_uncached(int freq_idx, int total_dim, float rope_theta) {
  constexpr float kTwoPi = 6.28318530717958647692f;
  int quarter_dim = total_dim / 4;
  if (quarter_dim <= 1) {
    return kTwoPi;
  }
  float exponent = -static_cast<float>(freq_idx) / static_cast<float>(quarter_dim - 1);
  return powf(rope_theta, exponent) * kTwoPi;
}

__device__ inline float rope_frequency_cached(
    const float* __restrict__ rope_freqs,
    int global_channel,
    int total_dim,
    float rope_theta) {
  int freq_idx = rope_frequency_index(global_channel, total_dim);
  return rope_freqs != nullptr ? rope_freqs[freq_idx] : rope_frequency_uncached(freq_idx, total_dim, rope_theta);
}

__device__ inline void init_rope_freqs(
    float* __restrict__ rope_freqs,
    int total_dim,
    float rope_theta) {
  if (rope_freqs == nullptr) {
    return;
  }
  int quarter_dim = total_dim / 4;
  for (int idx = threadIdx.x; idx < quarter_dim; idx += blockDim.x) {
    rope_freqs[idx] = rope_frequency_uncached(idx, total_dim, rope_theta);
  }
}

__device__ inline void rope_pair_channel(
    int head_idx,
    int channel,
    int heads,
    int dim,
    int& pair_head,
    int& pair_channel,
    float& sign) {
  int total_dim = heads * dim;
  int half_dim = total_dim / 2;
  int global_channel = head_idx * dim + channel;
  int pair_global;
  if (global_channel < half_dim) {
    pair_global = global_channel + half_dim;
    sign = -1.0f;
  } else {
    pair_global = global_channel - half_dim;
    sign = 1.0f;
  }
  pair_head = pair_global / dim;
  pair_channel = pair_global - pair_head * dim;
}

template <typename scalar_t>
__device__ inline float maybe_rope_key_load(
    const scalar_t* __restrict__ key,
    int batch_idx,
    int height,
    int width,
    int heads,
    int head_idx,
    int dim,
    int channel,
    int y,
    int x,
    float sample_y,
    float sample_x,
    bool use_bilinear,
    bool apply_key_rope,
    float pos_y,
    float pos_x,
    float rope_theta,
    const float* __restrict__ rope_freqs) {
  const scalar_t* k_ptr =
      key + ((((batch_idx * height + y) * width + x) * heads + head_idx) * dim);
  float kval = use_bilinear
      ? bilinear_load(key, batch_idx, height, width, heads, head_idx, dim, channel, sample_y, sample_x)
      : load_as_float(k_ptr + channel);
  if (!apply_key_rope) {
    return kval;
  }

  int pair_head, pair_channel;
  float sign;
  rope_pair_channel(head_idx, channel, heads, dim, pair_head, pair_channel, sign);
  const scalar_t* pair_ptr =
      key + ((((batch_idx * height + y) * width + x) * heads + pair_head) * dim);
  float pair_val = use_bilinear
      ? bilinear_load(key, batch_idx, height, width, heads, pair_head, dim, pair_channel, sample_y, sample_x)
      : load_as_float(pair_ptr + pair_channel);

  int total_dim = heads * dim;
  int global_channel = head_idx * dim + channel;
  int half_dim = total_dim / 2;
  float freq = rope_frequency_cached(rope_freqs, global_channel, total_dim, rope_theta);
  float angle = (global_channel < half_dim ? pos_y : pos_x) * freq;
  float sin_angle;
  float cos_angle;
  sincosf(angle, &sin_angle, &cos_angle);
  return kval * cos_angle + sign * pair_val * sin_angle;
}

template <typename scalar_t>
__device__ inline void maybe_rope_key_atomic_add(
    scalar_t* __restrict__ grad_key,
    int batch_idx,
    int height,
    int width,
    int heads,
    int head_idx,
    int dim,
    int channel,
    int y,
    int x,
    float sample_y,
    float sample_x,
    bool use_bilinear,
    bool apply_key_rope,
    float pos_y,
    float pos_x,
    float rope_theta,
    const float* __restrict__ rope_freqs,
    float grad_rotated,
    int numel) {
  if (!apply_key_rope) {
    if (use_bilinear) {
      bilinear_atomic_add(
          grad_key,
          batch_idx,
          height,
          width,
          heads,
          head_idx,
          dim,
          channel,
          sample_y,
          sample_x,
          grad_rotated,
          numel);
    } else {
      at::native::fastAtomicAdd(
          grad_key,
          ((((batch_idx * height + y) * width + x) * heads + head_idx) * dim) + channel,
          numel,
          static_cast<scalar_t>(grad_rotated),
          true);
    }
    return;
  }

  int total_dim = heads * dim;
  int global_channel = head_idx * dim + channel;
  int half_dim = total_dim / 2;
  float freq = rope_frequency_cached(rope_freqs, global_channel, total_dim, rope_theta);
  float angle = (global_channel < half_dim ? pos_y : pos_x) * freq;
  float sin_angle;
  float cos_angle;
  sincosf(angle, &sin_angle, &cos_angle);

  int pair_head, pair_channel;
  float sign;
  rope_pair_channel(head_idx, channel, heads, dim, pair_head, pair_channel, sign);
  float grad_same = grad_rotated * cos_angle;
  float grad_pair = grad_rotated * sign * sin_angle;

  if (use_bilinear) {
    bilinear_atomic_add(
        grad_key,
        batch_idx,
        height,
        width,
        heads,
        head_idx,
        dim,
        channel,
        sample_y,
        sample_x,
        grad_same,
        numel);
    bilinear_atomic_add(
        grad_key,
        batch_idx,
        height,
        width,
        heads,
        pair_head,
        dim,
        pair_channel,
        sample_y,
        sample_x,
        grad_pair,
        numel);
  } else {
    at::native::fastAtomicAdd(
        grad_key,
        ((((batch_idx * height + y) * width + x) * heads + head_idx) * dim) + channel,
        numel,
        static_cast<scalar_t>(grad_same),
        true);
    at::native::fastAtomicAdd(
        grad_key,
        ((((batch_idx * height + y) * width + x) * heads + pair_head) * dim) + pair_channel,
        numel,
        static_cast<scalar_t>(grad_pair),
        true);
  }
}

template <typename scalar_t, typename coord_t>
__global__ void sparse_na2d_forward_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const coord_t* __restrict__ coords,
    scalar_t* __restrict__ out,
    float* __restrict__ logsumexp,
    int batch,
    int num_queries,
    int height,
    int width,
    int heads,
    int dim,
    int dim_value,
    int kernel_h,
    int kernel_w,
    float scale,
    bool use_bilinear,
    bool apply_key_rope,
    float rope_theta) {
  extern __shared__ float smem[];
  float* probs = smem;
  float* rope_freqs = apply_key_rope ? probs + kernel_h * kernel_w : nullptr;

  int query_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int batch_idx = blockIdx.z;
  int k_tokens = kernel_h * kernel_w;

  const scalar_t* q_ptr =
      query + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim);
  const coord_t* coord_ptr = coords + ((batch_idx * num_queries + query_idx) * 2);
  int center_y = coord_to_index(coord_ptr[0], height);
  int center_x = coord_to_index(coord_ptr[1], width);
  float center_yf = coord_to_position(coord_ptr[0], height);
  float center_xf = coord_to_position(coord_ptr[1], width);
  init_rope_freqs(rope_freqs, heads * dim, rope_theta);
  __syncthreads();

  for (int offset_idx = threadIdx.x; offset_idx < k_tokens; offset_idx += blockDim.x) {
    int oy = (offset_idx / kernel_w) - (kernel_h / 2);
    int ox = (offset_idx % kernel_w) - (kernel_w / 2);
    int y = max(0, min(center_y + oy, height - 1));
    int x = max(0, min(center_x + ox, width - 1));
    float yf = fminf(fmaxf(center_yf + static_cast<float>(oy), 0.0f), static_cast<float>(height - 1));
    float xf = fminf(fmaxf(center_xf + static_cast<float>(ox), 0.0f), static_cast<float>(width - 1));
    float pos_y = (use_bilinear ? yf : static_cast<float>(y)) / static_cast<float>(max(height - 1, 1));
    float pos_x = (use_bilinear ? xf : static_cast<float>(x)) / static_cast<float>(max(width - 1, 1));
    float logit = 0.0f;
    for (int d = 0; d < dim; ++d) {
      float kval = maybe_rope_key_load(
          key,
          batch_idx,
          height,
          width,
          heads,
          head_idx,
          dim,
          d,
          y,
          x,
          yf,
          xf,
          use_bilinear,
          apply_key_rope,
          pos_y,
          pos_x,
          rope_theta,
          rope_freqs);
      logit += load_as_float(q_ptr + d) * kval;
    }
    probs[offset_idx] = logit * scale;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float max_logit = -INFINITY;
    for (int i = 0; i < k_tokens; ++i) {
      max_logit = fmaxf(max_logit, probs[i]);
    }
    float denom = 0.0f;
    for (int i = 0; i < k_tokens; ++i) {
      float p = expf(probs[i] - max_logit);
      probs[i] = p;
      denom += p;
    }
    float inv_denom = 1.0f / denom;
    for (int i = 0; i < k_tokens; ++i) {
      probs[i] *= inv_denom;
    }
    logsumexp[(batch_idx * num_queries + query_idx) * heads + head_idx] =
        logf(denom) + max_logit;
  }

  __syncthreads();

  scalar_t* out_ptr =
      out + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim_value);
  for (int dv = threadIdx.x; dv < dim_value; dv += blockDim.x) {
    float acc = 0.0f;
    int offset_idx = 0;
    for (int oy = -(kernel_h / 2); oy <= kernel_h / 2; ++oy) {
      int y = max(0, min(center_y + oy, height - 1));
      float yf = fminf(fmaxf(center_yf + static_cast<float>(oy), 0.0f), static_cast<float>(height - 1));
      for (int ox = -(kernel_w / 2); ox <= kernel_w / 2; ++ox, ++offset_idx) {
        int x = max(0, min(center_x + ox, width - 1));
        float xf = fminf(fmaxf(center_xf + static_cast<float>(ox), 0.0f), static_cast<float>(width - 1));
        const scalar_t* v_ptr =
            value + ((((batch_idx * height + y) * width + x) * heads + head_idx) * dim_value);
        float vval = use_bilinear
            ? bilinear_load(value, batch_idx, height, width, heads, head_idx, dim_value, dv, yf, xf)
            : load_as_float(v_ptr + dv);
        acc += probs[offset_idx] * vval;
      }
    }
    out_ptr[dv] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, typename coord_t>
__global__ void sparse_na2d_backward_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const coord_t* __restrict__ coords,
    const scalar_t* __restrict__ out,
    const scalar_t* __restrict__ grad_out,
    const float* __restrict__ logsumexp,
    scalar_t* __restrict__ grad_query,
    scalar_t* __restrict__ grad_key,
    scalar_t* __restrict__ grad_value,
    int batch,
    int num_queries,
    int height,
    int width,
    int heads,
    int dim,
    int dim_value,
    int kernel_h,
    int kernel_w,
    float scale,
    bool use_bilinear,
    bool apply_key_rope,
    float rope_theta) {
  extern __shared__ float smem[];
  float* probs = smem;
  float* dp = smem + kernel_h * kernel_w;
  float* reductions = dp + kernel_h * kernel_w;
  float* rope_freqs = apply_key_rope ? reductions + kThreads : nullptr;

  int query_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int batch_idx = blockIdx.z;
  int k_tokens = kernel_h * kernel_w;

  const scalar_t* q_ptr =
      query + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim);
  const scalar_t* o_ptr =
      out + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim_value);
  const scalar_t* go_ptr =
      grad_out + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim_value);
  const coord_t* coord_ptr = coords + ((batch_idx * num_queries + query_idx) * 2);
  int center_y = coord_to_index(coord_ptr[0], height);
  int center_x = coord_to_index(coord_ptr[1], width);
  float center_yf = coord_to_position(coord_ptr[0], height);
  float center_xf = coord_to_position(coord_ptr[1], width);
  float lse = logsumexp[(batch_idx * num_queries + query_idx) * heads + head_idx];
  init_rope_freqs(rope_freqs, heads * dim, rope_theta);
  __syncthreads();

  for (int offset_idx = threadIdx.x; offset_idx < k_tokens; offset_idx += blockDim.x) {
    int oy = (offset_idx / kernel_w) - (kernel_h / 2);
    int ox = (offset_idx % kernel_w) - (kernel_w / 2);
    int y = max(0, min(center_y + oy, height - 1));
    int x = max(0, min(center_x + ox, width - 1));
    float yf = fminf(fmaxf(center_yf + static_cast<float>(oy), 0.0f), static_cast<float>(height - 1));
    float xf = fminf(fmaxf(center_xf + static_cast<float>(ox), 0.0f), static_cast<float>(width - 1));
    float pos_y = (use_bilinear ? yf : static_cast<float>(y)) / static_cast<float>(max(height - 1, 1));
    float pos_x = (use_bilinear ? xf : static_cast<float>(x)) / static_cast<float>(max(width - 1, 1));
    float logit = 0.0f;
    for (int d = 0; d < dim; ++d) {
      float kval = maybe_rope_key_load(
          key,
          batch_idx,
          height,
          width,
          heads,
          head_idx,
          dim,
          d,
          y,
          x,
          yf,
          xf,
          use_bilinear,
          apply_key_rope,
          pos_y,
          pos_x,
          rope_theta,
          rope_freqs);
      logit += load_as_float(q_ptr + d) * kval;
    }
    probs[offset_idx] = expf(logit * scale - lse);
    dp[offset_idx] = 0.0f;
  }
  __syncthreads();

  float delta_part = 0.0f;
  for (int dv = threadIdx.x; dv < dim_value; dv += blockDim.x) {
    delta_part += load_as_float(go_ptr + dv) * load_as_float(o_ptr + dv);
  }
  reductions[threadIdx.x] = delta_part;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float delta = reductions[0];

  for (int offset_idx = 0; offset_idx < k_tokens; ++offset_idx) {
    int oy = (offset_idx / kernel_w) - (kernel_h / 2);
    int ox = (offset_idx % kernel_w) - (kernel_w / 2);
    int y = max(0, min(center_y + oy, height - 1));
    int x = max(0, min(center_x + ox, width - 1));
    float yf = fminf(fmaxf(center_yf + static_cast<float>(oy), 0.0f), static_cast<float>(height - 1));
    float xf = fminf(fmaxf(center_xf + static_cast<float>(ox), 0.0f), static_cast<float>(width - 1));
    const scalar_t* v_ptr =
        value + ((((batch_idx * height + y) * width + x) * heads + head_idx) * dim_value);

    float dp_part = 0.0f;
    for (int dv = threadIdx.x; dv < dim_value; dv += blockDim.x) {
      float vval = use_bilinear
          ? bilinear_load(value, batch_idx, height, width, heads, head_idx, dim_value, dv, yf, xf)
          : load_as_float(v_ptr + dv);
      dp_part += load_as_float(go_ptr + dv) * vval;
      float gv = probs[offset_idx] * load_as_float(go_ptr + dv);
      if (use_bilinear) {
        bilinear_atomic_add(
            grad_value,
            batch_idx,
            height,
            width,
            heads,
            head_idx,
            dim_value,
            dv,
            yf,
            xf,
            gv,
            batch * height * width * heads * dim_value);
      } else {
        at::native::fastAtomicAdd(
            grad_value,
            ((((batch_idx * height + y) * width + x) * heads + head_idx) * dim_value) + dv,
            batch * height * width * heads * dim_value,
            static_cast<scalar_t>(gv),
            true);
      }
    }
    reductions[threadIdx.x] = dp_part;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reductions[threadIdx.x] += reductions[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      dp[offset_idx] = probs[offset_idx] * (reductions[0] - delta) * scale;
    }
    __syncthreads();
  }

  scalar_t* gq_ptr =
      grad_query + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim);
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int offset_idx = 0; offset_idx < k_tokens; ++offset_idx) {
      int oy = (offset_idx / kernel_w) - (kernel_h / 2);
      int ox = (offset_idx % kernel_w) - (kernel_w / 2);
      int y = max(0, min(center_y + oy, height - 1));
      int x = max(0, min(center_x + ox, width - 1));
      float yf = fminf(fmaxf(center_yf + static_cast<float>(oy), 0.0f), static_cast<float>(height - 1));
      float xf = fminf(fmaxf(center_xf + static_cast<float>(ox), 0.0f), static_cast<float>(width - 1));
      float pos_y = (use_bilinear ? yf : static_cast<float>(y)) / static_cast<float>(max(height - 1, 1));
      float pos_x = (use_bilinear ? xf : static_cast<float>(x)) / static_cast<float>(max(width - 1, 1));
      float kval = maybe_rope_key_load(
          key,
          batch_idx,
          height,
          width,
          heads,
          head_idx,
          dim,
          d,
          y,
          x,
          yf,
          xf,
          use_bilinear,
          apply_key_rope,
          pos_y,
          pos_x,
          rope_theta,
          rope_freqs);
      acc += dp[offset_idx] * kval;
      float gk = dp[offset_idx] * load_as_float(q_ptr + d);
      maybe_rope_key_atomic_add(
          grad_key,
          batch_idx,
          height,
          width,
          heads,
          head_idx,
          dim,
          d,
          y,
          x,
          yf,
          xf,
          use_bilinear,
          apply_key_rope,
          pos_y,
          pos_x,
          rope_theta,
          rope_freqs,
          gk,
          batch * height * width * heads * dim);
    }
    gq_ptr[d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, typename coord_t>
__global__ void sparse_na2d_simple_forward_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const coord_t* __restrict__ coords,
    scalar_t* __restrict__ out,
    float* __restrict__ logsumexp,
    int batch,
    int num_queries,
    int height,
    int width,
    int heads,
    int dim,
    int dim_value,
    int kernel_h,
    int kernel_w,
    float scale) {
  extern __shared__ float probs[];

  int query_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int batch_idx = blockIdx.z;
  int k_tokens = kernel_h * kernel_w;

  const scalar_t* q_ptr =
      query + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim);
  const coord_t* coord_ptr = coords + ((batch_idx * num_queries + query_idx) * 2);
  int center_y = coord_to_index(coord_ptr[0], height);
  int center_x = coord_to_index(coord_ptr[1], width);

  for (int offset_idx = threadIdx.x; offset_idx < k_tokens; offset_idx += blockDim.x) {
    int oy = (offset_idx / kernel_w) - (kernel_h / 2);
    int ox = (offset_idx % kernel_w) - (kernel_w / 2);
    int y = max(0, min(center_y + oy, height - 1));
    int x = max(0, min(center_x + ox, width - 1));
    const scalar_t* k_ptr =
        key + ((((batch_idx * height + y) * width + x) * heads + head_idx) * dim);

    float logit = 0.0f;
    for (int d = 0; d < dim; ++d) {
      logit += load_as_float(q_ptr + d) * load_as_float(k_ptr + d);
    }
    probs[offset_idx] = logit * scale;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float max_logit = -INFINITY;
    for (int i = 0; i < k_tokens; ++i) {
      max_logit = fmaxf(max_logit, probs[i]);
    }
    float denom = 0.0f;
    for (int i = 0; i < k_tokens; ++i) {
      float p = expf(probs[i] - max_logit);
      probs[i] = p;
      denom += p;
    }
    float inv_denom = 1.0f / denom;
    for (int i = 0; i < k_tokens; ++i) {
      probs[i] *= inv_denom;
    }
    logsumexp[(batch_idx * num_queries + query_idx) * heads + head_idx] =
        logf(denom) + max_logit;
  }
  __syncthreads();

  scalar_t* out_ptr =
      out + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim_value);
  for (int dv = threadIdx.x; dv < dim_value; dv += blockDim.x) {
    float acc = 0.0f;
    int offset_idx = 0;
    for (int oy = -(kernel_h / 2); oy <= kernel_h / 2; ++oy) {
      int y = max(0, min(center_y + oy, height - 1));
      for (int ox = -(kernel_w / 2); ox <= kernel_w / 2; ++ox, ++offset_idx) {
        int x = max(0, min(center_x + ox, width - 1));
        const scalar_t* v_ptr =
            value + ((((batch_idx * height + y) * width + x) * heads + head_idx) * dim_value);
        acc += probs[offset_idx] * load_as_float(v_ptr + dv);
      }
    }
    out_ptr[dv] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, typename coord_t>
__global__ void sparse_na2d_simple_backward_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const coord_t* __restrict__ coords,
    const scalar_t* __restrict__ out,
    const scalar_t* __restrict__ grad_out,
    const float* __restrict__ logsumexp,
    scalar_t* __restrict__ grad_query,
    scalar_t* __restrict__ grad_key,
    scalar_t* __restrict__ grad_value,
    int batch,
    int num_queries,
    int height,
    int width,
    int heads,
    int dim,
    int dim_value,
    int kernel_h,
    int kernel_w,
    float scale) {
  extern __shared__ float smem[];
  float* probs = smem;
  float* dp = smem + kernel_h * kernel_w;
  float* reductions = dp + kernel_h * kernel_w;

  int query_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int batch_idx = blockIdx.z;
  int k_tokens = kernel_h * kernel_w;

  const scalar_t* q_ptr =
      query + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim);
  const scalar_t* o_ptr =
      out + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim_value);
  const scalar_t* go_ptr =
      grad_out + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim_value);
  const coord_t* coord_ptr = coords + ((batch_idx * num_queries + query_idx) * 2);
  int center_y = coord_to_index(coord_ptr[0], height);
  int center_x = coord_to_index(coord_ptr[1], width);
  float lse = logsumexp[(batch_idx * num_queries + query_idx) * heads + head_idx];

  for (int offset_idx = threadIdx.x; offset_idx < k_tokens; offset_idx += blockDim.x) {
    int oy = (offset_idx / kernel_w) - (kernel_h / 2);
    int ox = (offset_idx % kernel_w) - (kernel_w / 2);
    int y = max(0, min(center_y + oy, height - 1));
    int x = max(0, min(center_x + ox, width - 1));
    const scalar_t* k_ptr =
        key + ((((batch_idx * height + y) * width + x) * heads + head_idx) * dim);

    float logit = 0.0f;
    for (int d = 0; d < dim; ++d) {
      logit += load_as_float(q_ptr + d) * load_as_float(k_ptr + d);
    }
    probs[offset_idx] = expf(logit * scale - lse);
    dp[offset_idx] = 0.0f;
  }
  __syncthreads();

  float delta_part = 0.0f;
  for (int dv = threadIdx.x; dv < dim_value; dv += blockDim.x) {
    delta_part += load_as_float(go_ptr + dv) * load_as_float(o_ptr + dv);
  }
  reductions[threadIdx.x] = delta_part;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float delta = reductions[0];

  for (int offset_idx = 0; offset_idx < k_tokens; ++offset_idx) {
    int oy = (offset_idx / kernel_w) - (kernel_h / 2);
    int ox = (offset_idx % kernel_w) - (kernel_w / 2);
    int y = max(0, min(center_y + oy, height - 1));
    int x = max(0, min(center_x + ox, width - 1));
    const scalar_t* v_ptr =
        value + ((((batch_idx * height + y) * width + x) * heads + head_idx) * dim_value);

    float dp_part = 0.0f;
    for (int dv = threadIdx.x; dv < dim_value; dv += blockDim.x) {
      float vval = load_as_float(v_ptr + dv);
      dp_part += load_as_float(go_ptr + dv) * vval;
      float gv = probs[offset_idx] * load_as_float(go_ptr + dv);
      at::native::fastAtomicAdd(
          grad_value,
          ((((batch_idx * height + y) * width + x) * heads + head_idx) * dim_value) + dv,
          batch * height * width * heads * dim_value,
          static_cast<scalar_t>(gv),
          true);
    }
    reductions[threadIdx.x] = dp_part;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reductions[threadIdx.x] += reductions[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      dp[offset_idx] = probs[offset_idx] * (reductions[0] - delta) * scale;
    }
    __syncthreads();
  }

  scalar_t* gq_ptr =
      grad_query + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim);
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int offset_idx = 0; offset_idx < k_tokens; ++offset_idx) {
      int oy = (offset_idx / kernel_w) - (kernel_h / 2);
      int ox = (offset_idx % kernel_w) - (kernel_w / 2);
      int y = max(0, min(center_y + oy, height - 1));
      int x = max(0, min(center_x + ox, width - 1));
      const scalar_t* k_ptr =
          key + ((((batch_idx * height + y) * width + x) * heads + head_idx) * dim);
      acc += dp[offset_idx] * load_as_float(k_ptr + d);
      float gk = dp[offset_idx] * load_as_float(q_ptr + d);
      at::native::fastAtomicAdd(
          grad_key,
          ((((batch_idx * height + y) * width + x) * heads + head_idx) * dim) + d,
          batch * height * width * heads * dim,
          static_cast<scalar_t>(gk),
          true);
    }
    gq_ptr[d] = static_cast<scalar_t>(acc);
  }
}

template <typename coord_t>
__device__ inline void init_bilinear_neighborhood(
    const coord_t* __restrict__ coord_ptr,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int* __restrict__ y0s,
    int* __restrict__ y1s,
    int* __restrict__ x0s,
    int* __restrict__ x1s,
    float* __restrict__ w00s,
    float* __restrict__ w01s,
    float* __restrict__ w10s,
    float* __restrict__ w11s) {
  float center_yf = coord_to_position(coord_ptr[0], height);
  float center_xf = coord_to_position(coord_ptr[1], width);
  int k_tokens = kernel_h * kernel_w;

  for (int offset_idx = threadIdx.x; offset_idx < k_tokens; offset_idx += blockDim.x) {
    int oy = (offset_idx / kernel_w) - (kernel_h / 2);
    int ox = (offset_idx % kernel_w) - (kernel_w / 2);
    float yf = fminf(fmaxf(center_yf + static_cast<float>(oy), 0.0f), static_cast<float>(height - 1));
    float xf = fminf(fmaxf(center_xf + static_cast<float>(ox), 0.0f), static_cast<float>(width - 1));
    int y0, y1, x0, x1;
    float wy0, wy1, wx0, wx1;
    bilinear_indices_and_weights(yf, xf, height, width, y0, y1, x0, x1, wy0, wy1, wx0, wx1);
    y0s[offset_idx] = y0;
    y1s[offset_idx] = y1;
    x0s[offset_idx] = x0;
    x1s[offset_idx] = x1;
    w00s[offset_idx] = wy0 * wx0;
    w01s[offset_idx] = wy0 * wx1;
    w10s[offset_idx] = wy1 * wx0;
    w11s[offset_idx] = wy1 * wx1;
  }
}

template <typename scalar_t>
__device__ inline float bilinear_cached_load(
    const scalar_t* __restrict__ tensor,
    int batch_idx,
    int height,
    int width,
    int heads,
    int head_idx,
    int dim,
    int channel,
    int y0,
    int y1,
    int x0,
    int x1,
    float w00,
    float w01,
    float w10,
    float w11) {
  int base00 = ((((batch_idx * height + y0) * width + x0) * heads + head_idx) * dim) + channel;
  int base01 = ((((batch_idx * height + y0) * width + x1) * heads + head_idx) * dim) + channel;
  int base10 = ((((batch_idx * height + y1) * width + x0) * heads + head_idx) * dim) + channel;
  int base11 = ((((batch_idx * height + y1) * width + x1) * heads + head_idx) * dim) + channel;
  return w00 * load_as_float(tensor + base00) +
      w01 * load_as_float(tensor + base01) +
      w10 * load_as_float(tensor + base10) +
      w11 * load_as_float(tensor + base11);
}

template <typename scalar_t>
__device__ inline float bilinear_cached_dot(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    int batch_idx,
    int height,
    int width,
    int heads,
    int head_idx,
    int dim,
    int y0,
    int y1,
    int x0,
    int x1,
    float w00,
    float w01,
    float w10,
    float w11) {
  float acc = 0.0f;

  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    if ((dim & 1) == 0) {
      const __half2* q2 = reinterpret_cast<const __half2*>(query);
      const __half2* k00 = reinterpret_cast<const __half2*>(
          key + ((((batch_idx * height + y0) * width + x0) * heads + head_idx) * dim));
      const __half2* k01 = reinterpret_cast<const __half2*>(
          key + ((((batch_idx * height + y0) * width + x1) * heads + head_idx) * dim));
      const __half2* k10 = reinterpret_cast<const __half2*>(
          key + ((((batch_idx * height + y1) * width + x0) * heads + head_idx) * dim));
      const __half2* k11 = reinterpret_cast<const __half2*>(
          key + ((((batch_idx * height + y1) * width + x1) * heads + head_idx) * dim));
      for (int d = 0; d < dim / 2; ++d) {
        float2 qv = __half22float2(q2[d]);
        float2 v00 = __half22float2(k00[d]);
        float2 v01 = __half22float2(k01[d]);
        float2 v10 = __half22float2(k10[d]);
        float2 v11 = __half22float2(k11[d]);
        float kx = w00 * v00.x + w01 * v01.x + w10 * v10.x + w11 * v11.x;
        float ky = w00 * v00.y + w01 * v01.y + w10 * v10.y + w11 * v11.y;
        acc += qv.x * kx + qv.y * ky;
      }
      return acc;
    }
  }

  for (int d = 0; d < dim; ++d) {
    acc += load_as_float(query + d) *
        bilinear_cached_load(
               key,
               batch_idx,
               height,
               width,
               heads,
               head_idx,
               dim,
               d,
               y0,
               y1,
               x0,
               x1,
               w00,
               w01,
               w10,
               w11);
  }
  return acc;
}

template <typename scalar_t>
__device__ inline void bilinear_cached_atomic_add(
    scalar_t* __restrict__ grad,
    int batch_idx,
    int height,
    int width,
    int heads,
    int head_idx,
    int dim,
    int channel,
    int y0,
    int y1,
    int x0,
    int x1,
    float w00,
    float w01,
    float w10,
    float w11,
    float grad_value,
    int numel) {
  int idx00 = ((((batch_idx * height + y0) * width + x0) * heads + head_idx) * dim) + channel;
  int idx01 = ((((batch_idx * height + y0) * width + x1) * heads + head_idx) * dim) + channel;
  int idx10 = ((((batch_idx * height + y1) * width + x0) * heads + head_idx) * dim) + channel;
  int idx11 = ((((batch_idx * height + y1) * width + x1) * heads + head_idx) * dim) + channel;
  at::native::fastAtomicAdd(grad, idx00, numel, static_cast<scalar_t>(w00 * grad_value), true);
  at::native::fastAtomicAdd(grad, idx01, numel, static_cast<scalar_t>(w01 * grad_value), true);
  at::native::fastAtomicAdd(grad, idx10, numel, static_cast<scalar_t>(w10 * grad_value), true);
  at::native::fastAtomicAdd(grad, idx11, numel, static_cast<scalar_t>(w11 * grad_value), true);
}

template <typename scalar_t, typename coord_t>
__global__ void sparse_na2d_bilinear_forward_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const coord_t* __restrict__ coords,
    scalar_t* __restrict__ out,
    float* __restrict__ logsumexp,
    int batch,
    int num_queries,
    int height,
    int width,
    int heads,
    int dim,
    int dim_value,
    int kernel_h,
    int kernel_w,
    float scale) {
  extern __shared__ float smem[];
  int k_tokens = kernel_h * kernel_w;
  float* probs = smem;
  float* w00s = probs + k_tokens;
  float* w01s = w00s + k_tokens;
  float* w10s = w01s + k_tokens;
  float* w11s = w10s + k_tokens;
  int* y0s = reinterpret_cast<int*>(w11s + k_tokens);
  int* y1s = y0s + k_tokens;
  int* x0s = y1s + k_tokens;
  int* x1s = x0s + k_tokens;

  int query_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int batch_idx = blockIdx.z;

  const scalar_t* q_ptr =
      query + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim);
  const coord_t* coord_ptr = coords + ((batch_idx * num_queries + query_idx) * 2);
  init_bilinear_neighborhood(coord_ptr, height, width, kernel_h, kernel_w, y0s, y1s, x0s, x1s, w00s, w01s, w10s, w11s);
  __syncthreads();

  for (int offset_idx = threadIdx.x; offset_idx < k_tokens; offset_idx += blockDim.x) {
    float logit = bilinear_cached_dot(
        q_ptr,
        key,
        batch_idx,
        height,
        width,
        heads,
        head_idx,
        dim,
        y0s[offset_idx],
        y1s[offset_idx],
        x0s[offset_idx],
        x1s[offset_idx],
        w00s[offset_idx],
        w01s[offset_idx],
        w10s[offset_idx],
        w11s[offset_idx]);
    probs[offset_idx] = logit * scale;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float max_logit = -INFINITY;
    for (int i = 0; i < k_tokens; ++i) {
      max_logit = fmaxf(max_logit, probs[i]);
    }
    float denom = 0.0f;
    for (int i = 0; i < k_tokens; ++i) {
      float p = expf(probs[i] - max_logit);
      probs[i] = p;
      denom += p;
    }
    float inv_denom = 1.0f / denom;
    for (int i = 0; i < k_tokens; ++i) {
      probs[i] *= inv_denom;
    }
    logsumexp[(batch_idx * num_queries + query_idx) * heads + head_idx] =
        logf(denom) + max_logit;
  }
  __syncthreads();

  scalar_t* out_ptr =
      out + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim_value);
  for (int dv = threadIdx.x; dv < dim_value; dv += blockDim.x) {
    float acc = 0.0f;
    for (int offset_idx = 0; offset_idx < k_tokens; ++offset_idx) {
      acc += probs[offset_idx] *
          bilinear_cached_load(
                 value,
                 batch_idx,
                 height,
                 width,
                 heads,
                 head_idx,
                 dim_value,
                 dv,
                 y0s[offset_idx],
                 y1s[offset_idx],
                 x0s[offset_idx],
                 x1s[offset_idx],
                 w00s[offset_idx],
                 w01s[offset_idx],
                 w10s[offset_idx],
                 w11s[offset_idx]);
    }
    out_ptr[dv] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, typename coord_t>
__global__ void sparse_na2d_bilinear_backward_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const coord_t* __restrict__ coords,
    const scalar_t* __restrict__ out,
    const scalar_t* __restrict__ grad_out,
    const float* __restrict__ logsumexp,
    scalar_t* __restrict__ grad_query,
    scalar_t* __restrict__ grad_key,
    scalar_t* __restrict__ grad_value,
    int batch,
    int num_queries,
    int height,
    int width,
    int heads,
    int dim,
    int dim_value,
    int kernel_h,
    int kernel_w,
    float scale) {
  extern __shared__ float smem[];
  int k_tokens = kernel_h * kernel_w;
  float* probs = smem;
  float* dp = probs + k_tokens;
  float* reductions = dp + k_tokens;
  float* w00s = reductions + blockDim.x;
  float* w01s = w00s + k_tokens;
  float* w10s = w01s + k_tokens;
  float* w11s = w10s + k_tokens;
  int* y0s = reinterpret_cast<int*>(w11s + k_tokens);
  int* y1s = y0s + k_tokens;
  int* x0s = y1s + k_tokens;
  int* x1s = x0s + k_tokens;

  int query_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int batch_idx = blockIdx.z;

  const scalar_t* q_ptr =
      query + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim);
  const scalar_t* o_ptr =
      out + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim_value);
  const scalar_t* go_ptr =
      grad_out + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim_value);
  const coord_t* coord_ptr = coords + ((batch_idx * num_queries + query_idx) * 2);
  float lse = logsumexp[(batch_idx * num_queries + query_idx) * heads + head_idx];

  init_bilinear_neighborhood(coord_ptr, height, width, kernel_h, kernel_w, y0s, y1s, x0s, x1s, w00s, w01s, w10s, w11s);
  __syncthreads();

  for (int offset_idx = threadIdx.x; offset_idx < k_tokens; offset_idx += blockDim.x) {
    float logit = bilinear_cached_dot(
        q_ptr,
        key,
        batch_idx,
        height,
        width,
        heads,
        head_idx,
        dim,
        y0s[offset_idx],
        y1s[offset_idx],
        x0s[offset_idx],
        x1s[offset_idx],
        w00s[offset_idx],
        w01s[offset_idx],
        w10s[offset_idx],
        w11s[offset_idx]);
    probs[offset_idx] = expf(logit * scale - lse);
    dp[offset_idx] = 0.0f;
  }
  __syncthreads();

  float delta_part = 0.0f;
  for (int dv = threadIdx.x; dv < dim_value; dv += blockDim.x) {
    delta_part += load_as_float(go_ptr + dv) * load_as_float(o_ptr + dv);
  }
  reductions[threadIdx.x] = delta_part;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float delta = reductions[0];

  for (int offset_idx = 0; offset_idx < k_tokens; ++offset_idx) {
    float dp_part = 0.0f;
    for (int dv = threadIdx.x; dv < dim_value; dv += blockDim.x) {
      float vval = bilinear_cached_load(
          value,
          batch_idx,
          height,
          width,
          heads,
          head_idx,
          dim_value,
          dv,
          y0s[offset_idx],
          y1s[offset_idx],
          x0s[offset_idx],
          x1s[offset_idx],
          w00s[offset_idx],
          w01s[offset_idx],
          w10s[offset_idx],
          w11s[offset_idx]);
      dp_part += load_as_float(go_ptr + dv) * vval;
      float gv = probs[offset_idx] * load_as_float(go_ptr + dv);
      bilinear_cached_atomic_add(
          grad_value,
          batch_idx,
          height,
          width,
          heads,
          head_idx,
          dim_value,
          dv,
          y0s[offset_idx],
          y1s[offset_idx],
          x0s[offset_idx],
          x1s[offset_idx],
          w00s[offset_idx],
          w01s[offset_idx],
          w10s[offset_idx],
          w11s[offset_idx],
          gv,
          batch * height * width * heads * dim_value);
    }
    reductions[threadIdx.x] = dp_part;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reductions[threadIdx.x] += reductions[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      dp[offset_idx] = probs[offset_idx] * (reductions[0] - delta) * scale;
    }
    __syncthreads();
  }

  scalar_t* gq_ptr =
      grad_query + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim);
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int offset_idx = 0; offset_idx < k_tokens; ++offset_idx) {
      float kval = bilinear_cached_load(
          key,
          batch_idx,
          height,
          width,
          heads,
          head_idx,
          dim,
          d,
          y0s[offset_idx],
          y1s[offset_idx],
          x0s[offset_idx],
          x1s[offset_idx],
          w00s[offset_idx],
          w01s[offset_idx],
          w10s[offset_idx],
          w11s[offset_idx]);
      acc += dp[offset_idx] * kval;
      float gk = dp[offset_idx] * load_as_float(q_ptr + d);
      bilinear_cached_atomic_add(
          grad_key,
          batch_idx,
          height,
          width,
          heads,
          head_idx,
          dim,
          d,
          y0s[offset_idx],
          y1s[offset_idx],
          x0s[offset_idx],
          x1s[offset_idx],
          w00s[offset_idx],
          w01s[offset_idx],
          w10s[offset_idx],
          w11s[offset_idx],
          gk,
          batch * height * width * heads * dim);
    }
    gq_ptr[d] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, typename coord_t>
__global__ void sparse_na2d_bilinear_backward_value_dp_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    const coord_t* __restrict__ coords,
    const scalar_t* __restrict__ out,
    const scalar_t* __restrict__ grad_out,
    const float* __restrict__ logsumexp,
    float* __restrict__ dp_global,
    scalar_t* __restrict__ grad_value,
    int batch,
    int num_queries,
    int height,
    int width,
    int heads,
    int dim,
    int dim_value,
    int kernel_h,
    int kernel_w,
    float scale) {
  extern __shared__ float smem[];
  int k_tokens = kernel_h * kernel_w;
  float* probs = smem;
  float* reductions = probs + k_tokens;
  float* w00s = reductions + blockDim.x;
  float* w01s = w00s + k_tokens;
  float* w10s = w01s + k_tokens;
  float* w11s = w10s + k_tokens;
  int* y0s = reinterpret_cast<int*>(w11s + k_tokens);
  int* y1s = y0s + k_tokens;
  int* x0s = y1s + k_tokens;
  int* x1s = x0s + k_tokens;

  int query_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int batch_idx = blockIdx.z;

  const scalar_t* q_ptr =
      query + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim);
  const scalar_t* o_ptr =
      out + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim_value);
  const scalar_t* go_ptr =
      grad_out + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim_value);
  const coord_t* coord_ptr = coords + ((batch_idx * num_queries + query_idx) * 2);
  float* dp_ptr = dp_global + (((batch_idx * num_queries + query_idx) * heads + head_idx) * k_tokens);
  float lse = logsumexp[(batch_idx * num_queries + query_idx) * heads + head_idx];

  init_bilinear_neighborhood(coord_ptr, height, width, kernel_h, kernel_w, y0s, y1s, x0s, x1s, w00s, w01s, w10s, w11s);
  __syncthreads();

  for (int offset_idx = threadIdx.x; offset_idx < k_tokens; offset_idx += blockDim.x) {
    float logit = bilinear_cached_dot(
        q_ptr,
        key,
        batch_idx,
        height,
        width,
        heads,
        head_idx,
        dim,
        y0s[offset_idx],
        y1s[offset_idx],
        x0s[offset_idx],
        x1s[offset_idx],
        w00s[offset_idx],
        w01s[offset_idx],
        w10s[offset_idx],
        w11s[offset_idx]);
    probs[offset_idx] = expf(logit * scale - lse);
  }
  __syncthreads();

  float delta_part = 0.0f;
  for (int dv = threadIdx.x; dv < dim_value; dv += blockDim.x) {
    delta_part += load_as_float(go_ptr + dv) * load_as_float(o_ptr + dv);
  }
  reductions[threadIdx.x] = delta_part;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float delta = reductions[0];

  for (int offset_idx = 0; offset_idx < k_tokens; ++offset_idx) {
    float dp_part = 0.0f;
    for (int dv = threadIdx.x; dv < dim_value; dv += blockDim.x) {
      float vval = bilinear_cached_load(
          value,
          batch_idx,
          height,
          width,
          heads,
          head_idx,
          dim_value,
          dv,
          y0s[offset_idx],
          y1s[offset_idx],
          x0s[offset_idx],
          x1s[offset_idx],
          w00s[offset_idx],
          w01s[offset_idx],
          w10s[offset_idx],
          w11s[offset_idx]);
      float go = load_as_float(go_ptr + dv);
      dp_part += go * vval;
      bilinear_cached_atomic_add(
          grad_value,
          batch_idx,
          height,
          width,
          heads,
          head_idx,
          dim_value,
          dv,
          y0s[offset_idx],
          y1s[offset_idx],
          x0s[offset_idx],
          x1s[offset_idx],
          w00s[offset_idx],
          w01s[offset_idx],
          w10s[offset_idx],
          w11s[offset_idx],
          probs[offset_idx] * go,
          batch * height * width * heads * dim_value);
    }
    reductions[threadIdx.x] = dp_part;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        reductions[threadIdx.x] += reductions[threadIdx.x + stride];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      dp_ptr[offset_idx] = probs[offset_idx] * (reductions[0] - delta) * scale;
    }
    __syncthreads();
  }
}

template <typename scalar_t, typename coord_t>
__global__ void sparse_na2d_bilinear_backward_query_key_kernel(
    const scalar_t* __restrict__ query,
    const scalar_t* __restrict__ key,
    const coord_t* __restrict__ coords,
    const float* __restrict__ dp_global,
    scalar_t* __restrict__ grad_query,
    scalar_t* __restrict__ grad_key,
    int batch,
    int num_queries,
    int height,
    int width,
    int heads,
    int dim,
    int kernel_h,
    int kernel_w) {
  extern __shared__ float smem[];
  int k_tokens = kernel_h * kernel_w;
  float* w00s = smem;
  float* w01s = w00s + k_tokens;
  float* w10s = w01s + k_tokens;
  float* w11s = w10s + k_tokens;
  int* y0s = reinterpret_cast<int*>(w11s + k_tokens);
  int* y1s = y0s + k_tokens;
  int* x0s = y1s + k_tokens;
  int* x1s = x0s + k_tokens;

  int query_idx = blockIdx.x;
  int head_idx = blockIdx.y;
  int batch_idx = blockIdx.z;

  const scalar_t* q_ptr =
      query + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim);
  scalar_t* gq_ptr =
      grad_query + (((batch_idx * num_queries + query_idx) * heads + head_idx) * dim);
  const coord_t* coord_ptr = coords + ((batch_idx * num_queries + query_idx) * 2);
  const float* dp_ptr = dp_global + (((batch_idx * num_queries + query_idx) * heads + head_idx) * k_tokens);

  init_bilinear_neighborhood(coord_ptr, height, width, kernel_h, kernel_w, y0s, y1s, x0s, x1s, w00s, w01s, w10s, w11s);
  __syncthreads();

  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int offset_idx = 0; offset_idx < k_tokens; ++offset_idx) {
      float dp = dp_ptr[offset_idx];
      float kval = bilinear_cached_load(
          key,
          batch_idx,
          height,
          width,
          heads,
          head_idx,
          dim,
          d,
          y0s[offset_idx],
          y1s[offset_idx],
          x0s[offset_idx],
          x1s[offset_idx],
          w00s[offset_idx],
          w01s[offset_idx],
          w10s[offset_idx],
          w11s[offset_idx]);
      acc += dp * kval;
      bilinear_cached_atomic_add(
          grad_key,
          batch_idx,
          height,
          width,
          heads,
          head_idx,
          dim,
          d,
          y0s[offset_idx],
          y1s[offset_idx],
          x0s[offset_idx],
          x1s[offset_idx],
          w00s[offset_idx],
          w01s[offset_idx],
          w10s[offset_idx],
          w11s[offset_idx],
          dp * load_as_float(q_ptr + d),
          batch * height * width * heads * dim);
    }
    gq_ptr[d] = static_cast<scalar_t>(acc);
  }
}

void check_sparse_na2d_args(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& coords,
    const std::tuple<int32_t, int32_t>& kernel_size) {
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(coords);
  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(value);
  CHECK_CONTIGUOUS(coords);
  TORCH_CHECK(query.dim() == 4, "sparse_na2d query must be [B, N, heads, dim].");
  TORCH_CHECK(key.dim() == 5, "sparse_na2d key must be [B, Hk, Wk, heads, dim].");
  TORCH_CHECK(value.dim() == 5, "sparse_na2d value must be [B, Hk, Wk, heads, dim_value].");
  TORCH_CHECK(coords.dim() == 3, "sparse_na2d coords must be [B, N, 2].");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query and key dtypes must match.");
  TORCH_CHECK(query.scalar_type() == value.scalar_type(), "query and value dtypes must match.");
  TORCH_CHECK(query.size(0) == key.size(0), "query and key batch sizes must match.");
  TORCH_CHECK(query.size(0) == value.size(0), "query and value batch sizes must match.");
  TORCH_CHECK(query.size(0) == coords.size(0), "query and coords batch sizes must match.");
  TORCH_CHECK(query.size(1) == coords.size(1), "query and coords query counts must match.");
  TORCH_CHECK(coords.size(2) == 2, "coords last dimension must be 2.");
  TORCH_CHECK(key.size(1) == value.size(1), "key/value heights must match.");
  TORCH_CHECK(key.size(2) == value.size(2), "key/value widths must match.");
  TORCH_CHECK(query.size(2) == key.size(3), "query/key head counts must match.");
  TORCH_CHECK(query.size(2) == value.size(3), "query/value head counts must match.");
  TORCH_CHECK(query.size(3) == key.size(4), "query/key head dimensions must match.");
  TORCH_CHECK(std::get<0>(kernel_size) > 0 && std::get<1>(kernel_size) > 0, "kernel sizes must be positive.");
  TORCH_CHECK(std::get<0>(kernel_size) % 2 == 1 && std::get<1>(kernel_size) % 2 == 1, "kernel sizes must be odd.");
  TORCH_CHECK(
      query.scalar_type() == torch::kFloat16 ||
          query.scalar_type() == torch::kBFloat16 ||
          query.scalar_type() == torch::kFloat32,
      "sparse_na2d supports FP32, FP16, and BF16 tensors.");
  TORCH_CHECK(
      coords.scalar_type() == torch::kFloat16 ||
          coords.scalar_type() == torch::kBFloat16 ||
          coords.scalar_type() == torch::kFloat32,
      "sparse_na2d coords must be FP32, FP16, or BF16.");
}

} // namespace

void sparse_na2d_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& coords,
    at::Tensor& logsumexp,
    const std::tuple<int32_t, int32_t>& kernel_size,
    float attn_scale,
    bool use_bilinear,
    bool apply_key_rope,
    float rope_theta) {
  check_sparse_na2d_args(query, key, value, coords, kernel_size);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(logsumexp);
  CHECK_CUDA(out);
  CHECK_CUDA(logsumexp);

  at::cuda::OptionalCUDAGuard device_guard(query.device());
  const int batch = query.size(0);
  const int num_queries = query.size(1);
  const int height = key.size(1);
  const int width = key.size(2);
  const int heads = query.size(2);
  const int dim = query.size(3);
  const int dim_value = value.size(4);
  const int kernel_h = std::get<0>(kernel_size);
  const int kernel_w = std::get<1>(kernel_size);
  TORCH_CHECK(
      !apply_key_rope || (heads * dim) % 4 == 0,
      "sparse_na2d key RoPE requires heads * head_dim to be divisible by 4.");

  dim3 grid(num_queries, heads, batch);
  const int rope_freq_count = apply_key_rope ? (heads * dim) / 4 : 0;
  size_t smem_bytes = (kernel_h * kernel_w + rope_freq_count) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream(query.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "sparse_na2d_forward",
      [&] {
        using q_scalar_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            coords.scalar_type(),
            "sparse_na2d_forward_coords",
            [&] {
              using coord_scalar_t = scalar_t;
              sparse_na2d_forward_kernel<q_scalar_t, coord_scalar_t><<<grid, kThreads, smem_bytes, stream>>>(
                  query.data_ptr<q_scalar_t>(),
                  key.data_ptr<q_scalar_t>(),
                  value.data_ptr<q_scalar_t>(),
                  coords.data_ptr<coord_scalar_t>(),
                  out.data_ptr<q_scalar_t>(),
                  logsumexp.data_ptr<float>(),
                  batch,
                  num_queries,
                  height,
                  width,
                  heads,
                  dim,
                  dim_value,
                  kernel_h,
                  kernel_w,
                  attn_scale,
                  use_bilinear,
                  apply_key_rope,
                  rope_theta);
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_na2d_simple_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& coords,
    at::Tensor& logsumexp,
    const std::tuple<int32_t, int32_t>& kernel_size,
    float attn_scale) {
  check_sparse_na2d_args(query, key, value, coords, kernel_size);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(logsumexp);
  CHECK_CUDA(out);
  CHECK_CUDA(logsumexp);

  at::cuda::OptionalCUDAGuard device_guard(query.device());
  const int batch = query.size(0);
  const int num_queries = query.size(1);
  const int height = key.size(1);
  const int width = key.size(2);
  const int heads = query.size(2);
  const int dim = query.size(3);
  const int dim_value = value.size(4);
  const int kernel_h = std::get<0>(kernel_size);
  const int kernel_w = std::get<1>(kernel_size);

  dim3 grid(num_queries, heads, batch);
  size_t smem_bytes = kernel_h * kernel_w * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream(query.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "sparse_na2d_simple_forward",
      [&] {
        using q_scalar_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            coords.scalar_type(),
            "sparse_na2d_simple_forward_coords",
            [&] {
              using coord_scalar_t = scalar_t;
              sparse_na2d_simple_forward_kernel<q_scalar_t, coord_scalar_t>
                  <<<grid, kSimpleThreads, smem_bytes, stream>>>(
                      query.data_ptr<q_scalar_t>(),
                      key.data_ptr<q_scalar_t>(),
                      value.data_ptr<q_scalar_t>(),
                      coords.data_ptr<coord_scalar_t>(),
                      out.data_ptr<q_scalar_t>(),
                      logsumexp.data_ptr<float>(),
                      batch,
                      num_queries,
                      height,
                      width,
                      heads,
                      dim,
                      dim_value,
                      kernel_h,
                      kernel_w,
                      attn_scale);
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_na2d_bilinear_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& coords,
    at::Tensor& logsumexp,
    const std::tuple<int32_t, int32_t>& kernel_size,
    float attn_scale) {
  check_sparse_na2d_args(query, key, value, coords, kernel_size);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(logsumexp);
  CHECK_CUDA(out);
  CHECK_CUDA(logsumexp);

  at::cuda::OptionalCUDAGuard device_guard(query.device());
  const int batch = query.size(0);
  const int num_queries = query.size(1);
  const int height = key.size(1);
  const int width = key.size(2);
  const int heads = query.size(2);
  const int dim = query.size(3);
  const int dim_value = value.size(4);
  const int kernel_h = std::get<0>(kernel_size);
  const int kernel_w = std::get<1>(kernel_size);

  dim3 grid(num_queries, heads, batch);
  size_t smem_bytes = 5 * kernel_h * kernel_w * sizeof(float) +
      4 * kernel_h * kernel_w * sizeof(int);
  auto stream = at::cuda::getCurrentCUDAStream(query.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "sparse_na2d_bilinear_forward",
      [&] {
        using q_scalar_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            coords.scalar_type(),
            "sparse_na2d_bilinear_forward_coords",
            [&] {
              using coord_scalar_t = scalar_t;
              sparse_na2d_bilinear_forward_kernel<q_scalar_t, coord_scalar_t>
                  <<<grid, kSimpleThreads, smem_bytes, stream>>>(
                      query.data_ptr<q_scalar_t>(),
                      key.data_ptr<q_scalar_t>(),
                      value.data_ptr<q_scalar_t>(),
                      coords.data_ptr<coord_scalar_t>(),
                      out.data_ptr<q_scalar_t>(),
                      logsumexp.data_ptr<float>(),
                      batch,
                      num_queries,
                      height,
                      width,
                      heads,
                      dim,
                      dim_value,
                      kernel_h,
                      kernel_w,
                      attn_scale);
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_na2d_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& coords,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    const std::tuple<int32_t, int32_t>& kernel_size,
    float attn_scale,
    bool use_bilinear,
    bool apply_key_rope,
    float rope_theta) {
  check_sparse_na2d_args(query, key, value, coords, kernel_size);
  CHECK_CONTIGUOUS(grad_query);
  CHECK_CONTIGUOUS(grad_key);
  CHECK_CONTIGUOUS(grad_value);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(logsumexp);
  CHECK_CUDA(grad_query);
  CHECK_CUDA(grad_key);
  CHECK_CUDA(grad_value);
  CHECK_CUDA(out);
  CHECK_CUDA(grad_out);
  CHECK_CUDA(logsumexp);

  grad_key.zero_();
  grad_value.zero_();

  at::cuda::OptionalCUDAGuard device_guard(query.device());
  const int batch = query.size(0);
  const int num_queries = query.size(1);
  const int height = key.size(1);
  const int width = key.size(2);
  const int heads = query.size(2);
  const int dim = query.size(3);
  const int dim_value = value.size(4);
  const int kernel_h = std::get<0>(kernel_size);
  const int kernel_w = std::get<1>(kernel_size);
  TORCH_CHECK(
      !apply_key_rope || (heads * dim) % 4 == 0,
      "sparse_na2d key RoPE requires heads * head_dim to be divisible by 4.");

  dim3 grid(num_queries, heads, batch);
  const int rope_freq_count = apply_key_rope ? (heads * dim) / 4 : 0;
  size_t smem_bytes = (2 * kernel_h * kernel_w + kThreads + rope_freq_count) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream(query.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "sparse_na2d_backward",
      [&] {
        using q_scalar_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            coords.scalar_type(),
            "sparse_na2d_backward_coords",
            [&] {
              using coord_scalar_t = scalar_t;
              sparse_na2d_backward_kernel<q_scalar_t, coord_scalar_t><<<grid, kThreads, smem_bytes, stream>>>(
                  query.data_ptr<q_scalar_t>(),
                  key.data_ptr<q_scalar_t>(),
                  value.data_ptr<q_scalar_t>(),
                  coords.data_ptr<coord_scalar_t>(),
                  out.data_ptr<q_scalar_t>(),
                  grad_out.data_ptr<q_scalar_t>(),
                  logsumexp.data_ptr<float>(),
                  grad_query.data_ptr<q_scalar_t>(),
                  grad_key.data_ptr<q_scalar_t>(),
                  grad_value.data_ptr<q_scalar_t>(),
                  batch,
                  num_queries,
                  height,
                  width,
                  heads,
                  dim,
                  dim_value,
                  kernel_h,
                  kernel_w,
                  attn_scale,
                  use_bilinear,
                  apply_key_rope,
                  rope_theta);
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_na2d_simple_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& coords,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    const std::tuple<int32_t, int32_t>& kernel_size,
    float attn_scale) {
  check_sparse_na2d_args(query, key, value, coords, kernel_size);
  CHECK_CONTIGUOUS(grad_query);
  CHECK_CONTIGUOUS(grad_key);
  CHECK_CONTIGUOUS(grad_value);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(logsumexp);
  CHECK_CUDA(grad_query);
  CHECK_CUDA(grad_key);
  CHECK_CUDA(grad_value);
  CHECK_CUDA(out);
  CHECK_CUDA(grad_out);
  CHECK_CUDA(logsumexp);

  grad_key.zero_();
  grad_value.zero_();

  at::cuda::OptionalCUDAGuard device_guard(query.device());
  const int batch = query.size(0);
  const int num_queries = query.size(1);
  const int height = key.size(1);
  const int width = key.size(2);
  const int heads = query.size(2);
  const int dim = query.size(3);
  const int dim_value = value.size(4);
  const int kernel_h = std::get<0>(kernel_size);
  const int kernel_w = std::get<1>(kernel_size);

  dim3 grid(num_queries, heads, batch);
  size_t smem_bytes = (2 * kernel_h * kernel_w + kSimpleThreads) * sizeof(float);
  auto stream = at::cuda::getCurrentCUDAStream(query.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "sparse_na2d_simple_backward",
      [&] {
        using q_scalar_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            coords.scalar_type(),
            "sparse_na2d_simple_backward_coords",
            [&] {
              using coord_scalar_t = scalar_t;
              sparse_na2d_simple_backward_kernel<q_scalar_t, coord_scalar_t>
                  <<<grid, kSimpleThreads, smem_bytes, stream>>>(
                      query.data_ptr<q_scalar_t>(),
                      key.data_ptr<q_scalar_t>(),
                      value.data_ptr<q_scalar_t>(),
                      coords.data_ptr<coord_scalar_t>(),
                      out.data_ptr<q_scalar_t>(),
                      grad_out.data_ptr<q_scalar_t>(),
                      logsumexp.data_ptr<float>(),
                      grad_query.data_ptr<q_scalar_t>(),
                      grad_key.data_ptr<q_scalar_t>(),
                      grad_value.data_ptr<q_scalar_t>(),
                      batch,
                      num_queries,
                      height,
                      width,
                      heads,
                      dim,
                      dim_value,
                      kernel_h,
                      kernel_w,
                      attn_scale);
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_na2d_bilinear_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& coords,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    const std::tuple<int32_t, int32_t>& kernel_size,
    float attn_scale) {
  check_sparse_na2d_args(query, key, value, coords, kernel_size);
  CHECK_CONTIGUOUS(grad_query);
  CHECK_CONTIGUOUS(grad_key);
  CHECK_CONTIGUOUS(grad_value);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(logsumexp);
  CHECK_CUDA(grad_query);
  CHECK_CUDA(grad_key);
  CHECK_CUDA(grad_value);
  CHECK_CUDA(out);
  CHECK_CUDA(grad_out);
  CHECK_CUDA(logsumexp);

  grad_key.zero_();
  grad_value.zero_();

  at::cuda::OptionalCUDAGuard device_guard(query.device());
  const int batch = query.size(0);
  const int num_queries = query.size(1);
  const int height = key.size(1);
  const int width = key.size(2);
  const int heads = query.size(2);
  const int dim = query.size(3);
  const int dim_value = value.size(4);
  const int kernel_h = std::get<0>(kernel_size);
  const int kernel_w = std::get<1>(kernel_size);
  const int k_tokens = kernel_h * kernel_w;

  dim3 grid(num_queries, heads, batch);
  at::Tensor dp = at::empty({batch, num_queries, heads, k_tokens}, query.options().dtype(at::kFloat));
  size_t value_dp_smem_bytes = (5 * k_tokens + kSimpleThreads) * sizeof(float) +
      4 * k_tokens * sizeof(int);
  size_t query_key_smem_bytes = 4 * k_tokens * sizeof(float) +
      4 * k_tokens * sizeof(int);
  auto stream = at::cuda::getCurrentCUDAStream(query.device().index());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      query.scalar_type(),
      "sparse_na2d_bilinear_backward",
      [&] {
        using q_scalar_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            coords.scalar_type(),
            "sparse_na2d_bilinear_backward_coords",
            [&] {
              using coord_scalar_t = scalar_t;
              sparse_na2d_bilinear_backward_value_dp_kernel<q_scalar_t, coord_scalar_t>
                  <<<grid, kSimpleThreads, value_dp_smem_bytes, stream>>>(
                      query.data_ptr<q_scalar_t>(),
                      key.data_ptr<q_scalar_t>(),
                      value.data_ptr<q_scalar_t>(),
                      coords.data_ptr<coord_scalar_t>(),
                      out.data_ptr<q_scalar_t>(),
                      grad_out.data_ptr<q_scalar_t>(),
                      logsumexp.data_ptr<float>(),
                      dp.data_ptr<float>(),
                      grad_value.data_ptr<q_scalar_t>(),
                      batch,
                      num_queries,
                      height,
                      width,
                      heads,
                      dim,
                      dim_value,
                      kernel_h,
                      kernel_w,
                      attn_scale);
              sparse_na2d_bilinear_backward_query_key_kernel<q_scalar_t, coord_scalar_t>
                  <<<grid, kSimpleThreads, query_key_smem_bytes, stream>>>(
                      query.data_ptr<q_scalar_t>(),
                      key.data_ptr<q_scalar_t>(),
                      coords.data_ptr<coord_scalar_t>(),
                      dp.data_ptr<float>(),
                      grad_query.data_ptr<q_scalar_t>(),
                      grad_key.data_ptr<q_scalar_t>(),
                      batch,
                      num_queries,
                      height,
                      width,
                      heads,
                      dim,
                      kernel_h,
                      kernel_w);
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace natten
