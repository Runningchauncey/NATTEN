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
 **************************************************************************************************/

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_fp16.h>

#include <cmath>
#include <type_traits>

#include <natten/fna.h>
#include <natten/helpers.h>

namespace natten {
namespace {

constexpr int kQueryNeighborThreads = 128;

template <typename scalar_t>
__device__ inline float qn_load(const scalar_t* ptr) {
  return static_cast<float>(*ptr);
}

__device__ inline float qn_clamp(float x, float lo, float hi) {
  return fminf(fmaxf(x, lo), hi);
}

__device__ inline int qn_pair(int channel, int channels) {
  int half = channels / 2;
  return channel < half ? channel + half : channel - half;
}

__device__ inline float qn_pair_sign(int channel, int channels) {
  return channel < channels / 2 ? -1.0f : 1.0f;
}

__device__ inline void qn_bilinear_metadata(
    float sample_y,
    float sample_x,
    int height,
    int width,
    int& y0,
    int& y1,
    int& x0,
    int& x1,
    float& w00,
    float& w01,
    float& w10,
    float& w11) {
  float py = qn_clamp((sample_y + 1.0f) * 0.5f * height - 0.5f, 0.0f, height - 1.0f);
  float px = qn_clamp((sample_x + 1.0f) * 0.5f * width - 0.5f, 0.0f, width - 1.0f);
  y0 = static_cast<int>(floorf(py));
  x0 = static_cast<int>(floorf(px));
  y1 = min(y0 + 1, height - 1);
  x1 = min(x0 + 1, width - 1);
  float wy1 = py - y0;
  float wx1 = px - x0;
  float wy0 = 1.0f - wy1;
  float wx0 = 1.0f - wx1;
  w00 = wy0 * wx0;
  w01 = wy0 * wx1;
  w10 = wy1 * wx0;
  w11 = wy1 * wx1;
}

template <typename scalar_t>
__device__ inline float qn_bilinear_load(
    const scalar_t* tensor,
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
  int i00 = ((((batch_idx * height + y0) * width + x0) * heads + head_idx) * dim) + channel;
  int i01 = ((((batch_idx * height + y0) * width + x1) * heads + head_idx) * dim) + channel;
  int i10 = ((((batch_idx * height + y1) * width + x0) * heads + head_idx) * dim) + channel;
  int i11 = ((((batch_idx * height + y1) * width + x1) * heads + head_idx) * dim) + channel;
  return w00 * qn_load(tensor + i00) + w01 * qn_load(tensor + i01) +
      w10 * qn_load(tensor + i10) + w11 * qn_load(tensor + i11);
}

template <typename scalar_t>
__device__ inline void qn_bilinear_atomic_add(
    scalar_t* grad,
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
    float value,
    int numel) {
  int i00 = ((((batch_idx * height + y0) * width + x0) * heads + head_idx) * dim) + channel;
  int i01 = ((((batch_idx * height + y0) * width + x1) * heads + head_idx) * dim) + channel;
  int i10 = ((((batch_idx * height + y1) * width + x0) * heads + head_idx) * dim) + channel;
  int i11 = ((((batch_idx * height + y1) * width + x1) * heads + head_idx) * dim) + channel;
  at::native::fastAtomicAdd(grad, i00, numel, static_cast<scalar_t>(w00 * value), true);
  at::native::fastAtomicAdd(grad, i01, numel, static_cast<scalar_t>(w01 * value), true);
  at::native::fastAtomicAdd(grad, i10, numel, static_cast<scalar_t>(w10 * value), true);
  at::native::fastAtomicAdd(grad, i11, numel, static_cast<scalar_t>(w11 * value), true);
}

__device__ inline float2 qn_bilinear_load_half2(
    const at::Half* tensor,
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
  int i00 = ((((batch_idx * height + y0) * width + x0) * heads + head_idx) * dim) + channel;
  int i01 = ((((batch_idx * height + y0) * width + x1) * heads + head_idx) * dim) + channel;
  int i10 = ((((batch_idx * height + y1) * width + x0) * heads + head_idx) * dim) + channel;
  int i11 = ((((batch_idx * height + y1) * width + x1) * heads + head_idx) * dim) + channel;
  float2 v00 = __half22float2(*reinterpret_cast<const __half2*>(tensor + i00));
  float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(tensor + i01));
  float2 v10 = __half22float2(*reinterpret_cast<const __half2*>(tensor + i10));
  float2 v11 = __half22float2(*reinterpret_cast<const __half2*>(tensor + i11));
  return make_float2(
      w00 * v00.x + w01 * v01.x + w10 * v10.x + w11 * v11.x,
      w00 * v00.y + w01 * v01.y + w10 * v10.y + w11 * v11.y);
}

__device__ inline void qn_bilinear_atomic_add_half2(
    at::Half* grad,
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
    float2 value) {
  int i00 = ((((batch_idx * height + y0) * width + x0) * heads + head_idx) * dim) + channel;
  int i01 = ((((batch_idx * height + y0) * width + x1) * heads + head_idx) * dim) + channel;
  int i10 = ((((batch_idx * height + y1) * width + x0) * heads + head_idx) * dim) + channel;
  int i11 = ((((batch_idx * height + y1) * width + x1) * heads + head_idx) * dim) + channel;
  atomicAdd(reinterpret_cast<__half2*>(grad + i00), __floats2half2_rn(w00 * value.x, w00 * value.y));
  atomicAdd(reinterpret_cast<__half2*>(grad + i01), __floats2half2_rn(w01 * value.x, w01 * value.y));
  atomicAdd(reinterpret_cast<__half2*>(grad + i10), __floats2half2_rn(w10 * value.x, w10 * value.y));
  atomicAdd(reinterpret_cast<__half2*>(grad + i11), __floats2half2_rn(w11 * value.x, w11 * value.y));
}

template <typename scalar_t>
__device__ inline void qn_atomic_add(scalar_t* grad, int index, int numel, float value) {
  at::native::fastAtomicAdd(grad, index, numel, static_cast<scalar_t>(value), true);
}

__device__ inline float qn_block_sum(float value, float* reduction) {
  constexpr unsigned kFullWarpMask = 0xffffffffu;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  if (lane == 0) {
    reduction[warp] = value;
  }
  __syncthreads();

  float total = 0.0f;
  if (warp == 0) {
    total = lane < (blockDim.x + 31) / 32 ? reduction[lane] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      total += __shfl_down_sync(kFullWarpMask, total, offset);
    }
    if (lane == 0) {
      reduction[0] = total;
    }
  }
  __syncthreads();
  return reduction[0];
}

__device__ inline float qn_warp_sum(float value) {
  constexpr unsigned kFullWarpMask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset);
  }
  return value;
}

template <typename scalar_t>
__device__ inline float qn_query_raw(
    const scalar_t* query, int batch_idx, int query_idx, int num_queries, int channels, int channel) {
  return qn_load(query + ((batch_idx * num_queries + query_idx) * channels + channel));
}

template <typename scalar_t>
__device__ inline float qn_key_raw(
    const scalar_t* key,
    int batch_idx,
    int height,
    int width,
    int heads,
    int dim,
    int global_channel,
    int y0,
    int y1,
    int x0,
    int x1,
    float w00,
    float w01,
    float w10,
    float w11) {
  int head_idx = global_channel / dim;
  int channel = global_channel - head_idx * dim;
  return qn_bilinear_load(
      key, batch_idx, height, width, heads, head_idx, dim, channel,
      y0, y1, x0, x1, w00, w01, w10, w11);
}

template <typename scalar_t>
__device__ inline float qn_angle(
    const scalar_t* rope_freqs, int channels, int channel, float pos_y, float pos_x) {
  return pos_y * qn_load(rope_freqs + channel) +
      pos_x * qn_load(rope_freqs + channels + channel);
}

template <typename scalar_t>
__device__ inline float qn_query_rope(
    const scalar_t* query,
    const scalar_t* rope_freqs,
    int batch_idx,
    int query_idx,
    int num_queries,
    int channels,
    int channel,
    float pos_y,
    float pos_x) {
  int pair = qn_pair(channel, channels);
  float angle = qn_angle(rope_freqs, channels, channel, pos_y, pos_x);
  return qn_query_raw(query, batch_idx, query_idx, num_queries, channels, channel) * cosf(angle) +
      qn_pair_sign(channel, channels) *
      qn_query_raw(query, batch_idx, query_idx, num_queries, channels, pair) * sinf(angle);
}

template <typename scalar_t>
__device__ inline float qn_key_rope(
    const scalar_t* key,
    const scalar_t* rope_freqs,
    int batch_idx,
    int height,
    int width,
    int heads,
    int dim,
    int channels,
    int channel,
    int y0,
    int y1,
    int x0,
    int x1,
    float w00,
    float w01,
    float w10,
    float w11,
    float pos_y,
    float pos_x) {
  int pair = qn_pair(channel, channels);
  float angle = qn_angle(rope_freqs, channels, channel, pos_y, pos_x);
  return qn_key_raw(
             key, batch_idx, height, width, heads, dim, channel,
             y0, y1, x0, x1, w00, w01, w10, w11) * cosf(angle) +
      qn_pair_sign(channel, channels) *
      qn_key_raw(
             key, batch_idx, height, width, heads, dim, pair,
             y0, y1, x0, x1, w00, w01, w10, w11) * sinf(angle);
}

template <typename scalar_t>
__device__ inline float qn_query_transformed(
    const scalar_t* query,
    const scalar_t* q_weight,
    const scalar_t* rope_freqs,
    int batch_idx,
    int query_idx,
    int num_queries,
    int channels,
    int channel,
    float pos_y,
    float pos_x,
    float inv_rms,
    bool norm_before_rope) {
  if (!norm_before_rope) {
    return qn_query_rope(
               query, rope_freqs, batch_idx, query_idx, num_queries,
               channels, channel, pos_y, pos_x) * inv_rms * qn_load(q_weight + channel);
  }
  int pair = qn_pair(channel, channels);
  float angle = qn_angle(rope_freqs, channels, channel, pos_y, pos_x);
  float own = qn_query_raw(query, batch_idx, query_idx, num_queries, channels, channel) *
      inv_rms * qn_load(q_weight + channel);
  float paired = qn_query_raw(query, batch_idx, query_idx, num_queries, channels, pair) *
      inv_rms * qn_load(q_weight + pair);
  return own * cosf(angle) + qn_pair_sign(channel, channels) * paired * sinf(angle);
}

template <typename scalar_t>
__device__ inline float qn_key_transformed(
    const scalar_t* key,
    const scalar_t* k_weight,
    const scalar_t* rope_freqs,
    int batch_idx,
    int height,
    int width,
    int heads,
    int dim,
    int channels,
    int channel,
    int y0,
    int y1,
    int x0,
    int x1,
    float w00,
    float w01,
    float w10,
    float w11,
    float pos_y,
    float pos_x,
    float inv_rms,
    bool norm_before_rope) {
  if (!norm_before_rope) {
    return qn_key_rope(
               key, rope_freqs, batch_idx, height, width, heads, dim, channels,
               channel, y0, y1, x0, x1, w00, w01, w10, w11, pos_y, pos_x) *
        inv_rms * qn_load(k_weight + channel);
  }
  int pair = qn_pair(channel, channels);
  float angle = qn_angle(rope_freqs, channels, channel, pos_y, pos_x);
  float own = qn_key_raw(
                  key, batch_idx, height, width, heads, dim, channel,
                  y0, y1, x0, x1, w00, w01, w10, w11) *
      inv_rms * qn_load(k_weight + channel);
  float paired = qn_key_raw(
                     key, batch_idx, height, width, heads, dim, pair,
                     y0, y1, x0, x1, w00, w01, w10, w11) *
      inv_rms * qn_load(k_weight + pair);
  return own * cosf(angle) + qn_pair_sign(channel, channels) * paired * sinf(angle);
}

template <typename scalar_t>
__device__ inline float qn_query_square_partial(
    const scalar_t* query,
    const scalar_t* rope_freqs,
    int batch_idx,
    int query_idx,
    int num_queries,
    int channels,
    float pos_y,
    float pos_x,
    bool norm_before_rope) {
  float sum = 0.0f;
  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    if (norm_before_rope && (channels & 1) == 0) {
      const __half2* ptr = reinterpret_cast<const __half2*>(
          query + (batch_idx * num_queries + query_idx) * channels);
      for (int c2 = threadIdx.x; c2 < channels / 2; c2 += blockDim.x) {
        float2 values = __half22float2(ptr[c2]);
        sum += values.x * values.x + values.y * values.y;
      }
      return sum;
    }
  }
  for (int c = threadIdx.x; c < channels; c += blockDim.x) {
    float x = norm_before_rope
        ? qn_query_raw(query, batch_idx, query_idx, num_queries, channels, c)
        : qn_query_rope(
              query, rope_freqs, batch_idx, query_idx, num_queries,
              channels, c, pos_y, pos_x);
    sum += x * x;
  }
  return sum;
}

template <typename scalar_t>
__device__ inline float qn_key_square_partial(
    const scalar_t* key,
    const scalar_t* rope_freqs,
    int batch_idx,
    int height,
    int width,
    int heads,
    int dim,
    int channels,
    int y0,
    int y1,
    int x0,
    int x1,
    float w00,
    float w01,
    float w10,
    float w11,
    float pos_y,
    float pos_x,
    bool norm_before_rope) {
  float sum = 0.0f;
  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    if (norm_before_rope && (channels & 1) == 0) {
      const __half2* p00 = reinterpret_cast<const __half2*>(
          key + ((batch_idx * height + y0) * width + x0) * channels);
      const __half2* p01 = reinterpret_cast<const __half2*>(
          key + ((batch_idx * height + y0) * width + x1) * channels);
      const __half2* p10 = reinterpret_cast<const __half2*>(
          key + ((batch_idx * height + y1) * width + x0) * channels);
      const __half2* p11 = reinterpret_cast<const __half2*>(
          key + ((batch_idx * height + y1) * width + x1) * channels);
      for (int c2 = threadIdx.x; c2 < channels / 2; c2 += blockDim.x) {
        float2 v00 = __half22float2(p00[c2]);
        float2 v01 = __half22float2(p01[c2]);
        float2 v10 = __half22float2(p10[c2]);
        float2 v11 = __half22float2(p11[c2]);
        float x = w00 * v00.x + w01 * v01.x + w10 * v10.x + w11 * v11.x;
        float y = w00 * v00.y + w01 * v01.y + w10 * v10.y + w11 * v11.y;
        sum += x * x + y * y;
      }
      return sum;
    }
  }
  for (int c = threadIdx.x; c < channels; c += blockDim.x) {
    float x = norm_before_rope
        ? qn_key_raw(
              key, batch_idx, height, width, heads, dim, c,
              y0, y1, x0, x1, w00, w01, w10, w11)
        : qn_key_rope(
              key, rope_freqs, batch_idx, height, width, heads, dim, channels, c,
              y0, y1, x0, x1, w00, w01, w10, w11, pos_y, pos_x);
    sum += x * x;
  }
  return sum;
}

template <typename scalar_t>
__device__ inline float qn_key_square_partial_warp(
    const scalar_t* key,
    const scalar_t* rope_freqs,
    int batch_idx,
    int height,
    int width,
    int heads,
    int dim,
    int channels,
    int y0,
    int y1,
    int x0,
    int x1,
    float w00,
    float w01,
    float w10,
    float w11,
    float pos_y,
    float pos_x,
    bool norm_before_rope) {
  int lane = threadIdx.x & 31;
  float sum = 0.0f;
  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    if (norm_before_rope && (channels & 1) == 0) {
      const __half2* p00 = reinterpret_cast<const __half2*>(
          key + ((batch_idx * height + y0) * width + x0) * channels);
      const __half2* p01 = reinterpret_cast<const __half2*>(
          key + ((batch_idx * height + y0) * width + x1) * channels);
      const __half2* p10 = reinterpret_cast<const __half2*>(
          key + ((batch_idx * height + y1) * width + x0) * channels);
      const __half2* p11 = reinterpret_cast<const __half2*>(
          key + ((batch_idx * height + y1) * width + x1) * channels);
      for (int c2 = lane; c2 < channels / 2; c2 += 32) {
        float2 v00 = __half22float2(p00[c2]);
        float2 v01 = __half22float2(p01[c2]);
        float2 v10 = __half22float2(p10[c2]);
        float2 v11 = __half22float2(p11[c2]);
        float x = w00 * v00.x + w01 * v01.x + w10 * v10.x + w11 * v11.x;
        float y = w00 * v00.y + w01 * v01.y + w10 * v10.y + w11 * v11.y;
        sum += x * x + y * y;
      }
      return sum;
    }
  }
  for (int c = lane; c < channels; c += 32) {
    float x = norm_before_rope
        ? qn_key_raw(
              key, batch_idx, height, width, heads, dim, c,
              y0, y1, x0, x1, w00, w01, w10, w11)
        : qn_key_rope(
              key, rope_freqs, batch_idx, height, width, heads, dim, channels, c,
              y0, y1, x0, x1, w00, w01, w10, w11, pos_y, pos_x);
    sum += x * x;
  }
  return sum;
}

template <typename scalar_t, typename coord_t>
__global__ void qn_forward_kernel(
    const scalar_t* query,
    const scalar_t* key,
    const scalar_t* value,
    const coord_t* coords,
    const scalar_t* q_weight,
    const scalar_t* k_weight,
    const scalar_t* rope_freqs,
    scalar_t* out,
    float* logsumexp,
    int batch,
    int num_queries,
    int height,
    int width,
    int heads,
    int dim,
    int dim_value,
    int kernel_h,
    int kernel_w,
    float offset_scale_y,
    float offset_scale_x,
    float attn_scale,
    float norm_eps,
    bool norm_before_rope) {
  extern __shared__ unsigned char shared_raw[];
  int channels = heads * dim;
  int tokens = kernel_h * kernel_w;
  float* q_trans = reinterpret_cast<float*>(shared_raw);
  float* k_inv = q_trans + channels;
  float* probs = k_inv + tokens;
  float* w00s = probs + heads * tokens;
  float* w01s = w00s + tokens;
  float* w10s = w01s + tokens;
  float* w11s = w10s + tokens;
  float* pos_ys = w11s + tokens;
  float* pos_xs = pos_ys + tokens;
  float* reduction = pos_xs + tokens;
  int* y0s = reinterpret_cast<int*>(reduction + blockDim.x);
  int* y1s = y0s + tokens;
  int* x0s = y1s + tokens;
  int* x1s = x0s + tokens;

  int query_idx = blockIdx.x;
  int batch_idx = blockIdx.y;
  const coord_t* coord = coords + ((batch_idx * num_queries + query_idx) * 2);
  float coord_y = static_cast<float>(coord[0]);
  float coord_x = static_cast<float>(coord[1]);
  float q_pos_y = (coord_y + 1.0f) * 0.5f;
  float q_pos_x = (coord_x + 1.0f) * 0.5f;

  for (int token = threadIdx.x; token < tokens; token += blockDim.x) {
    int oy = token / kernel_w - kernel_h / 2;
    int ox = token % kernel_w - kernel_w / 2;
    float sy = qn_clamp(coord_y + oy * offset_scale_y, -1.0f, 1.0f);
    float sx = qn_clamp(coord_x + ox * offset_scale_x, -1.0f, 1.0f);
    pos_ys[token] = (sy + 1.0f) * 0.5f;
    pos_xs[token] = (sx + 1.0f) * 0.5f;
    qn_bilinear_metadata(
        sy, sx, height, width, y0s[token], y1s[token], x0s[token], x1s[token],
        w00s[token], w01s[token], w10s[token], w11s[token]);
  }
  __syncthreads();

  float q_sum = qn_query_square_partial(
      query, rope_freqs, batch_idx, query_idx, num_queries, channels,
      q_pos_y, q_pos_x, norm_before_rope);
  float q_inv = rsqrtf(qn_block_sum(q_sum, reduction) / channels + norm_eps);
  for (int c = threadIdx.x; c < channels; c += blockDim.x) {
    q_trans[c] = qn_query_transformed(
        query, q_weight, rope_freqs, batch_idx, query_idx, num_queries,
        channels, c, q_pos_y, q_pos_x, q_inv, norm_before_rope);
  }
  __syncthreads();

  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int warps = blockDim.x >> 5;
  for (int token = warp; token < tokens; token += warps) {
    float sum = qn_key_square_partial_warp(
        key, rope_freqs, batch_idx, height, width, heads, dim, channels,
        y0s[token], y1s[token], x0s[token], x1s[token],
        w00s[token], w01s[token], w10s[token], w11s[token],
        pos_ys[token], pos_xs[token], norm_before_rope);
    sum = qn_warp_sum(sum);
    if (lane == 0) {
      k_inv[token] = rsqrtf(sum / channels + norm_eps);
    }
  }
  __syncthreads();

  for (int index = threadIdx.x; index < heads * tokens; index += blockDim.x) {
    int head = index / tokens;
    int token = index - head * tokens;
    float logit = 0.0f;
    for (int d = 0; d < dim; ++d) {
      int c = head * dim + d;
      float kval = qn_key_transformed(
          key, k_weight, rope_freqs, batch_idx, height, width, heads, dim, channels, c,
          y0s[token], y1s[token], x0s[token], x1s[token],
          w00s[token], w01s[token], w10s[token], w11s[token],
          pos_ys[token], pos_xs[token], k_inv[token], norm_before_rope);
      logit += q_trans[c] * kval;
    }
    probs[index] = logit * attn_scale;
  }
  __syncthreads();

  for (int head = threadIdx.x; head < heads; head += blockDim.x) {
    float max_logit = -INFINITY;
    for (int token = 0; token < tokens; ++token) {
      max_logit = fmaxf(max_logit, probs[head * tokens + token]);
    }
    float denom = 0.0f;
    for (int token = 0; token < tokens; ++token) {
      float p = expf(probs[head * tokens + token] - max_logit);
      probs[head * tokens + token] = p;
      denom += p;
    }
    float inv_denom = 1.0f / denom;
    for (int token = 0; token < tokens; ++token) {
      probs[head * tokens + token] *= inv_denom;
    }
    logsumexp[(batch_idx * num_queries + query_idx) * heads + head] = logf(denom) + max_logit;
  }
  __syncthreads();

  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    if ((dim_value & 1) == 0) {
      __half2* out2 = reinterpret_cast<__half2*>(
          out + (batch_idx * num_queries + query_idx) * heads * dim_value);
      for (int c2 = threadIdx.x; c2 < heads * (dim_value / 2); c2 += blockDim.x) {
        int head = c2 / (dim_value / 2);
        int dv2 = c2 - head * (dim_value / 2);
        float2 acc = make_float2(0.0f, 0.0f);
        for (int token = 0; token < tokens; ++token) {
          float2 v = qn_bilinear_load_half2(
              value, batch_idx, height, width, heads, head, dim_value, dv2 * 2,
              y0s[token], y1s[token], x0s[token], x1s[token],
              w00s[token], w01s[token], w10s[token], w11s[token]);
          float p = probs[head * tokens + token];
          acc.x += p * v.x;
          acc.y += p * v.y;
        }
        out2[c2] = __floats2half2_rn(acc.x, acc.y);
      }
      return;
    }
  }

  for (int c = threadIdx.x; c < heads * dim_value; c += blockDim.x) {
    int head = c / dim_value;
    int dv = c - head * dim_value;
    float acc = 0.0f;
    for (int token = 0; token < tokens; ++token) {
      acc += probs[head * tokens + token] * qn_bilinear_load(
          value, batch_idx, height, width, heads, head, dim_value, dv,
          y0s[token], y1s[token], x0s[token], x1s[token],
          w00s[token], w01s[token], w10s[token], w11s[token]);
    }
    out[(batch_idx * num_queries + query_idx) * heads * dim_value + c] = static_cast<scalar_t>(acc);
  }
}

template <typename scalar_t, typename coord_t>
__global__ void qn_backward_value_kernel(
    const scalar_t* query,
    const scalar_t* key,
    const scalar_t* value,
    const coord_t* coords,
    const scalar_t* q_weight,
    const scalar_t* k_weight,
    const scalar_t* rope_freqs,
    const scalar_t* out,
    const scalar_t* grad_out,
    const float* logsumexp,
    float* d_logits,
    scalar_t* grad_value,
    int batch,
    int num_queries,
    int height,
    int width,
    int heads,
    int dim,
    int dim_value,
    int kernel_h,
    int kernel_w,
    float offset_scale_y,
    float offset_scale_x,
    float attn_scale,
    float norm_eps,
    bool norm_before_rope) {
  extern __shared__ unsigned char shared_raw[];
  int channels = heads * dim;
  int tokens = kernel_h * kernel_w;
  float* q_trans = reinterpret_cast<float*>(shared_raw);
  float* k_inv = q_trans + channels;
  float* probs = k_inv + tokens;
  float* w00s = probs + heads * tokens;
  float* w01s = w00s + tokens;
  float* w10s = w01s + tokens;
  float* w11s = w10s + tokens;
  float* pos_ys = w11s + tokens;
  float* pos_xs = pos_ys + tokens;
  float* reduction = pos_xs + tokens;
  int* y0s = reinterpret_cast<int*>(reduction + blockDim.x);
  int* y1s = y0s + tokens;
  int* x0s = y1s + tokens;
  int* x1s = x0s + tokens;

  int query_idx = blockIdx.x;
  int batch_idx = blockIdx.y;
  const coord_t* coord = coords + ((batch_idx * num_queries + query_idx) * 2);
  float coord_y = static_cast<float>(coord[0]);
  float coord_x = static_cast<float>(coord[1]);
  float q_pos_y = (coord_y + 1.0f) * 0.5f;
  float q_pos_x = (coord_x + 1.0f) * 0.5f;

  for (int token = threadIdx.x; token < tokens; token += blockDim.x) {
    int oy = token / kernel_w - kernel_h / 2;
    int ox = token % kernel_w - kernel_w / 2;
    float sy = qn_clamp(coord_y + oy * offset_scale_y, -1.0f, 1.0f);
    float sx = qn_clamp(coord_x + ox * offset_scale_x, -1.0f, 1.0f);
    pos_ys[token] = (sy + 1.0f) * 0.5f;
    pos_xs[token] = (sx + 1.0f) * 0.5f;
    qn_bilinear_metadata(
        sy, sx, height, width, y0s[token], y1s[token], x0s[token], x1s[token],
        w00s[token], w01s[token], w10s[token], w11s[token]);
  }
  __syncthreads();

  float q_sum = qn_query_square_partial(
      query, rope_freqs, batch_idx, query_idx, num_queries, channels,
      q_pos_y, q_pos_x, norm_before_rope);
  float q_inv = rsqrtf(qn_block_sum(q_sum, reduction) / channels + norm_eps);
  for (int c = threadIdx.x; c < channels; c += blockDim.x) {
    q_trans[c] = qn_query_transformed(
        query, q_weight, rope_freqs, batch_idx, query_idx, num_queries,
        channels, c, q_pos_y, q_pos_x, q_inv, norm_before_rope);
  }
  __syncthreads();

  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int warps = blockDim.x >> 5;
  for (int token = warp; token < tokens; token += warps) {
    float sum = qn_key_square_partial_warp(
        key, rope_freqs, batch_idx, height, width, heads, dim, channels,
        y0s[token], y1s[token], x0s[token], x1s[token],
        w00s[token], w01s[token], w10s[token], w11s[token],
        pos_ys[token], pos_xs[token], norm_before_rope);
    sum = qn_warp_sum(sum);
    if (lane == 0) {
      k_inv[token] = rsqrtf(sum / channels + norm_eps);
    }
  }
  __syncthreads();

  for (int index = threadIdx.x; index < heads * tokens; index += blockDim.x) {
    int head = index / tokens;
    int token = index - head * tokens;
    float logit = 0.0f;
    for (int d = 0; d < dim; ++d) {
      int c = head * dim + d;
      logit += q_trans[c] * qn_key_transformed(
          key, k_weight, rope_freqs, batch_idx, height, width, heads, dim, channels, c,
          y0s[token], y1s[token], x0s[token], x1s[token],
          w00s[token], w01s[token], w10s[token], w11s[token],
          pos_ys[token], pos_xs[token], k_inv[token], norm_before_rope);
    }
    float lse = logsumexp[(batch_idx * num_queries + query_idx) * heads + head];
    probs[index] = expf(logit * attn_scale - lse);
  }
  __syncthreads();

  for (int head = threadIdx.x; head < heads; head += blockDim.x) {
    float delta = 0.0f;
    const scalar_t* go = grad_out + (((batch_idx * num_queries + query_idx) * heads + head) * dim_value);
    const scalar_t* o = out + (((batch_idx * num_queries + query_idx) * heads + head) * dim_value);
    for (int dv = 0; dv < dim_value; ++dv) {
      delta += qn_load(go + dv) * qn_load(o + dv);
    }
    reduction[head] = delta;
  }
  __syncthreads();

  for (int index = threadIdx.x; index < heads * tokens; index += blockDim.x) {
    int head = index / tokens;
    int token = index - head * tokens;
    float dprob = 0.0f;
    const scalar_t* go = grad_out + (((batch_idx * num_queries + query_idx) * heads + head) * dim_value);
    if constexpr (std::is_same_v<scalar_t, at::Half>) {
      if ((dim_value & 1) == 0) {
        const __half2* go2 = reinterpret_cast<const __half2*>(go);
        for (int dv2 = 0; dv2 < dim_value / 2; ++dv2) {
          float2 g = __half22float2(go2[dv2]);
          float2 v = qn_bilinear_load_half2(
              value, batch_idx, height, width, heads, head, dim_value, dv2 * 2,
              y0s[token], y1s[token], x0s[token], x1s[token],
              w00s[token], w01s[token], w10s[token], w11s[token]);
          dprob += g.x * v.x + g.y * v.y;
        }
      } else {
        for (int dv = 0; dv < dim_value; ++dv) {
          float g = qn_load(go + dv);
          dprob += g * qn_bilinear_load(
              value, batch_idx, height, width, heads, head, dim_value, dv,
              y0s[token], y1s[token], x0s[token], x1s[token],
              w00s[token], w01s[token], w10s[token], w11s[token]);
        }
      }
    } else {
      for (int dv = 0; dv < dim_value; ++dv) {
        float g = qn_load(go + dv);
        dprob += g * qn_bilinear_load(
            value, batch_idx, height, width, heads, head, dim_value, dv,
            y0s[token], y1s[token], x0s[token], x1s[token],
            w00s[token], w01s[token], w10s[token], w11s[token]);
      }
    }
    d_logits[((batch_idx * num_queries + query_idx) * heads + head) * tokens + token] =
        probs[index] * (dprob - reduction[head]) * attn_scale;
  }
  __syncthreads();

  if constexpr (std::is_same_v<scalar_t, at::Half>) {
    if ((dim_value & 1) == 0) {
      for (int c2 = threadIdx.x; c2 < heads * (dim_value / 2); c2 += blockDim.x) {
        int head = c2 / (dim_value / 2);
        int dv2 = c2 - head * (dim_value / 2);
        const __half2* go = reinterpret_cast<const __half2*>(
            grad_out + (((batch_idx * num_queries + query_idx) * heads + head) * dim_value));
        float2 grad_pair = __half22float2(go[dv2]);
        for (int token = 0; token < tokens; ++token) {
          float p = probs[head * tokens + token];
          qn_bilinear_atomic_add_half2(
              grad_value, batch_idx, height, width, heads, head, dim_value, dv2 * 2,
              y0s[token], y1s[token], x0s[token], x1s[token],
              w00s[token], w01s[token], w10s[token], w11s[token],
              make_float2(p * grad_pair.x, p * grad_pair.y));
        }
      }
      return;
    }
  }

  int value_numel = batch * height * width * heads * dim_value;
  for (int c = threadIdx.x; c < heads * dim_value; c += blockDim.x) {
    int head = c / dim_value;
    int dv = c - head * dim_value;
    float go = qn_load(
        grad_out + (((batch_idx * num_queries + query_idx) * heads + head) * dim_value + dv));
    for (int token = 0; token < tokens; ++token) {
      qn_bilinear_atomic_add(
          grad_value, batch_idx, height, width, heads, head, dim_value, dv,
          y0s[token], y1s[token], x0s[token], x1s[token],
          w00s[token], w01s[token], w10s[token], w11s[token],
          probs[head * tokens + token] * go, value_numel);
    }
  }
}

template <typename scalar_t, typename coord_t>
__global__ void qn_backward_query_key_kernel(
    const scalar_t* query,
    const scalar_t* key,
    const coord_t* coords,
    const scalar_t* q_weight,
    const scalar_t* k_weight,
    const scalar_t* rope_freqs,
    const float* d_logits,
    scalar_t* grad_query,
    scalar_t* grad_key,
    scalar_t* grad_q_weight,
    scalar_t* grad_k_weight,
    scalar_t* grad_rope_freqs,
    int batch,
    int num_queries,
    int height,
    int width,
    int heads,
    int dim,
    int kernel_h,
    int kernel_w,
    float offset_scale_y,
    float offset_scale_x,
    float norm_eps,
    bool norm_before_rope) {
  extern __shared__ unsigned char shared_raw[];
  int channels = heads * dim;
  int tokens = kernel_h * kernel_w;
  float* q_trans = reinterpret_cast<float*>(shared_raw);
  float* k_inv = q_trans + channels;
  float* grad_stage = k_inv + tokens;
  float* grad_mid = grad_stage + channels;
  float* k_weight_acc = grad_mid + channels;
  float* rope_y_acc = k_weight_acc + channels;
  float* rope_x_acc = rope_y_acc + channels;
  float* w00s = rope_x_acc + channels;
  float* w01s = w00s + tokens;
  float* w10s = w01s + tokens;
  float* w11s = w10s + tokens;
  float* pos_ys = w11s + tokens;
  float* pos_xs = pos_ys + tokens;
  float* reduction = pos_xs + tokens;
  int* y0s = reinterpret_cast<int*>(reduction + blockDim.x);
  int* y1s = y0s + tokens;
  int* x0s = y1s + tokens;
  int* x1s = x0s + tokens;

  int query_idx = blockIdx.x;
  int batch_idx = blockIdx.y;
  const coord_t* coord = coords + ((batch_idx * num_queries + query_idx) * 2);
  float coord_y = static_cast<float>(coord[0]);
  float coord_x = static_cast<float>(coord[1]);
  float q_pos_y = (coord_y + 1.0f) * 0.5f;
  float q_pos_x = (coord_x + 1.0f) * 0.5f;

  for (int c = threadIdx.x; c < channels; c += blockDim.x) {
    k_weight_acc[c] = 0.0f;
    rope_y_acc[c] = 0.0f;
    rope_x_acc[c] = 0.0f;
  }
  for (int token = threadIdx.x; token < tokens; token += blockDim.x) {
    int oy = token / kernel_w - kernel_h / 2;
    int ox = token % kernel_w - kernel_w / 2;
    float sy = qn_clamp(coord_y + oy * offset_scale_y, -1.0f, 1.0f);
    float sx = qn_clamp(coord_x + ox * offset_scale_x, -1.0f, 1.0f);
    pos_ys[token] = (sy + 1.0f) * 0.5f;
    pos_xs[token] = (sx + 1.0f) * 0.5f;
    qn_bilinear_metadata(
        sy, sx, height, width, y0s[token], y1s[token], x0s[token], x1s[token],
        w00s[token], w01s[token], w10s[token], w11s[token]);
  }
  __syncthreads();

  float q_sum = qn_query_square_partial(
      query, rope_freqs, batch_idx, query_idx, num_queries, channels,
      q_pos_y, q_pos_x, norm_before_rope);
  float q_inv = rsqrtf(qn_block_sum(q_sum, reduction) / channels + norm_eps);
  for (int c = threadIdx.x; c < channels; c += blockDim.x) {
    q_trans[c] = qn_query_transformed(
        query, q_weight, rope_freqs, batch_idx, query_idx, num_queries,
        channels, c, q_pos_y, q_pos_x, q_inv, norm_before_rope);
  }
  __syncthreads();

  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int warps = blockDim.x >> 5;
  for (int token = warp; token < tokens; token += warps) {
    float sum = qn_key_square_partial_warp(
        key, rope_freqs, batch_idx, height, width, heads, dim, channels,
        y0s[token], y1s[token], x0s[token], x1s[token],
        w00s[token], w01s[token], w10s[token], w11s[token],
        pos_ys[token], pos_xs[token], norm_before_rope);
    sum = qn_warp_sum(sum);
    if (lane == 0) {
      k_inv[token] = rsqrtf(sum / channels + norm_eps);
    }
  }
  __syncthreads();

  for (int c = threadIdx.x; c < channels; c += blockDim.x) {
    int head = c / dim;
    float acc = 0.0f;
    for (int token = 0; token < tokens; ++token) {
      float dl = d_logits[((batch_idx * num_queries + query_idx) * heads + head) * tokens + token];
      acc += dl * qn_key_transformed(
          key, k_weight, rope_freqs, batch_idx, height, width, heads, dim, channels, c,
          y0s[token], y1s[token], x0s[token], x1s[token],
          w00s[token], w01s[token], w10s[token], w11s[token],
          pos_ys[token], pos_xs[token], k_inv[token], norm_before_rope);
    }
    grad_stage[c] = acc;
  }
  __syncthreads();

  float q_dot = 0.0f;
  if (norm_before_rope) {
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
      int pair = qn_pair(c, channels);
      float angle = qn_angle(rope_freqs, channels, c, q_pos_y, q_pos_x);
      float pair_angle = qn_angle(rope_freqs, channels, pair, q_pos_y, q_pos_x);
      float grad_norm = grad_stage[c] * cosf(angle) +
          grad_stage[pair] * qn_pair_sign(pair, channels) * sinf(pair_angle);
      grad_mid[c] = grad_norm;
      float x = qn_query_raw(query, batch_idx, query_idx, num_queries, channels, c);
      float u = x * q_inv;
      q_dot += grad_norm * qn_load(q_weight + c) * u;
      float n = u * qn_load(q_weight + c);
      float n_pair = qn_query_raw(query, batch_idx, query_idx, num_queries, channels, pair) *
          q_inv * qn_load(q_weight + pair);
      float dangle = grad_stage[c] *
          (-n * sinf(angle) + qn_pair_sign(c, channels) * n_pair * cosf(angle));
      qn_atomic_add(grad_q_weight, c, channels, grad_norm * u);
      rope_y_acc[c] += dangle * q_pos_y;
      rope_x_acc[c] += dangle * q_pos_x;
    }
    float correction = qn_block_sum(q_dot, reduction) / channels;
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
      float x = qn_query_raw(query, batch_idx, query_idx, num_queries, channels, c);
      float u = x * q_inv;
      float gx = q_inv * (grad_mid[c] * qn_load(q_weight + c) - u * correction);
      grad_query[(batch_idx * num_queries + query_idx) * channels + c] = static_cast<scalar_t>(gx);
    }
  } else {
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
      float r = qn_query_rope(
          query, rope_freqs, batch_idx, query_idx, num_queries,
          channels, c, q_pos_y, q_pos_x);
      float u = r * q_inv;
      q_dot += grad_stage[c] * qn_load(q_weight + c) * u;
      qn_atomic_add(grad_q_weight, c, channels, grad_stage[c] * u);
    }
    float correction = qn_block_sum(q_dot, reduction) / channels;
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
      float r = qn_query_rope(
          query, rope_freqs, batch_idx, query_idx, num_queries,
          channels, c, q_pos_y, q_pos_x);
      float u = r * q_inv;
      grad_mid[c] = q_inv * (grad_stage[c] * qn_load(q_weight + c) - u * correction);
    }
    __syncthreads();
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
      int pair = qn_pair(c, channels);
      float angle = qn_angle(rope_freqs, channels, c, q_pos_y, q_pos_x);
      float pair_angle = qn_angle(rope_freqs, channels, pair, q_pos_y, q_pos_x);
      float gx = grad_mid[c] * cosf(angle) +
          grad_mid[pair] * qn_pair_sign(pair, channels) * sinf(pair_angle);
      float x = qn_query_raw(query, batch_idx, query_idx, num_queries, channels, c);
      float x_pair = qn_query_raw(query, batch_idx, query_idx, num_queries, channels, pair);
      float dangle = grad_mid[c] *
          (-x * sinf(angle) + qn_pair_sign(c, channels) * x_pair * cosf(angle));
      grad_query[(batch_idx * num_queries + query_idx) * channels + c] = static_cast<scalar_t>(gx);
      rope_y_acc[c] += dangle * q_pos_y;
      rope_x_acc[c] += dangle * q_pos_x;
    }
  }
  __syncthreads();

  int key_numel = batch * height * width * heads * dim;
  for (int token = 0; token < tokens; ++token) {
    for (int c = threadIdx.x; c < channels; c += blockDim.x) {
      int head = c / dim;
      float dl = d_logits[((batch_idx * num_queries + query_idx) * heads + head) * tokens + token];
      grad_stage[c] = dl * q_trans[c];
    }
    __syncthreads();

    float k_dot = 0.0f;
    if (norm_before_rope) {
      for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        int pair = qn_pair(c, channels);
        float angle = qn_angle(rope_freqs, channels, c, pos_ys[token], pos_xs[token]);
        float pair_angle = qn_angle(rope_freqs, channels, pair, pos_ys[token], pos_xs[token]);
        float grad_norm = grad_stage[c] * cosf(angle) +
            grad_stage[pair] * qn_pair_sign(pair, channels) * sinf(pair_angle);
        grad_mid[c] = grad_norm;
        float x = qn_key_raw(
            key, batch_idx, height, width, heads, dim, c,
            y0s[token], y1s[token], x0s[token], x1s[token],
            w00s[token], w01s[token], w10s[token], w11s[token]);
        float u = x * k_inv[token];
        k_dot += grad_norm * qn_load(k_weight + c) * u;
        float n = u * qn_load(k_weight + c);
        float n_pair = qn_key_raw(
            key, batch_idx, height, width, heads, dim, pair,
            y0s[token], y1s[token], x0s[token], x1s[token],
            w00s[token], w01s[token], w10s[token], w11s[token]) *
            k_inv[token] * qn_load(k_weight + pair);
        float dangle = grad_stage[c] *
            (-n * sinf(angle) + qn_pair_sign(c, channels) * n_pair * cosf(angle));
        k_weight_acc[c] += grad_norm * u;
        rope_y_acc[c] += dangle * pos_ys[token];
        rope_x_acc[c] += dangle * pos_xs[token];
      }
      float correction = qn_block_sum(k_dot, reduction) / channels;
      for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        int head = c / dim;
        int d = c - head * dim;
        float x = qn_key_raw(
            key, batch_idx, height, width, heads, dim, c,
            y0s[token], y1s[token], x0s[token], x1s[token],
            w00s[token], w01s[token], w10s[token], w11s[token]);
        float u = x * k_inv[token];
        float gx = k_inv[token] * (grad_mid[c] * qn_load(k_weight + c) - u * correction);
        qn_bilinear_atomic_add(
            grad_key, batch_idx, height, width, heads, head, dim, d,
            y0s[token], y1s[token], x0s[token], x1s[token],
            w00s[token], w01s[token], w10s[token], w11s[token], gx, key_numel);
      }
    } else {
      for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        float r = qn_key_rope(
            key, rope_freqs, batch_idx, height, width, heads, dim, channels, c,
            y0s[token], y1s[token], x0s[token], x1s[token],
            w00s[token], w01s[token], w10s[token], w11s[token],
            pos_ys[token], pos_xs[token]);
        float u = r * k_inv[token];
        k_dot += grad_stage[c] * qn_load(k_weight + c) * u;
        k_weight_acc[c] += grad_stage[c] * u;
      }
      float correction = qn_block_sum(k_dot, reduction) / channels;
      for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        float r = qn_key_rope(
            key, rope_freqs, batch_idx, height, width, heads, dim, channels, c,
            y0s[token], y1s[token], x0s[token], x1s[token],
            w00s[token], w01s[token], w10s[token], w11s[token],
            pos_ys[token], pos_xs[token]);
        float u = r * k_inv[token];
        grad_mid[c] = k_inv[token] * (grad_stage[c] * qn_load(k_weight + c) - u * correction);
      }
      __syncthreads();
      for (int c = threadIdx.x; c < channels; c += blockDim.x) {
        int pair = qn_pair(c, channels);
        int head = c / dim;
        int d = c - head * dim;
        float angle = qn_angle(rope_freqs, channels, c, pos_ys[token], pos_xs[token]);
        float pair_angle = qn_angle(rope_freqs, channels, pair, pos_ys[token], pos_xs[token]);
        float gx = grad_mid[c] * cosf(angle) +
            grad_mid[pair] * qn_pair_sign(pair, channels) * sinf(pair_angle);
        float x = qn_key_raw(
            key, batch_idx, height, width, heads, dim, c,
            y0s[token], y1s[token], x0s[token], x1s[token],
            w00s[token], w01s[token], w10s[token], w11s[token]);
        float x_pair = qn_key_raw(
            key, batch_idx, height, width, heads, dim, pair,
            y0s[token], y1s[token], x0s[token], x1s[token],
            w00s[token], w01s[token], w10s[token], w11s[token]);
        float dangle = grad_mid[c] *
            (-x * sinf(angle) + qn_pair_sign(c, channels) * x_pair * cosf(angle));
        qn_bilinear_atomic_add(
            grad_key, batch_idx, height, width, heads, head, dim, d,
            y0s[token], y1s[token], x0s[token], x1s[token],
            w00s[token], w01s[token], w10s[token], w11s[token], gx, key_numel);
        rope_y_acc[c] += dangle * pos_ys[token];
        rope_x_acc[c] += dangle * pos_xs[token];
      }
    }
    __syncthreads();
  }
  for (int c = threadIdx.x; c < channels; c += blockDim.x) {
    qn_atomic_add(grad_k_weight, c, channels, k_weight_acc[c]);
    qn_atomic_add(grad_rope_freqs, c, 2 * channels, rope_y_acc[c]);
    qn_atomic_add(grad_rope_freqs, channels + c, 2 * channels, rope_x_acc[c]);
  }
}

void check_qn_args(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& coords,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& rope_freqs,
    const std::tuple<int32_t, int32_t>& kernel_size) {
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(coords);
  CHECK_CUDA(q_weight);
  CHECK_CUDA(k_weight);
  CHECK_CUDA(rope_freqs);
  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(value);
  CHECK_CONTIGUOUS(coords);
  CHECK_CONTIGUOUS(q_weight);
  CHECK_CONTIGUOUS(k_weight);
  CHECK_CONTIGUOUS(rope_freqs);
  TORCH_CHECK(query.scalar_type() == key.scalar_type() && query.scalar_type() == value.scalar_type(),
      "query, key, and value must have the same dtype");
  TORCH_CHECK(query.scalar_type() == q_weight.scalar_type() && query.scalar_type() == k_weight.scalar_type() &&
      query.scalar_type() == rope_freqs.scalar_type(), "normalization weights and RoPE frequencies must match query dtype");
  int channels = query.size(2) * query.size(3);
  TORCH_CHECK(channels % 2 == 0, "RoPE requires heads * head_dim to be even");
  TORCH_CHECK(q_weight.numel() == channels && k_weight.numel() == channels,
      "normalization weights must have heads * head_dim values");
  TORCH_CHECK(rope_freqs.dim() == 2 && rope_freqs.size(0) == 2 && rope_freqs.size(1) == channels,
      "rope_freqs must have shape [2, heads * head_dim]");
  TORCH_CHECK(std::get<0>(kernel_size) > 0 && std::get<1>(kernel_size) > 0,
      "kernel dimensions must be positive");
}

size_t qn_forward_smem(int channels, int heads, int tokens) {
  return (channels + tokens + heads * tokens + 6 * tokens + kQueryNeighborThreads) * sizeof(float) +
      4 * tokens * sizeof(int);
}

size_t qn_backward_qk_smem(int channels, int tokens) {
  return (6 * channels + tokens + 6 * tokens + kQueryNeighborThreads) * sizeof(float) +
      4 * tokens * sizeof(int);
}

} // namespace

void sparse_na2d_bilinear_query_neighbor_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& coords,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& rope_freqs,
    at::Tensor& logsumexp,
    const std::tuple<int32_t, int32_t>& kernel_size,
    float offset_scale_y,
    float offset_scale_x,
    float attn_scale,
    float norm_eps,
    bool norm_before_rope) {
  check_qn_args(query, key, value, coords, q_weight, k_weight, rope_freqs, kernel_size);
  at::cuda::OptionalCUDAGuard guard(query.device());
  int batch = query.size(0), num_queries = query.size(1), heads = query.size(2), dim = query.size(3);
  int height = key.size(1), width = key.size(2), dim_value = value.size(4);
  int kh = std::get<0>(kernel_size), kw = std::get<1>(kernel_size), tokens = kh * kw;
  dim3 grid(num_queries, batch);
  auto stream = at::cuda::getCurrentCUDAStream(query.device().index());
  size_t smem = qn_forward_smem(heads * dim, heads, tokens);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(),
      "sparse_na2d_bilinear_query_neighbor_forward", [&] {
        using q_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, coords.scalar_type(),
            "sparse_na2d_bilinear_query_neighbor_forward_coords", [&] {
              using c_t = scalar_t;
              qn_forward_kernel<q_t, c_t><<<grid, kQueryNeighborThreads, smem, stream>>>(
                  query.data_ptr<q_t>(), key.data_ptr<q_t>(), value.data_ptr<q_t>(), coords.data_ptr<c_t>(),
                  q_weight.data_ptr<q_t>(), k_weight.data_ptr<q_t>(), rope_freqs.data_ptr<q_t>(),
                  out.data_ptr<q_t>(), logsumexp.data_ptr<float>(), batch, num_queries, height, width,
                  heads, dim, dim_value, kh, kw, offset_scale_y, offset_scale_x, attn_scale, norm_eps,
                  norm_before_rope);
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void sparse_na2d_bilinear_query_neighbor_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    at::Tensor& grad_q_weight,
    at::Tensor& grad_k_weight,
    at::Tensor& grad_rope_freqs,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& coords,
    const at::Tensor& q_weight,
    const at::Tensor& k_weight,
    const at::Tensor& rope_freqs,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    const std::tuple<int32_t, int32_t>& kernel_size,
    float offset_scale_y,
    float offset_scale_x,
    float attn_scale,
    float norm_eps,
    bool norm_before_rope) {
  check_qn_args(query, key, value, coords, q_weight, k_weight, rope_freqs, kernel_size);
  grad_key.zero_();
  grad_value.zero_();
  grad_q_weight.zero_();
  grad_k_weight.zero_();
  grad_rope_freqs.zero_();
  at::cuda::OptionalCUDAGuard guard(query.device());
  int batch = query.size(0), num_queries = query.size(1), heads = query.size(2), dim = query.size(3);
  int height = key.size(1), width = key.size(2), dim_value = value.size(4);
  int kh = std::get<0>(kernel_size), kw = std::get<1>(kernel_size), tokens = kh * kw;
  at::Tensor d_logits = at::empty({batch, num_queries, heads, tokens}, query.options().dtype(at::kFloat));
  dim3 grid(num_queries, batch);
  auto stream = at::cuda::getCurrentCUDAStream(query.device().index());
  size_t value_smem = qn_forward_smem(heads * dim, heads, tokens);
  size_t qk_smem = qn_backward_qk_smem(heads * dim, tokens);
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(),
      "sparse_na2d_bilinear_query_neighbor_backward", [&] {
        using q_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, coords.scalar_type(),
            "sparse_na2d_bilinear_query_neighbor_backward_coords", [&] {
              using c_t = scalar_t;
              qn_backward_value_kernel<q_t, c_t><<<grid, kQueryNeighborThreads, value_smem, stream>>>(
                  query.data_ptr<q_t>(), key.data_ptr<q_t>(), value.data_ptr<q_t>(), coords.data_ptr<c_t>(),
                  q_weight.data_ptr<q_t>(), k_weight.data_ptr<q_t>(), rope_freqs.data_ptr<q_t>(),
                  out.data_ptr<q_t>(), grad_out.data_ptr<q_t>(), logsumexp.data_ptr<float>(),
                  d_logits.data_ptr<float>(), grad_value.data_ptr<q_t>(), batch, num_queries, height, width,
                  heads, dim, dim_value, kh, kw, offset_scale_y, offset_scale_x, attn_scale, norm_eps,
                  norm_before_rope);
              qn_backward_query_key_kernel<q_t, c_t><<<grid, kQueryNeighborThreads, qk_smem, stream>>>(
                  query.data_ptr<q_t>(), key.data_ptr<q_t>(), coords.data_ptr<c_t>(), q_weight.data_ptr<q_t>(),
                  k_weight.data_ptr<q_t>(), rope_freqs.data_ptr<q_t>(), d_logits.data_ptr<float>(),
                  grad_query.data_ptr<q_t>(), grad_key.data_ptr<q_t>(), grad_q_weight.data_ptr<q_t>(),
                  grad_k_weight.data_ptr<q_t>(), grad_rope_freqs.data_ptr<q_t>(), batch, num_queries,
                  height, width, heads, dim, kh, kw, offset_scale_y, offset_scale_x, norm_eps,
                  norm_before_rope);
            });
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace natten
