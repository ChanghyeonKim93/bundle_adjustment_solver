#ifndef _SIMD_LIBRARY_H_
#define _SIMD_LIBRARY_H_

#include <iostream>
#include <vector>

#include <eigen3/Eigen/Dense>

#include "immintrin.h"

// #define USE_SSE
#define USE_AVX

#if defined(USE_SSE)
#define _SIMD_BYTE_STEP 4
#define _SIMD_TYPE __m128
#define _SIMD_SET1_PS _mm_set1_ps
#define _SIMD_LOAD_PS _mm_load_ps
#define _SIMD_RCP_PS _mm_rcp_ps
#define _SIMD_MUL_PS _mm_mul_ps
#define _SIMD_ADD_PS _mm_add_ps
#define _SIMD_SUB_PS _mm_sub_ps
#define _SIMD_STORE_PS _mm_store_ps
#elif defined(USE_AVX)
#define _SIMD_BYTE_STEP 8
#define _SIMD_TYPE __m256
#define _SIMD_SET1_PS _mm256_set1_ps
#define _SIMD_LOAD_PS _mm256_load_ps
#define _SIMD_RCP_PS _mm256_rcp_ps
#define _SIMD_MUL_PS _mm256_mul_ps
#define _SIMD_ADD_PS _mm256_add_ps
#define _SIMD_SUB_PS _mm256_sub_ps
#define _SIMD_STORE_PS _mm256_store_ps
#endif

#define ALIGN_BYTES 64
// AVX, AVX2 (512 bits = 64 Bytes, 256 bits = 32 Bytes), SSE4.2 (128 bits = 16 Bytes)
/** \internal Like malloc, but the returned pointer is guaranteed to be 32-byte aligned.
 * Fast, but wastes 32 additional bytes of memory. Does not throw any exception.
 *
 * (256 bits) two LSB addresses of 32 bytes-aligned : 00, 20, 40, 60, 80, A0, C0, E0
 * (128 bits) two LSB addresses of 16 bytes-aligned : 00, 10, 20, 30, 40, 50, 60, 70, 80, 90, A0, B0, C0, D0, E0, F0
 */
inline void *custom_aligned_malloc(std::size_t size)
{
  void *original = std::malloc(size + ALIGN_BYTES); // size+ALIGN_BYTES��ŭ �Ҵ��ϰ�,
  if (original == 0)
    return nullptr; // if allocation is failed, return nullptr;
  void *aligned = reinterpret_cast<void *>((reinterpret_cast<std::size_t>(original) & ~(std::size_t(ALIGN_BYTES - 1))) + ALIGN_BYTES);
  *(reinterpret_cast<void **>(aligned) - 1) = original;
  return aligned;
};

/** \internal Frees memory allocated with handmade_aligned_malloc */
inline void custom_aligned_free(void *ptr)
{
  if (ptr)
    std::free(*(reinterpret_cast<void **>(ptr) - 1));
};

namespace simd
{
  class PointWarper
  {
  public:
    PointWarper();
    ~PointWarper();

    void Warp(const std::vector<Eigen::Vector3f> &points, const Eigen::Isometry3f &pose,
              std::vector<Eigen::Vector3f> &warped_points);

  private:
    float *buf_x_;
    float *buf_y_;
    float *buf_z_;

    float *buf_x_warped_;
    float *buf_y_warped_;
    float *buf_z_warped_;

    // x_warp = r11*x + r12*y + r13*z + t1
    // y_warp = r21*x + r22*y + r23*z + t2
    // z_warp = r31*x + r32*y + r33*z + t3
  };
};

#endif