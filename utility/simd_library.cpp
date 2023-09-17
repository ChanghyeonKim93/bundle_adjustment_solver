#include "simd_library.h"

namespace simd {
PointWarper::PointWarper() {
  buf_x_ = nullptr;
  buf_y_ = nullptr;
  buf_z_ = nullptr;

  buf_x_warped_ = nullptr;
  buf_y_warped_ = nullptr;
  buf_z_warped_ = nullptr;

  buf_x_ = (float *)custom_aligned_malloc(sizeof(float) * 100000000);
  buf_y_ = (float *)custom_aligned_malloc(sizeof(float) * 100000000);
  buf_z_ = (float *)custom_aligned_malloc(sizeof(float) * 100000000);

  buf_x_warped_ = (float *)custom_aligned_malloc(sizeof(float) * 100000000);
  buf_y_warped_ = (float *)custom_aligned_malloc(sizeof(float) * 100000000);
  buf_z_warped_ = (float *)custom_aligned_malloc(sizeof(float) * 100000000);

  // x_warp = r11*x + r12*y + r13*z + t1
  // y_warp = r21*x + r22*y + r23*z + t2
  // z_warp = r31*x + r32*y + r33*z + t3
}

PointWarper::~PointWarper() {
  if (buf_x_ != nullptr) custom_aligned_free((void *)buf_x_);
  if (buf_y_ != nullptr) custom_aligned_free((void *)buf_y_);
  if (buf_z_ != nullptr) custom_aligned_free((void *)buf_z_);

  if (buf_x_warped_ != nullptr) custom_aligned_free((void *)buf_x_warped_);
  if (buf_y_warped_ != nullptr) custom_aligned_free((void *)buf_y_warped_);
  if (buf_z_warped_ != nullptr) custom_aligned_free((void *)buf_z_warped_);
}

void PointWarper::Warp(const std::vector<Eigen::Vector3f> &points, const Eigen::Isometry3f &pose,
                       std::vector<Eigen::Vector3f> &warped_points) {
  const auto num_points = points.size();
  warped_points.resize(num_points);

  // Set pose
  _SIMD_TYPE _r11 = _SIMD_SET1_PS(pose.linear()(0, 0));
  _SIMD_TYPE _r12 = _SIMD_SET1_PS(pose.linear()(0, 1));
  _SIMD_TYPE _r13 = _SIMD_SET1_PS(pose.linear()(0, 2));
  _SIMD_TYPE _r21 = _SIMD_SET1_PS(pose.linear()(1, 0));
  _SIMD_TYPE _r22 = _SIMD_SET1_PS(pose.linear()(1, 1));
  _SIMD_TYPE _r23 = _SIMD_SET1_PS(pose.linear()(1, 2));
  _SIMD_TYPE _r31 = _SIMD_SET1_PS(pose.linear()(2, 0));
  _SIMD_TYPE _r32 = _SIMD_SET1_PS(pose.linear()(2, 1));
  _SIMD_TYPE _r33 = _SIMD_SET1_PS(pose.linear()(2, 2));
  _SIMD_TYPE _t1 = _SIMD_SET1_PS(pose.translation().x());
  _SIMD_TYPE _t2 = _SIMD_SET1_PS(pose.translation().y());
  _SIMD_TYPE _t3 = _SIMD_SET1_PS(pose.translation().z());

  float *x_ptr = buf_x_;
  float *y_ptr = buf_y_;
  float *z_ptr = buf_z_;
  float *warped_x_ptr = buf_x_warped_;
  float *warped_y_ptr = buf_y_warped_;
  float *warped_z_ptr = buf_z_warped_;

  size_t idx = 0;
  for (; idx < num_points; ++idx) {
    *x_ptr = points[idx].x();
    *y_ptr = points[idx].y();
    *z_ptr = points[idx].z();
    ++x_ptr;
    ++y_ptr;
    ++z_ptr;
  }

  // Warp
  idx = 0;
  x_ptr = buf_x_;
  y_ptr = buf_y_;
  z_ptr = buf_z_;
  for (; idx < num_points; idx += _SIMD_BYTE_STEP) {
    _SIMD_TYPE _x = _SIMD_LOAD_PS(x_ptr);
    _SIMD_TYPE _y = _SIMD_LOAD_PS(y_ptr);
    _SIMD_TYPE _z = _SIMD_LOAD_PS(z_ptr);

    _SIMD_TYPE _x_warp = _SIMD_ADD_PS(
        _SIMD_ADD_PS(_SIMD_ADD_PS(_SIMD_MUL_PS(_r11, _x), _SIMD_MUL_PS(_r12, _y)), _SIMD_MUL_PS(_r13, _z)), _t1);
    _SIMD_TYPE _y_warp = _SIMD_ADD_PS(
        _SIMD_ADD_PS(_SIMD_ADD_PS(_SIMD_MUL_PS(_r21, _x), _SIMD_MUL_PS(_r22, _y)), _SIMD_MUL_PS(_r23, _z)), _t2);
    _SIMD_TYPE _z_warp = _SIMD_ADD_PS(
        _SIMD_ADD_PS(_SIMD_ADD_PS(_SIMD_MUL_PS(_r31, _x), _SIMD_MUL_PS(_r32, _y)), _SIMD_MUL_PS(_r33, _z)), _t3);

    _x_warp = _SIMD_ADD_PS(_SIMD_ADD_PS(_SIMD_ADD_PS(_SIMD_MUL_PS(_r11, _x_warp), _SIMD_MUL_PS(_r12, _y_warp)),
                                        _SIMD_MUL_PS(_r13, _z_warp)),
                           _t1);
    _y_warp = _SIMD_ADD_PS(_SIMD_ADD_PS(_SIMD_ADD_PS(_SIMD_MUL_PS(_r21, _x_warp), _SIMD_MUL_PS(_r22, _y_warp)),
                                        _SIMD_MUL_PS(_r23, _z_warp)),
                           _t2);
    _z_warp = _SIMD_ADD_PS(_SIMD_ADD_PS(_SIMD_ADD_PS(_SIMD_MUL_PS(_r31, _x_warp), _SIMD_MUL_PS(_r32, _y_warp)),
                                        _SIMD_MUL_PS(_r33, _z_warp)),
                           _t3);

    _SIMD_STORE_PS(warped_x_ptr, _x_warp);
    _SIMD_STORE_PS(warped_y_ptr, _y_warp);
    _SIMD_STORE_PS(warped_z_ptr, _z_warp);

    x_ptr += _SIMD_BYTE_STEP;
    y_ptr += _SIMD_BYTE_STEP;
    z_ptr += _SIMD_BYTE_STEP;

    for (int j = 0; j < _SIMD_BYTE_STEP; ++j) {
      warped_points[idx + j].x() = *(warped_x_ptr + j);
      warped_points[idx + j].y() = *(warped_y_ptr + j);
      warped_points[idx + j].z() = *(warped_z_ptr + j);
    }
  }
  for (; idx < num_points; ++idx) {
    const Eigen::Vector3f warped_point = pose * points[idx];
    *warped_x_ptr = warped_point(0);
    *warped_y_ptr = warped_point(1);
    *warped_z_ptr = warped_point(2);

    warped_points[idx].x() = *(warped_x_ptr);
    warped_points[idx].y() = *(warped_y_ptr);
    warped_points[idx].z() = *(warped_z_ptr);
  }

  // warped_x_ptr = buf_x_warped_;
  // warped_y_ptr = buf_y_warped_;
  // warped_z_ptr = buf_z_warped_;
  // for (; idx < num_points; ++idx)
  // {
  //   warped_points[idx].x() = *warped_x_ptr;
  //   warped_points[idx].y() = *warped_y_ptr;
  //   warped_points[idx].z() = *warped_z_ptr;

  //   ++warped_x_ptr;
  //   ++warped_y_ptr;
  //   ++warped_z_ptr;
  // }
}

// int idx = 0;
// for (; idx < num_points; idx += 4)
// {
//   const auto &x = points[idx].x;

//   // u_warp4 = _mm_SIMD_LOAD_PS(buf_uel_warped + idx);
//   // v_warp4 = _mm_SIMD_LOAD_PS(buf_vel_warped + idx);
//   // u_match4 = _mm_SIMD_LOAD_PS(buf_uel_matched + idx);
//   // v_match4 = _mm_SIMD_LOAD_PS(buf_vel_matched + idx);
//   // gx4 = _mm_SIMD_LOAD_PS(buf_gxl_matched + idx);
//   // gy4 = _mm_SIMD_LOAD_PS(buf_gyl_matched + idx);

//   // // calculate residual left edge
//   // __m128 rrrr = _mm_SIMD_ADD_PS(_mm_SIMD_MUL_PS(gx4, _mm_sub_ps(u_warp4, u_match4)), _mm_SIMD_MUL_PS(gy4,
//   _mm_sub_ps(v_warp4, v_match4)));
//   // _mm_store_ps(buf_res_l_e + idx, rrrr);
// }

// if (idx < num_points)
// {
//   for (; idx < num_points; ++idx)
//   {
//     // const auto &x =

//     // u_warp4 = _mm_SIMD_LOAD_PS(buf_uel_warped + idx);
//     // v_warp4 = _mm_SIMD_LOAD_PS(buf_vel_warped + idx);
//     // u_match4 = _mm_SIMD_LOAD_PS(buf_uel_matched + idx);
//     // v_match4 = _mm_SIMD_LOAD_PS(buf_vel_matched + idx);
//     // gx4 = _mm_SIMD_LOAD_PS(buf_gxl_matched + idx);
//     // gy4 = _mm_SIMD_LOAD_PS(buf_gyl_matched + idx);

//     // // calculate residual left edge
//     // __m128 rrrr = _mm_SIMD_ADD_PS(_mm_SIMD_MUL_PS(gx4, _mm_sub_ps(u_warp4, u_match4)), _mm_SIMD_MUL_PS(gy4,
//     _mm_sub_ps(v_warp4, v_match4)));
//     // _mm_store_ps(buf_res_l_e + idx, rrrr);
//   }
// }
};  // namespace simd