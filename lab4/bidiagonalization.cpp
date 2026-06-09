// bidiagonalization.cpp
// 将 m×n 矩阵（本框架保证m ≥ n）通过 Householder 变换化为上双对角形
//
// 算法说明（你需要结合代码看）：
// 对上双对角化，需要交替从左侧和右侧应用 Householder 变换：
// 第 k 步（k = 0, 1, ..., n-1）：
//    - 从左侧作用 H_k，消去第 k 列中位置 (k+1,k), (k+2,k), ..., (m-1,k) 的元素
//    - 如果 k < n-2，从右侧作用 V_k，消去第 k 行中位置 (k,k+2), (k,k+3), ..., (k,n-1) 的元素
//
// 例如，对一个 4x4 矩阵 A，第一步 k=0：
//   - 从左侧作用 H_0，消去 A(1,0), A(2,0), A(3,0)，得到 B_0，同时更新 U = U * H_0
//   - 从右侧作用 V_0，消去 B_0(0,2)，B_0(0,3)，得到 B_1，同时更新 V = V * V_0
//
// 最终得到上双对角矩阵 B，只有主对角线和上次对角线有非零元素
//
// 本组件输出：A = U * B * V^T
// 其中 U（m×m）和 V（n×n）均为正交矩阵，B（m×n）为上双对角矩阵
#include "matrix.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <arm_neon.h>
// 辅助函数，计算向量的范数（平方和开根）
static double vector_norm(const std::vector<double> &v)
 {
      double sum = 0.0;
      for (double x : v)
      sum += x * x;
      return std::sqrt(sum);
 }
// 将 m×n 矩阵 A（m ≥ n）化为上双对角形，返回 B，同时输出 U（m×m）和 V（n×n）
Matrix to_bidiagonal(const Matrix &A, Matrix &U, Matrix &V)
{
if (A.rows() < A.cols())
 {
 throw std::invalid_argument("to_bidiagonal: requires m >= n");
 }
   const int m = A.rows();
   const int n = A.cols();
   Matrix B = A;

// U = I_m，V = I_n
U = Matrix(m, m, 0.0);
for (int i = 0; i < m; ++i)
    U.at(i, i) = 1.0;
V = Matrix(n, n, 0.0);
for (int i = 0; i < n; ++i)
    V.at(i, i) = 1.0;

for (int k = 0; k < n; ++k)
{
    // ================================================================
    // 步骤 1: 从左侧作用 Householder 变换，消去第 k 列中对角线以下的元素
    // ================================================================

    // 提取第 k 列从第 k 行往下的子向量
    // 例如：k=0 时提取 A(0:m-1, 0)，长度为 m-k+1 ; k=1 时提取 A(1:m-1, 1)
    std::vector<double> x(m - k);
    for (int i = 0; i < m - k; ++i)
    {
        x[i] = B.at(k + i, k);
    }

    double norm_x = vector_norm(x);

    if (norm_x > 1e-14 && k < m - 1)
    {
        // sign(x[0])：此处规定 x[0]==0 时取 +1
        double sigma = (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;

        // 实际上这里是+或者-都可以，手册里 Householder一节是 -αe_1
        // 但我们这里 sigma 取了 sign(x[0]) * norm_x，所以是 +sigma * e_1 的形式
        std::vector<double> v(x);      //v=x
        v[0] += sigma; // v = x + sigma * e_1

        // 计算 v^T v
        double vTv = 0.0;
        for (double vi : v)
            vTv += vi * vi;

        // TODO(SIMD编程)：此处的Householder变换可以通过 SIMD 指令加速，你可以尝试实现
        if (vTv > 1e-28)
{
    const double beta = 2.0 / vTv;
    const int rows = m - k;
    const int cols = n - k;
    // 第一步：w[j] = Σ_i v[i] * B[k+i, k+j]
    // 外i内j，行主序连续访问，真正的SIMD load
    std::vector<double> w(cols, 0.0);

    for (int i = 0; i < rows; ++i)
    {
        float64x2_t vi = vdupq_n_f64(v[i]);  // 广播 v[i]
        int j = 0;
        for (; j + 2 <= cols; j += 2)
        {
            float64x2_t bv = vld1q_f64(&B.at(k+i, k+j));  // 连续load
            float64x2_t wv = vld1q_f64(&w[j]);
            wv = vfmaq_f64(wv, vi, bv);                    // w += v[i]*B[i,j:j+2]
            vst1q_f64(&w[j], wv);
        }
        for (; j < cols; ++j)
            w[j] += v[i] * B.at(k+i, k+j);
    }

    // 第二步：B[k+i, k+j] -= beta * v[i] * w[j]
    // 外i内j，行连续访问
    for (int i = 0; i < rows; ++i)
    {
        float64x2_t scale = vdupq_n_f64(beta * v[i]);  // 广播 beta*v[i]
        int j = 0;
        for (; j + 2 <= cols; j += 2)
        {
            double*     bptr = &B.at(k+i, k+j);
            float64x2_t bv   = vld1q_f64(bptr);
            float64x2_t wv   = vld1q_f64(&w[j]);
            bv = vfmsq_f64(bv, scale, wv);              // B -= scale * w
            vst1q_f64(bptr, bv);
        }
        for (; j < cols; ++j)
            B.at(k+i, k+j) -= beta * v[i] * w[j];
    }


    // 第三步：wU[i] = Σ_j U[i, k+j] * v[j]
    // 外i内j，U行连续访问
    std::vector<double> wU(m, 0.0);

    for (int i = 0; i < m; ++i)
    {
        float64x2_t acc = vdupq_n_f64(0.0);
        int j = 0;
        for (; j + 2 <= rows; j += 2)
        {
            float64x2_t uv = vld1q_f64(&U.at(i, k+j));
            float64x2_t vv = vld1q_f64(&v[j]);
            acc = vfmaq_f64(acc, uv, vv);
        }
        // 水平规约
        wU[i] = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
        for (; j < rows; ++j)
            wU[i] += U.at(i, k+j) * v[j];
    }

   
    // 第四步：U[i, k+j] -= beta * wU[i] * v[j]
    // 外i内j，U行连续访问
    for (int i = 0; i < m; ++i)
    {
        float64x2_t scale = vdupq_n_f64(beta * wU[i]);
        int j = 0;
        for (; j + 2 <= rows; j += 2)
        {
            double*     uptr = &U.at(i, k+j);
            float64x2_t uv   = vld1q_f64(uptr);
            float64x2_t vv   = vld1q_f64(&v[j]);
            uv = vfmsq_f64(uv, scale, vv);
            vst1q_f64(uptr, uv);
        }
        for (; j < rows; ++j)
            U.at(i, k+j) -= beta * wU[i] * v[j];
    }
}
    }

    // 清除第 k 列中对角线以下的元素
    // 理论上应为 0，但不能完全保证全是 0，这里强制置零
    for (int i = k + 1; i < m; ++i)
    {
        B.at(i, k) = 0.0;
    }

    // ================================================================
    // 步骤 2: 从右侧作用 Householder 变换，消去第 k 行中 (k,k+2) 及右边的元素
    //        （只在 k < n-2 时需要）
    // ================================================================

    if (k < n - 2)
    {
        // 提取第 k 行从第 k+1 列往右的子向量（长度 n-k-1）
        std::vector<double> y(n - k - 1);
        for (int j = 0; j < n - k - 1; ++j)
        {
            y[j] = B.at(k, k + 1 + j);
        }

        // 与之前类似，计算模长
        double norm_y = vector_norm(y);

        if (norm_y > 1e-14)
        {
            double sigma = (y[0] >= 0.0 ? 1.0 : -1.0) * norm_y;

            // 构造 Householder 向量 v = y + sigma * e_1
            std::vector<double> v(y);
            v[0] += sigma;

            double vTv = 0.0;
            for (double vi : v)
                vTv += vi * vi;

            // TODO(SIMD编程)：此处的Householder变换可以通过 SIMD 指令加速，你可以尝试实现
            if (vTv > 1e-28)
            {
                 const double beta = 2.0 / vTv;
    const int rows  = m - k;      // B/V 的行数
    const int cols  = n - k - 1;  // 操作列数（j从0到n-k-2）

   
    // 第一步：w[i] = Σ_j B[k+i, k+1+j] * v[j]
    // 外i内j，B行连续访问
    std::vector<double> w(rows, 0.0);

    for (int i = 0; i < rows; ++i)
    {
        float64x2_t acc = vdupq_n_f64(0.0);
        int j = 0;
        for (; j + 2 <= cols; j += 2)
        {
            float64x2_t bv = vld1q_f64(&B.at(k+i, k+1+j));  // B行连续
            float64x2_t vv = vld1q_f64(&v[j]);
            acc = vfmaq_f64(acc, bv, vv);
        }
        // 水平规约
        w[i] = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
        for (; j < cols; ++j)
            w[i] += B.at(k+i, k+1+j) * v[j];
    }

   
    // 第二步：B[k+i, k+1+j] -= beta * w[i] * v[j]
    // 外i内j，B行连续访问

    for (int i = 0; i < rows; ++i)
    {
        float64x2_t scale = vdupq_n_f64(beta * w[i]);  // 广播 beta*w[i]
        int j = 0;
        for (; j + 2 <= cols; j += 2)
        {
            double*     bptr = &B.at(k+i, k+1+j);
            float64x2_t bv   = vld1q_f64(bptr);
            float64x2_t vv   = vld1q_f64(&v[j]);
            bv = vfmsq_f64(bv, scale, vv);              // B -= scale * v
            vst1q_f64(bptr, bv);
        }
        for (; j < cols; ++j)
            B.at(k+i, k+1+j) -= beta * w[i] * v[j];
    }

  
    // 第三步：wV[i] = Σ_j V[i, k+1+j] * v[j]
    // 外i内j，V行连续访问
    std::vector<double> wV(n, 0.0);

    for (int i = 0; i < n; ++i)
    {
        float64x2_t acc = vdupq_n_f64(0.0);
        int j = 0;
        for (; j + 2 <= cols; j += 2)
        {
            float64x2_t vmat = vld1q_f64(&V.at(i, k+1+j));  // V行连续
            float64x2_t vv   = vld1q_f64(&v[j]);
            acc = vfmaq_f64(acc, vmat, vv);
        }
        wV[i] = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
        for (; j < cols; ++j)
            wV[i] += V.at(i, k+1+j) * v[j];
    }

    // 第四步：V[i, k+1+j] -= beta * wV[i] * v[j]
    // 外i内j，V行连续访
    for (int i = 0; i < n; ++i)
    {
        float64x2_t scale = vdupq_n_f64(beta * wV[i]);
        int j = 0;
        for (; j + 2 <= cols; j += 2)
        {
            double*     vptr = &V.at(i, k+1+j);
            float64x2_t vmat = vld1q_f64(vptr);
            float64x2_t vv   = vld1q_f64(&v[j]);
            vmat = vfmsq_f64(vmat, scale, vv);
            vst1q_f64(vptr, vmat);
        }
        for (; j < cols; ++j)
            V.at(i, k+1+j) -= beta * wV[i] * v[j];
    }
            }
        }

        // 强制置零
        for (int j = k + 2; j < n; ++j)
        {
            B.at(k, j) = 0.0;
        }
    }
}

       return B;
}