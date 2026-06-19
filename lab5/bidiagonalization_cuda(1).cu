// bidiagonalization_cuda.cu
// CUDA hand-kernel version of to_bidiagonal().
// Compile with nvcc instead of g++ for this translation unit.
// Example:
//   nvcc -O2 -std=c++17 main.cpp bidiagonalization_cuda.cu gkh.cpp -o main_cuda

#include "matrix.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#ifndef SVD_GPU_PROFILE
#define SVD_GPU_PROFILE 1
#endif

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                               \
        if (_err != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error at ") + __FILE__ + \
                                     ":" + std::to_string(__LINE__) + " : " +   \
                                     cudaGetErrorString(_err));                  \
        }                                                                       \
    } while (0)

static double vector_norm(const std::vector<double> &v) {
    double sum = 0.0;
    for (double x : v) sum += x * x;
    return std::sqrt(sum);
}

static void copy_matrix_to_host(const Matrix &M, std::vector<double> &buf) {
    const int r = M.rows();
    const int c = M.cols();
    buf.resize(static_cast<size_t>(r) * c);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            buf[static_cast<size_t>(i) * c + j] = M.at(i, j);
        }
    }
}

static void copy_host_to_matrix(const std::vector<double> &buf, Matrix &M) {
    const int r = M.rows();
    const int c = M.cols();
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            M.at(i, j) = buf[static_cast<size_t>(i) * c + j];
        }
    }
}

__global__ void set_identity_kernel(double *A, int rows, int cols) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = rows * cols;
    while (idx < total) {
        int r = idx / cols;
        int c = idx - r * cols;
        A[idx] = (r == c) ? 1.0 : 0.0;
        idx += blockDim.x * gridDim.x;
    }
}

// left Householder: w[col] = sum_i v[i] * B[k+i, k+col]
__global__ void left_gemv_B_kernel(const double *B, const double *v, double *w,
                                   int m, int n, int k) {
    extern __shared__ double sdata[];
    int col = blockIdx.x;
    int tid = threadIdx.x;
    int rows = m - k;
    double sum = 0.0;
    for (int i = tid; i < rows; i += blockDim.x) {
        sum += v[i] * B[static_cast<size_t>(k + i) * n + (k + col)];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) w[col] = sdata[0];
}

// Bsub -= beta * v * w^T
__global__ void left_ger_B_kernel(double *B, const double *v, const double *w,
                                  double beta, int m, int n, int k) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int rows = m - k;
    int cols = n - k;
    if (row < rows && col < cols) {
        B[static_cast<size_t>(k + row) * n + (k + col)] -= beta * v[row] * w[col];
    }
}

// wU[row] = sum_j U[row, k+j] * v[j]
__global__ void left_gemv_U_kernel(const double *U, const double *v, double *wU,
                                   int m, int k) {
    extern __shared__ double sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int len = m - k;
    double sum = 0.0;
    for (int j = tid; j < len; j += blockDim.x) {
        sum += U[static_cast<size_t>(row) * m + (k + j)] * v[j];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) wU[row] = sdata[0];
}

// U[:, k:] -= beta * wU * v^T
__global__ void left_ger_U_kernel(double *U, const double *v, const double *wU,
                                  double beta, int m, int k) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int len = m - k;
    if (row < m && col < len) {
        U[static_cast<size_t>(row) * m + (k + col)] -= beta * wU[row] * v[col];
    }
}

// right Householder: w[row] = sum_j B[k+row, k+1+j] * v[j]
__global__ void right_gemv_B_kernel(const double *B, const double *v, double *w,
                                    int m, int n, int k) {
    extern __shared__ double sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int cols = n - k - 1;
    double sum = 0.0;
    for (int j = tid; j < cols; j += blockDim.x) {
        sum += B[static_cast<size_t>(k + row) * n + (k + 1 + j)] * v[j];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) w[row] = sdata[0];
}

// Bsub -= beta * w * v^T
__global__ void right_ger_B_kernel(double *B, const double *v, const double *w,
                                   double beta, int m, int n, int k) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int rows = m - k;
    int cols = n - k - 1;
    if (row < rows && col < cols) {
        B[static_cast<size_t>(k + row) * n + (k + 1 + col)] -= beta * w[row] * v[col];
    }
}

// wV[row] = sum_j V[row, k+1+j] * v[j]
__global__ void right_gemv_V_kernel(const double *V, const double *v, double *wV,
                                    int n, int k) {
    extern __shared__ double sdata[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int len = n - k - 1;
    double sum = 0.0;
    for (int j = tid; j < len; j += blockDim.x) {
        sum += V[static_cast<size_t>(row) * n + (k + 1 + j)] * v[j];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) wV[row] = sdata[0];
}

// V[:, k+1:] -= beta * wV * v^T
__global__ void right_ger_V_kernel(double *V, const double *v, const double *wV,
                                   double beta, int n, int k) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int len = n - k - 1;
    if (row < n && col < len) {
        V[static_cast<size_t>(row) * n + (k + 1 + col)] -= beta * wV[row] * v[col];
    }
}

__global__ void zero_below_col_kernel(double *B, int m, int n, int k) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int count = m - k - 1;
    while (idx < count) {
        B[static_cast<size_t>(k + 1 + idx) * n + k] = 0.0;
        idx += blockDim.x * gridDim.x;
    }
}

__global__ void zero_right_row_kernel(double *B, int n, int k) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int count = n - k - 2;
    while (idx < count) {
        B[static_cast<size_t>(k) * n + (k + 2 + idx)] = 0.0;
        idx += blockDim.x * gridDim.x;
    }
}

static int reduce_threads(int len) {
    int t = 1;
    while (t < len && t < 256) t <<= 1;
    return t < 32 ? 32 : t;
}

Matrix to_bidiagonal(const Matrix &A, Matrix &U, Matrix &V) {
    if (A.rows() < A.cols()) {
        throw std::invalid_argument("to_bidiagonal: requires m >= n");
    }

    const int m = A.rows();
    const int n = A.cols();
    Matrix B(m, n, 0.0);
    U = Matrix(m, m, 0.0);
    V = Matrix(n, n, 0.0);

    std::vector<double> hA;
    copy_matrix_to_host(A, hA);

    double *dB = nullptr, *dU = nullptr, *dV = nullptr;
    double *dv = nullptr, *dw = nullptr, *dw2 = nullptr;

    const size_t bytesB = static_cast<size_t>(m) * n * sizeof(double);
    const size_t bytesU = static_cast<size_t>(m) * m * sizeof(double);
    const size_t bytesV = static_cast<size_t>(n) * n * sizeof(double);
    const int maxLen = m; // because m >= n

#if SVD_GPU_PROFILE
    cudaEvent_t e0, e1;
    float h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));
#endif

    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dU, bytesU));
    CUDA_CHECK(cudaMalloc(&dV, bytesV));
    CUDA_CHECK(cudaMalloc(&dv,  static_cast<size_t>(maxLen) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dw,  static_cast<size_t>(maxLen) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dw2, static_cast<size_t>(maxLen) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, hA.data(), bytesB, cudaMemcpyHostToDevice));

    int totalU = m * m;
    int totalV = n * n;
    set_identity_kernel<<<(totalU + 255) / 256, 256>>>(dU, m, m);
    set_identity_kernel<<<(totalV + 255) / 256, 256>>>(dV, n, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

#if SVD_GPU_PROFILE
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, e0, e1));
#endif

    dim3 block2d(16, 16);

#if SVD_GPU_PROFILE
    CUDA_CHECK(cudaEventRecord(e0));
#endif

    for (int k = 0; k < n; ++k) {
        // Because v construction needs the current B column, copy only this column segment back.
        std::vector<double> x(m - k);
        std::vector<double> tmpB(bytesB / sizeof(double));
        CUDA_CHECK(cudaMemcpy(tmpB.data(), dB, bytesB, cudaMemcpyDeviceToHost));
        for (int i = 0; i < m - k; ++i) x[i] = tmpB[static_cast<size_t>(k + i) * n + k];

        double norm_x = vector_norm(x);
        if (norm_x > 1e-14 && k < m - 1) {
            double sigma = (x[0] >= 0.0 ? 1.0 : -1.0) * norm_x;
            std::vector<double> v(x);
            v[0] += sigma;
            double vTv = 0.0;
            for (double vi : v) vTv += vi * vi;

            if (vTv > 1e-28) {
                double beta = 2.0 / vTv;
                int rows = m - k;
                int cols = n - k;
                CUDA_CHECK(cudaMemcpy(dv, v.data(), static_cast<size_t>(rows) * sizeof(double),
                                      cudaMemcpyHostToDevice));

                int tB = reduce_threads(rows);
                left_gemv_B_kernel<<<cols, tB, tB * sizeof(double)>>>(dB, dv, dw, m, n, k);
                dim3 gridB((cols + block2d.x - 1) / block2d.x,
                           (rows + block2d.y - 1) / block2d.y);
                left_ger_B_kernel<<<gridB, block2d>>>(dB, dv, dw, beta, m, n, k);

                int tU = reduce_threads(rows);
                left_gemv_U_kernel<<<m, tU, tU * sizeof(double)>>>(dU, dv, dw2, m, k);
                dim3 gridU((rows + block2d.x - 1) / block2d.x,
                           (m + block2d.y - 1) / block2d.y);
                left_ger_U_kernel<<<gridU, block2d>>>(dU, dv, dw2, beta, m, k);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        zero_below_col_kernel<<<((m - k - 1) + 255) / 256, 256>>>(dB, m, n, k);
        CUDA_CHECK(cudaGetLastError());

        if (k < n - 2) {
            // Copy B back only to construct y. This is simple and safe for coursework baseline,
            // but it is also one source of overhead discussed in the report.
            std::vector<double> tmpB(bytesB / sizeof(double));
            CUDA_CHECK(cudaMemcpy(tmpB.data(), dB, bytesB, cudaMemcpyDeviceToHost));
            std::vector<double> y(n - k - 1);
            for (int j = 0; j < n - k - 1; ++j) {
                y[j] = tmpB[static_cast<size_t>(k) * n + (k + 1 + j)];
            }

            double norm_y = vector_norm(y);
            if (norm_y > 1e-14) {
                double sigma = (y[0] >= 0.0 ? 1.0 : -1.0) * norm_y;
                std::vector<double> v(y);
                v[0] += sigma;
                double vTv = 0.0;
                for (double vi : v) vTv += vi * vi;

                if (vTv > 1e-28) {
                    double beta = 2.0 / vTv;
                    int rows = m - k;
                    int cols = n - k - 1;
                    CUDA_CHECK(cudaMemcpy(dv, v.data(), static_cast<size_t>(cols) * sizeof(double),
                                          cudaMemcpyHostToDevice));

                    int tB = reduce_threads(cols);
                    right_gemv_B_kernel<<<rows, tB, tB * sizeof(double)>>>(dB, dv, dw, m, n, k);
                    dim3 gridB((cols + block2d.x - 1) / block2d.x,
                               (rows + block2d.y - 1) / block2d.y);
                    right_ger_B_kernel<<<gridB, block2d>>>(dB, dv, dw, beta, m, n, k);

                    int tV = reduce_threads(cols);
                    right_gemv_V_kernel<<<n, tV, tV * sizeof(double)>>>(dV, dv, dw2, n, k);
                    dim3 gridV((cols + block2d.x - 1) / block2d.x,
                               (n + block2d.y - 1) / block2d.y);
                    right_ger_V_kernel<<<gridV, block2d>>>(dV, dv, dw2, beta, n, k);
                    CUDA_CHECK(cudaGetLastError());
                }
            }
            zero_right_row_kernel<<<((n - k - 2) + 255) / 256, 256>>>(dB, n, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

#if SVD_GPU_PROFILE
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, e0, e1));
    CUDA_CHECK(cudaEventRecord(e0));
#endif

    std::vector<double> hB(static_cast<size_t>(m) * n);
    std::vector<double> hU(static_cast<size_t>(m) * m);
    std::vector<double> hV(static_cast<size_t>(n) * n);
    CUDA_CHECK(cudaMemcpy(hB.data(), dB, bytesB, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hU.data(), dU, bytesU, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hV.data(), dV, bytesV, cudaMemcpyDeviceToHost));

#if SVD_GPU_PROFILE
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, e0, e1));
    std::cerr << "[GPU bidiagonalization] H2D/init=" << h2d_ms
              << " ms, loop+kernels+inner_copy=" << kernel_ms
              << " ms, D2H=" << d2h_ms << " ms\n";
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
#endif

    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dU));
    CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dv));
    CUDA_CHECK(cudaFree(dw));
    CUDA_CHECK(cudaFree(dw2));

    B = Matrix(m, n, 0.0);
    copy_host_to_matrix(hB, B);
    copy_host_to_matrix(hU, U);
    copy_host_to_matrix(hV, V);
    return B;
}
