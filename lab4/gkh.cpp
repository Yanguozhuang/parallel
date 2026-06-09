#include "gkh.h"

#include "givens.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <arm_neon.h>
namespace
{

    // ===== MPI 初始化与输出控制（不修改 main.cpp 的版本） =====
    // main.cpp 无法修改时，MPI_Init 放在 gkh.cpp 内部兜底处理。
    // 注意：不能在 gkh_svd_from_bidiagonal() 每次结束时 MPI_Finalize，
    // 因为 main.cpp 会连续调用多次 GKH（5x5、8x8、10x8、1000x1000 等）。
    // 因此本文件只在第一次进入时初始化 MPI，并用 atexit 在程序退出时统一 finalize。
    static bool g_mpi_initialized_by_gkh = false;
    static bool g_mpi_atexit_registered = false;
    static bool g_stdout_redirected = false;

    // ===== Profiling 统计变量 =====
    // 这些变量在每一次 gkh_svd_from_bidiagonal() 调用开始时清零。
    // 统计范围：MPI 任务下发时的完整矩阵同步、worker 回传 patch、
    // worker 临时 patch 缓冲区申请量，以及通信/拷贝相关时间。
    static long long g_local_full_sync_count = 0;
    static long long g_local_full_state_bytes = 0;
    static long long g_local_patch_count = 0;
    static long long g_local_patch_bytes = 0;
    static long long g_local_temp_alloc_bytes = 0;
    static double g_local_comm_time = 0.0;
    static double g_local_copy_time = 0.0;

    static void reset_profile_counters()
    {
        g_local_full_sync_count = 0;
        g_local_full_state_bytes = 0;
        g_local_patch_count = 0;
        g_local_patch_bytes = 0;
        g_local_temp_alloc_bytes = 0;
        g_local_comm_time = 0.0;
        g_local_copy_time = 0.0;
    }

    static long long matrix_bytes(const Matrix &M)
    {
        return static_cast<long long>(M.rows()) * static_cast<long long>(M.cols()) *
               static_cast<long long>(sizeof(double));
    }

    static void finalize_mpi_at_exit()
    {
        int mpi_initialized = 0;
        int mpi_finalized = 0;

        MPI_Initialized(&mpi_initialized);
        MPI_Finalized(&mpi_finalized);

        if (mpi_initialized && !mpi_finalized && g_mpi_initialized_by_gkh)
        {
            MPI_Finalize();
        }
    }

    static void ensure_mpi_ready_and_control_output(int &rank, int &size)
    {
        int mpi_initialized = 0;
        int mpi_finalized = 0;

        MPI_Initialized(&mpi_initialized);
        MPI_Finalized(&mpi_finalized);

        if (mpi_finalized)
        {
            std::fprintf(stderr, "Error: MPI has already been finalized before gkh_svd_from_bidiagonal.\n");
            std::abort();
        }

        if (!mpi_initialized)
        {
            int argc = 0;
            char **argv = nullptr;
            MPI_Init(&argc, &argv);

            g_mpi_initialized_by_gkh = true;

            if (!g_mpi_atexit_registered)
            {
                std::atexit(finalize_mpi_at_exit);
                g_mpi_atexit_registered = true;
            }
        }

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // main.cpp 不能修改时，mpirun -np N 会导致 N 个进程都执行 cout。
        // 这里把非 rank 0 的 stdout 重定向到 /dev/null，避免多进程输出交叉混乱。
        // stderr 不重定向，因为负载统计 [mpi taskpool gkh load] 使用 stderr 输出。
        if (rank != 0 && !g_stdout_redirected)
        {
            std::fflush(stdout);
            freopen("/dev/null", "w", stdout);
            g_stdout_redirected = true;
        }
    }

    // 在 main() 开始前自动初始化 MPI 并关闭非 rank 0 的 stdout。
    // 这样即使 main.cpp 中第一行就 cout，也不会出现多进程重复输出。
    struct MpiAutoBootstrap
    {
        MpiAutoBootstrap()
        {
            int rank = 0;
            int size = 1;
            ensure_mpi_ready_and_control_output(rank, size);
        }
    };

    static MpiAutoBootstrap g_mpi_auto_bootstrap;


    // 活动块 [l, r]（闭区间）表示一个尚未完全收敛的上二对角子问题。
    // 在该区间内，超对角线元素非零，你可以认为通过这个抽象结构给矩阵“分块”。
    struct Block
    {
        int l;
        int r;
    };

    // 这些函数在后面定义，这里先声明，避免前面的 MPI 任务池函数调用时报“未声明”
static void cleanup_bidiagonal(Matrix &B, double tol);

static std::vector<Block> split_active_blocks(Matrix &B, int n, double tol);

static std::vector<Block> split_active_blocks_in_range(Matrix &B, int l, int r, double tol);

static void one_block_step(Matrix &U, Matrix &B, Matrix &V, int l, int r);

    // ===== MPI 主从式任务池版本参数与辅助函数 =====
    // 目标：满足“主进程管理任务池，工作进程空闲后动态领取任务”的要求。
    // rank 0 作为 master，只负责任务分发、结果回收和矩阵同步；
    // rank > 0 作为 worker，收到一个活动块 [l,r] 后执行 one_block_step，
    // 然后只把该活动块实际影响的 U/V 列块与 B 子块回传给 master。
    // 注意：结果头、耗时、U/B/V patch 使用不同 tag，避免同源连续消息在调试和维护时混淆。

    enum MpiTag
    {
        TAG_TASK = 100,
        TAG_STOP_ROUND = 101,
        TAG_RESULT_HEADER = 102,
        TAG_RESULT_TIME = 103,
        TAG_RESULT_U = 104,
        TAG_RESULT_B = 105,
        TAG_RESULT_V = 106,
        TAG_RESULT_BLOCKS = 107,
        TAG_STATE_U_DIMS = 108,
        TAG_STATE_U_DATA = 109,
        TAG_STATE_B_DIMS = 110,
        TAG_STATE_B_DATA = 111,
        TAG_STATE_V_DIMS = 112,
        TAG_STATE_V_DATA = 113
    };

    static long long block_work(const Block &blk)
    {
        const int len = blk.r - blk.l + 1;
        return 1LL * len * len;
    }

    static std::vector<double> flatten_matrix(const Matrix &M)
    {
        std::vector<double> x(M.rows() * M.cols());
        for (int i = 0; i < M.rows(); ++i)
        {
            for (int j = 0; j < M.cols(); ++j)
            {
                x[i * M.cols() + j] = M.at(i, j);
            }
        }
        return x;
    }

    static void fill_matrix_from_flat(Matrix &M, const std::vector<double> &buf)
    {
        for (int i = 0; i < M.rows(); ++i)
        {
            for (int j = 0; j < M.cols(); ++j)
            {
                M.at(i, j) = buf[i * M.cols() + j];
            }
        }
    }

    static void broadcast_matrix_from_rank0(Matrix &M)
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int dims[2] = {M.rows(), M.cols()};
        MPI_Bcast(dims, 2, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank != 0)
        {
            M = Matrix(dims[0], dims[1], 0.0);
        }

        std::vector<double> buf;
        if (rank == 0)
        {
            buf = flatten_matrix(M);
        }
        else
        {
            buf.assign(dims[0] * dims[1], 0.0);
        }

        MPI_Bcast(buf.data(),
                  static_cast<int>(buf.size()),
                  MPI_DOUBLE,
                  0,
                  MPI_COMM_WORLD);

        fill_matrix_from_flat(M, buf);
    }

    static void send_matrix_to_worker(int dest, const Matrix &M, int dims_tag, int data_tag)
    {
        int dims[2] = {M.rows(), M.cols()};
        std::vector<double> buf = flatten_matrix(M);

        MPI_Send(dims, 2, MPI_INT, dest, dims_tag, MPI_COMM_WORLD);
        MPI_Send(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, dest, data_tag, MPI_COMM_WORLD);
    }

    static void recv_matrix_from_master(Matrix &M, int dims_tag, int data_tag)
    {
        int dims[2] = {0, 0};
        MPI_Recv(dims, 2, MPI_INT, 0, dims_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        M = Matrix(dims[0], dims[1], 0.0);
        std::vector<double> buf(dims[0] * dims[1]);
        MPI_Recv(buf.data(), static_cast<int>(buf.size()), MPI_DOUBLE, 0, data_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        fill_matrix_from_flat(M, buf);
    }

    static void send_full_state_to_worker(int dest, const Matrix &U, const Matrix &B, const Matrix &V)
    {
        send_matrix_to_worker(dest, U, TAG_STATE_U_DIMS, TAG_STATE_U_DATA);
        send_matrix_to_worker(dest, B, TAG_STATE_B_DIMS, TAG_STATE_B_DATA);
        send_matrix_to_worker(dest, V, TAG_STATE_V_DIMS, TAG_STATE_V_DATA);
    }

    static void recv_full_state_from_master(Matrix &U, Matrix &B, Matrix &V)
    {
        recv_matrix_from_master(U, TAG_STATE_U_DIMS, TAG_STATE_U_DATA);
        recv_matrix_from_master(B, TAG_STATE_B_DIMS, TAG_STATE_B_DATA);
        recv_matrix_from_master(V, TAG_STATE_V_DIMS, TAG_STATE_V_DATA);
    }

    // 提取矩阵 M 的连续列 [l,r]，按“行优先 + 列偏移”存储。
    static std::vector<double> extract_cols(const Matrix &M, int l, int r)
    {
        const int rows = M.rows();
        const int len = r - l + 1;
        std::vector<double> patch(rows * len);

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < len; ++j)
            {
                patch[i * len + j] = M.at(i, l + j);
            }
        }
        return patch;
    }

    static void apply_cols(Matrix &M, int l, int r, const std::vector<double> &patch)
    {
        const int rows = M.rows();
        const int len = r - l + 1;

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < len; ++j)
            {
                M.at(i, l + j) = patch[i * len + j];
            }
        }
    }

    // 对 B 来说，一个活动块 [l,r] 的 bulge chasing 只应影响该块内部。
    // 这里回传 B[l:r, l:r]，避免每个 worker 都发送完整 B。
    static std::vector<double> extract_square_block(const Matrix &M, int l, int r)
    {
        const int len = r - l + 1;
        std::vector<double> patch(len * len);

        for (int i = 0; i < len; ++i)
        {
            for (int j = 0; j < len; ++j)
            {
                patch[i * len + j] = M.at(l + i, l + j);
            }
        }
        return patch;
    }

    static void apply_square_block(Matrix &M, int l, int r, const std::vector<double> &patch)
    {
        const int len = r - l + 1;

        for (int i = 0; i < len; ++i)
        {
            for (int j = 0; j < len; ++j)
            {
                M.at(l + i, l + j) = patch[i * len + j];
            }
        }
    }

    static void send_block_task(int dest, const Block &blk, int max_steps_for_task, const Matrix &U, const Matrix &B, const Matrix &V)
    {
        int msg[3] = {blk.l, blk.r, max_steps_for_task};

        const double t_comm0 = MPI_Wtime();
        MPI_Send(msg, 3, MPI_INT, dest, TAG_TASK, MPI_COMM_WORLD);

        // 为了在不修改 main.cpp 的前提下保证 worker 每次领取到的都是 master 最新状态，
        // 每个任务下发时同步一次完整 U/B/V。这样通信量较大，但逻辑最稳，便于满足任务池要求。
        send_full_state_to_worker(dest, U, B, V);
        const double t_comm1 = MPI_Wtime();

        g_local_comm_time += (t_comm1 - t_comm0);
        g_local_full_sync_count++;
        g_local_full_state_bytes += matrix_bytes(U) + matrix_bytes(B) + matrix_bytes(V);
    }

    static void send_stop_round(int dest)
    {
        int msg[3] = {-1, -1, 0};
        MPI_Send(msg, 3, MPI_INT, dest, TAG_STOP_ROUND, MPI_COMM_WORLD);
    }

    static void send_result_patch(int dest,
                                  const Block &blk,
                                  double elapsed,
                                  int steps,
                                  const std::vector<Block> &new_blocks,
                                  const Matrix &U,
                                  const Matrix &B,
                                  const Matrix &V)
    {
        const int len = blk.r - blk.l + 1;
        int header[5] = {blk.l, blk.r, len, static_cast<int>(new_blocks.size()), steps};

        const double t_copy0 = MPI_Wtime();
        std::vector<double> u_patch = extract_cols(U, blk.l, blk.r);
        std::vector<double> b_patch = extract_square_block(B, blk.l, blk.r);
        std::vector<double> v_patch = extract_cols(V, blk.l, blk.r);
        const double t_copy1 = MPI_Wtime();

        g_local_copy_time += (t_copy1 - t_copy0);

        const long long one_patch_bytes =
            static_cast<long long>(u_patch.size()) * static_cast<long long>(sizeof(double)) +
            static_cast<long long>(b_patch.size()) * static_cast<long long>(sizeof(double)) +
            static_cast<long long>(v_patch.size()) * static_cast<long long>(sizeof(double));
        g_local_patch_count++;
        g_local_patch_bytes += one_patch_bytes;
        g_local_temp_alloc_bytes += one_patch_bytes;

        std::vector<int> block_buf;
        block_buf.reserve(new_blocks.size() * 2);
        for (const Block &b : new_blocks)
        {
            block_buf.push_back(b.l);
            block_buf.push_back(b.r);
        }

        const double t_send0 = MPI_Wtime();
        MPI_Send(header, 5, MPI_INT, dest, TAG_RESULT_HEADER, MPI_COMM_WORLD);
        MPI_Send(&elapsed, 1, MPI_DOUBLE, dest, TAG_RESULT_TIME, MPI_COMM_WORLD);
        MPI_Send(u_patch.data(), static_cast<int>(u_patch.size()), MPI_DOUBLE, dest, TAG_RESULT_U, MPI_COMM_WORLD);
        MPI_Send(b_patch.data(), static_cast<int>(b_patch.size()), MPI_DOUBLE, dest, TAG_RESULT_B, MPI_COMM_WORLD);
        MPI_Send(v_patch.data(), static_cast<int>(v_patch.size()), MPI_DOUBLE, dest, TAG_RESULT_V, MPI_COMM_WORLD);
        if (!block_buf.empty())
        {
            MPI_Send(block_buf.data(), static_cast<int>(block_buf.size()), MPI_INT, dest, TAG_RESULT_BLOCKS, MPI_COMM_WORLD);
        }
        const double t_send1 = MPI_Wtime();
        g_local_comm_time += (t_send1 - t_send0);
    }

    static Block recv_result_patch(int source,
                                   double &elapsed,
                                   int &steps,
                                   std::vector<Block> &new_blocks,
                                   Matrix &U,
                                   Matrix &B,
                                   Matrix &V)
    {
        int header[5] = {0, 0, 0, 0, 0};
        MPI_Recv(header, 5, MPI_INT, source, TAG_RESULT_HEADER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&elapsed, 1, MPI_DOUBLE, source, TAG_RESULT_TIME, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        Block blk{header[0], header[1]};
        const int len = header[2];
        const int new_count = header[3];
        steps = header[4];

        std::vector<double> u_patch(U.rows() * len);
        std::vector<double> b_patch(len * len);
        std::vector<double> v_patch(V.rows() * len);

        const long long recv_patch_bytes =
            static_cast<long long>(u_patch.size()) * static_cast<long long>(sizeof(double)) +
            static_cast<long long>(b_patch.size()) * static_cast<long long>(sizeof(double)) +
            static_cast<long long>(v_patch.size()) * static_cast<long long>(sizeof(double));
        g_local_temp_alloc_bytes += recv_patch_bytes;

        const double t_recv0 = MPI_Wtime();
        MPI_Recv(u_patch.data(), static_cast<int>(u_patch.size()), MPI_DOUBLE, source, TAG_RESULT_U, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b_patch.data(), static_cast<int>(b_patch.size()), MPI_DOUBLE, source, TAG_RESULT_B, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(v_patch.data(), static_cast<int>(v_patch.size()), MPI_DOUBLE, source, TAG_RESULT_V, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        const double t_recv1 = MPI_Wtime();
        g_local_comm_time += (t_recv1 - t_recv0);

        const double t_apply0 = MPI_Wtime();
        apply_cols(U, blk.l, blk.r, u_patch);
        apply_square_block(B, blk.l, blk.r, b_patch);
        apply_cols(V, blk.l, blk.r, v_patch);
        const double t_apply1 = MPI_Wtime();
        g_local_copy_time += (t_apply1 - t_apply0);

        new_blocks.clear();
        if (new_count > 0)
        {
            std::vector<int> block_buf(new_count * 2);
            MPI_Recv(block_buf.data(), static_cast<int>(block_buf.size()), MPI_INT, source, TAG_RESULT_BLOCKS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < new_count; ++i)
            {
                Block b{block_buf[2 * i], block_buf[2 * i + 1]};
                if (b.r > b.l)
                {
                    new_blocks.push_back(b);
                }
            }
        }

        return blk;
    }

    static std::vector<Block> process_block_until_resplit(Matrix &U,
                                                          Matrix &B,
                                                          Matrix &V,
                                                          const Block &blk,
                                                          double tol,
                                                          int max_steps_for_task,
                                                          int &steps_done);

    // 新迭代逻辑下的 MPI 主从式任务池：
    // rank 0 不再按“全局扫描一轮 -> 每块只做一次 one_block_step -> 下一轮再扫描”的方式推进，
    // 而是维护一个持续变化的子矩阵池。worker 领取一个非 1x1 活动块后，会在该块内部
    // 连续执行 bulge chase，直到该块可以再次分裂；分裂出的非平凡子块再被 master 放回任务池。
    static bool mpi_master_worker_run_block_pool(Matrix &U,
                                                 Matrix &B,
                                                 Matrix &V,
                                                 const std::vector<Block> &initial_blocks,
                                                 int max_iter,
                                                 double tol,
                                                 int rank,
                                                 int size,
                                                 long long &local_tasks,
                                                 long long &local_work,
                                                 double &local_compute_time)
    {
        std::vector<Block> task_pool;
        for (int i = static_cast<int>(initial_blocks.size()) - 1; i >= 0; --i)
        {
            if (initial_blocks[i].r > initial_blocks[i].l)
            {
                task_pool.push_back(initial_blocks[i]);
            }
        }

        int global_steps = 0;

        // 单进程退化：仍使用同样的“子矩阵池”迭代逻辑，只是不走 MPI 通信。
        if (size == 1)
        {
            while (!task_pool.empty() && global_steps < max_iter)
            {
                Block blk = task_pool.back();
                task_pool.pop_back();

                int budget = std::max(1, max_iter - global_steps);
                int steps = 0;
                const double t0 = MPI_Wtime();
                std::vector<Block> new_blocks = process_block_until_resplit(U, B, V, blk, tol, budget, steps);
                const double t1 = MPI_Wtime();

                local_tasks++;
                local_work += block_work(blk) * std::max(1, steps);
                local_compute_time += (t1 - t0);
                global_steps += steps;

                for (int i = static_cast<int>(new_blocks.size()) - 1; i >= 0; --i)
                {
                    if (new_blocks[i].r > new_blocks[i].l)
                    {
                        task_pool.push_back(new_blocks[i]);
                    }
                }
            }
            return task_pool.empty();
        }

        if (rank == 0)
        {
            int active_workers = 0;

            auto dispatch_one = [&](int worker) -> bool
            {
                if (task_pool.empty() || global_steps >= max_iter)
                {
                    send_stop_round(worker);
                    return false;
                }

                Block blk = task_pool.back();
                task_pool.pop_back();

                int budget = std::max(1, max_iter - global_steps);
                send_block_task(worker, blk, budget, U, B, V);
                return true;
            };

            for (int p = 1; p < size; ++p)
            {
                if (dispatch_one(p))
                {
                    active_workers++;
                }
            }

            while (active_workers > 0)
            {
                MPI_Status status;
                MPI_Probe(MPI_ANY_SOURCE, TAG_RESULT_HEADER, MPI_COMM_WORLD, &status);
                const int source = status.MPI_SOURCE;

                double elapsed = 0.0;
                int steps = 0;
                std::vector<Block> new_blocks;
                Block done_blk = recv_result_patch(source, elapsed, steps, new_blocks, U, B, V);

                global_steps += steps;
                (void)done_blk;
                (void)elapsed;

                // worker 已经在该块内连续 chase 到再次分裂。把非 1x1 的新子块放回任务池。
                for (int i = static_cast<int>(new_blocks.size()) - 1; i >= 0; --i)
                {
                    if (new_blocks[i].r > new_blocks[i].l)
                    {
                        task_pool.push_back(new_blocks[i]);
                    }
                }

                if (!dispatch_one(source))
                {
                    active_workers--;
                }
            }
        }
        else
        {
            while (true)
            {
                MPI_Status status;
                int msg[3] = {0, 0, 0};
                MPI_Recv(msg, 3, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                if (status.MPI_TAG == TAG_STOP_ROUND)
                {
                    break;
                }

                Block blk{msg[0], msg[1]};
                const int max_steps_for_task = std::max(1, msg[2]);

                // 每个任务开始前接收 master 的最新矩阵状态，避免 worker 使用过期状态。
                recv_full_state_from_master(U, B, V);

                int steps = 0;
                const double t0 = MPI_Wtime();
                std::vector<Block> new_blocks = process_block_until_resplit(U, B, V, blk, tol, max_steps_for_task, steps);
                const double t1 = MPI_Wtime();

                local_tasks++;
                local_work += block_work(blk) * std::max(1, steps);
                local_compute_time += (t1 - t0);

                send_result_patch(0, blk, t1 - t0, steps, new_blocks, U, B, V);
            }
        }

        // master 已经合并所有完成任务的 patch。广播一次最终 U/B/V，保证所有 rank 状态一致。
        broadcast_matrix_from_rank0(U);
        broadcast_matrix_from_rank0(B);
        broadcast_matrix_from_rank0(V);

        int local_converged = 0;
        if (rank == 0)
        {
            cleanup_bidiagonal(B, tol);
            std::vector<Block> final_blocks = split_active_blocks(B, B.cols(), tol);
            local_converged = 1;
            for (const Block &b : final_blocks)
            {
                if (b.r > b.l)
                {
                    local_converged = 0;
                    break;
                }
            }
        }

        MPI_Bcast(&local_converged, 1, MPI_INT, 0, MPI_COMM_WORLD);
        return local_converged == 1;
    }

    // 对矩阵 M 的两行 r0, r1 左乘 Givens 旋转 [c s; -s c]。
    // 即 M <- L * M，其中 L 只作用在第 r0/r1 两行上。
    // 这类逐元素线性组合很适合向量化，SIMD/多线程中你也可以顺手的事把他们做了。
    static void apply_left_rows(Matrix &M, int r0, int r1, double c, double s)
    {
        const int cols = M.cols();
    double* row0 = &M.at(r0, 0);
    double* row1 = &M.at(r1, 0);

    float64x2_t vc  = vdupq_n_f64(c);
    float64x2_t vs  = vdupq_n_f64(s);
    float64x2_t vns = vdupq_n_f64(-s);

    int j = 0;
    for (; j + 2 <= cols; j += 2)
    {
        float64x2_t va = vld1q_f64(row0 + j);  // [a0, a1]
        float64x2_t vb = vld1q_f64(row1 + j);  // [b0, b1]

        // new_row0 = c*a + s*b
        float64x2_t new_a = vfmaq_f64(vmulq_f64(vs, vb), vc, va);

        // new_row1 = -s*a + c*b
        float64x2_t new_b = vfmaq_f64(vmulq_f64(vc, vb), vns, va);

        vst1q_f64(row0 + j, new_a);
        vst1q_f64(row1 + j, new_b);
    }

    // 尾部奇数列
    if (j < cols)
    {
        double a = row0[j];
        double b = row1[j];
        row0[j] =  c * a + s * b;
        row1[j] = -s * a + c * b;
    }
    }

    // 对矩阵 M 的两列 c0, c1 右乘 Givens 旋转 [c s; -s c]。
    // 即 M <- M * R，其中 R 只作用在第 c0/c1 两列上。
    static void apply_right_cols(Matrix &M, int c0, int c1, double c, double s)
    {
            const int rows = M.rows();
    const float64x2_t vc  = vdupq_n_f64(c);
    const float64x2_t vs  = vdupq_n_f64(s);
    const float64x2_t vns = vdupq_n_f64(-s);

    int i = 0;
    for (; i <= rows - 2; i += 2)
    {
        float64x2_t va = {M.at(i, c0), M.at(i+1, c0)};  // [M(i,c0),   M(i+1,c0)]
        float64x2_t vb = {M.at(i, c1), M.at(i+1, c1)};  // [M(i,c1),   M(i+1,c1)]

        // new_c0 = a*c - b*s
        float64x2_t new_c0 = vfmaq_f64(vmulq_f64(va, vc), vns, vb);  // 用 -s 变减为加
        // new_c1 = a*s + b*c
        float64x2_t new_c1 = vfmaq_f64(vmulq_f64(va, vs), vc, vb);
        // scatter 写回：同样不连续，逐元素写
        M.at(i,   c0) = vgetq_lane_f64(new_c0, 0);
        M.at(i+1, c0) = vgetq_lane_f64(new_c0, 1);
        M.at(i,   c1) = vgetq_lane_f64(new_c1, 0);
        M.at(i+1, c1) = vgetq_lane_f64(new_c1, 1);
    }
    // 尾部奇数行
    for (; i < rows; ++i)
    {
        double a = M.at(i, c0);
        double b = M.at(i, c1);
        M.at(i, c0) = a * c - b * s;
        M.at(i, c1) = a * s + b * c;
    }
    }
    static void accumulate_left_into_U(Matrix &U, int r0, int r1, double c, double s)
    {
        // 我们该怎样积累 U 和 V 的更新呢？
        // 以此处 U 的积累为例，让我们B <- L * B 时，我们必须维护的等式是 A = U * B * V^T
        // 如果 A = U * B * V^T 不成立，那么我们最终的SVD结果显然不是 A 的正确分解。
        // 由于正交矩阵和其转置的乘积是I，一个自然的想法是让 U <  // 这样就变成 A = (U * L^T) * (L * B) * V^T = U * B * V^T，等式得以保持。

        // 由于 L^T = [c -s; s c]，此处复用“右乘两列”接口并传入 -s。
        apply_right_cols(U, r0, r1, c, -s);
    }

    // 计算活动块 [l, r] 对应 B^T B 右下 2x2 主子块的 Wilkinson 偏移。
    // 偏移用于加速 QR 迭代收敛，并让 bulge chasing 过程更稳定。
    static double block_wilkinson_shift(const Matrix &B, int l, int r)
    {
        if (r == l)
        {
            return B.at(l, l) * B.at(l, l);
        }

        const double d1 = B.at(r - 1, r - 1);
        const double e1 = B.at(r - 1, r);
        const double d2 = B.at(r, r);
        const double e0 = (r - 1 > l) ? B.at(r - 2, r - 1) : 0.0;

        const double a = d1 * d1 + e0 * e0;
        const double b = d1 * e1;
        const double d = d2 * d2 + e1 * e1;

        const double tr = a + d;
        const double det = a * d - b * b;
        double disc = 0.25 * tr * tr - det;
        if (disc < 0.0)
        {
            disc = 0.0;
        }

        const double root = std::sqrt(disc);
        const double lam1 = 0.5 * tr + root;
        const double lam2 = 0.5 * tr - root;
        return (std::fabs(lam1 - d) <= std::fabs(lam2 - d)) ? lam1 : lam2;
    }

    // 将上二对角结构以外、且绝对值很小的元素强制置零。
    static void cleanup_bidiagonal(Matrix &B, double tol)
    {
        for (int i = 0; i < B.rows(); ++i)
        {
            for (int j = 0; j < B.cols(); ++j)
            {
                if (j != i && j != i + 1 && std::fabs(B.at(i, j)) <= tol)
                {
                    B.at(i, j) = 0.0;
                }
            }
        }
    }

    // 对活动块 [l, r] 执行一次“单块 GKH bulge chasing”迭代。
    // 流程：首次右乘引入 bulge -> 首次左乘消 bulge -> 交替右乘/左乘将 bulge 追赶到块末端。
    static void one_block_step(Matrix &U, Matrix &B, Matrix &V, int l, int r)
    {
        if (r <= l)
        {
            return;
        }

        const double mu = block_wilkinson_shift(B, l, r);

        double c = 1.0;
        double s = 0.0;
        double rr = 0.0;

        // 首次右乘：由 (d_l^2-mu, d_l*e_l) 构造。
        const double x = B.at(l, l) * B.at(l, l) - mu;
        const double z = B.at(l, l) * B.at(l, l + 1);
        givens_rotation(x, z, c, s, rr, false);
        apply_right_cols(B, l, l + 1, c, s);
        apply_right_cols(V, l, l + 1, c, s);

        // 首次左乘：消去 (l+1, l)。
        givens_rotation(B.at(l, l), B.at(l + 1, l), c, s, rr, true);
        apply_left_rows(B, l, l + 1, c, s);
        accumulate_left_into_U(U, l, l + 1, c, s);

        for (int k = l + 1; k <= r - 1; ++k)
        {
            // 右乘：消去 (k-1, k+1)
            givens_rotation(B.at(k - 1, k), B.at(k - 1, k + 1), c, s, rr, false);
            apply_right_cols(B, k, k + 1, c, s);
            apply_right_cols(V, k, k + 1, c, s);

            // 左乘：消去 (k+1, k)
            givens_rotation(B.at(k, k), B.at(k + 1, k), c, s, rr, true);
            apply_left_rows(B, k, k + 1, c, s);
            accumulate_left_into_U(U, k, k + 1, c, s);
        }
    }

    // 处理“对角元 d_k 近零但超对角 e_k 未近零”的情况。
    // 思路与单块追赶类似：先右乘把 e_i 消掉，再左乘清理新引入的次对角 bulge，
    // 把这个问题逐步向右传递，直到块末端。
    static bool chase_zero_diagonal(Matrix &U, Matrix &B, Matrix &V, int k, double tol)
    {
        const int m = B.rows();
        const int n = B.cols();
        if (k < 0 || k >= n - 1)
        {
            return false;
        }

        // d_k ~ 0 且 e_k 还未收敛时，按 lim_1 思路进行压缩追赶：
        // 1) 右乘消去第 k 行的 e_k；2) 左乘消去引入的次对角 bulge；
        // 然后把问题传递到下一行，直到末端。
        if (std::fabs(B.at(k, k + 1)) <= tol)
        {
            return false;
        }

        bool changed = false;
        for (int i = k; i <= n - 2; ++i)
        {
            double c = 1.0;
            double s = 0.0;
            double rr = 0.0;

            // 右乘：使第 i 行满足 [d_i, e_i] * G = [r, 0]。
            givens_rotation(B.at(i, i), B.at(i, i + 1), c, s, rr, false);
            apply_right_cols(B, i, i + 1, c, s);
            apply_right_cols(V, i, i + 1, c, s);

            // 左乘：消去 (i+1, i) 处由右乘引入的 bulge。
            if (i + 1 < m)
            {
                givens_rotation(B.at(i, i), B.at(i + 1, i), c, s, rr, true);
                apply_left_rows(B, i, i + 1, c, s);
                accumulate_left_into_U(U, i, i + 1, c, s);
            }

            changed = true;
        }

        cleanup_bidiagonal(B, tol);
        return changed;
    }

    // 扫描所有 d_k≈0 的位置：若对应 e_k 仍显著非零，则调用追赶过程压缩该异常结构。
    // 返回值表示本轮是否对 B/U/V 做了实际更新。
    static bool handle_diagonal_zeros(Matrix &U, Matrix &B, Matrix &V, double tol)
    {
        const int n = B.cols();
        bool changed = false;

        const double eps = std::numeric_limits<double>::epsilon();
        const double diag_tol = tol;
        const double super_tol = tol * (1.0 + 10.0 * eps);

        for (int k = 0; k < n - 1; ++k)
        {
            if (std::fabs(B.at(k, k)) <= diag_tol && std::fabs(B.at(k, k + 1)) > super_tol)
            {
                if (chase_zero_diagonal(U, B, V, k, tol))
                {
                    changed = true;
                }
            }
        }

        return changed;
    }

    // 根据超对角线是否“足够小”对问题进行分块。
    // 若 |e_k| <= tol*(|d_k|+|d_{k+1}|+1)，认为该位置可解耦并直接置零。
    // 最终会得到一系列小矩阵。
    static std::vector<Block> split_active_blocks(Matrix &B, int n, double tol)
    {
        for (int k = 0; k < n - 1; ++k)
        {
            const double a = std::fabs(B.at(k, k));
            const double d = std::fabs(B.at(k + 1, k + 1));
            const double crit = tol * (a + d + 1.0);
            if (std::fabs(B.at(k, k + 1)) <= crit)
            {
                B.at(k, k + 1) = 0.0;
            }
        }

        std::vector<Block> blocks;
        int l = 0;
        while (l < n)
        {
            int r = l;
            while (r < n - 1 && std::fabs(B.at(r, r + 1)) > 0.0)
            {
                ++r;
            }
            blocks.push_back({l, r});
            l = r + 1;
        }
        return blocks;
    }

    static std::vector<Block> split_active_blocks_in_range(Matrix &B, int l, int r, double tol)
    {
        if (r <= l)
        {
            return std::vector<Block>{Block{l, r}};
        }

        for (int k = l; k < r; ++k)
        {
            const double a = std::fabs(B.at(k, k));
            const double d = std::fabs(B.at(k + 1, k + 1));
            const double crit = tol * (a + d + 1.0);
            if (std::fabs(B.at(k, k + 1)) <= crit)
            {
                B.at(k, k + 1) = 0.0;
            }
        }

        std::vector<Block> blocks;
        int cur_l = l;
        while (cur_l <= r)
        {
            int cur_r = cur_l;
            while (cur_r < r && std::fabs(B.at(cur_r, cur_r + 1)) > 0.0)
            {
                ++cur_r;
            }
            blocks.push_back({cur_l, cur_r});
            cur_l = cur_r + 1;
        }
        return blocks;
    }

    static std::vector<Block> keep_nontrivial_blocks(const std::vector<Block> &blocks)
    {
        std::vector<Block> out;
        for (const Block &b : blocks)
        {
            if (b.r > b.l)
            {
                out.push_back(b);
            }
        }
        return out;
    }

    static bool same_single_block(const std::vector<Block> &blocks, const Block &blk)
    {
        return blocks.size() == 1 && blocks[0].l == blk.l && blocks[0].r == blk.r;
    }

    static std::vector<Block> process_block_until_resplit(Matrix &U,
                                                          Matrix &B,
                                                          Matrix &V,
                                                          const Block &blk,
                                                          double tol,
                                                          int max_steps_for_task,
                                                          int &steps_done)
    {
        steps_done = 0;

        if (blk.r <= blk.l)
        {
            return {};
        }

        cleanup_bidiagonal(B, tol);
        std::vector<Block> before = split_active_blocks_in_range(B, blk.l, blk.r, tol);
        if (!same_single_block(before, blk))
        {
            return keep_nontrivial_blocks(before);
        }

        // 7.1.2 要求：取出一个非 1x1 子矩阵后，不只做一轮 bulge chase，
        // 而是在该子矩阵上反复 chase，直到它能够再次分裂。为避免极端情况下死循环，
        // 这里仍受 max_iter 传入的步数预算约束；若预算耗尽仍未分裂，则把原块放回任务池。
        while (steps_done < max_steps_for_task)
        {
            one_block_step(U, B, V, blk.l, blk.r);
            ++steps_done;

            cleanup_bidiagonal(B, tol);
            std::vector<Block> after = split_active_blocks_in_range(B, blk.l, blk.r, tol);
            if (!same_single_block(after, blk))
            {
                return keep_nontrivial_blocks(after);
            }
        }

        // 没有在本次预算内分裂，交回 master 重新入池；这样不会破坏全局 max_iter 限制。
        return std::vector<Block>{blk};
    }

    // 收尾步骤：
    // 1) 把奇异值（对角元）统一调整为非负；
    // 2) 按降序重排奇异值，同时同步重排 U、V 对应列。
    // 最终得到常见的 SVD 规范形式：sigma_1 >= sigma_2 >= ... >= 0。
    // 这个函数你不用太在意，后续任务也不会明确涉及它。
    static void make_nonnegative_and_sort(Matrix &U, Matrix &B, Matrix &V)
    {
        const int m = B.rows();
        const int n = B.cols();

        for (int i = 0; i < n; ++i)
        {
            if (B.at(i, i) < 0.0)
            {
                B.at(i, i) = -B.at(i, i);
                for (int r = 0; r < m; ++r)
                {
                    U.at(r, i) = -U.at(r, i);
                }
            }
        }

        std::vector<int> idx(n);
        for (int i = 0; i < n; ++i)
        {
            idx[i] = i;
        }
        std::sort(idx.begin(), idx.end(), [&](int a, int b)
                  { return B.at(a, a) > B.at(b, b); });

        Matrix U2 = U;
        Matrix V2 = V;
        Matrix D(B.rows(), B.cols(), 0.0);

        for (int new_i = 0; new_i < n; ++new_i)
        {
            const int old_i = idx[new_i];
            D.at(new_i, new_i) = B.at(old_i, old_i);

            for (int r = 0; r < U.rows(); ++r)
            {
                U2.at(r, new_i) = U.at(r, old_i);
            }
            for (int r = 0; r < V.rows(); ++r)
            {
                V2.at(r, new_i) = V.at(r, old_i);
            }
        }

        U = U2;
        V = V2;
        B = D;
    }

} // namespace

// 从“上二对角矩阵 B”出发执行 Golub-Kahan SVD 迭代（MPI 主从式任务池版）：
// - 不要求 main.cpp 显式调用 MPI_Init；如果 main.cpp 没调用，本函数会自动初始化 MPI；
// - rank 0 是 master，维护任务池并动态分发活动块；
// - rank > 0 是 worker，空闲后从 master 领取任务；
// - worker 只回传本活动块影响到的 U/V 列块和 B 子块，避免每轮对完整 U/B/V 做 Allreduce；
// - 每轮结束时由 rank 0 广播合并后的完整 U/B/V，保证下一轮所有进程矩阵状态一致。
bool gkh_svd_from_bidiagonal(Matrix &U, Matrix &B, Matrix &V, int max_iter, double tol)
{
    int rank = 0;
    int size = 1;

    ensure_mpi_ready_and_control_output(rank, size);
    reset_profile_counters();

    // 统一初始 U/B/V。即使 main.cpp 中每个进程都生成了相同矩阵，这一步也安全；
    // 如果各进程随机数不同，这一步可以强制以 rank 0 的矩阵为准。
    broadcast_matrix_from_rank0(U);
    broadcast_matrix_from_rank0(B);
    broadcast_matrix_from_rank0(V);

    const int m = B.rows();
    const int n = B.cols();

    if (m < n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_mpi_taskpool: requires m >= n");
    }
    if (U.rows() != m || U.cols() != m)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_mpi_taskpool: U must be m x m");
    }
    if (V.rows() != n || V.cols() != n)
    {
        throw std::invalid_argument("gkh_svd_from_bidiagonal_mpi_taskpool: V must be n x n");
    }

    long long local_tasks = 0;
    long long local_work = 0;
    double local_compute_time = 0.0;

    // 只在 rank 0 上构造初始子矩阵池；随后通过任务下发把最新矩阵状态发给 worker。
    std::vector<Block> initial_blocks;
    if (rank == 0)
    {
        cleanup_bidiagonal(B, tol);
        handle_diagonal_zeros(U, B, V, tol);
        initial_blocks = split_active_blocks(B, n, tol);
    }

    // 让所有进程持有 rank 0 预处理后的初始状态。
    broadcast_matrix_from_rank0(U);
    broadcast_matrix_from_rank0(B);
    broadcast_matrix_from_rank0(V);

    if (rank != 0)
    {
        initial_blocks.clear();
    }

    bool converged = mpi_master_worker_run_block_pool(U,
                                                       B,
                                                       V,
                                                       initial_blocks,
                                                       max_iter,
                                                       tol,
                                                       rank,
                                                       size,
                                                       local_tasks,
                                                       local_work,
                                                       local_compute_time);

    long long global_tasks = 0;
    long long global_work = 0;
    double global_compute_time = 0.0;

    MPI_Reduce(&local_tasks, &global_tasks, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_work, &global_work, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_compute_time, &global_compute_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    long long global_full_sync_count = 0;
    long long global_full_state_bytes = 0;
    long long global_patch_count = 0;
    long long global_patch_bytes = 0;
    long long global_temp_alloc_bytes = 0;
    double global_comm_time = 0.0;
    double global_copy_time = 0.0;

    MPI_Reduce(&g_local_full_sync_count, &global_full_sync_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&g_local_full_state_bytes, &global_full_state_bytes, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&g_local_patch_count, &global_patch_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&g_local_patch_bytes, &global_patch_bytes, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&g_local_temp_alloc_bytes, &global_temp_alloc_bytes, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&g_local_comm_time, &global_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&g_local_copy_time, &global_copy_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

#ifdef SVD_PRINT_LOAD
    for (int p = 0; p < size; ++p)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == p)
        {
            std::fprintf(stderr,
                         "[mpi taskpool gkh load] rank %d: tasks=%lld, work=%lld, compute_time=%.6f s\n",
                         rank,
                         local_tasks,
                         local_work,
                         local_compute_time);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::fprintf(stderr,
                     "[mpi taskpool gkh total] tasks=%lld, work=%lld, sum_compute_time=%.6f s\n",
                     global_tasks,
                     global_work,
                     global_compute_time);
        std::fprintf(stderr,
                     "[mpi taskpool gkh overhead] full_sync_count=%lld, full_state=%.3f MB, "
                     "patch_count=%lld, patch=%.3f MB, temp_alloc=%.3f MB, "
                     "comm_time=%.6f s, copy_time=%.6f s\n",
                     global_full_sync_count,
                     global_full_state_bytes / 1024.0 / 1024.0,
                     global_patch_count,
                     global_patch_bytes / 1024.0 / 1024.0,
                     global_temp_alloc_bytes / 1024.0 / 1024.0,
                     global_comm_time,
                     global_copy_time);
    }
#endif

    // 收尾阶段以 rank 0 的结果为准，其他 rank 同步后进行同样的检查。
    if (rank == 0)
    {
        cleanup_bidiagonal(B, tol);
        for (int i = 0; i < n - 1; ++i)
        {
            B.at(i, i + 1) = 0.0;
        }
        make_nonnegative_and_sort(U, B, V);
    }

    broadcast_matrix_from_rank0(U);
    broadcast_matrix_from_rank0(B);
    broadcast_matrix_from_rank0(V);

    return converged;
}
