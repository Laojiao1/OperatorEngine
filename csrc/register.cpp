// 文件职责：集中定义 Dispatcher schema、CUDA backend 注册和 C++ Tensor 契约检查。

#include <ATen/Functions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <torch/library.h>
#include <cuda_runtime_api.h>

#include <optional>
#include <ATen/ops/scaled_dot_product_attention.h>

#include "vector_add/vector_add.h"
#include "softmax/softmax.h"
#include "gemm/gemm.h"
#include "attention/attention.h"


namespace my_ops {
    namespace {
        void check_gemm_inputs(const at::Tensor& a, const at::Tensor& b, bool transpose_b) {
            TORCH_CHECK(a.is_cuda(), "gemm: a 必须是 CUDA Tensor");
            TORCH_CHECK(b.is_cuda(), "gemm: b 必须是 CUDA Tensor");
            TORCH_CHECK(a.dim() == 2, "gemm: a 必须是二维 Tensor");
            TORCH_CHECK(b.dim() == 2, "gemm: b 必须是二维 Tensor");
            TORCH_CHECK(a.device() == b.device(), "gemm: a 和 b 必须位于同一 CUDA device");
            TORCH_CHECK(a.scalar_type() == b.scalar_type(), "gemm: a 和 b 的 dtype 必须相同");
            TORCH_CHECK(a.scalar_type() == at::kFloat || a.scalar_type() == at::kHalf, "gemm: 第一版只支持 FP32 和 FP16");
            TORCH_CHECK(a.is_contiguous(), "gemm: a 必须是 contiguous Tensor");
            TORCH_CHECK(b.is_contiguous(), "gemm: b 必须是 contiguous Tensor");
            TORCH_CHECK(!a.requires_grad() && !b.requires_grad(), "gemm: 第一版不支持 autograd");

            const std::int64_t K = a.size(1);
            const std::int64_t b_k = transpose_b ? b.size(1) : b.size(0);

            TORCH_CHECK(K == b_k, "gemm: a 和 b 的 K 维度不匹配");
            TORCH_CHECK(K > 0, "gemm: 第一版要求 K 大于 0");
        }

    }  // namespace

    at::Tensor vector_add_cuda(const at::Tensor& a, const at::Tensor& b) {
        // 1. 输入契约检查
        // Dispatcher 只按 dispatch key 选择后端，不会替自定义算子验证完整输入契约。
        // 特别是混合 CPU/CUDA 输入仍可能路由到这里，因此 wrapper 必须逐项检查。
        TORCH_CHECK(a.is_cuda(), "vector_add: a must be a CUDA tensor");
        TORCH_CHECK(b.is_cuda(), "vector_add: b must be a CUDA tensor");
        TORCH_CHECK(a.device() == b.device(), "vector_add: a and b must be on the same CUDA device");
        TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat, "vector_add: a and b must have dtype torch.float32");
        TORCH_CHECK(a.sizes() == b.sizes(), "vector_add: a and b must have the same shape");
        TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "vector_add: a and b must be contiguous");
        TORCH_CHECK(!a.requires_grad() && !b.requires_grad(), "vector_add: autograd is not supported");

        // 2. 保护 CUDA 设备上下文
        // PyTorch 可能在多卡程序中把 current device 留在别处。Guard 保证分配、
        // stream 查询和 kernel launch 都发生在输入 Tensor 所在的 device。
        const c10::cuda::CUDAGuard device_guard(a.device());

        // 3. 分配输出 Tensor 内存
        at::Tensor output = at::empty_like(a);

        // 4. 空 Tensor 保护
        // CUDA 不允许启动 0-block grid；空 Tensor 仍应返回合法的空输出。
        if (a.numel() == 0) {
            return output;
        }

        // 5. 获取当前的 CUDA Stream（流）
        // 不能写死 default stream；否则会破坏调用者在自定义 stream 上建立的执行顺序。
        const c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(a.get_device());

        // 6. 调用底层 Launcher
        launch_vector_add_cuda(a.const_data_ptr<float>(), b.const_data_ptr<float>(), output.mutable_data_ptr<float>(), a.numel(), stream.stream());

        return output;
    }

    at::Tensor softmax_cuda(const at::Tensor& input, std::int64_t dim) {
        // Dispatcher 只决定进入 CUDA backend，完整契约仍由 wrapper 检查。
        TORCH_CHECK(input.is_cuda(), "softmax: input 必须是 CUDA Tensor");

        TORCH_CHECK(input.dim() >= 1, "softmax: input 至少必须是一维 Tensor");

        const std::int64_t ndim = input.dim();

        // 同时接受 Python 常用的 dim=-1 和对应的正索引。
        const std::int64_t wrapped_dim = dim < 0 ? dim + ndim : dim;

        TORCH_CHECK(wrapped_dim == ndim - 1, "softmax: 第一版只支持沿最后一维计算");

        TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf, "softmax: 第一版只支持 FP32 和 FP16");

        TORCH_CHECK(input.is_contiguous(), "softmax: input 必须是 contiguous Tensor");

        TORCH_CHECK(!input.requires_grad(), "softmax: 第一版不支持 autograd");

        // 保护 CUDA 设备上下文：保证后续输出分配、stream 查询和 kernel launch 都发生在输入 Tensor 所在的 CUDA device。
        const c10::cuda::CUDAGuard device_guard(input.device());

        // 分配输出 Tensor 内存：调用 PyTorch 的内部显存分配器（Caching Allocator），在 GPU 上开辟一块与 a 大小相同的连续显存。绝不直接调用 cudaMalloc，因为直接调用 cudaMalloc 会破坏 PyTorch 的显存缓存池，导致极高的系统调用开销。
        at::Tensor output = at::empty_like(input);

        // 空 Tensor 保护：任意维度包含 0 时，numel 都是 0，不允许启动零 block kernel。
        if (input.numel() == 0) {
            return output;
        }

        const std::int64_t N = input.size(-1);
        const std::int64_t M = input.numel() / N;

        // 获取当前的 CUDA Stream（流）：不能写死 Default Stream，否则会破坏 PyTorch 调用者已经建立的执行顺序
        const c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(input.get_device());

        // 调用底层 Launcher
        launch_softmax_cuda(input, output, M, N, stream.stream());

        return output;
    }

    at::Tensor gemm_cuda(const at::Tensor& a, const at::Tensor& b, bool transpose_b) {
        check_gemm_inputs(a, b, transpose_b);

        const c10::cuda::CUDAGuard device_guard(a.device());

        const std::int64_t M = a.size(0);
        const std::int64_t K = a.size(1);
        const std::int64_t N = transpose_b ? b.size(0) : b.size(1);

        // FP16 Tensor Core 路径也会进行 FP32 累加，因此 GEMM 第一版统一输出 FP32。
        at::Tensor output = at::empty({M, N}, a.options().dtype(at::kFloat));

        // CUDA 不允许启动包含 0 个 block 的 grid。
        if (M == 0 || N == 0) {
            return output;
        }

        const c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(a.get_device());

        launch_gemm_cuda(a, b, output, M, N, K, transpose_b, stream.stream());

        return output;
    }

    at::Tensor attention_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, bool causal) {
        TORCH_CHECK(q.is_cuda(), "attention: q 必须是 CUDA Tensor");
        TORCH_CHECK(k.is_cuda(), "attention: k 必须是 CUDA Tensor");
        TORCH_CHECK(v.is_cuda(), "attention: v 必须是 CUDA Tensor");

        TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "attention: q、k、v 必须是四维 Tensor [B, H, N, D]");
        TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "attention: q、k、v 的 shape 必须完全相同");
        TORCH_CHECK(q.device() == k.device() && q.device() == v.device(), "attention: q、k、v 必须位于同一 CUDA device");
        TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf, "attention: 第一版只支持 FP16");
        TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "attention: q、k、v 必须是 contiguous Tensor");
        TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad(), "attention: 第一版不支持 autograd");

        const std::int64_t head_dim = q.size(-1);
        TORCH_CHECK(head_dim == 64 || head_dim == 128, "attention: 第一版只支持 head_dim=64 或 128");

        const c10::cuda::CUDAGuard device_guard(q.device());

        at::Tensor output = at::empty_like(q);

        // B、H 或 N 为 0 时直接返回空输出，避免把空 shape 交给后端 kernel selector。
        if (q.numel() == 0) {
            return output;
        }

        const std::int64_t sequence_length = q.size(-2);

        if (sequence_length <= 1024) {
            const c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(q.get_device());

            // Tensor Core 第一版固定 D=64。N>=32 时至少包含一个完整 Br/Bc Tile；
            // 更短的序列暂时保留 CUDA Core，最终阈值等 benchmark 后再校准。
            if (head_dim == 64 && sequence_length >= 32) {
                launch_attention_tensor_core(q, k, v, output, causal, stream.stream());
            } else {
                launch_attention_cuda_core(q, k, v, output, causal, stream.stream());
            }

            return output;
        }

        // 不传显式 mask 和 scale：SDPA 使用 1/sqrt(D)；dropout 固定为 0。
        return at::scaled_dot_product_attention(q, k, v, std::nullopt, 0.0, causal, std::nullopt);
    }

}  // namespace my_ops


// 1. 定义算子的 Schema（签名与规范）
// 注意：这里没有任何 CPU 或 CUDA 的硬件代码，只是声明了“有这样一个接口
TORCH_LIBRARY(my_ops, library) {
    // TORCH_LIBRARY 只定义全局 schema，不绑定某个具体设备实现。
    // 没有 alias 标注表示 functional operator：返回新 Tensor，不修改输入。
    library.def("vector_add(Tensor a, Tensor b) -> Tensor");

    // Softmax 是 functional operator，输出不与 input alias。
    library.def("softmax(Tensor input, int dim) -> Tensor");

    library.def("gemm(Tensor a, Tensor b, bool transpose_b=False) -> Tensor");

    library.def("attention(Tensor q, Tensor k, Tensor v, bool causal=False) -> Tensor");
}

// 2. 将 C++ 实现函数绑定到特定硬件类型（CUDA）
TORCH_LIBRARY_IMPL(my_ops, CUDA, library) {
    // 只有输入 dispatch key 包含 CUDA 时，Dispatcher 才会选择这个 wrapper。
    library.impl("vector_add", TORCH_FN(my_ops::vector_add_cuda));

    library.impl("softmax", TORCH_FN(my_ops::softmax_cuda));

    library.impl("gemm", TORCH_FN(my_ops::gemm_cuda));

    library.impl("attention", TORCH_FN(my_ops::attention_cuda));
}

// 3. 定义 Python 模块外壳
// 声明导出一个名为 _C 的模块。虽然花括号里没有导出具体的函数，但这个宏是 Python 识别该动态库必须的入口。
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    // 算子通过 Dispatcher 暴露，不额外维护一套 pybind 函数接口。
    module.doc() = "OperatorEngine dispatcher registration library";
}
