"""文件职责：
    配置 OperatorEngine 的 Python 包与 C++/CUDA Extension 构建。
    构建脚本，告诉编译器如何把上述 C++/CUDA 文件编译成二进制库

最终产物：
    在 Linux/WSL 系统中，编译完成后会在 my_ops/ 目录下生成一个共享库文件（动态链接库）：
    my_ops/_C.cpython-310-x86_64-linux-gnu.so
    这个 .so 文件是纯二进制机器码，里面包含了编译后的 C++ 函数和 GPU 二进制指令（SASS/PTX）
"""

from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


ROOT = Path(__file__).resolve().parent

if CUDA_HOME is None:
    raise RuntimeError(
        "OperatorEngine requires a CUDA Toolkit; CUDA_HOME was not detected."
    )

# 核心代码
setup(
    name="operator-engine",
    version="0.0.1",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension( # 这里接收两类文件：.cpp .cu
            name="my_ops._C",
            sources=[
                str(ROOT / "csrc" / "register.cpp"),
                str(ROOT / "csrc" / "vector_add" / "vector_add_cuda.cu"),
                str(ROOT / "csrc" / "softmax" / "softmax_cuda.cu"),
                str(ROOT / "csrc" / "gemm" / "gemm_cuda.cu"),
                str(ROOT / "csrc" / "attention" / "attention_cuda.cu"),
            ],
            include_dirs=[str(ROOT / "csrc")], # 包含目录：告诉编译器寻找算子的根目录 csrc
            extra_compile_args={
                "cxx": ["-O2"], # 02 编译优化
                "nvcc": ["-O3"], # 03 编译优化
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
    install_requires=["torch"],
    python_requires=">=3.10",
)
