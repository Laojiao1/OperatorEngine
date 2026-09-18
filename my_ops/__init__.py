"""文件职责：加载原生扩展与 FakeTensor 注册，并导出 OperatorEngine 的统一 Python 接口。"""

# 先加载 PyTorch 的原生共享库，否则动态加载器可能无法解析 _C 依赖的 libtorch。
import torch as _torch

# 导入动态库会执行它的模块初始化函数；后续 Dispatcher 注册也会在此时生效。
from . import _C as _C # Python 调用操作系统的 dlopen 函数加载编译好的 _C.*.so
from . import _meta as _meta
from .functional import attention


__all__ = ["_C", "attention"]
