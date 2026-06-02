"""文件功能：提供 pytest 共享夹具和后端切换辅助函数。
作者：Egor Izmaylov
时间：2026-06-02
"""

from nn import Ops


# 临时屏蔽 C 后端，让测试可以明确覆盖 Python fallback 或形状推断路径。
def _disable_c_backend(monkeypatch):
    monkeypatch.setattr(Ops, "_get_lib", classmethod(lambda cls: None))
