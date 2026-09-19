"""
Week 6 测试用例（30 条，骨架）
============================
分类：正常问答 10 + 拒答场景 10 + 溯源要求 10。
跑通：python test_cases.py
说明：这是 W9 上 Langfuse（自托管）前的基线。每个用例有：query、期望（答对/拒答/带来源）、检查函数。
"""
from rag import rag

# 用例格式：{query, expect, check}
CASES = [
    # ── 正常问答 10 条（基于 week5 docs 内容设计）──
    {"query": "我的购物清单里有什么", "expect": "answer", "check": lambda a, m: bool(a) and bool(m)},
    {"query": "常用命令有哪些", "expect": "answer", "check": lambda a, m: bool(a) and bool(m)},
    {"query": "本周有什么待办", "expect": "answer", "check": lambda a, m: bool(a) and bool(m)},
    {"query": "喜欢什么样的回答风格", "expect": "answer", "check": lambda a, m: bool(a) and bool(m)},
    {"query": "买了哪些食物", "expect": "answer", "check": lambda a, m: bool(a) and bool(m)},
    {"query": "会用什么编程工具", "expect": "answer", "check": lambda a, m: bool(a) and bool(m)},
    {"query": "安排了什么健康活动", "expect": "answer", "check": lambda a, m: bool(a) and bool(m)},
    {"query": "旅行要准备什么", "expect": "answer", "check": lambda a, m: bool(a) and bool(m)},
    {"query": "最近要交什么", "expect": "answer", "check": lambda a, m: bool(a) and bool(m)},
    {"query": "个人偏好是什么", "expect": "answer", "check": lambda a, m: bool(a) and bool(m)},
    # ── 拒答场景 10 条（库外话题应诚实拒答）──
    {"query": "量子力学的原理是什么", "expect": "refuse", "check": lambda a, m: (not m) or "不知道" in a},
    {"query": "火星上有没有生命", "expect": "refuse", "check": lambda a, m: (not m) or "不知道" in a},
    {"query": "今年的诺贝尔奖得主", "expect": "refuse", "check": lambda a, m: (not m) or "不知道" in a},
    {"query": "唐朝的历史事件", "expect": "refuse", "check": lambda a, m: (not m) or "不知道" in a},
    {"query": "如何造火箭", "expect": "refuse", "check": lambda a, m: (not m) or "不知道" in a},
    {"query": "梵高的代表作", "expect": "refuse", "check": lambda a, m: (not m) or "不知道" in a},
    {"query": "全球变暖的原因分析", "expect": "refuse", "check": lambda a, m: (not m) or "不知道" in a},
    {"query": "怎么治疗感冒", "expect": "refuse", "check": lambda a, m: (not m) or "不知道" in a},
    {"query": "股票怎么投资", "expect": "refuse", "check": lambda a, m: (not m) or "不知道" in a},
    {"query": "宇宙有多大", "expect": "refuse", "check": lambda a, m: (not m) or "不知道" in a},
    # ── 溯源要求 10 条（回答必须带来源 metadata）──
    {"query": "牛奶在哪个文档", "expect": "source", "check": lambda a, m: bool(m)},
    {"query": "git 相关记录", "expect": "source", "check": lambda a, m: bool(m)},
    {"query": "交周报的记录", "expect": "source", "check": lambda a, m: bool(m)},
    {"query": "跑步的安排", "expect": "source", "check": lambda a, m: bool(m)},
    {"query": "机票在哪", "expect": "source", "check": lambda a, m: bool(m)},
    {"query": "早餐吃什么", "expect": "source", "check": lambda a, m: bool(m)},
    {"query": "晚餐的记录", "expect": "source", "check": lambda a, m: bool(m)},
    {"query": "关于复习的安排", "expect": "source", "check": lambda a, m: bool(m)},
    {"query": "体检记录", "expect": "source", "check": lambda a, m: bool(m)},
    {"query": "买水果的记录", "expect": "source", "check": lambda a, m: bool(m)},
]


def run():
    passed = 0
    for i, c in enumerate(CASES, 1):
        try:
            ans, metas = rag(c["query"])
            ok = c["check"](ans, metas)
        except Exception as e:
            ok, ans, metas = False, f"异常:{e}", []
        print(f"{'PASS' if ok else 'FAIL'} #{i} [{c['expect']}] {c['query'][:20]}...")
        passed += ok
    total = len(CASES)
    print(f"\n通过率: {passed}/{total} = {passed / total * 100:.1f}%")


if __name__ == "__main__":
    run()
