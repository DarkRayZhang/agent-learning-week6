# Week 6 卫星项目：基础 RAG Agent

> **一句话卖点：可溯源 RAG + 诚实拒答**

第 6 周周末项目（详细拆解见 `../week6-daily-plan.md`）。检索→拼上下文→生成全链路，检索能力后续并入主线「备忘录检索」（V3 D7）。

## 功能

- **检索→拼上下文→生成**：问库里有答案的问题，基于检索结果回答并**附来源**
- **诚实拒答**：资料中没有相关信息时，直接说"不知道"，不编造
- **30 条测试用例**：正常问答 10 + 拒答场景 10 + 溯源要求 10，跑出通过率基线

## 运行步骤

1. 先确保 `../week5/chroma_db` 索引存在（跑过 week5 的 `make_docs.py` + `retrieval_app.py`）
2. `cd G:\agent学习\week6`
3. 激活 venv + `pip install -r requirements.txt`
4. `cp .env.example .env` 填 `DEEPSEEK_API_KEY`
5. `python rag.py` 启动交互，`exit` 退出
6. `python test_cases.py` 跑 30 条用例出通过率

## 架构

```
用户提问 → embedding → Chroma 检索 Top-K → 拼 <context> 上下文
        → LLM 生成（要求只基于上下文）→ 回答 + 附来源（metadata）
        → 相关度过低 → 诚实拒答
```

## 评估（V3 D3）

- 30 条用例：正常问答 10 + 拒答场景 10 + 溯源要求 10
- 每次改 Prompt/阈值后重跑，通过率记入 `notes.md` 评估趋势线
- 这是 W9 上 Langfuse（自托管，50+ 条 eval）前的基线

## 验收

1. 库内有答案 → 正确回答且带来源（溯源）
2. 库内没有的话题 → 诚实拒答，不编造
3. `python test_cases.py` 通过率记录基线（目标 ≥85%，如实记录）
