"""
Week 6 基础 RAG Agent（脚手架）
===============================
流程：检索 → 拼上下文 → 生成 → 溯源 → 诚实拒答。
跑通：先生成 docs/（用 make_docs.py 或自备）→ python rag.py
说明：直接复用 W5 的 Chroma 索引逻辑（week5/chroma_db），或在本目录重建。
"""
import os, glob, json
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import sys

load_dotenv()
# ── 两个 client：chat 走 DeepSeek，embedding 走百炼 ──
# 理由：DeepSeek 没有 /embeddings 端点（打过去是 404），两件事必须两个身份
chat_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)
MODEL = os.getenv("MODEL_NAME", "deepseek-v4-flash")

embedding_client = OpenAI(
    api_key=os.getenv("EMBEDDING_API_KEY"),
    base_url=os.getenv("EMBEDDING_BASE_URL"),
)
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

COLLECTION = "week5_docs"
PERSIST_DIR = "../week5/chroma_db"  # 复用 W5 索引；没有则先跑 week5

SIM_THRESHOLD = 0.35  # 相似度阈值，低于则拒答（需按 30 条用例调优）

# 拒答文案统一成常量：两道防线共用，避免"文案不一致"再次成为定位障碍
REFUSE_TEXT = "我不知道（资料中没有相关信息）。"


def embed(text: str):
    resp = embedding_client.embeddings.create(model=EMBED_MODEL, input=[text])
    return resp.data[0].embedding


def retrieve(query: str, top_k: int = 3):
    """检索 Top-K，返回 (docs, metas, distances)。"""
    chroma = chromadb.PersistentClient(path=PERSIST_DIR)
    col = chroma.get_collection(COLLECTION)  #这里也需要进行防御，如果col为空可以直接返回结果
    q = embed(query)  #这里也没有做预防因为有可能请求embedding模型会失败或者超时，当然也可以整体防御
    res = col.query(query_embeddings=[q], n_results=top_k)   
    return res["documents"][0], res["metadatas"][0], res["distances"][0]


def build_prompt(query: str, context: str):
    """三段式 prompt：上下文块 + 用户问题 + 输出契约。

    输出契约：模型必须返回 JSON —— {"found": bool, "answer": str}
      found=true  → answer 是回答正文
      found=false → 资料里找不到答案，answer 固定为 REFUSE_TEXT
    """
    return (
        "你是一个严格基于资料的助手。请只根据下面的【资料】回答【用户问题】。\n\n"
        f"【资料】\n{context}\n\n"
        f"【用户问题】{query}\n\n"
        "输出要求：只输出一个 JSON 对象，不要任何其他文字，不要 markdown 代码块。格式：\n"
        '{"found": true, "answer": "<给用户的回答>"}\n'
        "字段说明：\n"
        "- found=true：【资料】里有能回答【用户问题】的内容，answer 写回答正文，直接给内容，不要说"
        "\u201c根据资料\u201d这类套话。\n"
        "- found=false：【资料】里找不到相关内容。"
        "此时 answer 固定写"
        f"\u201c{REFUSE_TEXT}\u201d，绝对不要用你自己的知识、常识或猜测补充。\n"
    )


def rag(query: str, top_k: int = 3):
    #这个是通过embeddingAPI计算输入的问题的向量再从本地向量库中查询出topK的结果。流程的第一步，已经做过，这个是流程的R
    docs, metas, dists = retrieve(query, top_k)

    # ── 第一道防线：检索层（代码判，判据 = 相似度数字）──
    # 都太低 → 直接拒答，连 LLM 都不惊动。返回空 metas，main 会打印"无（已拒答）"
    # 这个是针对查询出来的召回结果进行相似度判断和拦截。相似度低于阈值的直接拦截，但是和相似度阈值高一点点的没法正确被拦截
    # 这里是防线1.用于将相似度低于阈值的topK直接排除掉
    if not docs or max([1 - d for d in dists]) < SIM_THRESHOLD:
        return REFUSE_TEXT, []
    #将文本和进行组装成context，作为提交给llm节点的优质数据源 这里是A
    context = "\n\n".join(f"[{m.get('source','?')}] {d}" for d, m in zip(docs, metas))
    #这里是chatAPI，也就是G请求调用LLM具体应该是build_prompt这个步骤，请求大模型这里没有设置超时时间是个问题
    resp = chat_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": build_prompt(query, context)}],  #build_prompt就是基于用户问题和查询的资料组装给大模型的上下文。这里面规定了json回答的格式
        temperature=0,
        response_format={"type": "json_object"},  # 结构化输出：让"模型拒答"对代码可见。这里重点是用 response_format 限制了大模型回复的消息契约。
    )
    #读取大模型的回答结果
    raw = (resp.choices[0].message.content or "").strip()

    
    try:
        data = json.loads(raw)
        if not "found" in data or not "answer" in data:  #如有字段但是内容为空应该如何处理
           print("契约字段缺失, 大模型返回为:", raw, file=sys.stderr)
           return REFUSE_TEXT, []
        answer = str(data.get("answer") or "").strip()
        found = bool(data.get("found", False))
    except (json.JSONDecodeError, AttributeError, TypeError):   #这里也是一个防线调用报错直接返回。但是没有设置超时时间
        # 兜底：模型没守输出契约。不吞错、也不编答案——直接返回拒绝消息，但是为了区分和方便后续验证，因此这里需要增加日志记录
        print("json解析报错, 大模型返回为:", raw, file=sys.stderr)
        return REFUSE_TEXT, []

    # ── 第二道防线：生成层（模型判，判据 = prompt 里的指令）──
    # 模型自己说找不到 → 代码据此清空 metas，来源不再乱打（契约不变，main 无需改）
    #第二道防线，根据大模型的执行来决定是返回回答还是走默认的拒绝策略
    if not found or not answer:
        return REFUSE_TEXT, []
    #return的都是给用户的回答和结果
    return answer, metas


def main():
    print("RAG Agent 已启动，输入 exit 退出")
    while True:
        q = input("问> ").strip()
        if q.lower() in {"exit", "q"}:
            break
        ans, metas = rag(q)
        print("答>", ans)
        if metas:
            print("来源>", ", ".join(m.get("source", "?") for m in metas))
        else:
            print("来源> 无（已拒答）")


if __name__ == "__main__":
    main()
