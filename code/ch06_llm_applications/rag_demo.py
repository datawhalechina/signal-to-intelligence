"""
第6章代码实验：RAG（检索增强生成）演示

本实验演示如何构建一个简单的RAG系统，包括：
1. 构建知识库
2. 用字符 n-gram 向量建立可离线运行的词法检索基线
3. 生成增强的回答
4. 可视化向量检索的语义空间
"""

from collections import Counter
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = "assets/ch06_rag_vector_search.png"


def character_ngrams(text: str, n: int = 2) -> Counter:
    """Convert text to a deterministic character n-gram count vector."""
    normalized = "".join(char.lower() for char in text if not char.isspace())
    if len(normalized) < n:
        return Counter([normalized]) if normalized else Counter()
    return Counter(normalized[index:index + n] for index in range(len(normalized) - n + 1))


def cosine_similarity(left: Counter, right: Counter) -> float:
    """Cosine similarity for sparse count vectors."""
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = np.sqrt(sum(value * value for value in left.values()))
    right_norm = np.sqrt(sum(value * value for value in right.values()))
    return float(dot / (left_norm * right_norm))


class SimpleRAG:
    """Offline lexical retrieval baseline plus prompt construction."""
    
    def __init__(self):
        """初始化RAG系统"""
        self.knowledge_base = [
            {
                "id": 1,
                "title": "什么是Transformer",
                "content": "Transformer是一种基于自注意力机制的神经网络架构。它由编码器和解码器组成，能够并行处理序列数据。"
            },
            {
                "id": 2,
                "title": "什么是LLM",
                "content": "大语言模型（LLM）是用大规模文本数据预训练的深度学习模型。它能够理解和生成自然语言。"
            },
            {
                "id": 3,
                "title": "什么是RAG",
                "content": "检索增强生成（RAG）是一种结合检索和生成的方法。它先从知识库检索相关文档，然后用这些文档增强生成过程。"
            },
            {
                "id": 4,
                "title": "什么是微调",
                "content": "微调是在预训练模型的基础上，用特定任务的数据进行训练。这样可以让模型适应特定的任务。"
            },
        ]
    
    def retrieve(self, query: str, top_k: int = 2) -> List[Dict]:
        """Retrieve documents using cosine similarity over n-gram vectors."""
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query_vector = character_ngrams(query)
        scores = []
        for doc in self.knowledge_base:
            doc_vector = character_ngrams(doc["title"] + doc["content"])
            score = cosine_similarity(query_vector, doc_vector)
            scores.append((doc, score))

        scores.sort(key=lambda item: (-item[1], item[0]["id"]))
        return [doc for doc, _ in scores[:top_k]]
    
    def generate_with_context(self, query: str, retrieved_docs: List[Dict]) -> str:
        """用检索到的文档生成回答"""
        # 构建增强的提示
        context = "\n".join([f"- {doc['title']}: {doc['content']}" for doc in retrieved_docs])
        
        prompt = f"""基于以下信息回答问题：

信息：
{context}

问题：{query}

回答："""
        
        return prompt
    
    def query(self, question: str) -> Dict:
        """完整的RAG流程"""
        # 1. 检索
        retrieved_docs = self.retrieve(question)
        
        # 2. 生成增强的提示
        enhanced_prompt = self.generate_with_context(question, retrieved_docs)
        
        return {
            "question": question,
            "retrieved_docs": retrieved_docs,
            "enhanced_prompt": enhanced_prompt,
            "num_retrieved": len(retrieved_docs)
        }


def demo_rag():
    """演示RAG系统"""
    print("=" * 50)
    print("第6章：LLM应用与微调 - RAG演示")
    print("=" * 50)
    print()
    
    # 初始化RAG系统
    rag = SimpleRAG()
    
    # 测试查询
    test_queries = [
        "什么是Transformer？",
        "如何微调模型？",
        "RAG有什么优势？"
    ]
    
    for query in test_queries:
        print(f"问题：{query}")
        print()
        
        result = rag.query(query)
        
        print(f"检索到 {result['num_retrieved']} 个相关文档：")
        for i, doc in enumerate(result['retrieved_docs'], 1):
            print(f"  {i}. {doc['title']}")
        print()
        
        print("增强的提示：")
        print(result['enhanced_prompt'])
        print()
        print("-" * 50)
        print()


def visualize_vector_search():
    """
    可视化向量检索的语义空间。
    用手工设计的 2D 向量做几何示意，展示：
    - 语义相近的文档聚在一起
    - 查询向量与相关文档方向相近（余弦相似度高）
    """
    np.random.seed(42)

    # ── 手工设计各语义簇的中心，确保聚类效果清晰 ──────────────────────────────
    # 簇1：退款/支付相关
    cluster_refund = np.array([0.85, 0.52])
    # 簇2：产品/规格相关
    cluster_product = np.array([-0.60, 0.80])
    # 簇3：配送/物流相关
    cluster_shipping = np.array([-0.75, -0.66])

    def jitter(center, n=3, scale=0.07):
        pts = center + np.random.randn(n, 2) * scale
        # 归一化到单位圆附近
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)

    docs_refund   = jitter(cluster_refund,   n=4, scale=0.06)
    docs_product  = jitter(cluster_product,  n=4, scale=0.06)
    docs_shipping = jitter(cluster_shipping, n=3, scale=0.06)

    # 查询向量：靠近退款簇
    query = np.array([0.90, 0.44])
    query = query / np.linalg.norm(query)

    # ── 余弦相似度计算 ────────────────────────────────────────────────────────
    all_docs = np.vstack([docs_refund, docs_product, docs_shipping])
    sims = all_docs @ query   # 单位向量点积 = 余弦相似度
    top2_idx = np.argsort(sims)[-2:][::-1]

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("RAG: Illustrative Vector Geometry",
                 fontsize=13, fontweight="bold")

    # --- 左图：向量空间 ---
    ax = axes[0]
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axhline(0, color="lightgray", linewidth=0.8)
    ax.axvline(0, color="lightgray", linewidth=0.8)

    # 单位圆
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="lightgray",
            linewidth=0.8, linestyle="--")

    cluster_info = [
        (docs_refund,   "#e74c3c", "Refund / Payment"),
        (docs_product,  "#3498db", "Product / Specs"),
        (docs_shipping, "#27ae60", "Shipping / Logistics"),
    ]

    doc_labels = ["Doc A", "Doc B", "Doc C", "Doc D",
                  "Doc E", "Doc F", "Doc G", "Doc H",
                  "Doc I", "Doc J", "Doc K"]
    label_idx = 0
    for pts, color, _ in cluster_info:
        for i, pt in enumerate(pts):
            is_retrieved = label_idx in top2_idx
            marker = "*" if is_retrieved else "o"
            size   = 180 if is_retrieved else 80
            ax.scatter(*pt, color=color, s=size, marker=marker,
                       zorder=5, edgecolors="white", linewidths=0.8)
            if is_retrieved:
                rank = list(top2_idx).index(label_idx)
                offset = (8, 10) if rank == 0 else (-38, -18)
                ax.annotate(doc_labels[label_idx], pt,
                            textcoords="offset points", xytext=offset,
                            fontsize=8, color=color, fontweight="bold")
            label_idx += 1

    # 查询向量（箭头）
    ax.annotate("", xy=query, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#e67e22", lw=2.2))
    ax.scatter(*query, color="#e67e22", s=120, zorder=6,
               edgecolors="white", linewidths=1)
    ax.annotate("Query\n\"refund policy?\"", query,
                textcoords="offset points", xytext=(8, -18),
                fontsize=9, color="#e67e22", fontweight="bold")

    # 标注 top-2 检索结果的连线
    for idx in top2_idx:
        ax.plot([query[0], all_docs[idx, 0]],
                [query[1], all_docs[idx, 1]],
                color="#e67e22", linewidth=1.2, linestyle=":", alpha=0.7)

    legend_patches = [mpatches.Patch(color=c, label=l)
                      for _, c, l in cluster_info]
    legend_patches.append(mpatches.Patch(color="#e67e22", label="Query vector"))
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")
    ax.set_title("Document Embeddings & Query Vector", fontsize=11)
    ax.set_xlabel("Embedding Dimension 1", fontsize=10)
    ax.set_ylabel("Embedding Dimension 2", fontsize=10)

    # --- 右图：余弦相似度柱状图 ---
    ax2 = axes[1]
    colors_bar = (["#e74c3c"] * 4 + ["#3498db"] * 4 + ["#27ae60"] * 3)
    bar_colors = ["#f1c40f" if i in top2_idx else c
                  for i, c in enumerate(colors_bar)]

    bars = ax2.barh(doc_labels[::-1], sims[::-1],
                    color=bar_colors[::-1], edgecolor="white", linewidth=0.6)
    ax2.axvline(x=0, color="gray", linewidth=0.8)
    ax2.set_xlabel("Cosine Similarity to Query", fontsize=10)
    ax2.set_title("Similarity Scores (Top-2 Retrieved = ★)", fontsize=11)

    for i, (bar, sim) in enumerate(zip(bars, sims[::-1])):
        ax2.text(sim + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{sim:.2f}", va="center", fontsize=8)

    # 标注 top-2
    for idx in top2_idx:
        rank_in_bar = len(doc_labels) - 1 - idx
        ax2.text(-0.02, rank_in_bar, "★",
                 va="center", ha="right", fontsize=11, color="#e67e22")

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"图表已保存：{OUTPUT_PATH}")


if __name__ == "__main__":
    demo_rag()
    visualize_vector_search()

    print("=" * 50)
    print("演示完成！")
    print("=" * 50)
