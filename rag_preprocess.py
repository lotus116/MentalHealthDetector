import torch
import os
from typing import Optional
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

# ===================== 预处理配置（与chain.py保持一致）=====================
# API密钥（仅预处理阶段初始化压缩器需要，后续主程序无需重复初始化）
DASHSCOPE_API_KEY = ""
QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# RAG源文档路径
RAG_DOCUMENTS = ["rag/SCL-90.txt", "rag/心理咨询与心理治疗.pdf"]
# Embeddings模型
EMBEDDINGS_MODEL = "text2vec-base-chinese"
# 检索配置
RAG_SEARCH_KWARGS = {"k": 3}  # 检索文档数量
# 本地保存路径（向量库+配置文件）
RAG_SAVE_DIR = "./saved_rag"  # 建议与主程序同目录，方便调用
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 确保保存目录存在
os.makedirs(RAG_SAVE_DIR, exist_ok=True)


# ===================== 工具函数（复用chain.py核心逻辑）=====================
def init_qwen_llm(model_name: str = "qwen-plus", temperature: float = 0.0) -> ChatOpenAI:
    """初始化Qwen模型（仅用于文档压缩器）"""
    return ChatOpenAI(
        model_name=model_name,
        api_key=DASHSCOPE_API_KEY,
        base_url=QWEN_API_BASE,
        temperature=temperature,
        max_tokens=1024,
        timeout=30,
        verbose=False
    )


def build_and_save_rag() -> Optional[bool]:
    """构建RAG向量库和检索器，并保存到本地"""
    try:
        # 1. 加载文档（与chain.py逻辑一致）
        print("1. 加载RAG源文档...")
        docs = []
        for doc_path in RAG_DOCUMENTS:
            if not os.path.exists(doc_path):
                print(f"⚠️ 文档不存在：{doc_path}，已跳过")
                continue
            try:
                if doc_path.endswith(".txt"):
                    loader = TextLoader(doc_path, encoding='utf-8')
                    docs.extend(loader.load())
                elif doc_path.endswith(".pdf"):
                    loader = PyPDFLoader(doc_path)
                    docs.extend(loader.load())
                print(f"✅ 加载成功：{doc_path}（累计{len(docs)}段）")
            except Exception as e:
                print(f"⚠️ 加载失败 {doc_path}：{str(e)[:50]}...")

        if not docs:
            print("❌ 未加载到任何文档，终止预处理")
            return False

        # 2. 分割文档（与chain.py逻辑一致）
        print(f"2. 分割文档（原始文档数：{len(docs)}）...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            separators=["。", "？", "！", "；", "\n", "，", " ", ""],
            length_function=len
        )
        splits = text_splitter.split_documents(docs)
        print(f"✅ 分割完成，得到 {len(splits)} 个文本片段")

        # 3. 构建Embeddings和向量库
        print(f"3. 初始化Embeddings模型（{EMBEDDINGS_MODEL}）...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDINGS_MODEL,
            model_kwargs={'device': DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )

        print("4. 构建FAISS向量库...")
        db = FAISS.from_documents(splits, embeddings)
        # 保存向量库到本地（关键步骤）
        vector_db_path = os.path.join(RAG_SAVE_DIR, "faiss_vector_db")
        db.save_local(vector_db_path)
        print(f"✅ 向量库保存成功：{vector_db_path}")

        # 5. 构建带压缩的检索器（仅需保存配置，无需序列化检索器本身）
        # 注：检索器依赖LLM和向量库，LLM每次启动需重新初始化，因此仅保存向量库+配置
        print("5. 验证检索器配置...")
        llm = init_qwen_llm(temperature=0.0)
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=db.as_retriever(search_kwargs=RAG_SEARCH_KWARGS)
        )
        # 测试检索器有效性(不要就注释掉吧)
        test_query = "什么是SCL-90？"
        test_docs = compression_retriever.get_relevant_documents(test_query)
        print(f"✅ 检索器验证成功：查询「{test_query}」返回 {len(test_docs)} 个相关文档")

        # 6. 保存检索器配置（方便主程序复用参数）
        config = {
            "embeddings_model": EMBEDDINGS_MODEL,
            "search_kwargs": RAG_SEARCH_KWARGS,
            "vector_db_path": vector_db_path
        }
        import json
        config_path = os.path.join(RAG_SAVE_DIR, "rag_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"✅ 检索器配置保存成功：{config_path}")

        print("\n🎉 RAG预处理完成！后续主程序可直接加载 ./saved_rag 目录")
        return True

    except Exception as e:
        print(f"❌ 预处理失败：{str(e)}")
        return False


if __name__ == "__main__":
    build_and_save_rag()