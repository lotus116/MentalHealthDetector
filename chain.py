import torch
import numpy as np
from typing import Tuple, Optional
import os
import json
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain_openai import ChatOpenAI
from transformers import BertTokenizer, BertForSequenceClassification
# RAG 相关导入
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
# 对话历史管理组件
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# ===================== 【核心配置】 =====================
DASHSCOPE_API_KEY = ""
QWEN_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
BERT_MODEL_PATH = "./bert_finetuned/bert_finetuned_20250908_135059"
BERT_DEFAULT_MODEL = "bert-base-chinese"
BERT_MAX_SEQ_LEN = 512  # 调整为更大的序列长度以容纳多轮用户输入
MAX_CONVERSATION_ROUNDS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 环境变量检查
if not DASHSCOPE_API_KEY or DASHSCOPE_API_KEY.startswith("YOUR_OWN"):
    raise ValueError("请替换DASHSCOPE_API_KEY为有效的API密钥")

# RAG 配置
RAG_SAVE_DIR = "./saved_rag"
RAG_CONFIG_PATH = os.path.join(RAG_SAVE_DIR, "rag_config.json")


# ===================== RAG工具函数 =====================
def init_rag() -> Optional[RetrievalQA]:
    """从本地加载预构建的RAG检索链"""
    try:
        if not os.path.exists(RAG_SAVE_DIR) or not os.path.exists(RAG_CONFIG_PATH):
            print(f"[警告] 本地RAG文件不存在（{RAG_SAVE_DIR}），请先运行rag_preprocess.py")
            return None

        # 加载RAG配置
        print("1. 加载本地RAG配置...")
        with open(RAG_CONFIG_PATH, "r", encoding="utf-8") as f:
            rag_config = json.load(f)
        embeddings_model = rag_config["embeddings_model"]
        search_kwargs = rag_config["search_kwargs"]
        vector_db_path = rag_config["vector_db_path"]
        print(f"[成功] 配置加载：Embeddings={embeddings_model}，检索数量={search_kwargs['k']}")

        # 初始化Embeddings
        print(f"2. 初始化Embeddings模型（{embeddings_model}）...")
        embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 加载FAISS向量库
        print(f"3. 加载本地FAISS向量库...")
        if not os.path.exists(vector_db_path):
            print(f"[警告] 向量库路径不存在：{vector_db_path}")
            return None

        db = FAISS.load_local(
            vector_db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"[成功] 向量库加载（含 {db.index.ntotal} 个文本片段）")

        # 构建压缩检索器
        print("4. 初始化文档压缩检索器...")
        llm = init_qwen_llm(temperature=0.0)
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=db.as_retriever(search_kwargs=search_kwargs)
        )

        # 重建检索链
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=compression_retriever,
            return_source_documents=True,
            verbose=False
        )
        print("[成功] RAG检索链加载完成")
        return rag_chain

    except Exception as e:
        print(f"[警告] RAG加载失败：{str(e)}，将继续运行但不使用RAG功能")
        return None


# ===================== 通用工具函数 =====================
def format_conversation(chat_history: InMemoryChatMessageHistory) -> str:
    """格式化完整对话历史（用于诊断报告）"""
    try:
        if not chat_history.messages:
            return "无有效对话记录"

        formatted_text = []
        for idx, msg in enumerate(chat_history.messages, 1):
            if isinstance(msg, HumanMessage):
                formatted_text.append(f"【用户{idx // 2 + 1}】：{msg.content.strip()}")
            elif isinstance(msg, AIMessage):
                formatted_text.append(f"【助手{idx // 2 + 1}】：{msg.content.strip()}")
        return "\n".join(formatted_text)
    except Exception as e:
        print(f"[警告] 格式化对话历史失败：{str(e)}")
        return "对话历史格式化失败"


def format_user_inputs(chat_history: InMemoryChatMessageHistory) -> str:
    """仅提取并格式化用户输入（用于BERT评估）"""
    try:
        if not chat_history.messages:
            return "无有效用户输入"

        user_inputs = []
        for msg in chat_history.messages:
            if isinstance(msg, HumanMessage):
                content = msg.content.strip()
                if content.lower() not in ["结束", "停止", "退出"]:
                    user_inputs.append(f"【用户{len(user_inputs) + 1}】：{content}")

        # 确保最多保留5轮用户输入
        return "\n".join(user_inputs[:MAX_CONVERSATION_ROUNDS])
    except Exception as e:
        print(f"[警告] 提取用户输入失败：{str(e)}")
        return "用户输入提取失败"


def validate_user_inputs(text: str) -> bool:
    """验证用户输入有效性（用于BERT评估）"""
    return len(text.strip()) >= 30 and "【用户" in text


# ===================== LLM初始化函数 =====================
def init_qwen_llm(model_name: str = "qwen-plus", temperature: float = 0.4) -> ChatOpenAI:
    """初始化Qwen模型"""
    return ChatOpenAI(
        model_name=model_name,
        api_key=DASHSCOPE_API_KEY,
        base_url=QWEN_API_BASE,
        temperature=temperature,
        max_tokens=1024,
        timeout=30,
        verbose=False
    )


# ===================== 对话交互函数 =====================
def run_llm_conversation(rag_chain) -> Tuple[InMemoryChatMessageHistory, str, str]:
    """带RAG支持的对话交互，返回完整历史、格式化历史、纯用户输入"""
    print("\n=== 精神健康信息收集（输入“结束”停止对话）===")
    print(f"提示：共支持 {MAX_CONVERSATION_ROUNDS} 轮对话，请尽量详细描述情况\n")

    # 初始化LLM和对话历史
    chat_llm = init_qwen_llm(model_name="qwen-plus", temperature=0.4)
    chat_history = InMemoryChatMessageHistory()

    # 定义Prompt模板
    prompt_template = PromptTemplate(
        input_variables=["history", "input", "docs"],
        template="""你是专业的心理咨询辅助师，仅执行以下任务：
1. 核心规则：每次提1-2个问题收集用户信息（聚焦情绪/身体症状/行为变化），不提前诊断、不给出建议；
2. 结束规则：若用户输入“结束”，立即回复“对话已结束，将为你生成评估报告”，无需额外内容；
3. 参考规则：可参考文档信息（非必须），但问题需基于用户当前输入设计，不脱离上下文。

参考文档：
{docs}

对话历史：
{history}

用户当前输入：
{input}

你的回复（仅1-2个问题或结束提示，不超过3句话，中文表达）：
"""
    )

    # 创建带历史的对话链
    base_chain = prompt_template | chat_llm
    conversation_chain = RunnableWithMessageHistory(
        base_chain,
        lambda session_id: chat_history,
        input_messages_key="input",
        history_messages_key="history",
        session_id="mental_health_chat_001"
    )

    # 多轮对话循环
    for round_idx in range(MAX_CONVERSATION_ROUNDS):
        try:
            user_input = input(f"\n【第{round_idx + 1}轮/共{MAX_CONVERSATION_ROUNDS}轮】请输入你的情况：").strip()
            if not user_input:
                print("【助手】：请描述你的具体情况（例如：最近是否经常失眠？情绪是否容易低落？）")
                continue

            # 处理结束指令
            if user_input.lower() in ["结束", "停止", "退出"]:
                print("【助手】：对话已结束，将为你生成评估报告")
                chat_history.add_user_message(user_input)
                chat_history.add_ai_message("对话已结束，将为你生成评估报告")
                break

            # 检索RAG文档
            docs_content = "无参考文档"
            if rag_chain:
                try:
                    print(f"[搜索中] 正在检索相关文档（第{round_idx + 1}轮）...")
                    rag_result = rag_chain.invoke({"query": user_input})
                    docs_content = "\n".join([
                        f"文档片段{idx + 1}：{doc.page_content.strip()[:200]}..."
                        for idx, doc in enumerate(rag_result["source_documents"])
                    ])
                except Exception as e:
                    print(f"[警告] 文档检索出错：{str(e)[:30]}...，将基于对话历史提问")
                    docs_content = "文档检索失败，无参考信息"

            # 调用LLM获取回复
            response = conversation_chain.invoke(
                {"input": user_input, "docs": docs_content},
                config={"configurable": {"session_id": "mental_health_chat_001"}}
            )

            # 解析回复并更新历史
            llm_response = response.content.strip() if hasattr(response, "content") else str(response)
            print(f"【助手】：{llm_response}")

        except KeyboardInterrupt:
            raise ValueError("用户手动中断对话")
        except Exception as e:
            error_msg = str(e)
            if "Invalid API key" in error_msg:
                raise ValueError("API密钥无效，请检查DASHSCOPE_API_KEY是否正确")
            elif "NetworkError" in error_msg:
                print("【助手】：当前网络不稳定，请重试一次")
                continue
            else:
                print(f"【助手】：对话出错：{error_msg[:50]}...，请重试")
                continue

    # 生成两种格式化结果
    full_history = format_conversation(chat_history)
    user_inputs_text = format_user_inputs(chat_history)

    # 验证用户输入有效性
    if not validate_user_inputs(user_inputs_text):
        raise ValueError(f"用户输入无效（内容：{user_inputs_text[:50]}...），无法进行BERT评估")

    return chat_history, full_history, user_inputs_text


# ===================== BERT相关函数 =====================
def load_bert_model() -> Tuple[BertTokenizer, BertForSequenceClassification]:
    """加载BERT模型"""
    print(f"\n3. 开始加载BERT模型（设备：{DEVICE}）...")
    try:
        # 优先加载本地模型
        if os.path.exists(BERT_MODEL_PATH) and len(os.listdir(BERT_MODEL_PATH)) > 3:
            tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
            model = BertForSequenceClassification.from_pretrained(
                BERT_MODEL_PATH,
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            model = model.to(DEVICE)
            print(f"[成功] 加载本地BERT模型：{BERT_MODEL_PATH}（设备：{DEVICE}）")
            return tokenizer, model
        else:
            print(f"[警告] 本地模型路径无效（{BERT_MODEL_PATH}），将加载默认模型：{BERT_DEFAULT_MODEL}")
    except Exception as e:
        print(f"[警告] 本地模型加载失败：{str(e)[:50]}...，将使用默认模型")

    # 加载默认BERT模型
    try:
        tokenizer = BertTokenizer.from_pretrained(BERT_DEFAULT_MODEL)
        model = BertForSequenceClassification.from_pretrained(
            BERT_DEFAULT_MODEL,
            num_labels=2
        )
        model = model.to(DEVICE)
        print(f"[成功] 加载默认BERT模型：{BERT_DEFAULT_MODEL}（设备：{DEVICE}）")
        return tokenizer, model
    except Exception as e:
        raise ValueError(f"BERT模型加载失败：{str(e)}（请确保网络通畅）")


def bert_classify(text: str, tokenizer: BertTokenizer, model: BertForSequenceClassification) -> Tuple[int, float]:
    """使用BERT进行分类"""
    print("\n4. 开始BERT精神健康风险分类...")
    try:
        # 文本编码
        inputs = tokenizer(
            text,
            max_length=BERT_MAX_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)

        # 模型推理
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # 解析结果
        predicted_label = int(np.argmax(probabilities))
        confidence = round(float(probabilities[predicted_label]), 4)
        label_desc = "存在精神健康问题风险" if predicted_label == 1 else "无精神健康问题风险"

        # 打印详细结果
        print(f" BERT分类结果：")
        print(f"   - 风险判断：{label_desc}")
        print(f"   - 预测标签：{predicted_label}（0=无风险，1=有风险）")
        print(f"   - 置信度：{confidence * 100:.2f}%")
        return predicted_label, confidence

    except Exception as e:
        raise ValueError(f"BERT分类失败：{str(e)}")


# ===================== 诊断建议生成函数 =====================
def generate_diagnosis(history_text: str, bert_label: int, confidence: float, rag_chain) -> str:
    """生成带RAG支持的诊断建议"""
    print("\n5. 开始生成精神健康评估报告...")
    # 初始化诊断用LLM
    diagnosis_llm = init_qwen_llm(model_name="qwen-plus", temperature=0.3)
    bert_result = "存在精神健康问题风险" if bert_label == 1 else "无精神健康问题风险"

    # 检索诊断相关文档
    docs_content = "无参考文档"
    if rag_chain:
        try:
            print("[搜索中] 正在检索诊断参考文档...")
            rag_result = rag_chain.invoke({"query": f"基于对话历史的精神健康评估：{history_text[:200]}..."})
            docs_content = []
            for idx, doc in enumerate(rag_result["source_documents"], 1):
                source = doc.metadata.get("source", "未知来源")
                page = doc.metadata.get("page", "未知页码")
                content = doc.page_content.strip()[:300] + "..."
                docs_content.append(f"参考文档{idx}（{source} 第{page}页）：{content}")
            docs_content = "\n\n".join(docs_content)
        except Exception as e:
            print(f"[警告] 诊断阶段文档检索出错：{str(e)}，将基于现有信息生成报告")
            docs_content = "文档检索失败，基于对话历史和BERT结果生成报告"

    confidence_percent = confidence * 100

    # 定义诊断报告Prompt
    diagnosis_prompt = PromptTemplate(
        input_variables=["history", "bert_result", "confidence_percent", "docs"],
        template="""你是专业的精神健康评估辅助顾问，基于以下信息生成结构化评估报告，语言温和、易懂。

【核心依据】
1. 用户对话历史（精简版）：
{history}

2. BERT模型评估结果：
{bert_result}（置信度：{confidence_percent:.2f}%）

3. 参考文档信息：
{docs}

【输出要求】
严格按照以下3个模块输出，每个模块2-5句话，不超过600字：
1. 【诊断结论】：结合对话中的具体症状，说明是否存在风险及风险类型；
2. 【评估建议】：给出具体的自我评估建议；
3. 【就医建议】：若存在风险，给出具体的就医指引；若无风险，给出健康维护建议。

【禁止内容】
- 不推荐具体药物或治疗方案；
- 不使用“确诊”“一定”等绝对化表述；
- 不扩展无关内容。

请生成评估报告：
"""
    )

    # 调用LLM生成报告
    diagnosis_chain = LLMChain(
        llm=diagnosis_llm,
        prompt=diagnosis_prompt,
        verbose=False
    )
    result = diagnosis_chain.invoke({
        "history": history_text[:1000],
        "bert_result": bert_result,
        "confidence_percent": confidence_percent,
        "docs": docs_content
    })

    return result.get("text", "评估报告生成失败，请重试")


# ===================== 主流程函数 =====================
def main():
    print("=" * 70)
    print(f"精神健康评估系统（{DEVICE}版）")
    print("=" * 70)
    print(f"当前配置：Qwen-Plus LLM / BERT分类 / 本地RAG检索")
    print("=" * 70 + "\n")

    try:
        # 步骤1：加载本地RAG检索链
        print("【步骤1/4】加载本地文档检索系统...")
        rag_chain = init_rag()

        # 步骤2：多轮对话收集信息
        print("\n【步骤2/4】开始对话信息收集...")
        chat_history, full_history, user_inputs_text = run_llm_conversation(rag_chain)
        print(f"\n📝 整理后的用户输入：\n{user_inputs_text}")

        # 步骤3：BERT模型分类评估（仅使用用户输入）
        print("\n【步骤3/4】开始BERT风险评估...")
        bert_tokenizer, bert_model = load_bert_model()
        bert_label, bert_confidence = bert_classify(user_inputs_text, bert_tokenizer, bert_model)

        # 步骤4：生成最终评估报告（使用完整历史）
        print("\n【步骤4/4】生成评估报告...")
        final_diagnosis = generate_diagnosis(full_history, bert_label, bert_confidence, rag_chain)

        # 输出最终报告
        print("\n" + "=" * 80)
        print("精神健康评估报告（辅助版）")
        print("=" * 80)
        print(final_diagnosis)
        print("=" * 80)
        print("\n【重要提示】本报告仅为AI辅助评估，不能替代专业医生诊断！")
        print("若存在明显不适，请及时前往正规医疗机构就诊。")
        print("=" * 80)

    except Exception as e:
        print(f"\n[Error!] 流程运行失败：{str(e)}")
        print("建议检查：1. API密钥有效性 2. 本地RAG资源完整性 3. 网络连接")


if __name__ == "__main__":
    main()