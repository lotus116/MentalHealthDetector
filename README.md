# MentalHealthDetector

## 概述
该系统是一个精神健康辅助评估工具，整合大语言模型 (LLM)、检索增强生成 (RAG) 和 BERT 分类模型，形成完整评估闭环：通过多轮对话收集用户信息，利用 RAG 提供专业文档支持，经微调 BERT 模型进行风险分类，最终生成包含诊断结论、评估及就医建议的结构化报告，实现从信息收集到专业评估的全流程自动化处理。

## 依赖项
- Hugging Face库（用于下载`bert-base-chinese`和`text2vec-base-chinese`模型）
- 百度翻译API（用于`data_preprocess.ipynb`中的可选数据预处理）
- LangChain（支持`chain.py`运行）

## 目录结构
- `bert-base-chinese/`：存放从Hugging Face下载的BERT基础模型
- `bert_finetuned/`：保存自建的微调后BERT模型
- `rag/`：存放下载的参考文献（txt/pdf格式）
- `saved_rag/`：保存处理后的RAG数据
- `text2vec-base-chinese/`：存放从Hugging Face下载的Text2Vec模型
- `data_preprocess.ipynb`：使用百度翻译API的备选数据预处理脚本
- `data_process.py`：主选的数据预处理脚本
- `chain.py`：主程序，辅助LangChain运行
- `rag_preprocess.py`：负责处理RAG相关数据
- `tran_bert.py`：负责微调BERT模型
- `Combined Data_p_with_translation.csv`：经分层随机抽样和AI处理后的5000条数据
- `final_data.csv`：最终用于微调BERT的数据集

## 使用方法
1. 从Hugging Face下载`bert-base-chinese`和`text2vec-base-chinese`模型，分别放入对应目录
2. 将参考文献放入`rag/`目录
3. 运行`data_process.py`进行主要数据预处理
4. 使用`tran_bert.py`结合`final_data.csv`微调BERT模型
5. 通过`rag_preprocess.py`处理RAG数据并保存到`saved_rag/`
6. 执行`chain.py`启动系统（基于LangChain）
