"""Streamlit demo for the Mental Health Information Support Assistant."""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
SESSION_ID = "streamlit"

st.set_page_config(page_title="心理健康信息支持助手", layout="wide")
st.title("心理健康信息支持助手")
st.info(
    "本助手仅提供一般信息、自我了解参考和求助路径，不提供医学诊断、疾病概率、药物建议或治疗决定。"
    "默认只保留当前会话内容，你可以随时清空。"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.expander("系统状态", expanded=False):
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.write(f"后端状态：{health.get('status', 'unknown')}")
        st.write(f"应用版本：{health.get('version', 'unknown')}")
        st.write(f"LLM 模式：{health.get('llm_provider', 'unknown')}")
    except requests.RequestException as exc:
        st.error(f"API 不可用：{exc}")

tab_chat, tab_survey, tab_resources, tab_eval = st.tabs(["对话", "问卷", "资源", "评估"])

with tab_chat:
    mode_labels = {
        "auto": "自动",
        "knowledge": "知识问答",
        "survey": "问卷引导",
        "resources": "资源导航",
        "support": "普通支持",
    }
    selected_label = st.segmented_control("模式", list(mode_labels.values()), default="自动")
    mode = next(key for key, value in mode_labels.items() if value == selected_label)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("引用来源"):
                    for source in msg["sources"]:
                        st.markdown(f"- **{source['title']}** `{source['source_id']}`")
                        st.caption(source["snippet"])
            if msg.get("meta"):
                st.caption(msg["meta"])

    prompt = st.chat_input("输入你的问题或当前困扰")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        try:
            response = requests.post(
                f"{API_URL}/chat",
                json={"message": prompt, "mode": mode, "session_id": SESSION_ID},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data.get("sources", []),
                    "meta": (
                        f"意图：{data['intent']} | 安全动作：{data['safety_action']} | 延迟：{data['latency_ms']} ms"
                    ),
                }
            )
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"API 请求失败：{exc}")

    col_clear, col_feedback = st.columns([1, 3])
    if col_clear.button("清空会话"):
        try:
            resp = requests.delete(f"{API_URL}/chat/{SESSION_ID}", timeout=5)
            resp.raise_for_status()
        except requests.RequestException as exc:
            st.error(f"清空会话失败：{exc}")
        else:
            st.session_state.messages = []
            st.rerun()

    if st.session_state.messages:
        feedback_labels = {
            "helpful": "有帮助",
            "not_helpful": "没有帮助",
            "inaccurate": "内容不准确",
            "unsafe": "内容不安全",
        }
        rating_label = col_feedback.radio("反馈", list(feedback_labels.values()), horizontal=True)
        rating = next(key for key, value in feedback_labels.items() if value == rating_label)
        if col_feedback.button("提交反馈"):
            requests.post(f"{API_URL}/feedback", json={"session_id": SESSION_ID, "rating": rating}, timeout=5)
            st.success("已记录反馈")

with tab_survey:
    try:
        survey = requests.get(f"{API_URL}/survey", timeout=10).json()
        st.subheader(survey["title"])
        st.caption(survey["description"])
        answers = {}
        default_options = survey.get("options", [])
        for question in survey["questions"]:
            options = question.get("options", default_options)
            labels = [opt["label"] for opt in options]
            scores = {opt["label"]: opt["score"] for opt in options}
            selected = st.radio(question["text"], labels, key=question["id"], horizontal=True)
            answers[question["id"]] = scores[selected]
        col1, col2 = st.columns(2)
        if col1.button("计算问卷结果"):
            result = requests.post(f"{API_URL}/survey/score", json={"answers": answers}, timeout=10).json()
            st.metric("分数", f"{result['score']} / {result['max_score']}")
            st.write(result["interpretation"])
            st.caption(result["disclaimer"])
        if col2.button("清空问卷选择"):
            for question in survey["questions"]:
                st.session_state.pop(question["id"], None)
            st.rerun()
    except requests.RequestException as exc:
        st.error(f"无法加载问卷：{exc}")

with tab_resources:
    st.write("如存在即时危险，请优先联系当地紧急服务或前往最近的急诊/安全地点。")
    if st.button("获取专业支持路径"):
        data = requests.post(
            f"{API_URL}/chat",
            json={"message": "我想寻找专业支持资源", "mode": "resources", "session_id": SESSION_ID},
            timeout=10,
        ).json()
        st.write(data["answer"])

with tab_eval:
    st.write("开发者评估请在命令行运行：`make evaluate` 或分别运行 `python evaluation/evaluate_*.py`。")
    st.write("评估脚本只报告小型合成集的实际结果，不代表临床效果或生产性能。")
