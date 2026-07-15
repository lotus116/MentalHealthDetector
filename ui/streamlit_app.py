"""Streamlit demo for the Mental Health Information Support Assistant."""

import requests
import streamlit as st

API_URL = st.sidebar.text_input("API URL", value="http://localhost:8000")

st.set_page_config(page_title="心理健康信息支持助手", layout="wide")
st.title("心理健康信息支持助手")
st.info("本助手仅提供一般信息、自我了解参考和求助路径，不提供医学诊断、疾病概率、药物建议或治疗决定。默认只保留当前会话内容。")

if "messages" not in st.session_state:
    st.session_state.messages = []

tab_chat, tab_survey, tab_resources, tab_eval = st.tabs(["对话", "问卷", "资源", "评估"])

with tab_chat:
    mode = st.segmented_control("模式", ["auto", "knowledge", "survey", "resources", "support"], default="auto")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    prompt = st.chat_input("输入你的问题或当前困扰")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        try:
            response = requests.post(f"{API_URL}/chat", json={"message": prompt, "mode": mode, "session_id": "streamlit"}, timeout=20)
            response.raise_for_status()
            data = response.json()
            st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
            st.rerun()
        except requests.RequestException as exc:
            st.error(f"API 请求失败：{exc}")
    if st.button("清空会话"):
        requests.delete(f"{API_URL}/chat/streamlit", timeout=5)
        st.session_state.messages = []
        st.rerun()
    if st.session_state.messages:
        rating = st.radio("反馈", ["helpful", "not_helpful", "inaccurate", "unsafe"], horizontal=True)
        if st.button("提交反馈"):
            requests.post(f"{API_URL}/feedback", json={"session_id": "streamlit", "rating": rating}, timeout=5)
            st.success("已记录反馈")

with tab_survey:
    try:
        survey = requests.get(f"{API_URL}/survey", timeout=10).json()
        st.subheader(survey["title"])
        answers = {}
        for question in survey["questions"]:
            labels = [opt["label"] for opt in question["options"]]
            scores = {opt["label"]: opt["score"] for opt in question["options"]}
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
        data = requests.post(f"{API_URL}/chat", json={"message": "我想寻找专业支持资源", "mode": "resources"}, timeout=10).json()
        st.write(data["answer"])

with tab_eval:
    st.write("开发者评估请在命令行运行：`make evaluate` 或分别运行 `python evaluation/evaluate_*.py`。")
    st.write("评估脚本只报告小型合成集的实际结果，不代表临床效果或生产性能。")

