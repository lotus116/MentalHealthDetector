# Interview Guide

## 2-Minute Project Introduction

I refactored a diagnostic-style mental health prototype into a non-diagnostic information support assistant. The v2 product provides cited knowledge answers, a deterministic self-understanding survey, help-seeking resource navigation and fixed crisis routing. It runs in Mock mode without API keys and includes automated safety tests.

## 5-Minute Technical Introduction

The backend is FastAPI with Pydantic v2 models. Safety routing runs before intent routing. Intent then routes to RAG, survey, resources or supportive response. RAG uses project-authored knowledge summaries with source metadata. Survey scoring is deterministic and cannot be changed by the LLM. ResponsePolicy blocks diagnostic and medication claims.

## 5-Minute BA/Product Introduction

The As-Is system mixed BERT, RAG and LLM into a diagnostic report flow. The To-Be flow focuses on explainability, source citation, user safety and clear scope. Requirements were split into must/should/could/out-of-scope, and acceptance criteria map to runnable tests.

## Why Not Use BERT as the Core Model

The legacy BERT classifier was trained on data with unclear source and labels. It is not clinically validated and can encourage false precision. In v2 it is isolated as an optional disabled adapter for offline comparison or intent fallback only.

## Why LLMs Cannot Diagnose Directly

LLMs can generate plausible unsupported claims and cannot verify clinical context. The product boundary explicitly forbids diagnosis, disease probability, medication advice and treatment decisions.

## How RAG Is Evaluated

The MVP uses 30 synthetic questions with expected source ids, reporting retrieval hit rate, citation completeness and latency. Human groundedness scoring is listed as not executed unless actually performed.

## Safety Routing Design

Deterministic rules catch explicit crisis and medication requests first. Structured LLM classification is available as a second layer. High-risk matches return a fixed template and do not continue ordinary conversation.

## Gap to Production

Production would need stronger multilingual safety evaluation, jurisdiction-aware resource lookup, authentication, observability, incident review, larger licensed knowledge sources and human governance.

## My Responsibility

In this refactor I designed the product boundary, architecture, safety policy, backend service layers, Streamlit demo, evaluation scripts, tests and migration documentation.

## Failures and Tradeoffs

The deterministic safety rules intentionally over-trigger on some academic/news mentions. This is documented as a known MVP limitation. The retrieval backend is lightweight for demo reliability and should be swapped for FAISS/Chroma in a larger deployment.

