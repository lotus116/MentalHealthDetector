import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from datetime import datetime


# 设置随机种子，保证结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


# ===================== 1. 数据准备 =====================
def load_and_split_data(file_path, text_col, label_col, test_size=0.2, val_size=0.5):
    """加载数据并划分为训练集、验证集和测试集"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据集文件不存在: {file_path}")

    data = pd.read_csv(file_path)

    # 检查必要的列是否存在
    if text_col not in data.columns:
        raise ValueError(f"数据集中不存在列: {text_col}")
    if label_col not in data.columns:
        raise ValueError(f"数据集中不存在列: {label_col}")

    # 划分训练集、验证集、测试集
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        data[text_col].tolist(),
        data[label_col].tolist(),
        test_size=test_size,
        random_state=42,
        stratify=data[label_col].tolist()  # 保持标签分布一致
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=val_size,
        random_state=42,
        stratify=temp_labels
    )

    print(f"数据集划分完成:")
    print(f"训练集: {len(train_texts)} 样本")
    print(f"验证集: {len(val_texts)} 样本")
    print(f"测试集: {len(test_texts)} 样本")

    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)


# ===================== 2. 加载分词器与预训练模型 =====================
def load_model_and_tokenizer(model_name, num_labels):
    """加载预训练模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )

    # 移动模型到可用设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"模型已加载，使用设备: {device}")

    return model, tokenizer, device


# ===================== 3. 数据预处理 =====================
def preprocess_data(texts, labels, tokenizer, max_length=512):  # 与test.py保持一致
    """预处理数据，转换为模型可接受的格式"""
    # 构建Dataset
    dataset = Dataset.from_dict({"text": texts, "label": labels})

    # 分词函数
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    # 应用分词
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 转换为PyTorch张量格式
    tokenized_dataset = tokenized_dataset.with_format("torch", columns=["input_ids", "attention_mask", "label"])

    return tokenized_dataset


# ===================== 4. 定义评估函数 =====================
def compute_metrics(pred):
    """计算评估指标：准确率、精确率、召回率、F1值"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # 计算指标
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"  # 使用加权平均处理可能的不平衡数据
    )

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


# ===================== 5. 训练模型 =====================
def train_model(model, tokenizer, train_dataset, val_dataset, num_epochs=3, batch_size=16):
    """训练模型并返回训练器"""
    # 创建带时间戳的输出目录，与test.py中的BERT_MODEL_PATH格式保持一致
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./bert_finetuned/bert_finetuned_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 训练参数配置
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,  # 评估时可用更大的batch size
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"./logs_{timestamp}",
        logging_steps=10,  # 更频繁地记录日志
        evaluation_strategy="epoch",  # 每轮结束后评估
        save_strategy="epoch",  # 每轮结束后保存模型
        load_best_model_at_end=True,  # 训练结束后加载最优模型
        metric_for_best_model="f1",  # 以F1作为最优模型的评判指标
        greater_is_better=True,  # F1值越大越好
        report_to="none"  # 不使用wandb等报告工具
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    # 开始训练
    print("开始训练...")
    trainer.train()

    return trainer, output_dir


# ===================== 6. 评估模型 =====================
def evaluate_model(trainer, test_dataset):
    """在测试集上评估模型"""
    print("\n在测试集上评估模型...")
    metrics = trainer.evaluate(test_dataset)

    print("\n测试集评估结果:")
    print(f"准确率: {metrics['eval_accuracy']:.4f}")
    print(f"精确率: {metrics['eval_precision']:.4f}")
    print(f"召回率: {metrics['eval_recall']:.4f}")
    print(f"F1值: {metrics['eval_f1']:.4f}")

    return metrics


# ===================== 7. 保存模型 =====================
def save_model(trainer, tokenizer, output_dir):
    """保存完整模型和分词器，确保与test.py的加载要求兼容"""
    # 保存分词器
    tokenizer.save_pretrained(output_dir)
    # 保存模型权重
    trainer.save_model(output_dir)
    print(f"\n模型和分词器已保存到: {output_dir}")

    # 生成模型路径信息，方便用户替换test.py中的BERT_MODEL_PATH
    print(f"\n请将test.py中的BERT_MODEL_PATH设置为: {output_dir}")
    return output_dir


# ===================== 8. 推理函数：增强版 =====================
def predict_text(text, model, tokenizer, device, max_length=512, return_prob=False):  # 与test.py保持一致
    """
    单条文本推理（增强版）
    return_prob: 是否返回所有类别的概率
    """
    model.eval()  # 设置为评估模式，关闭Dropout等训练特有的层
    # 文本编码（与训练时预处理逻辑一致）
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )

    # 移动到模型所在设备（CPU/GPU）
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 关闭梯度计算，减少内存消耗和计算时间
    with torch.no_grad():
        outputs = model(**inputs)

    # 解析结果
    logits = outputs.logits  # 模型输出的原始分数
    probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]  # 转换为概率（0-1）
    predicted_class = int(np.argmax(probabilities))  # 取概率最大的类别
    confidence = round(probabilities[predicted_class], 4)  # 对应类别的置信度

    # 标签含义映射（与test.py保持一致：0=无风险，1=有风险）
    label_map = {0: "无精神健康问题风险", 1: "存在精神健康问题风险"}
    result = {
        "文本": text,
        "预测标签": predicted_class,
        "标签含义": label_map[predicted_class],
        "置信度": confidence,
        "置信度百分比": f"{confidence * 100:.2f}%"
    }

    # 若需要返回所有类别概率，补充信息
    if return_prob:
        result["各类别概率"] = {
            label_map[i]: round(prob, 4) for i, prob in enumerate(probabilities)
        }

    return result


def batch_predict_texts(texts, model, tokenizer, device, max_length=512):  # 与test.py保持一致
    """
    批量文本推理（高效处理多条文本）
    """
    model.eval()
    # 批量编码文本
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # 解析批量结果
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
    predicted_classes = np.argmax(probabilities, axis=1).tolist()
    confidences = [round(prob[i], 4) for i, prob in zip(predicted_classes, probabilities)]

    # 整理结果
    label_map = {0: "无精神健康问题风险", 1: "存在精神健康问题风险"}
    results = []
    for idx, (text, cls, conf, prob) in enumerate(zip(texts, predicted_classes, confidences, probabilities)):
        results.append({
            "序号": idx + 1,
            "文本": text,
            "预测标签": cls,
            "标签含义": label_map[cls],
            "置信度": conf,
            "置信度百分比": f"{conf * 100:.2f}%",
            "各类别概率": {label_map[i]: round(p, 4) for i, p in enumerate(prob)}
        })
    return results


def test_edge_cases(model, tokenizer, device, max_length=512):  # 与test.py保持一致
    """
    测试边缘案例（异常文本场景）
    """
    # 边缘案例集合：空文本、极短文本、超长文本、特殊字符文本、无关文本
    edge_texts = [
        "",  # 空文本
        "好",  # 极短文本（1个字符）
        "我最近感觉很不好，每天晚上都睡不着，一闭眼就会想很多事情，担心工作做不好，担心家人身体，白天上班没精神，注意力根本集中不起来，有时候甚至会突然想哭，这种情况已经持续快一个月了，我尝试过运动、听音乐，但都没什么用，不知道该怎么办才好" * 5,
        # 超长文本
        "！@#￥%……&*（）——+=-{}[]|、：；“‘<>？，。/`",  # 纯特殊字符
        "今天天气很好，适合去公园散步，顺便买杯咖啡"  # 无关文本（无精神健康相关内容）
    ]

    print("\n=== 边缘案例推理测试 ===")
    for text in edge_texts:
        # 处理空文本，避免编码错误
        input_text = text if text.strip() else "无有效文本"
        result = predict_text(input_text, model, tokenizer, device, max_length, return_prob=True)
        # 打印结果（简化超长文本的展示）
        display_text = result["文本"][:50] + "..." if len(result["文本"]) > 50 else result["文本"]
        print(f"\n【文本】：{display_text}")
        print(f"【预测结果】：{result['标签含义']}（置信度：{result['置信度百分比']}）")
        print(f"【各类别概率】：{result['各类别概率']}")


# ===================== 9. 推理测试主函数 =====================
def run_comprehensive_inference_test(model, tokenizer, device, max_length=512):  # 与test.py保持一致
    """
    综合推理测试：包含单条、批量、边缘案例测试
    """
    print("\n" + "=" * 60)
    print("=== 模型综合推理测试 ===")
    print("=" * 60)

    # 1. 单条文本推理测试（覆盖不同场景）
    print("\n【1. 单条文本推理测试】")
    single_test_texts = [
        "我每天要量10次体温，总觉得自己生病了，即使体温正常也会反复确认，影响了正常生活",
        "身体状况良好，每天按时作息，工作和家庭都很顺心，没有任何不适的感觉",
        "最近半个月总是食欲不振，吃一点就饱，体重下降了5斤，对以前喜欢的事情也提不起兴趣",
        "晚上经常做噩梦，醒来后很难再入睡，白天容易烦躁，和同事沟通时经常忍不住发脾气"
    ]
    for text in single_test_texts:
        result = predict_text(text, model, tokenizer, device, max_length, return_prob=True)
        print(f"\n【文本】：{result['文本']}")
        print(f"【预测标签】：{result['预测标签']}（{result['标签含义']}）")
        print(f"【置信度】：{result['置信度百分比']}")
        print(f"【各类别概率】：{result['各类别概率']}")

    # 2. 批量文本推理测试（模拟实际批量处理场景）
    print("\n" + "-" * 60)
    print("【2. 批量文本推理测试】")
    batch_test_texts = [
        "我总是担心出门会遇到危险，所以尽量不出门，已经快一个月没去上班了",
        "周末会和朋友去爬山、看电影，心情一直很轻松，没有什么压力",
        "最近经常头晕、心慌，去医院检查没发现身体问题，但还是控制不住担心自己得重病",
        "每天都很开心，和家人相处融洽，工作效率也很高，没有任何情绪困扰",
        "晚上失眠已经持续3周了，靠安眠药才能睡3-4小时，白天注意力无法集中"
    ]
    batch_results = batch_predict_texts(batch_test_texts, model, tokenizer, device, max_length)
    # 打印批量结果（表格形式更清晰）
    print(f"\n批量推理结果（共{len(batch_results)}条）：")
    print(f"{'序号':<4} {'预测标签':<8} {'标签含义':<16} {'置信度':<12} {'文本预览':<30}")
    print("-" * 80)
    for res in batch_results:
        text_preview = res["文本"][:25] + "..." if len(res["文本"]) > 25 else res["文本"]
        print(
            f"{res['序号']:<4} {res['预测标签']:<8} {res['标签含义']:<16} {res['置信度百分比']:<12} {text_preview:<30}")

    # 3. 边缘案例测试（验证模型鲁棒性）
    print("\n" + "-" * 60)
    test_edge_cases(model, tokenizer, device, max_length)

    print("\n" + "=" * 60)
    print("=== 综合推理测试完成 ===")
    print("=" * 60)


# ===================== 主函数 =====================
def main():
    # 配置参数 - 与test.py保持一致
    DATA_PATH = "./final_data.csv"  # 替换为你的数据路径
    TEXT_COLUMN = "translate_chinese"  # 文本列名
    LABEL_COLUMN = "status"  # 标签列名
    MODEL_NAME = "bert-base-chinese"  # 与test.py的BERT_DEFAULT_MODEL一致
    NUM_LABELS = 2  # 分类类别数（与test.py保持一致）
    MAX_LENGTH = 512  # 与test.py的BERT_MAX_SEQ_LEN保持一致
    NUM_EPOCHS = 3  # 训练轮数
    BATCH_SIZE = 16  # 批次大小

    try:
        # 1. 加载和划分数据
        (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = load_and_split_data(
            DATA_PATH, TEXT_COLUMN, LABEL_COLUMN
        )

        # 2. 加载模型和分词器
        model, tokenizer, device = load_model_and_tokenizer(MODEL_NAME, NUM_LABELS)

        # 3. 预处理数据
        print("预处理数据...")
        train_dataset = preprocess_data(train_texts, train_labels, tokenizer, MAX_LENGTH)
        val_dataset = preprocess_data(val_texts, val_labels, tokenizer, MAX_LENGTH)
        test_dataset = preprocess_data(test_texts, test_labels, tokenizer, MAX_LENGTH)

        # 4. 训练模型
        trainer, output_dir = train_model(
            model, tokenizer, train_dataset, val_dataset, NUM_EPOCHS, BATCH_SIZE
        )

        # 5. 在测试集上评估
        evaluate_model(trainer, test_dataset)

        # 6. 保存模型
        save_model(trainer, tokenizer, output_dir)

        # 7. 增强版推理测试
        run_comprehensive_inference_test(model, tokenizer, device, MAX_LENGTH)

    except Exception as e:
        print(f"发生错误: {str(e)}")
        # 打印详细错误堆栈，便于调试
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()