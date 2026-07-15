"""Output policy checks for non-diagnostic behavior."""

import re

FORBIDDEN_PATTERNS = [
    r"你(患有|得了|就是).{0,8}(抑郁症|焦虑症|双相|精神分裂)",
    r"诊断结论",
    r"自杀概率|疾病概率",
    r"应该服用|剂量为|停用.*药",
]


class ResponsePolicy:
    """Blocks unsafe medical claims from final responses."""

    def validate(self, answer: str) -> str:
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, answer):
                return "我不能提供诊断、疾病概率、药物剂量或治疗决定。可以提供一般信息、问卷参考和求助路径。"
        return answer

