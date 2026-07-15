"""Safety labels and fixed crisis response template."""

from enum import StrEnum


class SafetyAction(StrEnum):
    continue_normal = "continue_normal"
    fixed_crisis_response = "fixed_crisis_response"
    refuse_medical_advice = "refuse_medical_advice"
    clarify = "clarify"


CRISIS_RESPONSE = (
    "我很抱歉你正在经历这么强烈的痛苦。这个系统不能处理紧急危机，也不能替代专业人员。"
    "如果你现在可能伤害自己或他人，请立即联系当地紧急服务，或马上去最近的急诊/安全地点。"
    "如果可以，请现在联系一个你信任的人，让对方陪在你身边，并远离可能造成伤害的物品。"
    "我可以继续提供一般信息和求助路径，但此刻最重要的是让现实中的人尽快介入。"
)

NON_DIAGNOSTIC_DISCLAIMER = (
    "本助手仅提供心理健康信息支持和自我了解参考，不提供医学诊断、疾病概率、药物剂量或治疗决定。"
)

