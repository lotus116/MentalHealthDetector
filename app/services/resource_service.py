"""Region-aware support resource text."""


class ResourceService:
    """Returns non-time-sensitive help-seeking guidance."""

    def get_resources(self, region: str = "generic") -> str:
        return (
            "可以考虑的支持路径包括：联系当地紧急服务处理即时危险；预约持证心理咨询师、精神科或全科医生；"
            "联系学校/雇主员工支持计划；让可信赖的家人或朋友陪同求助。"
            "热线号码会随地区变化，本演示不硬编码可能过期的号码，请以当地政府、医院或公益机构官网为准。"
        )

