"""Region-aware support resource text."""

import json
from pathlib import Path


class ResourceService:
    """Returns non-time-sensitive help-seeking guidance from configuration."""

    def __init__(self, config_path: Path = Path("resources/support_resources.json")):
        self.config_path = config_path
        self.resources = self._load()

    def _load(self) -> dict:
        if not self.config_path.exists():
            return {}
        return json.loads(self.config_path.read_text(encoding="utf-8"))

    def get_resources(self, region: str = "generic") -> str:
        item = self.resources.get(region) or self.resources.get("generic")
        if not item:
            return (
                "可以考虑联系当地紧急服务、持证心理咨询师、精神科或全科医生、学校/单位支持服务，"
                "并让可信赖的人陪同求助。"
            )
        steps = "；".join(item["steps"])
        return f"{item['intro']}{steps}。{item['note']}"
