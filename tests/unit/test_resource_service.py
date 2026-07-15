from pathlib import Path

from app.services.resource_service import ResourceService


def test_resource_service_uses_generic_config():
    text = ResourceService(Path("resources/support_resources.json")).get_resources("unknown")
    assert "不硬编码" in text
    assert "当地紧急服务" in text
