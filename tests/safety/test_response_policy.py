from app.services.response_policy import ResponsePolicy


def test_blocks_diagnostic_claim():
    answer = ResponsePolicy().validate("你患有抑郁症")
    assert "不能提供诊断" in answer


def test_blocks_medication_instruction():
    answer = ResponsePolicy().validate("应该服用某药，剂量为10mg")
    assert "药物剂量" in answer

