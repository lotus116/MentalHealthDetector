from app.classifiers.optional_bert import OptionalBertClassifier


def test_optional_bert_disabled_does_not_load_or_route():
    classifier = OptionalBertClassifier(enabled=False, threshold=0.82)
    assert classifier.classify("我想了解压力") is None


def test_optional_bert_enabled_without_model_is_non_blocking():
    classifier = OptionalBertClassifier(enabled=True, threshold=0.82)
    assert classifier.classify("我想了解压力") is None
