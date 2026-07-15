"""Intent labels for routing user requests."""

from enum import StrEnum


class IntentLabel(StrEnum):
    knowledge_query = "knowledge_query"
    survey_request = "survey_request"
    resource_request = "resource_request"
    supportive_conversation = "supportive_conversation"
    clarification_needed = "clarification_needed"
    crisis_signal = "crisis_signal"
    out_of_scope = "out_of_scope"

