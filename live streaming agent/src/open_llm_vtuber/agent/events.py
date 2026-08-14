LLM_FIRST_TOKEN_EVENT_TYPE = "llm-first-token"
LLM_FINAL_ERROR_FALLBACK_EVENT_TYPE = "llm-final-error-fallback"
WEB_SEARCH_START_EVENT_TYPE = "web-search-start"
WEB_SEARCH_TIMING_EVENT_TYPE = "web-search-timing"


def make_llm_first_token_event() -> dict[str, str]:
    return {"type": LLM_FIRST_TOKEN_EVENT_TYPE}


def make_llm_final_error_fallback_event() -> dict[str, str]:
    return {"type": LLM_FINAL_ERROR_FALLBACK_EVENT_TYPE}


def is_web_search_event(value: object) -> bool:
    return isinstance(value, dict) and value.get("type") in {
        WEB_SEARCH_START_EVENT_TYPE,
        WEB_SEARCH_TIMING_EVENT_TYPE,
    }
