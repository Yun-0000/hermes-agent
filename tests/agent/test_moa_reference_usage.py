"""Reference-slot usage must survive the chat-shaped adapter (#123157).

``call_llm`` rebuilds Codex and Anthropic payloads into OpenAI Chat usage.
Pricing that object with the slot's native shape records every bucket as 0,
so advisor tokens never fold into the session.
"""

from types import SimpleNamespace

from agent.auxiliary_client import _parse_codex_final_response
from agent.moa_loop import _price_reference_response


def _priced(usage, *, provider: str, api_mode: str, model: str = "advisor"):
    usage, _cost, _status, _source = _price_reference_response(
        SimpleNamespace(usage=usage),
        {"model": model, "provider": provider},
        {"provider": provider, "api_mode": api_mode, "model": model},
    )
    return usage


def test_codex_reference_slot_keeps_adapter_chat_usage():
    """A Codex advisor records the chat usage ``_parse_codex_final_response`` emits.

    The native ``codex_responses`` shape looks for ``input_tokens`` and would
    zero every bucket, including plain prompt and completion totals.
    """
    raw = {
        "input_tokens": 20000,
        "input_tokens_details": {"cached_tokens": 18000},
        "output_tokens": 1500,
        "output_tokens_details": {"reasoning_tokens": 1200},
        "total_tokens": 21500,
    }
    _text, _calls, adapted = _parse_codex_final_response(SimpleNamespace(output=[], usage=raw))

    usage = _priced(adapted, provider="openai-codex", api_mode="codex_responses")

    # Fresh input plus cache buckets must conserve the chat prompt total.
    assert usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens == adapted.prompt_tokens
    assert usage.output_tokens == adapted.completion_tokens
    assert adapted.prompt_tokens > 0
    assert usage.output_tokens > 0


def test_anthropic_reference_slot_keeps_adapter_chat_usage():
    """An Anthropic advisor records the chat usage the auxiliary adapter returns.

    ``provider='anthropic'`` would otherwise select the native Messages shape,
    which does not read ``prompt_tokens`` / ``completion_tokens``.
    """
    adapted = SimpleNamespace(prompt_tokens=562, completion_tokens=400, total_tokens=962)

    usage = _priced(adapted, provider="anthropic", api_mode="anthropic_messages", model="claude-opus")

    assert usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens == adapted.prompt_tokens
    assert usage.output_tokens == adapted.completion_tokens
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0
