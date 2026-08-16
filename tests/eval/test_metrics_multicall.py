from llm_internal.eval.benchmark import BenchmarkCase
from llm_internal.eval.metrics import score_case

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def test_tool_selection_preserves_duplicate_call_multiplicity():
    case = BenchmarkCase(
        id="same_tool_twice",
        category="parallel_tool_calls",
        description="Call the same tool independently for two cities",
        tools=[_WEATHER_TOOL],
        messages=[{"role": "user", "content": "Weather in Paris and Berlin?"}],
        expects_tool_call=True,
        expected_tool_calls=[
            {"name": "get_weather", "arguments": {"city": "Paris"}},
            {"name": "get_weather", "arguments": {"city": "Berlin"}},
        ],
    )
    predicted = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'

    result = score_case(case, predicted)

    assert result.tool_selection_correct is False
    assert result.call_tp == 1
    assert result.call_fn == 1
    assert result.exact_match is False
