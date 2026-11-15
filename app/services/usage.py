import json


def record(model: str, input_tokens: int = 0, output_tokens: int = 0, search_units: int = 0) -> None:
    """Log the billable usage of one external model call (no content), for scripts/cost_report.py."""
    print(
        "USAGE " + json.dumps({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "search_units": search_units,
        }),
        flush=True,
    )
