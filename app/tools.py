"""
Task 2 -- Deterministic calculation, exposed to the model as a real tool.

NOI must be computed here, in code -- never by the language model. But
unlike a hand-rolled keyword router, the MODEL decides when to call this,
via its own native function-calling interface (OpenAI/Anthropic-style
tool schemas, or Ollama's tool support for models like qwen2.5 / llama3.1).
"""
import pandas

def calculate_noi(property_id: int, period: str, csv_path: str = "data/properties.csv") -> float:
    """Return revenue - operating_expenses for the given property_id and
    period, read directly from data/properties.csv. No model involved.
    """

    properties = pandas.read_csv(csv_path)
    row = properties.loc[(properties['property_id'] == property_id) & (properties['period'] == period)]

    return float(row.iloc[0]["revenue"] - row.iloc[0]["operating_expenses"])



# A provider-agnostic description of this tool, in JSON-Schema-ish shape.
# Translate this into whatever your chosen model's API actually expects
# (OpenAI/Anthropic tool schemas differ slightly in envelope, not in spirit).
CALCULATE_NOI_TOOL_SCHEMA = {
    "name": "calculate_noi",
    "description": (
        "Compute Net Operating Income (revenue - operating expenses) for a "
        "property and period. Always use this tool for NOI questions -- "
        "never estimate or compute NOI yourself."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "property_id": {"type": "integer", "description": "The property's numeric id"},
            "period": {"type": "string", "description": "e.g. 'Q3-2026'"},
        },
        "required": ["property_id", "period"],
    },
}
