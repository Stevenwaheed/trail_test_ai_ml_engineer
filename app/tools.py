"""
Task 2 -- Deterministic calculation, exposed to the model as a real tool.

NOI must be computed here, in code -- never by the language model. But
unlike a hand-rolled keyword router, the MODEL decides when to call this,
via its own native function-calling interface (OpenAI/Anthropic-style
tool schemas, or Ollama's tool support for models like qwen2.5 / llama3.1).
"""
import pandas
import difflib
import re


async def calculate_noi(property_id: int, period: str, csv_path: str = "data/properties.csv") -> float:
    """Return revenue - operating_expenses for the given property_id and
    period, read directly from data/properties.csv. No model involved.
    """

    properties = pandas.read_csv(csv_path)
    row = properties.loc[(properties['property_id'] == property_id) & (properties['period'] == period)]

    return float(row.iloc[0]["revenue"] - row.iloc[0]["operating_expenses"])


def normalize_period(raw_period: str) -> str:
    """Takes formats like '2026 q3' or 'Q3 2026' and returns 'Q3-2026'"""
    s = str(raw_period).upper().replace(" ", "")
    if m := re.fullmatch(r"Q([1-4])[-/]?(\d{4})", s):  
        return f"Q{m[1]}-{m[2]}"
    elif m := re.fullmatch(r"(\d{4})[-/]?Q([1-4])", s):  
        return f"Q{m[2]}-{m[1]}"
    return raw_period # Return as-is if we can't parse it



def find_property(name_query: str, csv_path: str = "data/properties.csv") -> list:
    """Fuzzy search for a property by name, returning the top matches and their IDs."""
    properties = pandas.read_csv(csv_path)
    all_names = properties['name'].tolist()
    
    # Find the top 3 closest matches, allowing for typos
    closest_matches = difflib.get_close_matches(
        word=name_query, 
        possibilities=all_names, 
        n=3,          # return up to 3 matches
        cutoff=0.5    # 0.5 is very forgiving for typos
    )
    
    if not closest_matches:
        return {"error": "No properties found matching that name."}
        
    # Get the IDs for those matched names
    results = []
    for match in closest_matches:
        row = properties[properties['name'] == match].iloc[0]
        results.append({
            "property_id": int(row['property_id']),
            "name": row['name']
        })
        
    return {"matches": results}


# A provider-agnostic description of this tool, in JSON-Schema-ish shape.
# Translate this into whatever your chosen model's API actually expects
# (OpenAI/Anthropic tool schemas differ slightly in envelope, not in spirit).
CALCULATE_NOI_TOOL_SCHEMA = {
    "type": "function",
    "function": {
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
}


FIND_PROPERTY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "find_property",
        "description": "Fuzzy search for a property by name, returning the top matches and their IDs.",
        "parameters": {
            "type": "object",
            "properties": {
                "name_query": {"type": "string", "description": "The Property name that the user is asking about"},
                "csv_path": {"type": "string", "description": "the path of the properties csv file"},
            },
            "required": ["name_query"],
        }
    }
}


NORMALISE_PERIOD_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "normalize_period",
        "description": "Normalizes the format of a period string to a consistent format (e.g. '2026 q3' -> 'Q3-2026').",
        "parameters": {
            "type": "object",
            "properties": {
                "raw_period": {"type": "string", "description": "The period string to normalize."},
            },
            "required": ["raw_period"],
        }
    }
}


tools = [CALCULATE_NOI_TOOL_SCHEMA, FIND_PROPERTY_TOOL_SCHEMA, NORMALISE_PERIOD_TOOL_SCHEMA]
