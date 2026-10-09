"""
Tasks 1 & 2 -- Real generation, grounding, and model-driven tool-calling.

Wire this up to an ACTUAL language model that supports native tool/function
calling (a hosted API, or a local model via Ollama such as qwen2.5 or
llama3.1) -- not a stub, and not a hand-rolled keyword router. Which model
you use isn't graded; whether you can actually work with a real model's
real behavior is.
"""

import json
import inspect
from typing import List, Dict
from openai import AsyncOpenAI
from .retrieval import retrieve, Chunk
from .tools import calculate_noi, find_property, normalize_period, tools


client = AsyncOpenAI(base_url="http://host.docker.internal:11434/v1", api_key="ollama")

SYSTEM_PROMPT = """
You are a retrieval-grounded assistant for property and financial questions.

You answer using two sources:
1. Authorized context chunks, provided in the user's message under "CONTEXT".
2. Tools, which you call to look up IDs, normalize periods, and calculate NOI.

## Grounding rules

1. Context chunks are your only source of facts. Answer ONLY from information explicitly stated in them or returned by a tool.
2. Do NOT use general or world knowledge to fill gaps. If the context does not contain the answer, say what is missing.
3. Do NOT invent or assume facts, numbers, dates, names, property details, financial values, or relationships between entities.
4. Use the most relevant chunks; you do not need to use all of them.
5. Combine chunks only when they are consistent. If chunks conflict, do not pick one silently: state the conflict and cite the chunks that support each side.

## Citations

- Each chunk in CONTEXT has an id, shown as [chunk_id].
- Cite every fact taken from the context with its id, for example: "Revenue was $120,000 [3-1]."
- Use only ids that appear in CONTEXT. Never invent an id.
- Facts that come from a tool result are cited as (tool: tool_name), not as a chunk.

## Available tools

- `find_property(name_query)`: fuzzy-matches a property name and returns candidate property IDs.
- `normalize_period(raw_period)`: converts a period such as "2026 q3" or "third quarter 2026" into the standard format, e.g. "Q3-2026".
- `calculate_noi(property_id, period)`: returns the Net Operating Income for a property and period.

## When to use tools

- Use `find_property` when you need a property ID and the user gave only a name. Do not guess IDs.
- Use `normalize_period` when the user's period is not already in the form "Q3-2026". Always pass its output to `calculate_noi`.
- Use `calculate_noi` for ANY question that asks for NOI. Never compute NOI yourself from revenue and expenses, never do the arithmetic, and never state an NOI value before the tool returns it. Do not modify the returned value.
- Do NOT use tools for questions that are not about NOI (for example revenue, expenses, occupancy, lease terms). Answer those from the context chunks. A property ID or period from the chunks is only needed if a later NOI call requires it.

## Calling tools in order

- Some calls depend on others: `calculate_noi` needs the `property_id` from `find_property` and the period from `normalize_period`. Wait for those results before calling it.
- Independent calls, such as `find_property` and `normalize_period`, may be made together in the same turn.
- You can make as many tool calls as you need, one round at a time, until you have everything required.
- Do not repeat a call with the same arguments. Reuse the earlier result.
- If a tool returns an error, read the message, fix the arguments, and retry once. If it still fails, tell the user what went wrong.

## Handling ambiguity and missing information

- If `find_property` returns several plausible matches and the user's name does not clearly identify one, ask the user which property they mean. Do not choose for them.
- If `find_property` returns no match, say so.
- If the property or period cannot be determined from the question, the context, or the tools, ask for the missing information instead of guessing.

## Final answer

- Be concise and answer the question directly.
- State the property name, period, and value so the answer is unambiguous.
- Include citations as described above.
- Do not describe your internal reasoning or the tool-calling steps unless the user asks.

## Example

User: "What is the NOI for Maple Ridge Apartments in 2026 q3?"
Steps: call `find_property("Maple Ridge Apartments")` and `normalize_period("2026 q3")`, then call `calculate_noi` with the returned property_id and "Q3-2026", then answer with the tool's value.

User: "What was Maple Ridge Apartments' total revenue in Q3 2026?"
Steps: no NOI is requested, so answer from the context chunks with citations. If the chunks do not contain it, say the available context does not include that figure.
"""


GROUNDING_SYSTEM_PROMPT = """
You are a grounding verifier for a retrieval-augmented generation system.

Your task is to determine whether the ANSWER is fully supported by the
provided CONTEXT.

Rules:

1. Evaluate only whether the claims in the ANSWER are supported by the
   CONTEXT.

2. Do not use your own knowledge or outside information.

3. Every factual claim in the ANSWER must be supported by the CONTEXT.

4. If the ANSWER contains a factual claim that is not supported by the
   CONTEXT, the answer is not grounded.

5. Paraphrasing is allowed. The answer does not need to use the exact
   wording from the context.

6. Reasonable mathematical transformations are allowed when they are
   directly supported by the context.

7. If the answer contains no unsupported factual claims, return grounded=true.

8. If the ANSWER is a refusal or states that it cannot answer based on the provided context, return grounded=true.

Return ONLY valid JSON in this format:

{
    "grounded": true,
    "unsupported_claims": []
}

or:

{
    "grounded": false,
    "unsupported_claims": [
        "The unsupported claim here."
    ]
}
"""


TOOL_REGISTRY = {
    "calculate_noi": calculate_noi,
    "find_property": find_property,
    "normalize_period": normalize_period,
}


async def call_model(prompt: str, tools: list | None = None) -> Dict:
    """Call your chosen real language model.

    Return a normalized dict shaped like:
        {"text": str | None, "tool_call": {"name": str, "arguments": dict} | None}

    - If the model just answers, `text` is set and `tool_call` is None.
    - If the model wants to call a tool (only possible when `tools` is
      passed), `tool_call` is set and `text` may be None.

    You will need to translate this normalized shape to/from whatever your
    provider's actual API expects (OpenAI, Anthropic, and Ollama all differ
    slightly in envelope). That translation is the point of this task.
    """ 

    messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ]
    executed = []

    while True:
        completion = await client.chat.completions.create(
            model="glm-4.7-flash:latest",
            messages=messages,
            tools=tools,
        )

        message = completion.choices[0].message
        if not message.tool_calls:
            answer = message.content

            return {
                "text": answer,
                "tool_call": executed[-1] if executed else None,
                "tool_calls": executed
            }

        tool_calls = message.tool_calls


        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
                "content": message.content or "",
            }
        )
        
        for tool_call in tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            name = tool_call.function.name

            func = TOOL_REGISTRY.get(name)
            if func is None:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({"error": f"Unknown tool: '{name}'. Did you misspell it?"}),
                })
                continue
            if inspect.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)

            executed.append({"name": name, "arguments": arguments})

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"result": result}),
            })

    return {
        "text": "Stopped: too many tool-calling steps.",
        "tool_call": executed[-1] if executed else None,
        "tool_calls": executed,
    }


async def is_grounded(answer: str, chunks: List[Chunk]) -> bool:
    """Task 1 grounding check: verify the answer's key claims (names,
    numbers, dates) actually appear in the retrieved chunks it was
    supposedly based on. A simple substring/entity check is enough -- it
    needs to exist, not be sophisticated.
    """

    messages = [
        {
            "role": "system",
            "content": GROUNDING_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"""
                    CONTEXT:
                        <context>
                            {[{'id': c.doc_id, 'text': c.text} for c in chunks]}
                        </context>

                    ANSWER:
                        <answer>
                            {answer}
                        </answer>
                """,
              },
          ]

    completion = await client.chat.completions.create(
        model="glm-4.7-flash:latest",
        messages=messages,
        temperature=0,
    )

    content = completion.choices[0].message.content

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return False

    return result.get("grounded", False) is True 


async def answer_question(query: str, user_groups: List[str]) -> Dict:
    """Orchestration for both tasks:

    - Retrieve authorized context for the query via retrieve().
    - Call the model WITH the calculate_noi tool available
      (CALCULATE_NOI_TOOL_SCHEMA) so the model itself decides whether this
      question needs the tool -- do not pre-classify the query yourself
      with keywords/regex and route around the model.
    - If the model requests the tool call, execute calculate_noi() in code,
      send the result back to the model, and get its final narrated answer.
      The model must never be allowed to just state the NOI number itself.
    - If the model answers directly instead, apply is_grounded() to check
      it against the retrieved chunks.

    Return a dict shaped like:
        {"answer": str, "citations": [...], "grounded": bool, "used_tool": bool}
    """


    chunks = await retrieve(query, user_groups)
    
    context_str = "\\n".join([f"[{c.doc_id}] {c.text}" for c in chunks])
    prompt = f" Authorized context: {context_str} \\n User question:{query}"
    response = await call_model(prompt, tools)
    
    text = response['text']
    import re
    cited_ids = set()
    for match in re.finditer(r'\[(.*?)\]', text):
        pieces = re.split(r'[,\s]+', match.group(1))
        cited_ids.update(pieces)
        
    cited_chunks = [c for c in chunks if str(c.doc_id) in cited_ids]
    formatted_citations = [f"[{c.doc_id}] {c.text}" for c in cited_chunks]

    return {
          "answer": text,
          "citations": formatted_citations,
          "grounded": await is_grounded(text, cited_chunks),
          "used_tool": response['tool_call'] != None
         }
