"""
Tasks 1 & 2 -- Real generation, grounding, and model-driven tool-calling.

Wire this up to an ACTUAL language model that supports native tool/function
calling (a hosted API, or a local model via Ollama such as qwen2.5 or
llama3.1) -- not a stub, and not a hand-rolled keyword router. Which model
you use isn't graded; whether you can actually work with a real model's
real behavior is.
"""

import json
from typing import List, Dict
from openai import OpenAI
from .retrieval import retrieve, Chunk
from .tools import calculate_noi, CALCULATE_NOI_TOOL_SCHEMA


client = OpenAI()
tools = [CALCULATE_NOI_TOOL_SCHEMA]



SYSTEM_PROMPT="""
You are a retrieval-grounded assistant.

Your job is to answer the user's question using the authorized context chunks provided to you.

## Grounding rules

1. The provided context chunks are your authoritative source of information for answering the user's question.

2. For normal questions, answer ONLY using information explicitly supported by the provided chunks.

3. Do NOT use your general/world knowledge to fill in missing information. If the answer cannot be determined from the provided chunks, clearly say that the available context does not contain enough information to answer the question.

4. Do NOT invent, assume, infer, or fabricate:
   - facts
   - numbers
   - property information
   - dates
   - financial values
   - names
   - relationships between entities
   - explanations that are not supported by the chunks

5. When answering, use the most relevant chunks rather than attempting to use every chunk.

6. If multiple chunks contain relevant information, combine them only when the information is consistent and supported by the chunks.

7. If the chunks contain conflicting information, do not silently choose one. Explicitly mention the conflict and identify which chunks support the conflicting claims.

## Citations

When answering from the retrieved context, cite the relevant chunk(s).

Use the chunk identifiers supplied in the context. Do not invent citation identifiers.

Every factual claim that comes from the retrieved context should be traceable to one or more retrieved chunks.

## NOI calculation tool

You have access to a `calculate_noi` tool.

Use the tool when the user's question requires calculating or retrieving Net Operating Income (NOI) for a property and period.

Examples of questions that should generally require the tool:

- "What is the NOI for property 123 in 2026-Q1?"
- "Calculate the NOI for property 10 for January."
- "What was the property's NOI in Q2?"
- "How much NOI did property X generate?"

Do NOT calculate NOI yourself from revenue and operating expenses.

Do NOT perform the arithmetic yourself.

Do NOT state an NOI value before receiving the result from the `calculate_noi` tool.

When the tool is required:

1. Identify the required `property_id` and `period` from the user's question and/or authorized context.
2. Call `calculate_noi` with those values.
3. Wait for the tool result.
4. Use the tool result when producing the final answer.
5. Do not independently recompute or modify the returned NOI value.

The `calculate_noi` tool is the authoritative source for the calculated NOI value.

If the required property ID or period cannot be determined from the user's question and authorized context, do not guess. Ask the user for the missing information.

## Important distinction

The retrieved chunks are the source of factual context.

The `calculate_noi` tool is the source of truth for NOI calculations.

Do not substitute one for the other.

For example, if the chunks contain:

property_id=123
period=2026-Q1
revenue=100000
operating_expenses=30000

and the user asks for NOI, call the `calculate_noi` tool instead of answering "$70,000" yourself.

## Final answer behavior

Be concise and directly answer the user's question.

If the question can be answered from the chunks, answer it using only supported information.

If the question requires NOI, use the tool and then provide the tool's result in a natural response.

If the necessary information is unavailable, say what information is missing rather than making an unsupported assumption.



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



def call_model(prompt: str, tools: list | None = None) -> Dict:
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

    completion = client.chat.completions.create(
        model="Qwen/Qwen3.5-27B",
        messages=messages,
        tools=tools,
    )

    message = completion.choices[0].message

    # Model decided that no tool is necessary.
    if not message.tool_calls:
        answer = message.content

        return {
          "text": answer,
          "tool_call": None,
      }

    # Model requested a tool.
    tool_messages = []
    tool_call = message.tool_calls[0]
    
    if tool_call.function.name != "calculate_noi":
        raise ValueError(
            f"Unknown tool requested: {tool_call.function.name}"
        )

    arguments = json.loads(tool_call.function.arguments)

    result = calculate_noi(
        property_id=arguments["property_id"],
        period=arguments["period"],
    )

    tool_messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps({
                "noi": result
            }),
        }
    )

    messages.append(message)
    messages.extend(tool_messages)

    final_completion = client.chat.completions.create(
        model="Qwen/Qwen3.5-27B",
        messages=messages,
        tools=tools,
    )

    final_message = final_completion.choices[0].message

    return {
        "text": final_message.content,
        "tool_call": {"name": tool_call.function.name, "arguments": arguments},
    }


def is_grounded(answer: str, chunks: List[Chunk]) -> bool:
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
      {chunks}
      </context>

      ANSWER:
      <answer>
      {answer}
      </answer>
      """,
              },
          ]

    completion = client.chat.completions.create(
        model="Qwen/Qwen3.5-27B",
        messages=messages,
        temperature=0,
    )

    content = completion.choices[0].message.content

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return False

    return result.get("grounded", False) is True 


def answer_question(query: str, user_groups: List[str]) -> Dict:
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


    chunks = retrieve(query, user_groups)
    prompt = f" Authorized context: {chunks} \n User question:{query}"
    response = call_model(prompt, tools)
    

    return {
          "answer": response['text'],
          "citations": chunks,
          "grounded": is_grounded(response['text'], chunks),
          "used_tool": response['tool_called'] != None
         }
