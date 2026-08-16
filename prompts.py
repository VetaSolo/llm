"""Prompt templates. Classification and per-route replies live here, not in pipeline.py."""

from __future__ import annotations

from dataclasses import dataclass

from schemas import Category


@dataclass(frozen=True)
class PromptSpec:
    key: str
    title: str
    idea: str
    system: str
    user_template: str
    json_mode: bool = False

    def build_user(self, **fields: str) -> str:
        text = self.user_template
        for key, value in fields.items():
            text = text.replace("{" + key + "}", str(value))
        return text


# ---------------------------------------------------------------------------
# Day 2 prompt comparison (not used by the live pipeline).
# Same 5 inputs, format score /30: A 23, B 30, C ~0.
# Winner: B — hard limits in system, user message is only the raw text.
# C often sounded nicer but wrote "## SUMMARY" and the parser got nothing.
# After Day 3 the winner was rewritten as JSON (EXTRACT / CLASSIFY / ROUTES).
# ---------------------------------------------------------------------------
#
# OUTPUT_LAYOUT = """SUMMARY:
# <summary>
#
# KEY_POINTS:
# - <point 1>
# - <point 2>
# - <point 3>
#
# RESPONSE:
# <helpful reply>"""
#
# VARIANT_A = PromptSpec(
#     key="a",
#     title="baseline_user_only",
#     idea="Everything in one user prompt. Simple, but the model treats rules as optional advice.",
#     system="Follow the user instructions exactly. Do not add extra sections.",
#     user_template="""Process the user text below.
#
# Rules:
# - Write the summary, key points, and response in the same language as the user text.
# - Summary: at most 2 sentences.
# - Key points: exactly 3 bullet points. Each one sentence.
# - Response: a short helpful reply to the author, at most 4 sentences.
#
# Use this exact layout and headings (do not add extra sections):
#
# SUMMARY:
# <summary>
#
# KEY_POINTS:
# - <point 1>
# - <point 2>
# - <point 3>
#
# RESPONSE:
# <helpful reply>
#
# User text:
# {user_text}
# """,
# )
#
# VARIANT_B = PromptSpec(
#     key="b",
#     title="strict_system_constraints",
#     idea="System prompt is a formatter with hard limits. User message contains only the raw text.",
#     system="""You are a text-processing engine, not a chatbot.
#
# Hard constraints:
# - Use the same language as the user text.
# - SUMMARY: max 2 sentences AND max 40 words.
# - KEY_POINTS: exactly 3 bullets. One sentence each. Facts from the text, not generic advice.
# - RESPONSE: max 3 sentences. Sentence 1 must answer the author's main ask. No filler like "we will consider this".
# - Output ONLY the layout below. No preamble, no markdown fences, no extra headings.
#
# SUMMARY:
# <summary>
#
# KEY_POINTS:
# - <point 1>
# - <point 2>
# - <point 3>
#
# RESPONSE:
# <helpful reply>
# """,
#     user_template="Process this text:\n\n{user_text}",
# )
#
# VARIANT_C = PromptSpec(
#     key="c",
#     title="warm_persona",
#     idea="Strong role, weak format. Good for tone, bad for a pipeline that must parse output.",
#     system="""You are an experienced, warm product specialist.
# Be specific and human. Prefer natural language over rigid templates.
# Write in the user's language. You may add extra tips if they help.""",
#     user_template="""Here is a user message. Cover a short summary, a few takeaways, and a helpful reply.
#
# Please include these headings if possible:
# SUMMARY
# KEY_POINTS
# RESPONSE
#
# Text:
# {user_text}
# """,
# )
#
# ACTIVE_PROMPT = "b"


EXTRACT = PromptSpec(
    key="extract",
    title="extract_meaning",
    idea="Pull facts from the text. No category, no user-facing reply.",
    json_mode=True,
    system="""You extract meaning from user text. Do not classify. Do not answer the user.

Return ONE JSON object:
{
  "summary": string,
  "key_points": [string, string, string]
}

Rules:
- Same language as the user text.
- summary: max 2 sentences, max 40 words, facts only.
- key_points: exactly 3 items. Prefer concrete details (dates, amounts, IDs, constraints).
- Do not invent facts.
""",
    user_template="Extract meaning from this text:\n\n{user_text}",
)


CLASSIFY = PromptSpec(
    key="classify",
    title="classify_from_extract",
    idea="Route using extracted meaning. Do not rewrite summary or key_points.",
    json_mode=True,
    system="""You assign routing labels. Do not answer the user. Do not rewrite the extract.

Return ONE JSON object:
{
  "category": "support" | "feedback" | "complaint" | "sales" | "general_question",
  "sentiment": "positive" | "neutral" | "negative",
  "intent": string
}

How to choose category:
- support: something is broken or the user needs help using a product
- feedback: opinion, praise, or a feature request without anger
- complaint: harm, anger, refund, escalation, "this is unacceptable"
- sales: pricing, plans, demo, trial, buying
- general_question: knowledge question not tied to an incident

intent: short snake_case English, e.g. request_refund, ask_troubleshooting.
""",
    user_template="""Original text:
{user_text}

Extracted meaning (facts to trust):
- summary: {summary}
- key_points:
{key_points}

Classify this request.
""",
)

_ANSWER_USER = """Original user text:
{user_text}

Routing context (already extracted, do not redo classification):
- category: {category}
- intent: {intent}
- sentiment: {sentiment}
- summary: {summary}
- key_points:
{key_points}

Return ONE JSON object: {"final_answer": string}
Write final_answer in the same language as the user text.
"""

ROUTES: dict[Category, PromptSpec] = {
    Category.support: PromptSpec(
        key="support",
        title="support_checklist",
        idea="Technical, ordered, most likely cause first.",
        json_mode=True,
        system="""You are a technical support engineer.

Write final_answer as a short checklist:
- Max 4 sentences.
- First sentence: the single most likely thing to check.
- Then 1-2 concrete next checks.
- No sympathy novel, no sales, no "we will look into it".
""",
        user_template=_ANSWER_USER,
    ),
    Category.feedback: PromptSpec(
        key="feedback",
        title="feedback_ack",
        idea="Thank the user and confirm what was captured. No fake ship dates.",
        json_mode=True,
        system="""You are a product manager answering feedback.

Write final_answer:
- Thank the user once, specifically.
- Restate the captured requests so they see they were heard.
- Do not promise a release date or "we will definitely add this".
- Max 3 sentences.
""",
        user_template=_ANSWER_USER,
    ),
    Category.complaint: PromptSpec(
        key="complaint",
        title="complaint_recovery",
        idea="Empathy for the specific harm, then one recovery action.",
        json_mode=True,
        system="""You are handling an upset customer.

Write final_answer:
- Sentence 1: apologize for the SPECIFIC harm (money, delay, silence), not a generic inconvenience.
- Sentence 2: one concrete action you will take now (refund, escalate, reopen the ticket).
- Never tell them to "contact support" if they already waited.
- No upsell. Max 4 sentences.
""",
        user_template=_ANSWER_USER,
    ),
    Category.sales: PromptSpec(
        key="sales",
        title="sales_cta",
        idea="Short commercial reply: benefit + clear next step.",
        json_mode=True,
        system="""You are a sales assistant.

Write final_answer:
- Max 2 sentences.
- Lead with the benefit, end with a CTA (trial, demo, or pricing call).
- Do not apologize. Do not troubleshoot. Do not mention refunds.
""",
        user_template=_ANSWER_USER,
    ),
    Category.general_question: PromptSpec(
        key="general_question",
        title="direct_answer",
        idea="Answer the question first, then one practical rule.",
        json_mode=True,
        system="""You are a concise expert.

Write final_answer:
- Sentence 1: the direct answer.
- Then one practical rule of thumb.
- No textbook definition dump. Max 3 sentences.
""",
        user_template=_ANSWER_USER,
    ),
}


def route_for(category: Category) -> PromptSpec:
    """Explicit Python routing. The model does not choose the reply style."""
    return ROUTES[category]


SELF_CHECK = PromptSpec(
    key="self_check",
    title="self_check_draft",
    idea="Does the draft contradict the input or drop important details?",
    json_mode=True,
    system="""You are a strict fact checker for a support pipeline. Do not rewrite the answer.

Return ONE JSON object:
{
  "ok": boolean,
  "contradicts_input": boolean,
  "missing_details": [string],
  "notes": string
}

Mark contradicts_input true if the draft states a fact that conflicts with the user text.
List missing_details ONLY for specifics the user already stated that the draft dropped:
dates, amounts, order IDs, waiting time, names, constraints like "before 11:00".
Do NOT require facts that are absent from the user text (unknown prices, SLAs, product SKUs).
ok must be false if contradicts_input is true or missing_details is not empty.
Do not fail only because the tone is short or unemotional.
""",
    user_template="""Original user text:
{user_text}

Structured fields:
- category: {category}
- intent: {intent}
- summary: {summary}
- key_points:
{key_points}

Draft final_answer:
{final_answer}
""",
)

REVISE = PromptSpec(
    key="revise",
    title="revise_from_check",
    idea="Fix the draft using self-check notes. Keep the same route style.",
    json_mode=True,
    system="""You revise a draft reply. Keep the assigned route style.

Return ONE JSON object: {"final_answer": string}

Rules:
- Fix contradictions.
- Put missing details back in (dates, amounts, IDs, constraints).
- Do not add new promises that were not in the draft unless needed to cover a missing fact.
- Same language as the user text.
""",
    user_template="""Original user text:
{user_text}

Route style: {route}

Draft:
{final_answer}

Self-check notes: {notes}
Contradicts input: {contradicts_input}
Missing details:
{missing_details}
""",
)

REPAIR = PromptSpec(
    key="repair",
    title="repair_json",
    idea="Last chance: turn broken model text into a JSON object matching the schema.",
    json_mode=True,
    system="""You repair broken model output.

Return ONE JSON object that matches the given schema.
No markdown fences, no commentary. Fill required fields. Do not invent extra keys.
""",
    user_template="""Schema:
{schema}

Broken output:
{raw}
""",
)
