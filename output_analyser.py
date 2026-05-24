from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a senior data analyst embedded in a business intelligence platform called Insightiq.

Your job is to write a concise, plain-English narrative that explains what a set of SQL \
query results means for a non-technical business stakeholder. You will receive:

  1. The original user question / analysis requirement.
  2. The scenario / analyst instructions that guided the SQL generation (may be empty).
  3. The SQL query that was executed.
  4. The actual query result data in CSV format (may be truncated for large results).

Output guidelines:
  - Write 2–4 short paragraphs maximum.
  - Open with a single sentence that directly answers the user's question using the data.
  - In the body, highlight the most important numbers, trends, comparisons, or anomalies \
    visible in the data. Be specific — quote actual values where they add clarity.
  - If the data is empty or has only one row, say so clearly and explain what that implies.
  - Close with a one-sentence so-what / business implication if it is obvious from the data; \
    otherwise skip it.
  - Do NOT reproduce raw CSV rows or table dumps.
  - Do NOT include SQL syntax or technical jargon.
  - Do NOT add headers, bullet points, or markdown — return flowing prose only.
  - Be factual. Do not invent insights that are not supported by the data.
"""

_USER_TEMPLATE = """\
## User Question / Analysis Requirement
{user_input}

## Scenario / Analyst Instructions
{scenario_instruction}

## Executed SQL Query
```sql
{sql_query}
```

## Query Result Data (CSV)
```
{csv_data}
```

Write a plain-English narrative summary of what this data is showing relative to the \
user's question. Follow the output guidelines strictly.
"""
_MAX_CSV_CHARS: int = 8_000


class OutputAnalyserAgent:
    def __init__(
        self,
        groq_api_key: str,
        groq_model: str,
        base_url: str = "https://api.groq.com/openai/v1",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> None:
        self._api_key = groq_api_key
        self._base_url = base_url.rstrip("/")
        self._model = groq_model
        self._max_tokens = max_tokens
        self._temperature = temperature
    def analyse(
        self,
        user_input: str,
        sql_query: str,
        scenario_instruction: str,
        csv_data: str,
    ) -> str:
        """Generate a plain-English narrative for the given query results.

        Parameters
        ----------
        user_input:
            The full resolved user question sent to the pipeline (may include prior
            follow-up context that was prepended for multi-turn sessions).
        sql_query:
            The SQL query produced by TextToSQLAgent (formatted / pretty-printed).
        scenario_instruction:
            The scenario instruction string returned by ScenarioIdentifierAgent
            (empty string if no scenario was matched above the similarity threshold).
        csv_data:
            Full CSV string of the query result returned by PostgresConnector.
            Will be truncated to ``_MAX_CSV_CHARS`` before being sent to the LLM.

        Returns
        -------
        str
            The narrative text. On failure, a short error message is returned
            instead of raising, so the UI always has something to display.
        """
        truncated_csv, was_truncated = self._truncate_csv(csv_data)

        if was_truncated:
            logger.info(
                "OutputAnalyserAgent: CSV truncated to %d chars (original: %d chars).",
                _MAX_CSV_CHARS,
                len(csv_data),
            )

        user_msg = _USER_TEMPLATE.format(
            user_input=user_input.strip() or "(not provided)",
            scenario_instruction=scenario_instruction.strip() or "(none)",
            sql_query=sql_query.strip() or "(not provided)",
            csv_data=truncated_csv,
        )

        logger.info(
            "OutputAnalyserAgent: sending narrative request "
            "(prompt len=%d, model=%s).",
            len(user_msg),
            self._model,
        )

        try:
            if self._openai_client is not None:
                return self._call_via_openai_sdk(user_msg)
            return self._call_via_requests(user_msg)
        except Exception as exc:  # noqa: BLE001
            logger.exception("OutputAnalyserAgent: LLM call failed: %s", exc)
            return (
                f"Narrative generation failed: {exc}. "
                "Please inspect the query results directly."
            )
    @staticmethod
    def _truncate_csv(csv_data: str) -> tuple[str, bool]:
        """Return (truncated_csv, was_truncated)."""
        if not csv_data:
            return "(no data returned)", False
        if len(csv_data) <= _MAX_CSV_CHARS:
            return csv_data, False
        truncated = csv_data[:_MAX_CSV_CHARS]
        # Try to cut at a clean line boundary to avoid partial rows.
        last_newline = truncated.rfind("\n")
        if last_newline > 0:
            truncated = truncated[:last_newline]
        truncated += "\n… (result set truncated — only the first rows are shown above)"
        return truncated, True

    def _messages(self, user_msg: str) -> list[dict]:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    def _call_via_openai_sdk(self, user_msg: str) -> str:
        resp = self._openai_client.chat.completions.create(
            model=self._model,
            messages=self._messages(user_msg),
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return resp.choices[0].message.content.strip()

    def _call_via_requests(self, user_msg: str) -> str:
        import requests  # noqa: PLC0415

        resp = requests.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": self._messages(user_msg),
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            },
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()