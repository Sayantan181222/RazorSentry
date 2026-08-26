# LLM is NEVER in the decision path — it only drafts notes after the model has already decided REVIEW

import os
import json
import urllib.request

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
FALLBACK_NOTE = "Analyst note unavailable — manual review required"


# Builds the prompt string sent to Groq from transaction context and model output
def _build_prompt(transaction: dict, score: float, reasons: list) -> str:
    reasons_text = "; ".join(reasons) if reasons else "none provided"
    return (
        f"You are a fraud analyst assistant at a payments company.\n"
        f"A transaction has been flagged for REVIEW by the fraud detection model.\n\n"
        f"Transaction details:\n"
        f"  Amount: {transaction.get('amount', 0):,.2f}\n"
        f"  Type: {transaction.get('transaction_type', transaction.get('type', 'UNKNOWN'))}\n"
        f"  Fraud score: {score:.4f}\n"
        f"  Top risk reasons: {reasons_text}\n\n"
        f"Write exactly 2 lines:\n"
        f"Line 1: A one-sentence risk summary for the analyst.\n"
        f"Line 2: A one-sentence suggested action for the analyst.\n"
        f"No headings, no bullet points, no extra text."
    )


# Calls Groq API using urllib to generate a 2-line analyst note for a REVIEW-queue transaction
def generate_analyst_note(transaction: dict, score: float, reasons: list) -> str:
    try:
        if not GROQ_API_KEY:
            return FALLBACK_NOTE
        prompt = _build_prompt(transaction, score, reasons)
        payload = json.dumps({
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 120,
            "temperature": 0.3,
        }).encode("utf-8")
        req = urllib.request.Request(
            GROQ_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return FALLBACK_NOTE
