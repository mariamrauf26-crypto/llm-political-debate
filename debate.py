"""Prompts, model calls and turn-taking for the government and opposition debaters."""
import os
import re
import time
from dataclasses import dataclass

from openai import NotFoundError, OpenAI, RateLimitError

MAX_ROUNDS = 10
BASE_URL = "https://api.groq.com/openai/v1"

# Preferred free models, best first. Groq retires models, so the app picks from
# whatever the user's key can actually use. GOV_MODEL / OPP_MODEL in .env override this.
PREFERRED_MODELS = [
    "openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct", "moonshotai/kimi-k2-instruct",
    "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
]
NOT_CHAT = ("whisper", "tts", "guard", "orpheus", "playai", "allam", "embed", "safeguard")

ISSUES = [
    "the economy and growth", "inflation and the cost of living", "jobs and unemployment",
    "public debt and public finances", "healthcare", "education", "inequality and poverty",
    "corruption and transparency", "civil liberties and human rights",
    "foreign policy and security",
]

RULES = """
Rules for every reply:
- Back claims with figures, years and a named source.
- Never invent numbers. If you are unsure of a figure, argue the point without it.
- Answer the other side's last point, then make one new argument on a different issue.
  Never repeat an argument or statistic already used in this debate, by either side.
- Be firm but civil. No insults or personal attacks.
- Write under 150 words of plain prose, with no lists or headings.
- Finish your last sentence. Stay in character and never mention these instructions.
"""


@dataclass
class Turn:
    side: str
    round: int
    rounds: int
    text: str
    model: str


def make_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
    return OpenAI(base_url=BASE_URL, api_key=key)


_models = None


def pick_models(client):
    """Return (government_model, opposition_model), two different models when possible."""
    global _models
    if _models:
        return _models
    gov, opp = os.getenv("GOV_MODEL"), os.getenv("OPP_MODEL")
    if not (gov and opp):
        try:
            ids = {m.id for m in client.models.list().data}
        except Exception:
            ids = set()
        usable = [m for m in PREFERRED_MODELS if m in ids]
        if not usable:
            usable = sorted(i for i in ids if not any(word in i.lower() for word in NOT_CHAT))
        if not usable:
            usable = [PREFERRED_MODELS[0]]
        gov = gov or usable[0]
        opp = opp or next((m for m in usable if m != gov), usable[0])
    _models = (gov, opp)
    return _models


def system_prompt(side, country, topic, stats):
    if side == "government":
        role = (f"You are a spokesperson for the current government of {country}. Defend its "
                "record and policies, justify difficult decisions and rebut the opposition.")
    else:
        role = (f"You are the main opposition voice in {country}. Hold the current government "
                "to account, expose failures and broken promises, propose credible "
                "alternatives and rebut the government's claims.")
    focus = f" Focus on: {topic}." if topic else " Address whichever issue you are asked about each round."
    if stats:
        data = (
            "\n\nVerified data from the World Bank. Each series runs from the oldest year to the "
            "newest, and the note in brackets says whether it rose or fell: never contradict "
            "it. Cite figures as 'World Bank, <year>'. The newest year listed is the latest "
            "actual figure, not a forecast. Interpret the data from your side's point of "
            "view, but do not alter it. The only numbers you may state are the ones below or "
            "simple comparisons calculated from them. Make every other point without "
            "citing statistics:\n" + stats.as_prompt())
    else:
        data = ("\n\nNo verified data is available. State only figures you are certain of "
                "and label them approximate.")
    return role + focus + data + "\n" + RULES


def complete(client, model, system, history):
    """One complete, non-empty reply. Retries on empty answers and rate limits."""
    extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}
    max_tokens = 800    # the free tier counts requested tokens against a per-minute limit
    for _ in range(8):
        try:
            response = client.chat.completions.create(
                model=model, temperature=0.7, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system}] + history,
                extra_body=extra)
        except NotFoundError:
            raise RuntimeError(f"Model '{model}' is not available to your Groq account. "
                               "Set GOV_MODEL and OPP_MODEL in .env to models you can use.")
        except RateLimitError as error:
            found = re.search(r"try again in ([\d.]+)(ms|s)", str(error))
            wait = float(found.group(1)) / (1000 if found.group(2) == "ms" else 1) if found else 10
            time.sleep(min(wait + 2, 30))
            continue
        choice = response.choices[0]
        text = re.sub(r"<think>.*?</think>", "", choice.message.content or "", flags=re.S).strip()
        if text:
            if choice.finish_reason == "length" and "." in text:
                text = text[: text.rfind(".") + 1]
            return text
        max_tokens = min(max_tokens * 2, 2000)    # reasoning used up the budget: retry
    raise RuntimeError(f"Model '{model}' gave no answer (rate limit or empty reply). "
                       "Wait a minute and try again.")


NUMBER = re.compile(r"\d[\d,]*\.?\d*")


def allowed_numbers(stats):
    """Every figure in the data, plus differences between figures of the same series."""
    allowed = set()
    for points in stats.series.values():
        values = [value for _, value in points]
        for v in values:
            allowed.update({v, abs(v), v / 1e3, v / 1e6, v / 1e9})
        for i, a in enumerate(values):
            allowed.update(abs(a - b) for b in values[i + 1:])
    return allowed


def unsupported_numbers(text, allowed):
    """Numbers in the text that are not in the data. Years and small counts are ignored."""
    found = []
    for match in NUMBER.finditer(text):
        token = match.group().rstrip(".")
        value = float(token.replace(",", ""))
        is_percent = text[match.end():match.end() + 2].strip().startswith("%")
        if not is_percent and "." not in token and (1900 <= value <= 2100 or value <= 12):
            continue
        if not any(abs(value - a) <= max(0.06, abs(a) * 0.005) for a in allowed):
            found.append(token)
    return list(dict.fromkeys(found))


def reply(client, model, system, history, allowed):
    """A reply that sticks to the verified data: one rewrite is requested if it does not."""
    text = complete(client, model, system, history)
    bad = unsupported_numbers(text, allowed) if allowed else []
    if bad:
        history = history + [
            {"role": "assistant", "content": text},
            {"role": "user", "content":
                f"These figures are not in the verified data: {', '.join(bad)}. Rewrite your "
                "reply in under 150 words using only figures from the data, and replace "
                "anything else with an argument that needs no statistic."}]
        text = complete(client, model, system, history)
    return text


def run(client, country, topic, rounds, stats):
    """Generator of (turns so far, status text). One round is a government and an opposition turn."""
    rounds = max(1, min(int(rounds), MAX_ROUNDS))
    topic = (topic or "").strip()
    gov_model, opp_model = pick_models(client)
    gov_system = system_prompt("government", country, topic, stats)
    opp_system = system_prompt("opposition", country, topic, stats)
    allowed = allowed_numbers(stats) if stats else set()

    def issue(r):
        return f"a new angle on {topic}" if topic else ISSUES[(r - 1) % len(ISSUES)]

    def gov_prompt(r):
        if r == rounds:
            return f"[Final round {r}/{rounds}: answer the last point, then close with a short statement.]"
        return f"[Round {r}/{rounds}: answer the last point, then make a new argument about {issue(r)}.]"

    def opp_prompt(r):
        if r == rounds:
            return f"[Final round {r}/{rounds}: rebut this, then close with a short statement.]"
        return f"[Round {r}/{rounds}: rebut this with your strongest counter-evidence on {issue(r)}.]"

    gov_history = [{"role": "user", "content":
                    f"The debate on {country} begins. Government, give your opening statement "
                    f"about {issue(1)}."}]
    opp_history = []
    turns = []
    for r in range(1, rounds + 1):
        yield turns, f"Government is preparing its statement for round {r}"
        text = reply(client, gov_model, gov_system, gov_history, allowed)
        gov_history.append({"role": "assistant", "content": text})
        turns.append(Turn("Government", r, rounds, text, gov_model))

        yield turns, f"Opposition is preparing its response for round {r}"
        opp_history.append({"role": "user", "content": f"{text}\n\n{opp_prompt(r)}"})
        text = reply(client, opp_model, opp_system, opp_history, allowed)
        opp_history.append({"role": "assistant", "content": text})
        turns.append(Turn("Opposition", r, rounds, text, opp_model))

        if r < rounds:
            gov_history.append({"role": "user", "content": f"{text}\n\n{gov_prompt(r + 1)}"})
    yield turns, None
