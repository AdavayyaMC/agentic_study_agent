#!/usr/bin/env python3
import os, json, datetime, requests, sys, re
from pathlib import Path

# --- LLM imports (LangChain) ---
try:
    from langchain.llms import HuggingFaceHub
    from langchain import LLMChain, PromptTemplate
except Exception as e:
    print("LangChain imports failed:", e)
    raise

# --- Config (via env or defaults) ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
HF_MODEL = os.environ.get("HF_MODEL", "google/flan-t5-large")  # change if desired
EXAM_DATE = os.environ.get("EXAM_DATE", None)  # format YYYY-MM-DD
DAILY_HOURS = float(os.environ.get("DAILY_HOURS", "2"))

ROOT = Path(__file__).parent
SYLLABUS_FILE = ROOT / "syllabus.json"
PLAN_FILE = ROOT / "plan.json"

# --- Utilities ---
def today_str(offset=0):
    d = datetime.date.today() + datetime.timedelta(days=offset)
    return d.isoformat()

def read_json_file(path):
    if not path.exists(): return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json_file(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token or chat id missing; skipping send.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        r = requests.post(url, json=payload, timeout=15)
        print("Telegram status:", r.status_code, r.text)
    except Exception as e:
        print("Telegram send failed:", e)

# --- LLM calling helpers ---
def make_hf_llm():
    if not HF_TOKEN:
        raise RuntimeError("HUGGINGFACEHUB_API_TOKEN required in env")
    llm = HuggingFaceHub(
        repo_id=HF_MODEL,
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"temperature": 0.0, "max_length": 512}
    )
    return llm

plan_prompt_template = """
You are a planning assistant. Inputs:
- exam_date: {exam_date}
- start_date: {start_date}
- daily_hours: {daily_hours}
- syllabus: {syllabus_json}

Create a JSON array (only the JSON) of objects with fields:
- date (YYYY-MM-DD)
- topic (string)
- duration_hours (number)
- status (one of: pending, done)
- notes (string, optional)

Constraints:
1) Place study sessions from start_date up to exam_date (exclusive).
2) Do not exceed daily_hours total duration per date.
3) Prioritize 'high' priority topics earlier; include at least one review day every 6 days.
4) Try to spread long topics across multiple days.
5) Output valid JSON (an array). Keep entries ordered by date.

Return only the JSON array (no commentary).
"""

rebalanced_prompt_template = """
You are an assistant that rebalances an existing study plan by inserting missed sessions.
Inputs:
- existing_plan: {existing_plan}
- missed_sessions: {missed}
- start_date: {start_date}
- exam_date: {exam_date}
- daily_hours: {daily_hours}

Produce a new valid JSON array (only the JSON) with fields:
- date, topic, duration_hours, status, notes
Keep previously 'done' items unchanged. Insert missed sessions fairly (no daily_hours exceed).
Keep output strictly JSON.
"""

def call_llm_generate_plan(syllabus, start_date, exam_date, daily_hours):
    llm = make_hf_llm()
    prompt = PromptTemplate(
        input_variables=["exam_date","start_date","daily_hours","syllabus_json"],
        template=plan_prompt_template
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    syllabus_json = json.dumps(syllabus, ensure_ascii=False)
    out = chain.run({"exam_date":exam_date, "start_date":start_date, "daily_hours":str(daily_hours), "syllabus_json": syllabus_json})
    return extract_json_from_text(out)

def call_llm_rebalance(existing_plan, missed, start_date, exam_date, daily_hours):
    llm = make_hf_llm()
    prompt = PromptTemplate(
        input_variables=["existing_plan","missed","start_date","exam_date","daily_hours"],
        template=rebalanced_prompt_template
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    out = chain.run({
        "existing_plan": json.dumps(existing_plan, ensure_ascii=False),
        "missed": json.dumps(missed, ensure_ascii=False),
        "start_date": start_date,
        "exam_date": exam_date,
        "daily_hours": str(daily_hours)
    })
    return extract_json_from_text(out)

def extract_json_from_text(text):
    # best-effort extract JSON array from LLM output
    text = text.strip()
    # find first '[' and last ']'
    first = text.find('[')
    last = text.rfind(']')
    if first != -1 and last != -1 and last > first:
        candidate = text[first:last+1]
    else:
        candidate = text
    try:
        return json.loads(candidate)
    except Exception as e:
        # attempt to fix trailing commas etc then parse
        candidate = re.sub(r',\s*}', '}', candidate)
        candidate = re.sub(r',\s*\]', ']', candidate)
        try:
            return json.loads(candidate)
        except Exception as e2:
            print("Failed to parse LLM output as JSON. Raw output:")
            print(text)
            raise

# --- Main flow ---
def main():
    if not EXAM_DATE:
        print("Set EXAM_DATE environment variable (YYYY-MM-DD). Aborting.")
        sys.exit(1)

    syllabus = read_json_file(SYLLABUS_FILE)
    if syllabus is None:
        print("No syllabus.json found. Create one (see README). Aborting.")
        sys.exit(1)

    plan = read_json_file(PLAN_FILE) or []

    # if plan empty -> generate a new plan from today to EXAM_DATE
    if not plan:
        print("No plan found; generating a new plan with the LLM...")
        plan = call_llm_generate_plan(syllabus, today_str(0), EXAM_DATE, DAILY_HOURS)
        write_json_file(PLAN_FILE, plan)
        send_telegram(f"✅ Generated a study plan from {today_str(0)} to {EXAM_DATE}. Check plan.json in repo.")
        return

    # send today's tasks
    today = today_str(0)
    todays = [p for p in plan if p.get("date") == today and p.get("status","pending")!="done"]
    if todays:
        msg = f"📚 Study plan for {today}:\n"
        for i,t in enumerate(todays,1):
            msg += f"{i}) {t.get('topic')} — {t.get('duration_hours')} hrs\n"
        msg += "\nTo mark tasks done: edit plan.json in the repo and set status to \"done\" for that entry."
        send_telegram(msg)
    else:
        print("No tasks scheduled for today.")

    # detect missed sessions (date < today and status != done)
    missed = [p for p in plan if p.get("date") < today and p.get("status","pending")!="done"]
    if missed:
        print(f"Detected {len(missed)} missed session(s). Asking LLM to rebalance...")
        # mark missed with note
        for m in missed:
            m.setdefault("notes", "")
            m["notes"] = ("reschedule: missed on " + today + "; " + m.get("notes","")).strip()
            m["status"] = "missed"
        # Rebalance
        new_plan = call_llm_rebalance(plan, missed, today, EXAM_DATE, DAILY_HOURS)
        write_json_file(PLAN_FILE, new_plan)
        send_telegram(f"🔁 Rebalanced plan to accommodate {len(missed)} missed session(s). Check the updated plan.json.")
    else:
        print("No missed sessions detected.")

if __name__ == "__main__":
    main()
