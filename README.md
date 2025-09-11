# Agentic Study Planner

## How it works
- `syllabus.json` contains topics and estimated hours.
- Action runs daily and writes `plan.json`.
- To mark sessions done: edit `plan.json` in the repo (GitHub web UI) and set `"status": "done"`.

## Required GitHub secrets
- TELEGRAM_TOKEN
- TELEGRAM_CHAT_ID
- HUGGINGFACEHUB_API_TOKEN

Optionally set repository Variables:
- EXAM_DATE (YYYY-MM-DD)
- DAILY_HOURS
