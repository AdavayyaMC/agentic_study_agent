#!/usr/bin/env python3
"""
Deterministic Study Planner (no LLM).
- Reads syllabus.json (list of {"topic","est_hours", "priority"(optional)})
- Generates a daily plan from today up to EXAM_DATE (exclusive)
- Sends today's tasks to Telegram
- Detects missed sessions (date < today and status != done) and reschedules them
- Writes plan.json (GitHub Action can commit changes back)
"""

import os
import json
import datetime
from pathlib import Path
import math
import requests
import sys
import re

# CONFIG (from env)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
EXAM_DATE = os.environ.get("EXAM_DATE")  # required: YYYY-MM-DD
DAILY_HOURS = float(os.environ.get("DAILY_HOURS", "2"))
REVIEW_EVERY_DAYS = int(os.environ.get("REVIEW_EVERY_DAYS", "6"))  # add review day every N days
MAX_CHUNK_PER_TOPIC_PER_DAY = float(os.environ.get("MAX_CHUNK_PER_TOPIC_PER_DAY", "2.0"))

ROOT = Path(__file__).parent
SYLLABUS_FILE = ROOT / "syllabus.json"
PLAN_FILE = ROOT / "plan.json"

# ---------------- utils ----------------
def today_date():
    return datetime.date.today()

def today_str(offset=0):
    return (today_date() + datetime.timedelta(days=offset)).isoformat()

def parse_date(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d").date()

def read_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured; skipping send.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
    try:
        r = requests.post(url, json=payload, timeout=15)
        print("Telegram status:", r.status_code, r.text)
    except Exception as e:
        print("Failed to send Telegram:", e)

# ---------------- planning logic ----------------
def normalize_syllabus(syllabus):
    normalized = []
    for item in syllabus:
        topic = item.get("topic") or item.get("name")
        if not topic:
            continue
        est = float(item.get("est_hours", item.get("hours", 1)))
        priority = str(item.get("priority", "medium")).lower()
        weight = {"high": 3, "medium": 2, "low": 1}.get(priority, 2)
        normalized.append({"topic": topic, "remaining": est, "priority": priority, "weight": weight})
    # sort by weight desc (higher priority first), stable
    normalized.sort(key=lambda x: (-x["weight"], x["topic"]))
    return normalized

def generate_plan(syllabus, start_date_str, exam_date_str, daily_hours):
    start_date = parse_date(start_date_str)
    exam_date = parse_date(exam_date_str)
    if exam_date <= start_date:
        raise ValueError("EXAM_DATE must be after start_date (today).")

    days = (exam_date - start_date).days
    if days <= 0:
        raise ValueError("No days available to plan.")

    topics = normalize_syllabus(syllabus)
    total_hours_needed = sum(t["remaining"] for t in topics)
    capacity = days * daily_hours

    plan = []
    current_topic_index = 0

    # quick warning flag if not enough time
    not_enough_time = total_hours_needed > capacity

    for day_offset in range(days):
        date = (start_date + datetime.timedelta(days=day_offset)).isoformat()
        capacity_left = daily_hours

        # Add periodic review slot (reserve 1 hour if possible)
        if (day_offset + 1) % REVIEW_EVERY_DAYS == 0:
            review_dur = min(1.0, capacity_left)
            if review_dur > 0:
                plan.append({"date": date, "topic": "Review & Practice", "duration_hours": round(review_dur,2), "status":"pending", "notes":"Periodic review day"})
                capacity_left -= review_dur

        # allocate topics in round-robin starting from current_topic_index
        attempts = 0
        num_topics = len(topics)
        # if no topics left break
        if not any(t["remaining"] > 0 for t in topics):
            # optional: add buffer or break
            continue

        while capacity_left > 0 and any(t["remaining"] > 0 for t in topics) and attempts < num_topics * 4:
            # find next topic with remaining > 0
            found = False
            for i in range(num_topics):
                idx = (current_topic_index + i) % num_topics
                t = topics[idx]
                if t["remaining"] > 0:
                    # chunk size: don't give more than MAX_CHUNK_PER_TOPIC_PER_DAY to a topic per day
                    max_chunk = min(MAX_CHUNK_PER_TOPIC_PER_DAY, capacity_left)
                    alloc = min(t["remaining"], max_chunk, capacity_left)
                    # allocate minimum granularity 0.25 hours if alloc is tiny
                    if alloc <= 0:
                        continue
                    alloc = round(float(alloc), 2)
                    plan.append({"date": date, "topic": t["topic"], "duration_hours": alloc, "status":"pending", "notes":""})
                    t["remaining"] = round(t["remaining"] - alloc, 2)
                    capacity_left = round(capacity_left - alloc, 2)
                    # next time continue from next topic
                    current_topic_index = (idx + 1) % num_topics
                    found = True
                    break
            if not found:
                break
            attempts += 1

    # after scheduling until exam_date, collect any remaining topic hours as UNSCHEDULED
    unscheduled = []
    for t in topics:
        if t["remaining"] > 0.0001:
            unscheduled.append({"topic": t["topic"], "remaining_hours": round(t["remaining"],2), "priority": t["priority"]})

    # if unscheduled, append them as entries with date "UNSCHEDULED" so user sees overflow
    for u in unscheduled:
        plan.append({"date":"UNSCHEDULED", "topic": u["topic"], "duration_hours": u["remaining_hours"], "status":"pending", "notes":"Not enough days before exam; increase daily hours or start earlier."})

    # attach a top-level metadata note as first element? Instead, include no metadata in plan file but return flag
    return plan, not_enough_time, total_hours_needed, capacity

# ---------------- rescheduling missed ----------------
def reschedule_missed(plan, start_date_str, exam_date_str, daily_hours):
    today = today_date()
    start_date = parse_date(start_date_str)
    exam_date = parse_date(exam_date_str)
    days = (exam_date - today).days
    if days < 0:
        days = 0

    # find missed sessions: date < today and status not 'done'
    missed = [p for p in plan if p.get("date") not in (None,"UNSCHEDULED") and p.get("date") != "" and p.get("date") < today.isoformat() and p.get("status","pending") != "done" and p.get("status","pending") != "missed"]
    if not missed:
        return plan, 0

    # mark them missed
    for m in missed:
        m["status"] = "missed"
        m.setdefault("notes", "")
        m["notes"] = ("marked missed on " + today.isoformat() + "; " + m["notes"]).strip()

    # build occupancy map for dates from today to exam_date-1
    occupancy = {}
    for d_offset in range(days):
        d = (today + datetime.timedelta(days=d_offset)).isoformat()
        occupancy[d] = 0.0
    # include future planned sessions (pending) into occupancy
    for p in plan:
        d = p.get("date")
        if d in occupancy and p.get("status","pending") != "missed":
            try:
                occupancy[d] += float(p.get("duration_hours",0))
            except:
                occupancy[d] += 0.0

    placed = 0
    unscheduled = []
    for m in missed:
        dur = float(m.get("duration_hours", 0))
        placed_flag = False
        # try to find earliest day from today to exam_date-1 with space
        for d_offset in range(days):
            d = (today + datetime.timedelta(days=d_offset)).isoformat()
            if occupancy.get(d, 0.0) + dur <= daily_hours + 1e-6:
                # place
                new_entry = {"date": d, "topic": m.get("topic"), "duration_hours": dur, "status":"pending", "notes":"rescheduled from " + str(m.get("date"))}
                plan.append(new_entry)
                occupancy[d] += dur
                placed_flag = True
                placed += 1
                break
        if not placed_flag:
            # cannot place before exam_date
            unscheduled.append(m)
    # unscheduled remain marked as missed in plan; optionally add entries with date UNSCHEDULED
    for u in unscheduled:
        plan.append({"date":"UNSCHEDULED", "topic": u.get("topic"), "duration_hours": float(u.get("duration_hours", u.get("duration_hours",0) or u.get("duration_hours",0) or 0)), "status":"pending", "notes":"reschedule failed: not enough free days before exam."})
    return plan, placed

# ---------------- main ----------------
def main():
    if not EXAM_DATE:
        print("ERROR: EXAM_DATE env var not set (YYYY-MM-DD).")
        sys.exit(1)

    syllabus = read_json(SYLLABUS_FILE)
    if syllabus is None:
        print(f"ERROR: {SYLLABUS_FILE} not found. Create a syllabus.json.")
        sys.exit(1)

    plan = read_json(PLAN_FILE) or []

    start_date_str = today_str(0)
    exam_date_str = EXAM_DATE

    # if no plan exist, generate one
    if not plan:
        print("No plan.json found or empty -> generating plan (deterministic)...")
        plan, not_enough_time, total_needed, capacity = generate_plan(syllabus, start_date_str, exam_date_str, DAILY_HOURS)
        write_json(PLAN_FILE, plan)
        summary = f"Generated plan from {start_date_str} to {exam_date_str}. Required hours: {total_needed}. Capacity: {capacity}."
        if not_enough_time:
            summary += " WARNING: not enough time to finish all topics before exam."
        print(summary)
        send_telegram("✅ Study plan generated.\n" + summary)
        return

    # send today's tasks
    today = today_str(0)
    todays = [p for p in plan if p.get("date") == today and p.get("status","pending") != "done"]
    if todays:
        msg = f"📚 Study plan for {today}:\n"
        for i,t in enumerate(todays,1):
            msg += f"{i}) {t.get('topic')} — {t.get('duration_hours')} hrs\n"
        msg += "\nTo mark tasks done: edit plan.json in the repo and set status to \"done\" for that entry."
        send_telegram(msg)
    else:
        print("No tasks scheduled for today.")

    # detect missed sessions and reschedule them
    updated_plan, placed = reschedule_missed(plan, start_date_str, exam_date_str, DAILY_HOURS)
    if placed > 0:
        write_json(PLAN_FILE, updated_plan)
        send_telegram(f"🔁 Rescheduled {placed} missed session(s). Check updated plan.json.")
    else:
        print("No missed sessions to reschedule.")

if __name__ == "__main__":
    main()
