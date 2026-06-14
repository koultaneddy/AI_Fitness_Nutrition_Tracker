import os
import json
from datetime import date, datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Load environment ──────────────────────────────────────────────
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── File helpers ──────────────────────────────────────────────────
def load_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def save_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

# ── Daily total from nutrition log ────────────────────────────────
def get_daily_totals(log_data, target_date=None):
    if target_date is None:
        target_date = str(date.today())
    totals = {"protein_g": 0, "carbs_g": 0, "fat_g": 0, "calories": 0, "meals": 0}
    for entry in log_data["log"]:
        if entry.get("date") == target_date:
            totals["protein_g"] += entry.get("protein_g", 0)
            totals["carbs_g"] += entry.get("carbs_g", 0)
            totals["fat_g"] += entry.get("fat_g", 0)
            totals["calories"] += entry.get("calories", 0)
            totals["meals"] += 1
    return totals

# ── Build system prompt ───────────────────────────────────────────
def build_system_prompt():
    profile = load_json("data/profile.json")
    workout_log = load_json("data/workout_log.json")
    nutrition_log = load_json("data/nutrition_log.json")
    daily_totals = get_daily_totals(nutrition_log)

    return f"""
You are a personal fitness and nutrition coach with access to the user's
complete profile, workout history, and nutrition history.

Always read from the provided data rather than relying on conversation history.
Be concise, specific, and practical in your responses.

## User Profile
{json.dumps(profile, indent=2)}

## Recent Workout Log (last 7 entries)
{json.dumps(workout_log["log"][-7:], indent=2)}

## Recent Nutrition Log (last 7 entries)
{json.dumps(nutrition_log["log"][-7:], indent=2)}

## Today's Running Totals ({date.today()})
Meals logged today: {daily_totals["meals"]}
Protein: {daily_totals["protein_g"]}g (floor: {profile["nutrition_targets"]["protein_g_floor"]}g)
Carbs: {daily_totals["carbs_g"]}g
Fat: {daily_totals["fat_g"]}g
Calories: {daily_totals["calories"]} kcal (ceiling: {profile["nutrition_targets"]["calories_ceiling"]} kcal)

## Your capabilities
- Tell the user what their next workout is based on current_position in profile
- Log a workout session when the user provides it
- Log meal entries — after EVERY meal log, output a JSON block so the app can save it
- Show running daily totals after every meal entry
- Summarize today's nutrition progress
- Answer questions about their routine, progression flags, or goals

## IMPORTANT: Meal logging format
When the user logs a meal, always:
1. Show the breakdown per food item with protein, carbs, fat, and calories
2. Output a JSON block in this EXACT format so the app can save it — do NOT add any text or running totals after the JSON block, the app handles that:
```json
{{
  "MEAL_LOG": {{
    "date": "YYYY-MM-DD",
    "time": "HH:MM",
    "description": "brief meal description",
    "protein_g": 0,
    "carbs_g": 0,
    "fat_g": 0,
    "calories": 0,
    "items": []
  }}
}}
```
3. Show the updated running daily total including this meal

Today's date: {date.today()}
Current time: {datetime.now().strftime("%H:%M")}
"""

# ── Parse and save meal log from agent response ───────────────────
def extract_and_save_meal(response_text):
    try:
        start = response_text.find('{"MEAL_LOG"')
        if start == -1:
            start = response_text.find('{\n  "MEAL_LOG"')
        if start == -1:
            return False

        # Find the matching closing brace
        depth = 0
        end = start
        for i, ch in enumerate(response_text[start:]):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = start + i + 1
                    break

        json_str = response_text[start:end]
        parsed = json.loads(json_str)
        meal_entry = parsed["MEAL_LOG"]

        nutrition_log = load_json("data/nutrition_log.json")
        nutrition_log["log"].append(meal_entry)
        save_json("data/nutrition_log.json", nutrition_log)
        return True
    except Exception:
        return False

# ── Format daily totals for display ──────────────────────────────
def format_daily_totals():
    profile = load_json("data/profile.json")
    nutrition_log = load_json("data/nutrition_log.json")
    t = get_daily_totals(nutrition_log)
    protein_floor = profile["nutrition_targets"]["protein_g_floor"]
    cal_ceiling = profile["nutrition_targets"]["calories_ceiling"]
    protein_remaining = protein_floor - t["protein_g"]
    cal_remaining = cal_ceiling - t["calories"]

    return (
        f"\n📊 Today's Running Total ({t['meals']} meal(s) logged):\n"
        f"  Protein : {t['protein_g']}g / {protein_floor}g floor "
        f"({'✅' if t['protein_g'] >= protein_floor else f'{protein_remaining}g to go'})\n"
        f"  Carbs   : {t['carbs_g']}g\n"
        f"  Fat     : {t['fat_g']}g\n"
        f"  Calories: {t['calories']} / {cal_ceiling} kcal "
        f"({'⚠️ over!' if t['calories'] > cal_ceiling else f'{cal_remaining} remaining'})\n"
    )

# ── Main conversation loop ────────────────────────────────────────
def main():
    print("\n💪 Fitness & Nutrition Agent")
    print("Type 'quit' to exit\n")

    system_prompt = build_system_prompt()
    history = []

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Agent: See you next session. Keep pushing.")
            break

        if not user_input:
            continue

        history.append(
            types.Content(role="user", parts=[types.Part(text=user_input)])
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(system_instruction=system_prompt),
            contents=history
        )

        reply = response.text

        # Check if a meal was logged and save it
        meal_saved = extract_and_save_meal(reply)

        # Clean display output
        display_reply = reply
        if meal_saved:
            # Remove the JSON block
            start = display_reply.find('{"MEAL_LOG"')
            if start == -1:
                start = display_reply.find('{\n  "MEAL_LOG"')
            if start != -1:
                depth = 0
                end = start
                for i, ch in enumerate(display_reply[start:]):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = start + i + 1
                            break
                display_reply = display_reply[:start] + display_reply[end:]

            # Strip leftover markdown code fences
            import re
            display_reply = re.sub(r'```json\s*```', '', display_reply)
            display_reply = re.sub(r'```\s*```', '', display_reply)
            display_reply = display_reply.strip()

            # Strip agent's own running total section to avoid duplicate
            import re
            display_reply = re.sub(
                r"\*{0,2}Today's Running Totals?.*",
                "",
                display_reply,
                flags=re.DOTALL | re.IGNORECASE
            )
            # Strip any leftover markdown headers
            display_reply = re.sub(r'\n#+\s*$', '', display_reply).strip()

        history.append(
            types.Content(role="model", parts=[types.Part(text=reply)])
        )

        print(f"\nAgent: {display_reply.strip()}\n")

        # Print running totals after any meal log
        if meal_saved:
            print(format_daily_totals())

if __name__ == "__main__":
    main()