"""
Memory layer — what makes Mirror personal.

Unlike Yunjue Agent (which only evolves tools), Mirror also evolves:
  1. Preferences: structured key-value pairs learned from interactions
  2. Persona: a model of the user's behavior patterns, communication style
  3. Episodic memory: important past events and decisions
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger("mirror.memory")

# ── Preference Extraction ──────────────────────

PREFERENCE_EXTRACTION_PROMPT = """Analyze this user interaction and extract any new preferences or facts about the user.

User message: {user_message}
Agent response: {agent_response}

Output ONLY valid JSON with this structure:
{
    "preferences": {"key": "value", ...},  // new or updated preferences
    "persona_updates": {"trait": "value", ...},  // personality/behavior observations
    "important_facts": ["fact1", "fact2"]  // things worth remembering long-term
}

If nothing new is learned, return empty collections.
"""


def extract_memory_updates(
    user_message: str,
    agent_response: str,
    llm_call: callable,
) -> dict[str, Any]:
    """
    Analyze an interaction to extract new knowledge about the user.

    Returns:
        {
            "preferences": {...},
            "persona_updates": {...},
            "important_facts": [...]
        }
    """
    prompt = PREFERENCE_EXTRACTION_PROMPT.format(
        user_message=user_message,
        agent_response=agent_response,
    )

    try:
        result = llm_call(prompt)
        return json.loads(result)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Memory extraction failed: {e}")
        return {"preferences": {}, "persona_updates": {}, "important_facts": []}


# ── Health Data Integration ────────────────────

def parse_apple_health_export(xml_path: str) -> dict[str, Any]:
    """
    Parse an Apple Health export XML to extract key metrics.

    Extracts: steps, heart_rate, sleep, weight, workouts
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    metrics = {
        "steps": [],
        "heart_rate": [],
        "sleep": [],
        "weight": [],
        "workouts": [],
    }

    type_map = {
        "HKQuantityTypeIdentifierStepCount": "steps",
        "HKQuantityTypeIdentifierHeartRate": "heart_rate",
        "HKCategoryTypeIdentifierSleepAnalysis": "sleep",
        "HKQuantityTypeIdentifierBodyMass": "weight",
        "HKWorkoutTypeIdentifier": "workouts",
    }

    for record in root.findall(".//Record"):
        rec_type = record.get("type", "")
        target = type_map.get(rec_type)
        if not target:
            continue

        value = record.get("value")
        start = record.get("startDate", "")
        end = record.get("endDate", "")

        metrics[target].append({
            "value": float(value) if value else 0,
            "start": start,
            "end": end,
        })

    # Aggregate
    summary = {}
    for key, records in metrics.items():
        if not records:
            continue
        values = [r["value"] for r in records]
        summary[key] = {
            "count": len(records),
            "total": sum(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    return summary


# ── Daily Insight Generator ────────────────────

def generate_daily_insight(
    health_data: dict[str, Any],
    preferences: dict[str, Any],
    persona: dict[str, Any],
    llm_call: callable,
) -> str:
    """
    Generate a personalized daily insight combining health data and preferences.
    """
    prompt = f"""You are Mirror, a personal AI. Generate a friendly, personalized daily insight.

User's health data today:
{json.dumps(health_data, ensure_ascii=False, indent=2)}

What you know about this person:
Preferences: {json.dumps(preferences, ensure_ascii=False)}
Personality: {json.dumps(persona, ensure_ascii=False)}

Write a warm, concise insight (2-3 sentences) that connects their health data
with what you know about them. Be supportive, not judgmental. Speak in Chinese."""

    return llm_call(prompt)
