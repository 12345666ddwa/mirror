"""
Insight Engine — Proactive pattern discovery.

Unlike reactive agents that only answer questions, the Insight Engine
continuously analyzes user data and surfaces things the user SHOULD know
but hasn't asked about.

This is the "卧槽" moment generator.
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional


@dataclass
class Insight:
    """A single insight discovered by the engine."""
    id: str
    category: str  # sleep, activity, productivity, mood, health, pattern
    title: str
    description: str
    severity: str  # info, warning, positive, critical
    data: dict = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "data": self.data,
            "created_at": self.created_at,
        }


class InsightEngine:
    """
    Analyzes time-series personal data for patterns, anomalies, and trends.

    Methods work without LLM — pure statistical analysis first,
    LLM-enhanced insights second (for natural language generation).
    """

    def __init__(self, llm_generate: Optional[callable] = None):
        self.llm = llm_generate
        self.previous_insights: list[Insight] = []

    def analyze(
        self,
        health_data: dict[str, list[dict]],
        screen_time: Optional[list[dict]] = None,
        preferences: Optional[dict] = None,
    ) -> list[Insight]:
        """
        Analyze all available data and generate insights.

        Args:
            health_data: {"steps": [...], "sleep": [...], "heart_rate": [...]}
            screen_time: [{"date": ..., "minutes": ..., "app": ...}, ...]

        Returns:
            List of discovered insights
        """
        insights = []

        # Sleep analysis
        if "sleep" in health_data:
            insights.extend(self._analyze_sleep(health_data["sleep"]))

        # Activity analysis
        if "steps" in health_data:
            insights.extend(self._analyze_activity(health_data["steps"]))

        # Heart rate analysis
        if "heart_rate" in health_data:
            insights.extend(self._analyze_heart_rate(health_data["heart_rate"]))

        # Screen time
        if screen_time:
            insights.extend(self._analyze_screen_time(screen_time))

        # Cross-domain correlations
        if len(health_data) >= 2:
            insights.extend(self._cross_domain_analysis(health_data, screen_time))

        # Sort by severity
        severity_order = {"critical": 0, "warning": 1, "positive": 2, "info": 3}
        insights.sort(key=lambda x: severity_order.get(x.severity, 99))

        self.previous_insights = insights
        return insights

    # ── Sleep Analysis ────────────────────────

    def _analyze_sleep(self, data: list[dict]) -> list[Insight]:
        insights = []
        if len(data) < 3:
            return insights

        # Extract values
        durations = []
        bedtimes = []
        for d in data:
            dur = d.get("duration_hours") or d.get("value", 0)
            if dur > 0:
                durations.append(dur)
            bt = d.get("bedtime") or d.get("start", "")
            if bt:
                bedtimes.append(bt)

        if not durations:
            return insights

        # Trend: is sleep getting worse?
        if len(durations) >= 7:
            recent = durations[-7:]
            older = durations[-14:-7] if len(durations) >= 14 else durations[:7]
            recent_avg = sum(recent) / len(recent)
            older_avg = sum(older) / len(older)

            if recent_avg < older_avg - 0.5:
                change_pct = abs((recent_avg - older_avg) / older_avg * 100)
                insights.append(Insight(
                    id="sleep_decline",
                    category="sleep",
                    title="睡眠时间在减少",
                    description=f"最近7天平均睡眠 {recent_avg:.1f} 小时，比之前减少了 {change_pct:.0f}%。长期睡眠不足会影响记忆力和免疫力。",
                    severity="warning",
                    data={"recent_avg": recent_avg, "older_avg": older_avg, "change_pct": change_pct},
                ))

        # Consistency
        if len(durations) >= 5:
            avg = sum(durations) / len(durations)
            variance = sum((d - avg) ** 2 for d in durations) / len(durations)
            std = variance ** 0.5
            if std > 1.5:
                insights.append(Insight(
                    id="sleep_irregular",
                    category="sleep",
                    title="睡眠时间不稳定",
                    description=f"你的入睡和起床时间波动较大（标准差 {std:.1f} 小时）。规律的睡眠对深度睡眠质量影响很大。",
                    severity="info",
                    data={"std_dev": std},
                ))

        # Positive: good sleep
        if len(durations) >= 7 and sum(durations) / len(durations) >= 7.5:
            insights.append(Insight(
                id="sleep_good",
                category="sleep",
                title="睡眠质量不错",
                description=f"最近平均睡眠 {sum(durations)/len(durations):.1f} 小时，保持在健康范围。继续保持！",
                severity="positive",
            ))

        return insights

    # ── Activity Analysis ─────────────────────

    def _analyze_activity(self, data: list[dict]) -> list[Insight]:
        insights = []
        if len(data) < 3:
            return insights

        steps = []
        for d in data:
            s = d.get("steps") or d.get("value", 0)
            if s > 0:
                steps.append(s)

        if not steps:
            return insights

        avg = sum(steps) / len(steps)

        # Sedentary warning
        if len(steps) >= 7 and avg < 5000:
            insights.append(Insight(
                id="low_activity",
                category="activity",
                title="活动量偏低",
                description=f"最近日均 {avg:.0f} 步，低于推荐的 8000 步。久坐与多种健康风险相关。试试每天多走 15 分钟？",
                severity="warning",
                data={"avg_steps": avg},
            ))

        # Weekend warrior pattern
        if len(steps) >= 14:
            weekdays = steps[-10:-3] if len(steps) >= 10 else steps[:5]
            weekends = [steps[i] for i in range(len(steps)) if i % 7 in (5, 6)]
            if weekdays and weekends and sum(weekends) / len(weekends) > sum(weekdays) / len(weekdays) * 1.5:
                insights.append(Insight(
                    id="weekend_warrior",
                    category="activity",
                    title="周末集中运动模式",
                    description="你周末运动量远大于工作日。分散到每天会更有利于心血管健康。",
                    severity="info",
                ))

        return insights

    # ── Heart Rate Analysis ───────────────────

    def _analyze_heart_rate(self, data: list[dict]) -> list[Insight]:
        insights = []
        if len(data) < 5:
            return insights

        rates = [d.get("bpm") or d.get("value", 0) for d in data]
        rates = [r for r in rates if 40 < r < 200]

        if not rates:
            return insights

        avg = sum(rates) / len(rates)
        resting = min(rates)

        if resting > 75:
            insights.append(Insight(
                id="high_resting_hr",
                category="health",
                title="静息心率偏高",
                description=f"静息心率 {resting:.0f} bpm，高于正常范围（60-75）。可能和压力、睡眠不足、缺乏运动有关。",
                severity="warning",
                data={"resting_hr": resting},
            ))

        if resting < 55:
            insights.append(Insight(
                id="low_resting_hr",
                category="health",
                title="静息心率偏低",
                description=f"静息心率 {resting:.0f} bpm。如果你是经常运动的人，这是心肺功能好的表现！",
                severity="positive",
                data={"resting_hr": resting},
            ))

        return insights

    # ── Screen Time ───────────────────────────

    def _analyze_screen_time(self, data: list[dict]) -> list[Insight]:
        insights = []
        if len(data) < 3:
            return insights

        totals = []
        for d in data:
            m = d.get("minutes") or d.get("value", 0)
            if m > 0:
                totals.append(m)

        if not totals:
            return insights

        avg = sum(totals) / len(totals)

        if avg > 360:  # >6 hours/day
            insights.append(Insight(
                id="high_screen_time",
                category="productivity",
                title="屏幕时间偏高",
                description=f"日均屏幕使用 {avg/60:.1f} 小时。其中大部分可能不是工作相关的。要不要试试设置App限额？",
                severity="warning",
                data={"avg_hours": avg / 60},
            ))

        return insights

    # ── Cross-Domain Correlations ─────────────

    def _cross_domain_analysis(
        self, health: dict, screen: Optional[list[dict]]
    ) -> list[Insight]:
        insights = []

        # Sleep + Activity correlation
        if "sleep" in health and "steps" in health:
            sleep_data = health["sleep"]
            step_data = health["steps"]

            if len(sleep_data) >= 7 and len(step_data) >= 7:
                # Simplified correlation: high activity days → better sleep?
                sleep_vals = [d.get("duration_hours", 0) for d in sleep_data[-7:]]
                step_vals = [d.get("steps", d.get("value", 0)) for d in step_data[-7:]]

                both = [(s, t) for s, t in zip(sleep_vals, step_vals) if s > 0 and t > 0]
                if len(both) >= 5:
                    high_activity_days = [s for s, t in both if t > 8000]
                    low_activity_days = [s for s, t in both if t <= 5000]

                    if high_activity_days and low_activity_days:
                        high_avg = sum(high_activity_days) / len(high_activity_days)
                        low_avg = sum(low_activity_days) / len(low_activity_days)

                        if high_avg > low_avg + 0.5:
                            insights.append(Insight(
                                id="activity_sleep_link",
                                category="pattern",
                                title="运动日睡得更香",
                                description=f"你运动量大的日子（>8000步）平均睡眠 {high_avg:.1f}h，比少运动的日子多 {high_avg-low_avg:.1f}h。运动真的能帮你睡得更好。",
                                severity="positive",
                                data={"high_activity_sleep": high_avg, "low_activity_sleep": low_avg},
                            ))

        return insights

    def generate_narrative(self, insights: list[Insight]) -> str:
        """Generate a natural-language daily briefing from insights."""
        if not insights:
            return "今天没有特别的变化，一切正常。保持现有节奏就好 ☀️"

        parts = []
        for i in insights:
            emoji = {"warning": "⚠️", "positive": "✨", "info": "💡", "critical": "🚨"}
            parts.append(f"{emoji.get(i.severity, '')} {i.title}: {i.description}")

        return "\n\n".join(parts)


# ── Demo Data Generator ───────────────────────

def generate_demo_data(days: int = 30) -> dict:
    """
    Generate realistic demo data so new users immediately see value.
    This is critical for the "first-use wow moment".
    """
    import random
    random.seed(42)

    today = datetime.now()
    data = {
        "sleep": [],
        "steps": [],
        "heart_rate": [],
        "screen_time": [],
    }

    for i in range(days):
        date = today - timedelta(days=days - 1 - i)
        date_str = date.strftime("%Y-%m-%d")

        # Sleep: 7.5h baseline, STRONG decline last 7 days (ensure warning triggers)
        base_sleep = 7.5
        if i >= days - 7:
            base_sleep -= (i - (days - 7)) * 0.22  # Steep decline
        sleep_hours = max(4.0, min(9, base_sleep + random.gauss(0, 0.4)))
        bedtime = date.replace(hour=23, minute=random.randint(0, 59))

        data["sleep"].append({
            "date": date_str,
            "duration_hours": round(sleep_hours, 1),
            "bedtime": bedtime.strftime("%H:%M"),
            "deep_sleep_pct": random.uniform(15, 25),
        })

        # Realistic steps: weekday LOW, weekend HIGH (to trigger sedentary warning)
        is_weekend = date.weekday() >= 5
        steps_base = random.randint(8000, 12000) if is_weekend else random.randint(2000, 5500)
        data["steps"].append({
            "date": date_str,
            "steps": steps_base,
        })

        # Heart rate
        resting = random.randint(58, 78)
        data["heart_rate"].append({
            "date": date_str,
            "bpm": resting,
            "resting": resting - random.randint(3, 8),
        })

        # Screen time: intentionally high to trigger warning
        data["screen_time"].append({
            "date": date_str,
            "minutes": random.randint(300, 540) if is_weekend else random.randint(240, 480),
        })

    return data
