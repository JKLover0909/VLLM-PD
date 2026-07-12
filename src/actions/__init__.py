"""Action agents (report/calendar/alert) tách khỏi luồng hỏi đáp RAG/MES.

Giai đoạn A: Report Agent — planner deterministic, thực thi SQL qua guardrail
``MesSqlAgent``, renderer Python thuần (số liệu không qua LLM).
"""
