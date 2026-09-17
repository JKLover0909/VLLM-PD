"""Adversarial tests for the CCTVAI Postgres SQL agent.

Two jobs here. First, every query in the semantic model's
``verified_query_examples`` must PASS validation — those are measured-correct
queries from the handoff document, and a validator that rejects them is useless
however safe it is. Second, every shape in ``rejected_query_examples`` plus the
three measured fan-out shapes from handoff §6.1 must be REJECTED.

The fan-out cases matter most: they are syntactically valid, they execute
successfully against the replica, and they return inflated numbers (11,986
instead of 8,305, +44%). Nothing downstream can detect that, so the validator
is the only line of defence.
"""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from src.integrations import cctvai_contract as contract
from src.integrations.cctvai_sql_agent import (
    CctvaiSqlAgent,
    CctvaiSqlAgentError,
    CctvaiSqlQueryResult,
)


SEMANTIC_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "cctvai_semantic_model.json"
)


@pytest.fixture()
def agent() -> CctvaiSqlAgent:
    return CctvaiSqlAgent(SEMANTIC_MODEL_PATH, max_rows=50)


@pytest.fixture(scope="module")
def semantic_model() -> dict:
    return json.loads(SEMANTIC_MODEL_PATH.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Contract consistency
# ----------------------------------------------------------------------


def test_semantic_model_matches_python_table_allowlist(semantic_model):
    assert set(semantic_model["tables"]) == set(contract.REPORTABLE_TABLES)


def test_semantic_model_columns_match_readable_columns(semantic_model):
    for table, spec in semantic_model["tables"].items():
        assert set(spec["columns"]) == set(contract.readable_columns(table)), table


def test_semantic_model_soft_delete_flags_match_contract(semantic_model):
    for table, spec in semantic_model["tables"].items():
        assert spec.get("soft_delete", False) is contract.requires_soft_delete_filter(
            table
        ), table


def test_semantic_model_leaks_no_credentials(semantic_model):
    payload = json.dumps(semantic_model, ensure_ascii=False)
    for blocked in contract.BLOCKED_CAMERA_COLUMNS:
        # Blocked column names may be *named* as forbidden, but only inside the
        # policy text, never inside a table's readable column list.
        for spec in semantic_model["tables"].values():
            assert blocked not in spec["columns"]
    assert "55434" not in payload
    assert "cctvai_llm_ro" not in payload


def test_agent_available_when_model_present(agent):
    assert agent.available is True


def test_from_env_returns_none_when_disabled(monkeypatch):
    monkeypatch.delenv("CCTVAI_SQL_AGENT_ENABLED", raising=False)
    assert CctvaiSqlAgent.from_env() is None
    monkeypatch.setenv("CCTVAI_SQL_AGENT_ENABLED", "false")
    assert CctvaiSqlAgent.from_env() is None


def test_from_env_returns_agent_when_enabled(monkeypatch):
    monkeypatch.setenv("CCTVAI_SQL_AGENT_ENABLED", "true")
    monkeypatch.setenv("CCTVAI_SEMANTIC_MODEL_PATH", str(SEMANTIC_MODEL_PATH))
    built = CctvaiSqlAgent.from_env()
    assert built is not None and built.available


# ----------------------------------------------------------------------
# The verified queries must survive validation
# ----------------------------------------------------------------------


def test_every_verified_example_passes_validation(agent, semantic_model):
    for example in semantic_model["verified_query_examples"]:
        safe_sql, tables, _ = agent.validate_sql(example["sql"])
        assert safe_sql
        assert tables
        assert "LIMIT" in safe_sql.upper()


def test_verified_examples_keep_del_flag_inside_on(agent, semantic_model):
    for example in semantic_model["verified_query_examples"]:
        safe_sql, _, _ = agent.validate_sql(example["sql"])
        # Rendering must not relocate the predicate out of the ON clause.
        if "LEFT JOIN" in safe_sql.upper():
            assert "del_flag" in safe_sql


def test_missing_limit_is_injected(agent):
    safe_sql, _, _ = agent.validate_sql(
        "SELECT e.id FROM cctvai.event_snapshots e ORDER BY e.detected_time DESC"
    )
    assert "LIMIT 50" in safe_sql.upper()


def test_existing_limit_is_preserved(agent):
    safe_sql, _, _ = agent.validate_sql(
        "SELECT e.id FROM cctvai.event_snapshots e LIMIT 7"
    )
    assert "LIMIT 7" in safe_sql.upper()


def test_referenced_tables_are_reported(agent):
    _, tables, _ = agent.validate_sql(
        "SELECT count(e.id) FROM cctvai.event_snapshots e "
        "LEFT JOIN cctvai.cctvai_cameras c "
        "ON lower(c.camera_id) = lower(e.camera_id) AND NOT c.del_flag LIMIT 10"
    )
    assert tables == ("cctvai_cameras", "event_snapshots")


# ----------------------------------------------------------------------
# The measured fan-out shapes — handoff §6.1
# ----------------------------------------------------------------------


FANOUT_SHAPES = [
    pytest.param(
        "SELECT count(*) FROM cctvai.event_snapshots e "
        "JOIN cctvai.cctvai_cameras c ON c.camera_id = e.camera_id "
        "JOIN cctvai.cctvai_violation_types vt ON vt.violation_code = e.violation_type",
        id="three_master_join_no_del_flag_11986_rows",
    ),
    pytest.param(
        "SELECT count(e.id) FROM cctvai.event_snapshots e "
        "JOIN cctvai.cctvai_violation_types vt ON vt.violation_code = e.violation_type",
        id="violation_types_only_no_del_flag_8782_rows",
    ),
    pytest.param(
        "SELECT count(e.id) FROM cctvai.event_snapshots e "
        "LEFT JOIN cctvai.cctvai_cameras c ON lower(c.camera_id) = lower(e.camera_id) "
        "LEFT JOIN cctvai.cctvai_lines l ON l.id = c.line_ref_id "
        "LEFT JOIN cctvai.cctvai_violation_types vt "
        "ON lower(vt.violation_code) = lower(e.violation_type)",
        id="left_join_three_masters_no_del_flag",
    ),
]


@pytest.mark.parametrize("sql", FANOUT_SHAPES)
def test_measured_fanout_shapes_are_rejected(agent, sql):
    with pytest.raises(CctvaiSqlAgentError):
        agent.validate_sql(sql)


def test_del_flag_in_where_instead_of_on_is_rejected(agent):
    """The subtle one: converts LEFT JOIN to INNER JOIN, drops 26 orphans."""
    with pytest.raises(CctvaiSqlAgentError, match="ON"):
        agent.validate_sql(
            "SELECT c.camera_name, count(e.id) FROM cctvai.event_snapshots e "
            "LEFT JOIN cctvai.cctvai_cameras c "
            "ON lower(c.camera_id) = lower(e.camera_id) "
            "WHERE NOT c.del_flag GROUP BY 1 LIMIT 20"
        )


def test_leading_soft_delete_table_without_where_is_rejected(agent):
    with pytest.raises(CctvaiSqlAgentError, match="del_flag"):
        agent.validate_sql(
            "SELECT c.camera_id, c.camera_name FROM cctvai.cctvai_cameras c LIMIT 50"
        )


def test_leading_soft_delete_table_with_where_passes(agent):
    safe_sql, _, _ = agent.validate_sql(
        "SELECT c.camera_id, c.camera_name FROM cctvai.cctvai_cameras c "
        "WHERE NOT c.del_flag LIMIT 50"
    )
    assert safe_sql


@pytest.mark.parametrize(
    "predicate",
    [
        "NOT c.del_flag",
        "c.del_flag = FALSE",
        "c.del_flag IS FALSE",
        "c.del_flag IS NOT TRUE",
    ],
)
def test_accepted_del_flag_negation_spellings(agent, predicate):
    safe_sql, _, _ = agent.validate_sql(
        f"SELECT c.camera_id FROM cctvai.cctvai_cameras c WHERE {predicate} LIMIT 5"
    )
    assert safe_sql


def test_exists_subquery_requires_del_flag(agent):
    with pytest.raises(CctvaiSqlAgentError, match="del_flag"):
        agent.validate_sql(
            "SELECT e.camera_id FROM cctvai.event_snapshots e WHERE NOT EXISTS ("
            "SELECT 1 FROM cctvai.cctvai_cameras c "
            "WHERE lower(c.camera_id) = lower(e.camera_id)) LIMIT 50"
        )


def test_exists_subquery_with_del_flag_passes(agent):
    safe_sql, _, _ = agent.validate_sql(
        "SELECT e.camera_id, count(e.id) AS events FROM cctvai.event_snapshots e "
        "WHERE NOT EXISTS (SELECT 1 FROM cctvai.cctvai_cameras c "
        "WHERE lower(c.camera_id) = lower(e.camera_id) AND NOT c.del_flag) "
        "GROUP BY 1 LIMIT 50"
    )
    assert safe_sql


def test_implicit_cross_join_on_soft_delete_table_is_rejected(agent):
    with pytest.raises(CctvaiSqlAgentError):
        agent.validate_sql(
            "SELECT e.id FROM cctvai.event_snapshots e, cctvai.cctvai_cameras c "
            "WHERE NOT c.del_flag LIMIT 10"
        )


def test_del_flag_of_wrong_alias_does_not_satisfy_the_check(agent):
    """Two soft-delete tables, only one de-duplicated."""
    with pytest.raises(CctvaiSqlAgentError):
        agent.validate_sql(
            "SELECT count(e.id) FROM cctvai.event_snapshots e "
            "LEFT JOIN cctvai.cctvai_cameras c "
            "ON lower(c.camera_id) = lower(e.camera_id) AND NOT c.del_flag "
            "LEFT JOIN cctvai.cctvai_violation_types vt "
            "ON lower(vt.violation_code) = lower(e.violation_type) AND NOT c.del_flag "
            "LIMIT 10"
        )


# ----------------------------------------------------------------------
# Statement-shape and injection rejects
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("INSERT INTO cctvai.event_snapshots (id) VALUES (1)", id="insert"),
        pytest.param("UPDATE cctvai.event_snapshots SET id = 1", id="update"),
        pytest.param("DELETE FROM cctvai.event_snapshots", id="delete"),
        pytest.param("DROP TABLE cctvai.event_snapshots", id="drop"),
        pytest.param("TRUNCATE cctvai.event_snapshots", id="truncate"),
        pytest.param("SET search_path TO cctvai", id="set_search_path"),
        pytest.param("COPY cctvai.event_snapshots TO PROGRAM 'sh'", id="copy_program"),
        pytest.param(
            "SELECT e.id FROM cctvai.event_snapshots e FOR UPDATE", id="for_update"
        ),
        pytest.param(
            "SELECT e.id FROM cctvai.event_snapshots e; SELECT 1", id="two_statements"
        ),
        pytest.param("SELECT pg_sleep(10)", id="pg_sleep"),
        pytest.param("SELECT pg_read_file('/etc/passwd')", id="pg_read_file"),
        pytest.param("SELECT dblink('x', 'y')", id="dblink"),
        pytest.param(
            "SELECT table_name FROM information_schema.tables", id="information_schema"
        ),
        pytest.param(
            "SELECT relname FROM pg_catalog.pg_class", id="pg_catalog"
        ),
        pytest.param(
            "SELECT camera_id FROM cctvai_cameras WHERE NOT del_flag",
            id="missing_schema_prefix",
        ),
        pytest.param(
            "SELECT r.user_id FROM cctvai.cctvai_line_responsibles r "
            "JOIN identity.users u ON u.id = r.user_id",
            id="nonexistent_identity_schema",
        ),
        pytest.param(
            "SELECT o.id FROM cctvai.cctvai_notification_outbox o", id="out_of_scope"
        ),
        pytest.param(
            "SELECT * FROM cctvai.cctvai_cameras c WHERE NOT c.del_flag",
            id="select_star",
        ),
        pytest.param(
            "SELECT count(*) FROM cctvai.event_snapshots e", id="count_star"
        ),
        pytest.param(
            "SELECT c.camera_id, c.password FROM cctvai.cctvai_cameras c "
            "WHERE NOT c.del_flag",
            id="blocked_credential_column",
        ),
        pytest.param(
            "SELECT c.camera_id, c.main_cam_url FROM cctvai.cctvai_cameras c "
            "WHERE NOT c.del_flag",
            id="blocked_url_column",
        ),
        pytest.param(
            "SELECT e.nonexistent_column FROM cctvai.event_snapshots e", id="bad_column"
        ),
        pytest.param(
            "SELECT e.id FROM otherdb.cctvai.event_snapshots e", id="cross_database"
        ),
        pytest.param("SELECT 1", id="no_table"),
        pytest.param("", id="empty"),
    ],
)
def test_rejected_statements(agent, sql):
    with pytest.raises(CctvaiSqlAgentError):
        agent.validate_sql(sql)


def test_every_rejected_example_from_the_model_is_rejected(agent, semantic_model):
    for example in semantic_model["rejected_query_examples"]:
        with pytest.raises(CctvaiSqlAgentError):
            agent.validate_sql(example["sql"])


# ----------------------------------------------------------------------
# Advanced bypass shapes: CTE, UNION, LATERAL, quoted identifiers.
#
# The shape-check in _validate_soft_delete_shapes walks every exp.Select the
# statement contains (statement.find_all(exp.Select)), so a CTE body or a
# UNION branch is itself a scope subject to the same rules — these tests
# confirm that recursion actually holds instead of assuming it from reading
# the code. LATERAL join targets are derived tables (not a bare exp.Table),
# which is not one of the three accepted del_flag shapes, so they are
# rejected unconditionally — including a correctly-filtered one, which is a
# false positive by design (reject-by-default), not a bypass.
# ----------------------------------------------------------------------


def test_cte_leading_soft_delete_table_without_filter_is_rejected(agent):
    with pytest.raises(CctvaiSqlAgentError, match="del_flag"):
        agent.validate_sql(
            "WITH cams AS ("
            "  SELECT camera_id, camera_name, del_flag FROM cctvai.cctvai_cameras"
            ") "
            "SELECT e.id FROM cctvai.event_snapshots e "
            "JOIN cams c ON c.camera_id = e.camera_id LIMIT 10"
        )


def test_cte_prefiltered_leading_table_then_outer_join_passes(agent):
    """A CTE that already filters del_flag itself needs no further check."""
    safe_sql, _, _ = agent.validate_sql(
        "WITH cams AS ("
        "  SELECT cam.camera_id, cam.camera_name FROM cctvai.cctvai_cameras cam "
        "  WHERE NOT cam.del_flag"
        ") "
        "SELECT e.id, c.camera_name FROM cctvai.event_snapshots e "
        "JOIN cams c ON c.camera_id = e.camera_id LIMIT 10"
    )
    assert safe_sql


def test_cte_disguised_fanout_all_three_masters_is_rejected(agent):
    """The exact +44% fan-out shape (handoff §6.1), hidden inside a WITH."""
    with pytest.raises(CctvaiSqlAgentError):
        agent.validate_sql(
            "WITH joined AS ("
            "  SELECT e.id FROM cctvai.event_snapshots e "
            "  JOIN cctvai.cctvai_cameras c ON c.camera_id = e.camera_id "
            "  JOIN cctvai.cctvai_violation_types vt "
            "    ON vt.violation_code = e.violation_type"
            ") "
            "SELECT count(id) FROM joined LIMIT 10"
        )


def test_union_branch_missing_del_flag_is_rejected(agent):
    with pytest.raises(CctvaiSqlAgentError, match="del_flag"):
        agent.validate_sql(
            "SELECT count(e.id) FROM cctvai.event_snapshots e "
            "LEFT JOIN cctvai.cctvai_cameras c "
            "ON lower(c.camera_id) = lower(e.camera_id) AND NOT c.del_flag "
            "UNION ALL "
            "SELECT count(e.id) FROM cctvai.event_snapshots e "
            "JOIN cctvai.cctvai_violation_types vt "
            "ON vt.violation_code = e.violation_type LIMIT 10"
        )


def test_union_both_branches_clean_passes(agent):
    safe_sql, _, _ = agent.validate_sql(
        "SELECT e.camera_id FROM cctvai.event_snapshots e "
        "WHERE e.record_status = 'A' "
        "UNION "
        "SELECT e.camera_id FROM cctvai.event_snapshots e "
        "WHERE e.record_status = 'B' LIMIT 10"
    )
    assert safe_sql


def test_union_blocked_column_in_second_branch_is_rejected(agent):
    with pytest.raises(CctvaiSqlAgentError, match="password"):
        agent.validate_sql(
            "SELECT c.camera_id FROM cctvai.cctvai_cameras c "
            "WHERE NOT c.del_flag "
            "UNION ALL "
            "SELECT c.password FROM cctvai.cctvai_cameras c "
            "WHERE NOT c.del_flag LIMIT 10"
        )


@pytest.mark.parametrize("with_del_flag", [False, True], ids=["unfiltered", "filtered"])
def test_lateral_join_to_soft_delete_table_is_always_rejected(agent, with_del_flag):
    """Not one of the three accepted del_flag shapes — rejected either way."""
    filter_clause = "AND NOT c.del_flag " if with_del_flag else ""
    with pytest.raises(CctvaiSqlAgentError, match="subquery"):
        agent.validate_sql(
            "SELECT e.id, sub.camera_name FROM cctvai.event_snapshots e "
            "JOIN LATERAL ("
            "  SELECT c.camera_name FROM cctvai.cctvai_cameras c "
            f"  WHERE c.camera_id = e.camera_id {filter_clause}"
            ") sub ON true LIMIT 10"
        )


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            'SELECT e.id FROM "CCTVAI".event_snapshots e LIMIT 10',
            id="quoted_schema",
        ),
        pytest.param(
            'SELECT c.camera_id FROM cctvai."Cctvai_Cameras" c '
            "WHERE NOT c.del_flag LIMIT 10",
            id="quoted_table",
        ),
        pytest.param(
            'SELECT c."camera_id" FROM cctvai.cctvai_cameras c '
            "WHERE NOT c.del_flag LIMIT 10",
            id="quoted_column",
        ),
        pytest.param(
            'SELECT e.id FROM cctvai.event_snapshots "e" LIMIT 10',
            id="quoted_alias",
        ),
    ],
)
def test_quoted_identifiers_are_rejected(agent, sql):
    """A quoted identifier is case-*preserving* in Postgres (unlike unquoted,
    which Postgres folds to lowercase) — every check elsewhere in this class
    lower()s before comparing, so a quoted "CCTVAI" would otherwise pass this
    validator while addressing a different, case-sensitive object at
    execution time. Reject quoting outright instead of relying on no such
    object existing today."""
    with pytest.raises(CctvaiSqlAgentError, match="ngoặc kép"):
        agent.validate_sql(sql)


def test_unquoted_mixed_case_schema_still_passes(agent):
    """Postgres folds unquoted identifiers to lowercase regardless of source
    casing, so this is the same object as cctvai.event_snapshots — confirms
    the quoted-identifier rejection above does not also reject this."""
    safe_sql, _, _ = agent.validate_sql(
        "SELECT e.id FROM CcTvAi.event_snapshots e LIMIT 10"
    )
    assert safe_sql


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("SELECT Pg_Sleep(1)", id="mixed_case"),
        pytest.param("SELECT PG_SLEEP(1)", id="uppercase"),
    ],
)
def test_denied_function_prefix_is_case_insensitive(agent, sql):
    with pytest.raises(CctvaiSqlAgentError, match="pg_sleep"):
        agent.validate_sql(sql)


# ----------------------------------------------------------------------
# Epoch-millisecond and open-event traps
# ----------------------------------------------------------------------


def test_to_timestamp_without_divisor_is_rejected(agent):
    with pytest.raises(CctvaiSqlAgentError, match="milliseconds"):
        agent.validate_sql(
            "SELECT e.camera_id FROM cctvai.event_snapshots e "
            "WHERE to_timestamp(e.detected_time) >= now() - interval '1 day' LIMIT 10"
        )


def test_to_timestamp_with_wrong_divisor_is_rejected(agent):
    with pytest.raises(CctvaiSqlAgentError, match="1000.0"):
        agent.validate_sql(
            "SELECT e.camera_id FROM cctvai.event_snapshots e "
            "WHERE to_timestamp(e.detected_time / 60.0) >= now() LIMIT 10"
        )


def test_to_timestamp_with_correct_divisor_passes(agent):
    safe_sql, _, _ = agent.validate_sql(
        "SELECT e.camera_id FROM cctvai.event_snapshots e "
        "WHERE to_timestamp(e.detected_time / 1000.0) >= now() - interval '7 days' "
        "LIMIT 10"
    )
    assert safe_sql


def test_duration_without_null_guard_is_rejected(agent):
    with pytest.raises(CctvaiSqlAgentError, match="end_time"):
        agent.validate_sql(
            "SELECT e.id, (e.end_time - e.detected_time) / 1000.0 AS duration_s "
            "FROM cctvai.event_snapshots e ORDER BY duration_s DESC LIMIT 10"
        )


def test_duration_with_null_guard_passes(agent):
    safe_sql, _, _ = agent.validate_sql(
        "SELECT e.id, (e.end_time - e.detected_time) / 1000.0 AS duration_s "
        "FROM cctvai.event_snapshots e WHERE e.end_time IS NOT NULL "
        "ORDER BY duration_s DESC LIMIT 10"
    )
    assert safe_sql


def test_oversized_sql_is_rejected(agent):
    with pytest.raises(CctvaiSqlAgentError):
        agent.validate_sql("SELECT e.id FROM cctvai.event_snapshots e -- " + "x" * 9000)


# ----------------------------------------------------------------------
# Advisory reason codes
# ----------------------------------------------------------------------


def test_inner_join_to_master_flags_orphan_drop(agent):
    _, _, reason_codes = agent.validate_sql(
        "SELECT count(e.id) FROM cctvai.event_snapshots e "
        "JOIN cctvai.cctvai_cameras c "
        "ON lower(c.camera_id) = lower(e.camera_id) AND NOT c.del_flag LIMIT 10"
    )
    assert contract.REASON_INNER_JOIN_DROPS_ORPHANS in reason_codes


def test_left_join_to_master_has_no_orphan_warning(agent):
    _, _, reason_codes = agent.validate_sql(
        "SELECT count(e.id) FROM cctvai.event_snapshots e "
        "LEFT JOIN cctvai.cctvai_cameras c "
        "ON lower(c.camera_id) = lower(e.camera_id) AND NOT c.del_flag LIMIT 10"
    )
    assert contract.REASON_INNER_JOIN_DROPS_ORPHANS not in reason_codes


# ----------------------------------------------------------------------
# Plan parsing
# ----------------------------------------------------------------------


def test_parse_plan_accepts_fenced_json():
    plan = CctvaiSqlAgent.parse_plan(
        '```json\n{"can_answer": true, "sql": "SELECT 1", "reason": "ok"}\n```'
    )
    assert plan.can_answer and plan.sql == "SELECT 1"


def test_parse_plan_accepts_refusal():
    plan = CctvaiSqlAgent.parse_plan('{"can_answer": false, "reason": "thiếu cột"}')
    assert plan.can_answer is False and plan.reason == "thiếu cột"


@pytest.mark.parametrize(
    "content",
    ["not json", "", '{"can_answer": true, "sql": ""}', '["a"]'],
)
def test_parse_plan_rejects_malformed(content):
    with pytest.raises(CctvaiSqlAgentError):
        CctvaiSqlAgent.parse_plan(content)


# ----------------------------------------------------------------------
# Prompts and answer shaping
# ----------------------------------------------------------------------


def test_planner_prompt_carries_the_del_flag_rule(agent):
    messages = agent.planner_messages("7 ngày qua có bao nhiêu sự kiện?")
    system = messages[0]["content"]
    assert "del_flag" in system
    assert "ON" in system
    assert "count(*)" in system
    assert "1000.0" in system
    assert "cctvai." in system


def test_planner_prompt_includes_previous_error(agent):
    messages = agent.planner_messages("x", previous_error="thiếu del_flag")
    assert "thiếu del_flag" in messages[1]["content"]


def test_answer_prompt_demands_low_confidence_disclosure(agent):
    messages = agent.answer_messages(
        "x", CctvaiSqlQueryResult(columns=["total"], rows=[{"total": 5}])
    )
    assert "replica" in messages[0]["content"].lower()
    assert "thấp" in messages[0]["content"]


def test_answer_is_natural_rejects_leaked_sql():
    assert CctvaiSqlAgent.answer_is_natural("Có 5 sự kiện trong 7 ngày qua.")
    assert not CctvaiSqlAgent.answer_is_natural(
        "SELECT count(e.id) FROM cctvai.event_snapshots"
    )
    assert not CctvaiSqlAgent.answer_is_natural("")


def test_answer_matches_result_requires_identifiers():
    result = CctvaiSqlQueryResult(
        columns=["camera_id", "total"],
        rows=[{"camera_id": "cam007", "total": 42}],
    )
    assert CctvaiSqlAgent.answer_matches_result(
        "Camera cam007 ghi nhận 42 sự kiện.", result
    )
    assert not CctvaiSqlAgent.answer_matches_result("Có một số sự kiện.", result)


def test_answer_matches_result_requires_empty_disclaimer():
    empty = CctvaiSqlQueryResult(columns=["total"], rows=[])
    assert CctvaiSqlAgent.answer_matches_result("Không có dữ liệu phù hợp.", empty)
    assert not CctvaiSqlAgent.answer_matches_result("Có 5 sự kiện.", empty)


def test_fallback_answer_always_states_low_confidence(agent):
    result = CctvaiSqlQueryResult(
        columns=["camera_id", "total"],
        rows=[{"camera_id": "cam007", "total": 42}],
    )
    answer = agent.fallback_answer(result)
    assert "cam007" in answer
    # Wording is user-facing copy; what must never disappear are the two
    # disclosures: the query was generated (unverified) and the data is not
    # realtime. The word "replica" itself is internal jargon, not a contract.
    assert "chưa được kiểm chứng" in answer
    assert "không phải thời gian thực" in answer


def test_fallback_answer_surfaces_orphan_drop_notice(agent):
    result = CctvaiSqlQueryResult(
        columns=["total"],
        rows=[{"total": 1}],
        reason_codes=(contract.REASON_INNER_JOIN_DROPS_ORPHANS,),
    )
    assert "INNER JOIN" in agent.fallback_answer(result)


def test_fallback_answer_surfaces_fanout_warning(agent):
    result = CctvaiSqlQueryResult(
        columns=["total"],
        rows=[{"total": 11986}],
        reason_codes=(contract.REASON_FANOUT_SUSPECTED,),
    )
    assert "nhân bản" in agent.fallback_answer(result)


def test_fallback_answer_handles_empty_result(agent):
    assert "không có dữ liệu" in agent.fallback_answer(
        CctvaiSqlQueryResult(columns=["total"], rows=[])
    ).lower()


def test_prompt_payload_never_claims_realtime(agent):
    payload = CctvaiSqlQueryResult(
        columns=["total"], rows=[{"total": 1}]
    ).prompt_payload()
    assert payload["replica_lag_state"] == contract.REPLICA_LAG_STATE_UNVERIFIED
