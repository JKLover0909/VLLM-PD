# MSM/MSA Text-to-SQL context

Concise authoring reference for SQLite queries against the **development mock**. It describes schema facts and fixture behavior; it is not an implementation of a SQL agent or a runtime security boundary.

## Query contract

- Target only the nine business tables listed below. `schema_metadata` is SQLite fixture metadata, not a business source.
- Generate one read-only SQLite `SELECT` (a `WITH ... SELECT` is fine). Do not generate writes, DDL, `PRAGMA`, `ATTACH`, multiple statements, or queries against SQLite system tables.
- Name columns explicitly; bind user-provided values as parameters. Do not interpolate them into SQL. Put a reasonable `LIMIT` on detail results; a limit on grouped output does not replace necessary filters.
- These instructions do **not** provide runtime table allowlisting, SQL validation, read-only enforcement, or time/row limits. Those must be implemented separately.

## Schema facts

The source of truth for SQLite names, types, keys, defaults, and declared foreign keys is [`schema/msm_msa.sql`](schema/msm_msa.sql). It defines nine business tables; the DDL has no CHECK constraints describing status domains or date semantics.

| Table | Row grain | Declared relation / key useful for queries |
|---|---|---|
| `MSM_MACHINE_DEPARTMENT` | Department/code | `DEPARTMENT_ID` is unique; referenced by machine and MSA rows |
| `MSM_USER_PERMISSION` | User name | `USER_NAME` is unique; no FK to Oracle core `USERS` in this mock |
| `MSM_PLAN_FILE` | Plan-file row | Has `YEAR` and `MONTH`; no unique constraint on the pair and no machine FK |
| `MSM_TRACE_DIAGRAM` | Customer | `CUSTOMER_ID` is unique; no machine FK |
| `MSM_MACHINE` | Measurement machine | `MACHINE_ID` is unique; `DEPARTMENT_ID` references the department table |
| `MSM_USER_EMAIL` | Email setting | `USER_NAME` references `MSM_USER_PERMISSION.USER_NAME` and is nullable; multiple rows per user are possible |
| `MSM_CALIB_JUDGE` | Calibration/history or schedule row | `MACHINE_ID` references `MSM_MACHINE.MACHINE_ID` and is nullable; multiple rows per machine are possible |
| `MSA_MACHINE` | MSA configuration | `MACHINE_ID` is unique, NOT NULL, and references `MSM_MACHINE.MACHINE_ID`; department FK also declared |
| `MSA_JUDGE` | MSA plan/result row | Nullable `MACHINE_ID` references `MSA_MACHINE.MACHINE_ID`; department FK also declared; multiple rows per machine are possible |

Use only these declared joins:

```text
MSM_MACHINE.DEPARTMENT_ID       → MSM_MACHINE_DEPARTMENT.DEPARTMENT_ID
MSM_CALIB_JUDGE.MACHINE_ID      → MSM_MACHINE.MACHINE_ID
MSA_MACHINE.MACHINE_ID          → MSM_MACHINE.MACHINE_ID
MSA_MACHINE.DEPARTMENT_ID       → MSM_MACHINE_DEPARTMENT.DEPARTMENT_ID
MSA_JUDGE.MACHINE_ID            → MSA_MACHINE.MACHINE_ID
MSA_JUDGE.DEPARTMENT_ID         → MSM_MACHINE_DEPARTMENT.DEPARTMENT_ID
MSM_USER_EMAIL.USER_NAME        → MSM_USER_PERMISSION.USER_NAME
```

`MSM_PLAN_FILE` and `MSM_TRACE_DIAGRAM` are standalone in this schema. Do not invent joins by names, product, customer, or attachment fields. The mock requires each `MSA_MACHINE` to reference an MSM machine, so it cannot represent all possible Oracle configurations.

## Query choices and safe patterns

- Use `MACHINE_ID` for machine joins, not machine names. A machine name or serial number is not declared unique.
- Distinguish row counts from entity counts: `COUNT(*)` counts rows at the selected table grain; use `COUNT(DISTINCT MACHINE_ID)` when counting machines.
- Do not join `MSM_CALIB_JUDGE` and `MSA_JUDGE` as raw child rows before aggregating; each can have many rows per machine and their combination multiplies counts. Aggregate each child table by machine first.
- A row in `MSA_MACHINE` is the configuration record. `MSM_MACHINE.IS_MSA` is a separate column; do not assume the two agree outside the generated fixture.
- Keep nullable values nullable. A plan date is not evidence that an MSA result exists. For mock rows with a recorded result date, `RESULT_DATE_UNIX` is non-empty and numerically greater than zero; this indicates recorded fixture data, not validated measurement quality.
- Return the stored status/flag values unless the user requests a filter and its meaning is confirmed. Do not infer API access from `HAS_PERMISSION`, that an email was sent from `SEND_EMAIL`, or details of a failure from sheet flags or notes.

### Example: machine lookup

```sql
SELECT m.MACHINE_ID, m.MACHINE_NAME, m.STATUS,
       m.CALIBRATION_STATUS, d.DEPARTMENT_NAME
FROM MSM_MACHINE AS m
LEFT JOIN MSM_MACHINE_DEPARTMENT AS d
  ON d.DEPARTMENT_ID = m.DEPARTMENT_ID
WHERE m.MACHINE_ID = ?
LIMIT 1;
```

### Example: aggregate child rows before joining

```sql
WITH calibration AS (
    SELECT MACHINE_ID, COUNT(*) AS calibration_rows
    FROM MSM_CALIB_JUDGE
    GROUP BY MACHINE_ID
), msa AS (
    SELECT MACHINE_ID, COUNT(*) AS msa_rows
    FROM MSA_JUDGE
    GROUP BY MACHINE_ID
)
SELECT m.MACHINE_ID,
       COALESCE(c.calibration_rows, 0) AS calibration_rows,
       COALESCE(a.msa_rows, 0) AS msa_rows
FROM MSM_MACHINE AS m
LEFT JOIN calibration AS c ON c.MACHINE_ID = m.MACHINE_ID
LEFT JOIN msa AS a ON a.MACHINE_ID = m.MACHINE_ID
WHERE m.MACHINE_ID = ?
LIMIT 1;
```

## Fixture conventions (not production contracts)

The generator [`../scripts/create_msm_mock_database.py`](../scripts/create_msm_mock_database.py) creates synthetic, deterministic data for development. The following describe that fixture only:

- It stores date-only epoch values as Unix **seconds** (generated at 12:00 UTC). In `MSA_JUDGE`, `PLAN_DATE_UNIX` and `RESULT_DATE_UNIX` are SQLite `TEXT` containing stringified seconds; cast to integer for numeric comparison. Other epoch fields are generally `INTEGER` as declared by the schema. `CREATE_DATE`/`EDIT_DATE` are text timestamps formatted `YYYY-MM-DD HH:MM:SS`, without a timezone suffix.
- Some date-like fields use `0` as a missing-date sentinel; some result fields use `NULL`. For text epoch fields, a mock-safe numeric predicate is:

```sql
CAST(NULLIF(TRIM(j.RESULT_DATE_UNIX), '') AS INTEGER) > 0
```

  SQLite can cast malformed text to `0`; validate data before applying this to any non-fixture database. The production timezone and local-day boundary convention are not established here.
- MSA conclusions, pass/fail values, statuses, IDs, notes, attachments, and populations are synthetic. In particular, numbers embedded in `MSA_JUDGE.CONCLUSION` are simulated text, not raw measurements or valid KPI inputs. Do not parse or aggregate them as real metrics.
- The fixture's status distributions and generated flag patterns are implementation details of the generator, not definitions of operational business rules.

## Unverified business assumptions — do not encode as universal filters

The SQLite DDL alone does not establish the meaning or valid domain of flags/statuses, what qualifies as a submitted or completed calibration, which machine date is the authoritative due date, how reschedules affect due dates, production timezone rules, whether duplicated department/name fields are synchronized, or MSA acceptance thresholds. Column names and `DEFAULT` values do not settle these questions.

When an answer depends on one of these definitions, preserve/return the source fields, ask for clarification, or state that the schema does not establish the rule. Do not fabricate 5M1E/checksheet details, employee names, authentication/RBAC data, sent-email history, realtime state, or entities absent from these tables. Avoid returning email addresses or other identifiers unless explicitly necessary and permitted.

The Oracle schema path previously referenced for comments, `database/schema/MES_HOABINH_MSM_MSA_SCHEMA.sql`, is absent from this development worktree. The SQLite DDL says it is derived from nine Oracle business tables, but Oracle comments and Oracle-to-SQLite differences have not been independently verified here.

## Sources and maintenance

- SQLite schema: [`schema/msm_msa.sql`](schema/msm_msa.sql)
- Synthetic fixture generator: [`../scripts/create_msm_mock_database.py`](../scripts/create_msm_mock_database.py)
- Fixture setup and limitations: [`README.md`](README.md)

When updating this guide, verify table/column names and constraints against the SQLite DDL. Keep generator behavior and business-owner-confirmed rules in separate sections; never promote a fixture pattern or an unverified assumption into a production contract.
