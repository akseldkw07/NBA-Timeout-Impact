"""Shared helper functions for the NBA Data Explorer app."""

import json
import re
import time as _time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import duckdb
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st
from anthropic import Anthropic

from nba_timeout_impact.constants import NBAConstants
from nba_timeout_impact.webapp.schema_context import build_system_prompt

DATA_DIR = NBAConstants.NBA_DATA_DIR
WEBAPP_DIR = Path(__file__).parent
QUERY_LOG = WEBAPP_DIR / "query_log.jsonl"

MODEL_SONNET = "claude-sonnet-4-6"
MODEL_OPUS = "claude-opus-4-6"
LLM_TIMEOUT_SECS = 180
QUERY_TIMEOUT_SECS = 120
MAX_DISPLAY_ROWS = 2000


def _to_ascii(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")


@st.cache_resource
def load_data():
    pbp = pl.read_parquet(DATA_DIR / "cdnnba_enriched.parquet")

    # Compute columns expected by schema_context but missing from our pipeline
    pbp = (
        pbp.with_columns(
            pl.col("seconds_remaining").alias("clock_seconds"),
            pl.col("subType").alias("subType_clean"),
            pl.col("descriptor").alias("descriptor_clean"),
            pl.col("qualifiers").alias("qualifiers_clean"),
        )
        .sort("gameId", "orderNumber")
        .with_columns(
            pl.col("subType").shift(1).over("gameId").alias("prev_sub_type"),
            pl.col("subType").shift(1).over("gameId").alias("prev_sub_type_clean"),
        )
    )

    player_stats = pl.read_parquet(DATA_DIR / "player_season_stats.parquet")
    player_advanced = pl.read_parquet(DATA_DIR / "player_advanced_stats.parquet")

    _rename = {
        "PLAYER_ID": "personId",
        "PLAYER_NAME": "playerName",
        "TEAM_ID": "teamId",
        "TEAM_ABBREVIATION": "teamTricode",
    }
    player_stats = player_stats.rename({k: v for k, v in _rename.items() if k in player_stats.columns})
    player_advanced = player_advanced.rename({k: v for k, v in _rename.items() if k in player_advanced.columns})

    # Add player_name_ascii where missing
    if "player_name_ascii" not in player_advanced.columns and "playerName" in player_advanced.columns:
        player_advanced = player_advanced.with_columns(
            pl.col("playerName").map_elements(_to_ascii, return_dtype=pl.String).alias("player_name_ascii")
        )
    if "player_name_ascii" not in player_stats.columns and "playerName" in player_stats.columns:
        player_stats = player_stats.with_columns(
            pl.col("playerName").map_elements(_to_ascii, return_dtype=pl.String).alias("player_name_ascii")
        )

    conn = duckdb.connect()
    conn.register("pbp", pbp.to_arrow())
    conn.register("player_stats", player_stats.to_arrow())
    conn.register("player_advanced", player_advanced.to_arrow())

    stints_path = DATA_DIR / "stints.parquet"
    if stints_path.exists():
        stints = pl.read_parquet(stints_path)
        conn.register("stints", stints.to_arrow())

    boxscores_path = DATA_DIR / "boxscores.parquet"
    if boxscores_path.exists():
        boxscores = pl.read_parquet(boxscores_path)
        conn.register("boxscores", boxscores.to_arrow())

    rotations_path = DATA_DIR / "rotations.parquet"
    if rotations_path.exists():
        rotations = pl.read_parquet(rotations_path)
        conn.register("rotations", rotations.to_arrow())

    return conn, pbp, player_stats, player_advanced


def get_system_prompt(pbp, player_stats, player_advanced):
    stints_path = DATA_DIR / "stints.parquet"
    stints_rows = pl.read_parquet(stints_path).height if stints_path.exists() else None
    return build_system_prompt(pbp.height, player_stats.height, player_advanced.height, stints_rows)


def get_client():
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    return Anthropic(api_key=api_key)


def call_llm(client, messages, system=None, model=None, max_tokens=4096):
    kwargs = dict(
        model=model or MODEL_SONNET,
        max_tokens=max_tokens,
        messages=messages,
        timeout=LLM_TIMEOUT_SECS,
    )
    if system:
        kwargs["system"] = system
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(client.messages.create, **kwargs)
        timer = st.empty()
        start = _time.time()
        while not future.done():
            elapsed = _time.time() - start
            timer.caption(f"Waiting for response... {elapsed:.0f}s")
            _time.sleep(0.5)
        timer.empty()
        return future.result().content[0].text


def build_messages(history, user_msg):
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": user_msg})
    return messages


def summarize_result(df, max_rows=5):
    lines = [f"Rows: {df.height}, Columns: {df.columns}"]
    lines.append(f"First {min(max_rows, df.height)} rows:\n{df.head(max_rows)}")
    numeric_cols = [c for c in df.columns if df[c].dtype.is_integer() or df[c].dtype.is_float()]
    if numeric_cols:
        desc = df.select(numeric_cols).describe()
        lines.append(f"Stats:\n{desc}")
    return "\n".join(lines)


SANITY_SYSTEM = """You are a data quality reviewer for NBA play-by-play queries.

CRITICAL DATA MODEL FACTS:
- The `possession` column is a TEAM ID (e.g. 1610612737), NOT a possession number. Each game only has 3 values: 0, team1_id, team2_id. GROUP BY gameId, possession does NOT give individual possessions — it groups ALL events for one team in one game.
- To get individual possessions, you must detect changes in the possession column using window functions.
- player_stats contains SEASON TOTALS, not per-game stats. Values like PTS=1932 are correct for a season.
- shotResult is only non-null for field goal attempts (isFieldGoal=1).

Review the SQL logic and results. Flag issues like:
- Grouping by possession column directly (wrong — gives per-team-per-game, not per-possession)
- Unreasonable row counts or values
- Missing filters that would change results
- Logic errors in CTEs (columns not carried through, wrong joins)
Be brief — 2-4 sentences max."""


PROMPT_SUGGESTION_SYSTEM = """You are a system prompt engineer. Given a failure (SQL error or incorrect results) from an LLM generating DuckDB SQL over NBA data, suggest a concise addition to the system prompt that would prevent this class of error in the future. Return ONLY the text to add — 1-3 sentences, written as an instruction. Example: "When using CTEs, always SELECT all columns needed by downstream CTEs, not just the columns for the current step." """


TITLE_SYSTEM = 'Generate a short title (max 60 chars) and 1-sentence description for this NBA data query result. Return JSON: {"title": "...", "description": "..."}'


def sanity_check(client, question, sql, df):
    summary = summarize_result(df)
    msg = (
        f'The user asked: "{question}"\n\n'
        f"SQL:\n{sql}\n\n"
        f"Result:\n{summary}\n\n"
        "Is this correct? Flag any issues. Start your response with PASS or FAIL."
    )
    return call_llm(client, [{"role": "user", "content": msg}], system=SANITY_SYSTEM, max_tokens=512)


def sanity_check_opus(client, question, sql, df):
    summary = summarize_result(df)
    msg = (
        f'The user asked: "{question}"\n\n'
        f"SQL:\n{sql}\n\n"
        f"Result:\n{summary}\n\n"
        "The initial review flagged potential issues. Do a thorough review. "
        "Is this correct? Start with PASS or FAIL, then explain."
    )
    return call_llm(client, [{"role": "user", "content": msg}], system=SANITY_SYSTEM, model=MODEL_OPUS, max_tokens=1024)


def suggest_prompt_addition(client, question, sql, error):
    msg = f'Question: "{question}"\nSQL:\n{sql}\nError/Issue: {error}'
    return call_llm(client, [{"role": "user", "content": msg}], system=PROMPT_SUGGESTION_SYSTEM, max_tokens=256)


def generate_query_title(client, question, sql, explanation):
    msg = f'Question: "{question}"\nSQL:\n{sql}\nExplanation: {explanation}'
    raw = call_llm(client, [{"role": "user", "content": msg}], system=TITLE_SYSTEM, max_tokens=128)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"title": question[:60], "description": explanation or question}


def parse_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    def _try_parse(s):
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            fixed = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            try:
                parsed = json.loads(fixed)
            except json.JSONDecodeError:
                return None
        if isinstance(parsed, dict) and parsed.get("sql"):
            parsed["sql"] = sanitize_sql(parsed["sql"])
        return parsed

    result = _try_parse(text)
    if result:
        return result

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        result = _try_parse(match.group())
        if result:
            return result

    return {"sql": "", "explanation": "Failed to parse LLM response", "error": text}


def sanitize_sql(sql):
    sql = sql.replace("\u2014", "--").replace("\u2013", "--")
    sql = sql.replace("\u2264", "<=").replace("\u2265", ">=").replace("\u2260", "!=")
    sql = sql.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    return sql


def lookup_players(conn, query_text):
    words = query_text.split()
    skip = {
        "show",
        "me",
        "the",
        "for",
        "this",
        "year",
        "season",
        "from",
        "what",
        "give",
        "how",
        "many",
        "all",
        "per",
        "game",
        "stats",
        "averages",
        "compare",
        "plot",
        "chart",
        "list",
        "top",
        "best",
        "worst",
        "most",
        "least",
        "with",
        "and",
        "points",
        "rebounds",
        "assists",
        "steals",
        "blocks",
        "turnovers",
        "shooting",
        "percentage",
    }
    candidates = []
    for w in words:
        clean = w.strip("'\".,!?()").lower()
        if len(clean) >= 3 and clean not in skip and not clean.isdigit():
            candidates.append(clean)
    if not candidates:
        return ""
    results = []
    seen = set()
    for term in candidates:
        if not re.match(r"^[a-z0-9]+$", term):
            continue
        try:
            rows = conn.execute(
                "SELECT DISTINCT personId, playerName, player_name_ascii FROM player_advanced "
                "WHERE player_name_ascii ILIKE '%' || $1 || '%' LIMIT 5",
                [term],
            ).fetchall()
            for row in rows:
                if row[0] not in seen:
                    seen.add(row[0])
                    results.append(f"{row[0]}: {row[1]} ({row[2]})")
        except Exception:
            pass
    if results:
        return "Matching players found in database:\n" + "\n".join(results[:10])
    return ""


def execute_sql(conn, sql):
    def _run():
        return conn.execute(sanitize_sql(sql)).pl()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run)
        try:
            return future.result(timeout=QUERY_TIMEOUT_SECS)
        except FuturesTimeout:
            conn.interrupt()
            raise TimeoutError(
                f"Query exceeded {QUERY_TIMEOUT_SECS}s timeout. Simplify the query — avoid correlated subqueries and self-joins on pbp."
            )


_b = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
SAFE_BUILTINS = dict(_b)


def draw_court(fig=None):
    import numpy as np

    if fig is None:
        fig = go.Figure()
    _c = "rgba(255,255,255,0.7)"
    theta_top = np.linspace(0, np.pi, 100)
    theta_bot = np.linspace(np.pi, 2 * np.pi, 100)
    # Free throw circle (top half solid, bottom half dashed)
    fig.add_trace(
        go.Scatter(
            x=(60 * np.cos(theta_top)).tolist(),
            y=(138 + 60 * np.sin(theta_top)).tolist(),
            mode="lines",
            line=dict(color=_c, width=3),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=(60 * np.cos(theta_bot)).tolist(),
            y=(138 + 60 * np.sin(theta_bot)).tolist(),
            mode="lines",
            line=dict(color=_c, width=3, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # Restricted area arc
    fig.add_trace(
        go.Scatter(
            x=(40 * np.cos(theta_top)).tolist(),
            y=(40 * np.sin(theta_top)).tolist(),
            mode="lines",
            line=dict(color=_c, width=3),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # Three-point arc
    t3 = np.linspace(np.arccos(220 / 237.5), np.pi - np.arccos(220 / 237.5), 200)
    fig.add_trace(
        go.Scatter(
            x=(237.5 * np.cos(t3)).tolist(),
            y=(237.5 * np.sin(t3)).tolist(),
            mode="lines",
            line=dict(color=_c, width=3),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        shapes=[
            dict(type="rect", x0=-250, y0=-52, x1=250, y1=470, line=dict(color=_c, width=3), fillcolor="rgba(0,0,0,0)"),
            dict(type="circle", x0=-7.5, y0=-7.5, x1=7.5, y1=7.5, line=dict(color="orange", width=3)),
            dict(type="line", x0=-30, y0=-7.5, x1=30, y1=-7.5, line=dict(color=_c, width=3)),
            dict(type="rect", x0=-80, y0=-52, x1=80, y1=138, line=dict(color=_c, width=3), fillcolor="rgba(0,0,0,0)"),
            dict(type="line", x0=-220, y0=-52, x1=-220, y1=89.5, line=dict(color=_c, width=3)),
            dict(type="line", x0=220, y0=-52, x1=220, y1=89.5, line=dict(color=_c, width=3)),
        ],
        xaxis=dict(range=[-300, 300], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(
            range=[-80, 500],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
        width=700,
        height=650,
    )
    return fig


def shot_chart(result_df, hex_size=15, min_attempts=5, title="Shot Chart"):
    """Hexagonal binned shot chart. result_df needs x_court, y_court, shotResult."""
    from collections import Counter

    import numpy as np

    x = result_df["x_court"].cast(pl.Float64).to_numpy()
    y = result_df["y_court"].cast(pl.Float64).to_numpy()
    made = (result_df["shotResult"] == "Made").to_numpy()

    w = np.sqrt(3) * hex_size
    h = 1.5 * hex_size
    rows = np.round(y / h).astype(int)
    offsets = np.where(rows % 2 != 0, w / 2, 0.0)
    cols = np.round((x - offsets) / w).astype(int)

    keys = list(zip(rows, cols))
    att_counter = Counter(keys)
    made_counter = Counter()
    for i in range(len(x)):
        if made[i]:
            made_counter[keys[i]] += 1

    cx, cy, pcts, atts = [], [], [], []
    for (r, c), att in att_counter.items():
        if att < min_attempts:
            continue
        hx = c * w + (w / 2 if r % 2 != 0 else 0.0)
        hy = r * h
        if hy > 470 or hy < -60 or abs(hx) > 260:
            continue
        cx.append(hx)
        cy.append(hy)
        pcts.append(made_counter.get((r, c), 0) / att * 100)
        atts.append(att)

    fig = go.Figure()
    if not atts:
        draw_court(fig)
        fig.update_layout(template="plotly_dark", title=title)
        return fig

    # Add hex data first, then court lines on top
    fig.add_trace(
        go.Scatter(
            x=cx,
            y=cy,
            mode="markers+text",
            marker=dict(
                size=hex_size * 1.8,
                color=pcts,
                colorscale=[[0, "#c62828"], [0.45, "#fdd835"], [1, "#2e7d32"]],
                cmin=25,
                cmax=65,
                colorbar=dict(title="FG%"),
                symbol="hexagon",
                line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
            ),
            text=[str(a) for a in atts],
            textfont=dict(size=7, color="white"),
            textposition="middle center",
            hovertext=[f"FG%: {p:.1f}%<br>Attempts: {a}" for p, a in zip(pcts, atts)],
            hoverinfo="text",
            showlegend=False,
        )
    )
    draw_court(fig)
    fig.update_layout(template="plotly_dark", title=title)
    return fig


def execute_plot(plot_code, result_df, client=None):
    # Shot chart data is always rendered by shot_chart() — LLM hex code is unreliable
    if {"x_court", "y_court", "shotResult"}.issubset(set(result_df.columns)) and "shot_chart(" not in plot_code:
        return shot_chart(result_df)

    plot_code = plot_code.replace("fig.show()", "").replace("fig.show(", "# fig.show(")
    local_vars = {
        "result_df": result_df,
        "px": px,
        "go": go,
        "np": __import__("numpy"),
        "fig": None,
        "draw_court": draw_court,
        "shot_chart": shot_chart,
    }
    try:
        exec(plot_code, {"__builtins__": SAFE_BUILTINS}, local_vars)
        return local_vars.get("fig")
    except (ValueError, TypeError) as e:
        if client is None:
            raise
        fix_msg = (
            f"This Plotly code failed:\n```\n{plot_code}\n```\n"
            f"Error: {e}\n\n"
            "Fix the code. Use only modern Plotly API (no deprecated properties). "
            "Return ONLY the fixed Python code, no explanation, no markdown fences."
        )
        fixed = call_llm(client, [{"role": "user", "content": fix_msg}], max_tokens=1024)
        fixed = fixed.strip()
        if fixed.startswith("```"):
            fixed = re.sub(r"^```(?:python)?\s*\n?", "", fixed)
            fixed = re.sub(r"\n?```\s*$", "", fixed)
        local_vars2 = {
            "result_df": result_df,
            "px": px,
            "go": go,
            "np": __import__("numpy"),
            "fig": None,
            "draw_court": draw_court,
            "shot_chart": shot_chart,
        }
        exec(fixed, {"__builtins__": SAFE_BUILTINS}, local_vars2)
        return local_vars2.get("fig")


def format_df_for_display(df):
    styled_cols = []
    for col in df.columns:
        dtype = df[col].dtype
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            styled_cols.append(pl.col(col).cast(pl.Int64))
        elif dtype in (pl.Float32, pl.Float64):
            styled_cols.append(pl.col(col).cast(pl.Float64))
        else:
            styled_cols.append(pl.col(col))
    display_df = df.select(styled_cols).to_pandas()
    col_config = {}
    for col in df.columns:
        dtype = df[col].dtype
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            col_config[col] = st.column_config.NumberColumn(format="%,d")
        elif dtype in (pl.Float32, pl.Float64):
            col_config[col] = st.column_config.NumberColumn(format="%,.2f")
    return display_df, col_config


def display_result(df, parsed, client=None, key_suffix="default"):
    total = df.height
    if total > MAX_DISPLAY_ROWS:
        st.caption(f"Showing {MAX_DISPLAY_ROWS:,} of {total:,} rows. Download CSV for full data.")
    display_pdf, col_cfg = format_df_for_display(df.head(MAX_DISPLAY_ROWS))
    st.dataframe(display_pdf, column_config=col_cfg, width="stretch")
    csv = df.write_csv()
    st.download_button("Download CSV", csv, "result.csv", "text/csv", key=f"dl_{key_suffix}")
    fig = None
    if parsed.get("plot"):
        try:
            fig = execute_plot(parsed["plot"], df, client)
            if fig:
                st.plotly_chart(fig, width="stretch", theme="streamlit")
        except Exception as e:
            st.warning(f"Plot generation failed: {e}")
    return fig


def log_query(question, sql, explanation, error=None, rows=None, tables_used=None, columns_used=None):
    from datetime import datetime

    entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "sql": sql,
        "explanation": explanation,
        "error": error,
        "rows": rows,
        "tables_used": tables_used,
        "columns_used": columns_used,
    }
    with open(QUERY_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
