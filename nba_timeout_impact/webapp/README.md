# NBA Play-by-Play Data Explorer

Natural language SQL explorer over NBA play-by-play data, powered by Claude API and Streamlit.

Live at: **https://nba-claude-app.com**

## How it works

1. User types a question in plain English (e.g. "Show me LeBron's shooting percentage by season")
2. Claude (Sonnet 4.6) generates DuckDB SQL from the question using a schema-aware system prompt
3. SQL executes over in-memory tables registered from parquet files
4. Claude validates results (Sonnet first, escalates to Opus 4.6 if issues found)
5. Plotly charts generated automatically when appropriate

## Data

All data lives in `$DATA_DIR/NBA/` (resolved via `NBAConstants`):

| Table | Source file | Rows | Description |
|---|---|---|---|
| `pbp` | cdnnba_enriched.parquet | 4.2M | Play-by-play events (2020-2025, regular + playoffs) |
| `player_stats` | player_season_stats.parquet | 5.2K | Per-player per-season counting totals |
| `player_advanced` | player_advanced_stats.parquet | 3.4K | Advanced metrics + bio (through 2024) |
| `stints` | stints.parquet | 2.1M | Player on/off court stints |
| `boxscores` | boxscores.parquet | 217K | Per-player per-game box scores |
| `rotations` | rotations.parquet | 1.9M | Raw rotation data |

At load time, `helpers.py` computes columns expected by the schema prompt but missing from our pipeline:
- `clock_seconds` (alias of `seconds_remaining`)
- `subType_clean`, `descriptor_clean`, `qualifiers_clean` (aliases of raw columns)
- `prev_sub_type`, `prev_sub_type_clean` (lagged subType within each game)
- `player_name_ascii` on `player_advanced` (Unicode-normalized names for ILIKE search)

## Pages

- **Query** — natural language to SQL, with auto-fix on errors and multi-tier validation
- **Hypothesis Playground** — structured statistical testing with Claude assistance
- **Saved Queries** — browse, pin, reload past analyses
- **Conversations** — auto-saved chat history

## Origin

Adapted from [NBA_Genie](/home/Akseldkw/coding/nba/NBA_Genie/). Key changes:
- Data paths use `NBAConstants.NBA_DATA_DIR` instead of relative `datasets/` dir
- All imports are package-qualified (`nba_timeout_impact.webapp.*`)
- Missing enriched columns computed at load time
- Fetch/pipeline scripts not copied (data already exists)

## Running

```bash
cd /home/Akseldkw/coding/nba/NBA-Timeout-Impact

# Start app
DATA_DIR=/home/Akseldkw/coding/data_kretsinger \
PYTHONPATH=/home/Akseldkw/coding/kretsinger:/home/Akseldkw/coding/nba/NBA-Timeout-Impact \
streamlit run nba_timeout_impact/webapp/app.py &

# Start tunnel (persistent URL)
cloudflared tunnel run nba-claude-app &
```

## Requirements

- `streamlit >= 1.36`
- `duckdb`
- `anthropic`
- `polars`
- `plotly`
- `scipy`
- `ANTHROPIC_API_KEY` in `.streamlit/secrets.toml`

## Cloudflare Tunnel

Named tunnel `nba-claude-app` (ID: `a1fa7b7c-33f0-4714-bae0-2e598c549c9d`) routes `nba-claude-app.com` to `localhost:8501`. Config at `~/.cloudflared/config.yml`.
