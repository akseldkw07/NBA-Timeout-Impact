"""Main query page — natural language to SQL over NBA data."""

import json

import streamlit as st

from nba_timeout_impact.webapp import storage
from nba_timeout_impact.webapp.helpers import (
    MAX_DISPLAY_ROWS,
    MODEL_OPUS,
    build_messages,
    call_llm,
    display_result,
    execute_sql,
    format_df_for_display,
    generate_query_title,
    get_client,
    get_system_prompt,
    load_data,
    log_query,
    lookup_players,
    parse_response,
    sanitize_sql,
    sanity_check,
    sanity_check_opus,
    suggest_prompt_addition,
    summarize_result,
)


def _init_state():
    for key, default in [
        ("messages", []),
        ("results", []),
        ("opus_retry", None),
        ("last_query_ctx", None),
        ("conversation_id", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


def _handle_load_saved_query():
    qid = st.session_state.pop("load_saved_query", None)
    if not qid:
        return
    q = storage.load_saved_query(qid)
    st.session_state.messages = [
        {"role": "user", "content": q.get("description", q.get("title", ""))},
        {"role": "assistant", "content": f"{q.get('explanation', '')}\n\nSQL: {q['sql']}"},
    ]
    st.session_state.results = [
        {
            "sql": q["sql"],
            "explanation": q.get("explanation", ""),
            "tables_used": q.get("tables_used", []),
            "columns_used": q.get("columns_used", {}),
            "df": q.get("df"),
            "fig": q.get("fig"),
        }
    ]
    st.session_state.conversation_id = None


def _handle_load_conversation():
    cid = st.session_state.pop("load_conversation", None)
    if not cid:
        return
    conv = storage.load_conversation(cid)
    st.session_state.messages = conv.get("messages", [])
    st.session_state.results = conv.get("results", [])
    st.session_state.conversation_id = cid


def _auto_save_conversation():
    if not st.session_state.messages:
        return
    if not st.session_state.conversation_id:
        st.session_state.conversation_id = storage.generate_id()
    storage.save_conversation(
        st.session_state.conversation_id,
        st.session_state.messages,
        st.session_state.results,
    )


def query_page():
    _init_state()
    _handle_load_saved_query()
    _handle_load_conversation()

    st.title("NBA Play-by-Play Data Explorer")
    import socket

    try:
        _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _s.connect(("8.8.8.8", 80))
        _ip = _s.getsockname()[0]
        _s.close()
    except Exception:
        _ip = "localhost"
    st.caption(f"Ask questions in natural language. Get SQL + results. | Network: http://{_ip}:8501")

    client = get_client()
    if not client:
        st.error("Set ANTHROPIC_API_KEY in .streamlit/secrets.toml")
        return

    conn, pbp, player_stats, player_advanced = load_data()
    SYSTEM_PROMPT = get_system_prompt(pbp, player_stats, player_advanced)

    # Display chat history
    result_idx = 0
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and result_idx < len(st.session_state.results):
                result = st.session_state.results[result_idx]
                result_idx += 1
                if result.get("sql"):
                    st.code(result["sql"], language="sql")
                if result.get("explanation"):
                    st.caption(result["explanation"])
                if result.get("df") is not None:
                    total = result["df"].height
                    if total > MAX_DISPLAY_ROWS:
                        st.caption(f"Showing {MAX_DISPLAY_ROWS:,} of {total:,} rows. Download CSV for full data.")
                    hist_pdf, hist_cfg = format_df_for_display(result["df"].head(MAX_DISPLAY_ROWS))
                    st.dataframe(hist_pdf, column_config=hist_cfg, width="stretch")
                    csv = result["df"].write_csv()
                    st.download_button("Download CSV", csv, "result.csv", "text/csv", key=f"dl_hist_{i}_{result_idx}")
                if result.get("fig"):
                    st.plotly_chart(result["fig"], width="stretch", theme="streamlit")
                if result.get("error"):
                    st.error(result["error"])

    # Chat input
    query = st.chat_input("Ask a question about NBA data...")
    if hasattr(st.session_state, "pending_query"):
        query = st.session_state.pending_query
        del st.session_state.pending_query

    if query:
        st.session_state.opus_retry = None
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Generating SQL..."):
                msgs = build_messages(st.session_state.messages[:-1], query)
                raw = call_llm(client, msgs, system=SYSTEM_PROMPT)
                parsed = parse_response(raw)

            result_entry = {
                "sql": parsed.get("sql", ""),
                "explanation": parsed.get("explanation", ""),
                "tables_used": parsed.get("tables_used", []),
                "columns_used": parsed.get("columns_used", {}),
            }

            if parsed.get("sql"):
                st.code(parsed["sql"], language="sql")
                if parsed.get("explanation"):
                    st.caption(parsed["explanation"])
                if parsed.get("tables_used") or parsed.get("columns_used"):
                    with st.expander("Tables & columns used"):
                        for tbl in parsed.get("tables_used", []):
                            cols = parsed.get("columns_used", {}).get(tbl, [])
                            st.markdown(f"**{tbl}**: {', '.join(cols) if cols else '(all)'}")

                df = None
                try:
                    with st.spinner("Running query..."):
                        df = execute_sql(conn, parsed["sql"])
                except Exception as e:
                    err_msg = str(e)
                    result_entry["error"] = err_msg
                    st.error(f"SQL error: {err_msg}")
                    with st.spinner("Attempting to fix..."):
                        failed_response = json.dumps(
                            {"sql": parsed["sql"], "explanation": parsed.get("explanation", "")}
                        )
                        fix_messages = [
                            {"role": "user", "content": query},
                            {"role": "assistant", "content": failed_response},
                            {
                                "role": "user",
                                "content": f"That SQL failed with this error:\n\n{err_msg}\n\nFix the SQL. Make sure all columns referenced in later CTEs or clauses are included in the SELECT of earlier CTEs.",
                            },
                        ]
                        raw_fix = call_llm(client, fix_messages, system=SYSTEM_PROMPT)
                        parsed_fix = parse_response(raw_fix)
                        if parsed_fix.get("sql"):
                            parsed_fix["sql"] = sanitize_sql(parsed_fix["sql"])
                            st.markdown("**Fixed SQL:**")
                            st.code(parsed_fix["sql"], language="sql")
                            if parsed_fix.get("explanation"):
                                st.caption(parsed_fix["explanation"])
                            try:
                                df = execute_sql(conn, parsed_fix["sql"])
                                result_entry["sql"] = parsed_fix["sql"]
                                result_entry["explanation"] = parsed_fix.get("explanation", "")
                                result_entry.pop("error", None)
                                parsed = parsed_fix
                            except Exception as e2:
                                result_entry["error"] = f"Original: {err_msg}\nFix attempt: {e2}"
                                st.error(f"Fix also failed: {e2}")
                        else:
                            st.warning("Could not auto-fix. Try rephrasing your question.")
                    with st.expander("Suggested prompt addition"):
                        try:
                            suggestion = suggest_prompt_addition(client, query, parsed["sql"], err_msg)
                            st.code(suggestion, language="text")
                        except Exception:
                            st.write("Could not generate suggestion.")

                if df is not None and df.height == 0:
                    player_info = lookup_players(conn, query)
                    if player_info:
                        with st.spinner("No results — checking player names..."):
                            st.caption("0 rows returned. Retrying with player name lookup...")
                            retry_messages = [
                                {"role": "user", "content": query},
                                {
                                    "role": "assistant",
                                    "content": json.dumps(
                                        {"sql": parsed["sql"], "explanation": parsed.get("explanation", "")}
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": f"That query returned 0 rows. The issue may be a player name mismatch. Use the player_name_ascii column for ILIKE matching.\n\n{player_info}\n\nRewrite the query.",
                                },
                            ]
                            raw_retry = call_llm(client, retry_messages, system=SYSTEM_PROMPT)
                            parsed_retry = parse_response(raw_retry)
                            if parsed_retry.get("sql"):
                                st.markdown("**Retried SQL:**")
                                st.code(parsed_retry["sql"], language="sql")
                                try:
                                    df = execute_sql(conn, parsed_retry["sql"])
                                    result_entry["sql"] = parsed_retry["sql"]
                                    result_entry["explanation"] = parsed_retry.get("explanation", "")
                                    parsed = parsed_retry
                                except Exception:
                                    pass

                if df is not None:
                    result_entry["df"] = df
                    fig = display_result(df, parsed, client, key_suffix="latest")
                    if fig:
                        result_entry["fig"] = fig
                    st.success(f"{df.height:,} rows returned")

                    with st.spinner("Checking results..."):
                        try:
                            check = sanity_check(client, query, result_entry["sql"], df)
                            result_entry["sanity_check"] = check
                            is_fail = check.strip().upper().startswith("FAIL")
                            if is_fail:
                                with st.spinner("Escalating review to Opus..."):
                                    check_opus = sanity_check_opus(client, query, result_entry["sql"], df)
                                    result_entry["sanity_check"] = check_opus
                                    opus_fail = check_opus.strip().upper().startswith("FAIL")
                                    with st.expander("Result check (Opus review)", expanded=True):
                                        st.markdown(check_opus)
                                    if opus_fail:
                                        with st.spinner("Opus is rewriting the query..."):
                                            rewrite_messages = [
                                                {"role": "user", "content": query},
                                                {
                                                    "role": "assistant",
                                                    "content": json.dumps(
                                                        {
                                                            "sql": result_entry["sql"],
                                                            "explanation": result_entry.get("explanation", ""),
                                                        }
                                                    ),
                                                },
                                                {
                                                    "role": "user",
                                                    "content": f"A reviewer found issues with that query:\n\n{check_opus}\n\nRewrite the SQL to fix these issues.",
                                                },
                                            ]
                                            raw_rewrite = call_llm(
                                                client, rewrite_messages, system=SYSTEM_PROMPT, model=MODEL_OPUS
                                            )
                                            parsed_rewrite = parse_response(raw_rewrite)
                                        if parsed_rewrite.get("sql"):
                                            st.divider()
                                            st.markdown("**Corrected SQL (Opus):**")
                                            st.code(parsed_rewrite["sql"], language="sql")
                                            if parsed_rewrite.get("explanation"):
                                                st.caption(parsed_rewrite["explanation"])
                                            try:
                                                df2 = execute_sql(conn, parsed_rewrite["sql"])
                                                result_entry["sql"] = parsed_rewrite["sql"]
                                                result_entry["explanation"] = parsed_rewrite.get("explanation", "")
                                                result_entry["df"] = df2
                                                fig2 = display_result(df2, parsed_rewrite, client, key_suffix="opus")
                                                if fig2:
                                                    result_entry["fig"] = fig2
                                                st.success(f"{df2.height:,} rows returned (Opus rewrite)")
                                                with st.spinner("Checking Opus rewrite..."):
                                                    recheck = sanity_check_opus(client, query, result_entry["sql"], df2)
                                                    recheck_fail = recheck.strip().upper().startswith("FAIL")
                                                    with st.expander(
                                                        "Result check (Opus rewrite)", expanded=recheck_fail
                                                    ):
                                                        st.markdown(recheck)
                                                if recheck_fail:
                                                    st.session_state.opus_retry = {
                                                        "query": query,
                                                        "sql": result_entry["sql"],
                                                        "review": recheck,
                                                        "explanation": result_entry.get("explanation", ""),
                                                    }
                                            except Exception as e3:
                                                st.error(f"Opus rewrite failed: {e3}")
                                                st.session_state.opus_retry = {
                                                    "query": query,
                                                    "sql": parsed_rewrite["sql"],
                                                    "review": f"SQL execution error: {e3}",
                                                    "explanation": "",
                                                }
                                        else:
                                            st.warning("Opus could not generate a fix.")
                                            st.session_state.opus_retry = {
                                                "query": query,
                                                "sql": result_entry["sql"],
                                                "review": check_opus,
                                                "explanation": result_entry.get("explanation", ""),
                                            }
                                        with st.expander("Suggested prompt addition"):
                                            try:
                                                suggestion = suggest_prompt_addition(
                                                    client, query, result_entry["sql"], check_opus
                                                )
                                                result_entry["prompt_suggestion"] = suggestion
                                                st.code(suggestion, language="text")
                                            except Exception:
                                                st.write("Could not generate suggestion.")
                            else:
                                with st.expander("Result check"):
                                    st.markdown(check)
                        except Exception:
                            pass
            else:
                result_entry["error"] = "No SQL generated"
                if parsed.get("error"):
                    st.error("Could not parse a SQL query from the response.")
                    with st.expander("Raw response"):
                        st.text(parsed["error"][:500])
                else:
                    st.error("No SQL generated. Try rephrasing your question.")

            # Save context for manual Opus review
            if result_entry.get("df") is not None and result_entry.get("sql"):
                st.session_state.last_query_ctx = {
                    "query": query,
                    "sql": result_entry["sql"],
                    "explanation": result_entry.get("explanation", ""),
                }

            # Build assistant message for history
            parts = []
            if result_entry.get("explanation"):
                parts.append(result_entry["explanation"])
            if result_entry.get("sql"):
                parts.append(f"SQL: {result_entry['sql']}")
            if result_entry.get("df") is not None:
                parts.append(f"Result summary:\n{summarize_result(result_entry['df'], max_rows=100)}")
            if result_entry.get("sanity_check"):
                parts.append(f"Quality check: {result_entry['sanity_check']}")
            if result_entry.get("prompt_suggestion"):
                parts.append(f"Suggested improvement: {result_entry['prompt_suggestion']}")
            if result_entry.get("error"):
                parts.append(f"Error: {result_entry['error']}")
            assistant_msg = "\n\n".join(parts) if parts else "(no response)"
            st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
            st.session_state.results.append(result_entry)

            log_query(
                question=query,
                sql=result_entry.get("sql", ""),
                explanation=result_entry.get("explanation", ""),
                error=result_entry.get("error"),
                rows=result_entry["df"].height if result_entry.get("df") is not None else None,
                tables_used=result_entry.get("tables_used"),
                columns_used=result_entry.get("columns_used"),
            )
            _auto_save_conversation()

    # Save query button
    if st.session_state.results and st.session_state.results[-1].get("df") is not None:
        last = st.session_state.results[-1]
        if st.button("Save this query", key="save_query_btn"):
            # Generate title immediately, then show form on rerun
            with st.spinner("Generating title..."):
                question = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else ""
                gen = generate_query_title(client, question, last.get("sql", ""), last.get("explanation", ""))
                st.session_state.save_title = gen.get("title", "")
                st.session_state.save_desc = gen.get("description", "")
                st.session_state.show_save_dialog = True
            st.rerun()

    if st.session_state.get("show_save_dialog") and st.session_state.results:
        last = st.session_state.results[-1]
        with st.form("save_query_form"):
            title = st.text_input("Title", value=st.session_state.get("save_title", ""))
            desc = st.text_area("Description", value=st.session_state.get("save_desc", ""), height=80)
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Confirm Save"):
                    storage.save_query(
                        title=title,
                        description=desc,
                        sql=last.get("sql", ""),
                        explanation=last.get("explanation", ""),
                        df=last["df"],
                        fig=last.get("fig"),
                        metadata={
                            "tables_used": last.get("tables_used", []),
                            "columns_used": last.get("columns_used", {}),
                        },
                    )
                    st.session_state.show_save_dialog = False
                    st.session_state.pop("save_title", None)
                    st.session_state.pop("save_desc", None)
                    st.success("Query saved!")
                    st.rerun()
            with col2:
                if st.form_submit_button("Cancel"):
                    st.session_state.show_save_dialog = False
                    st.session_state.pop("save_title", None)
                    st.session_state.pop("save_desc", None)
                    st.rerun()

    # Manual "Review with Opus" button
    if not st.session_state.opus_retry and st.session_state.last_query_ctx:
        ctx = st.session_state.last_query_ctx
        st.markdown("---")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Review with Opus", key="manual_opus_review"):
                st.session_state.opus_retry = {
                    "query": ctx["query"],
                    "sql": ctx["sql"],
                    "review": "User requested Opus review.",
                    "explanation": ctx.get("explanation", ""),
                }
                st.rerun()

    # Opus retry / feedback
    if st.session_state.opus_retry:
        ctx = st.session_state.opus_retry
        st.markdown("---")
        feedback = st.text_area("Feedback for Opus (optional):", height=100, key="opus_feedback_input")
        if st.button("Send to Opus", key="opus_send_btn"):
            if feedback:
                with st.chat_message("user"):
                    st.markdown(feedback)
                st.session_state.messages.append({"role": "user", "content": feedback})
                user_msg = f"Previous review:\n{ctx['review']}\n\nUser feedback: {feedback}\n\nRewrite the SQL addressing this feedback."
            else:
                user_msg = f"A reviewer found issues:\n\n{ctx['review']}\n\nRewrite the SQL to fix these issues."
            with st.chat_message("assistant"):
                with st.spinner("Opus is rewriting..."):
                    rewrite_messages = [
                        {"role": "user", "content": ctx["query"]},
                        {
                            "role": "assistant",
                            "content": json.dumps({"sql": ctx["sql"], "explanation": ctx["explanation"]}),
                        },
                        {"role": "user", "content": user_msg},
                    ]
                    raw_opus = call_llm(client, rewrite_messages, system=SYSTEM_PROMPT, model=MODEL_OPUS)
                    parsed_opus = parse_response(raw_opus)
                if parsed_opus.get("sql"):
                    st.code(parsed_opus["sql"], language="sql")
                    if parsed_opus.get("explanation"):
                        st.caption(parsed_opus["explanation"])
                    try:
                        df_opus = execute_sql(conn, parsed_opus["sql"])
                        fig_opus = display_result(df_opus, parsed_opus, client, key_suffix="opus_retry")
                        st.success(f"{df_opus.height:,} rows returned")
                        result_entry_opus = {
                            "sql": parsed_opus["sql"],
                            "explanation": parsed_opus.get("explanation", ""),
                            "tables_used": parsed_opus.get("tables_used", []),
                            "columns_used": parsed_opus.get("columns_used", {}),
                            "df": df_opus,
                        }
                        if fig_opus:
                            result_entry_opus["fig"] = fig_opus
                        with st.spinner("Checking..."):
                            recheck = sanity_check_opus(client, ctx["query"], parsed_opus["sql"], df_opus)
                            recheck_fail = recheck.strip().upper().startswith("FAIL")
                            with st.expander("Result check", expanded=recheck_fail):
                                st.markdown(recheck)
                            if recheck_fail:
                                st.session_state.opus_retry = {
                                    "query": ctx["query"],
                                    "sql": parsed_opus["sql"],
                                    "review": recheck,
                                    "explanation": parsed_opus.get("explanation", ""),
                                }
                            else:
                                st.session_state.opus_retry = None
                        assistant_msg = parsed_opus.get("explanation", "")
                        st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
                        st.session_state.results.append(result_entry_opus)
                        st.session_state.last_query_ctx = {
                            "query": ctx["query"],
                            "sql": parsed_opus["sql"],
                            "explanation": parsed_opus.get("explanation", ""),
                        }
                    except Exception as e_opus:
                        st.error(f"Failed: {e_opus}")
                else:
                    st.warning("Opus could not generate SQL.")
            _auto_save_conversation()
            st.rerun()
