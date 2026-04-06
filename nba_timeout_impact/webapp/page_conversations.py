"""Conversation log browser page."""

import streamlit as st

from nba_timeout_impact.webapp import storage


def conversations_page():
    st.title("Conversation Log")

    index = storage.load_conversations_index()
    if not index:
        st.info("No conversations yet. Start asking questions on the Query page.")
        return

    # Sort newest first
    index.sort(key=lambda e: e.get("last_activity", ""), reverse=True)

    # Summary stats
    total_size = sum(e.get("size_bytes", 0) for e in index)
    st.caption(f"{len(index)} conversations | {storage.format_bytes(total_size)} total")

    # Collect prompt suggestions across all conversations
    all_suggestions = []

    for entry in index:
        cid = entry["id"]
        first_q = entry.get("first_question", cid)[:80]
        msgs = entry.get("message_count", 0)
        errors = entry.get("error_count", 0)
        size = storage.format_bytes(entry.get("size_bytes", 0))
        date = entry.get("last_activity", "")[:16]
        error_tag = f" | {errors} errors" if errors else ""

        with st.expander(f"{first_q} ({msgs} msgs{error_tag} | {size} | {date})"):
            try:
                conv = storage.load_conversation(cid)
            except Exception:
                st.error("Could not load conversation file.")
                continue

            # Show messages
            for msg in conv.get("messages", []):
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    st.markdown(f"**You:** {content[:500]}")
                else:
                    # Truncate long assistant messages
                    preview = content[:300] + ("..." if len(content) > 300 else "")
                    st.markdown(f"**Assistant:** {preview}")

            # Show errors
            for r in conv.get("results", []):
                if r.get("error"):
                    st.error(f"Error: {r['error'][:200]}")
                if r.get("prompt_suggestion"):
                    all_suggestions.append(r["prompt_suggestion"])

            # Actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Resume", key=f"resume_{cid}"):
                    st.session_state.load_conversation = cid
                    st.rerun()
            with col2:
                if st.button("Delete", key=f"del_conv_{cid}"):
                    storage.delete_conversation(cid)
                    st.rerun()

    # Prompt suggestions section
    if all_suggestions:
        st.markdown("---")
        st.markdown("### Prompt Improvement Suggestions")
        st.caption("Collected from conversations with errors:")
        for i, s in enumerate(all_suggestions):
            st.code(s, language="text")
