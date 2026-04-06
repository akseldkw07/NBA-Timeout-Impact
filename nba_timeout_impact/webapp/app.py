"""
NBA Play-by-Play Data Explorer — Streamlit multi-page app.

Usage: streamlit run app.py
"""

import sys

import streamlit as st

from nba_timeout_impact.webapp import storage
from nba_timeout_impact.webapp.helpers import load_data
from nba_timeout_impact.webapp.page_conversations import conversations_page
from nba_timeout_impact.webapp.page_hypothesis import hypothesis_page
from nba_timeout_impact.webapp.page_query import query_page
from nba_timeout_impact.webapp.page_saved import saved_queries_page

st.set_page_config(page_title="NBA Data Explorer", layout="wide")

# Navigation
pg = st.navigation(
    {
        "": [st.Page(query_page, title="Query", icon=":material/search:", default=True)],
        "Analysis": [
            st.Page(hypothesis_page, title="Hypothesis Playground", icon=":material/science:"),
        ],
        "History": [
            st.Page(saved_queries_page, title="Saved Queries", icon=":material/bookmark:"),
            st.Page(conversations_page, title="Conversations", icon=":material/chat:"),
        ],
    }
)

# Shared sidebar
conn, pbp, player_stats, player_advanced = load_data()

# Pinned queries
pinned = storage.get_pinned_queries()
if pinned:
    st.sidebar.markdown("### Pinned")
    for q in pinned:
        if st.sidebar.button(q["title"][:50], key=f"pin_{q['id']}"):
            st.session_state.load_saved_query = q["id"]
            st.rerun()

# Recent saved queries
recent = storage.get_recent_queries(5)
recent = [r for r in recent if not r.get("pinned")]
if recent:
    st.sidebar.markdown("### Recent")
    for q in recent:
        if st.sidebar.button(q["title"][:50], key=f"recent_{q['id']}"):
            st.session_state.load_saved_query = q["id"]
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Data:** {pbp.height:,} play-by-play events")
st.sidebar.markdown(f"**Seasons:** 2019-2025 (rg + po)")
st.sidebar.markdown(f"**Players:** {player_advanced['personId'].n_unique():,} with advanced stats")

# Storage sizes
sizes = storage.get_storage_sizes()
st.sidebar.caption(
    f"Storage: {storage.format_bytes(sizes['datasets'])} data, "
    f"{storage.format_bytes(sizes['saved_queries'])} saved, "
    f"{storage.format_bytes(sizes['conversations'])} logs"
)

st.sidebar.markdown("---")
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.results = []
    st.session_state.conversation_id = None
    st.rerun()
if st.sidebar.button("Stop App"):
    sys.exit(0)

pg.run()
