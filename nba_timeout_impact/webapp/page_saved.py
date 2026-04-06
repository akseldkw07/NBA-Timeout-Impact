"""Saved queries management page."""

import streamlit as st

from nba_timeout_impact.webapp import storage


def saved_queries_page():
    st.title("Saved Queries")

    index = storage.load_saved_queries_index()
    if not index:
        st.info("No saved queries yet. Save a query from the Query page.")
        return

    # Sort/filter controls
    col1, col2 = st.columns([2, 3])
    with col1:
        sort_by = st.selectbox("Sort by", ["Date (newest)", "Date (oldest)", "Title", "Pinned first"], key="sq_sort")
    with col2:
        search = st.text_input("Search", key="sq_search", placeholder="Filter by title or description...")

    # Apply filter
    if search:
        search_lower = search.lower()
        index = [
            e
            for e in index
            if search_lower in e.get("title", "").lower() or search_lower in e.get("description", "").lower()
        ]

    # Apply sort
    if sort_by == "Date (newest)":
        index.sort(key=lambda e: e.get("updated_at", ""), reverse=True)
    elif sort_by == "Date (oldest)":
        index.sort(key=lambda e: e.get("updated_at", ""))
    elif sort_by == "Title":
        index.sort(key=lambda e: e.get("title", "").lower())
    elif sort_by == "Pinned first":
        index.sort(key=lambda e: (not e.get("pinned", False), e.get("updated_at", "")), reverse=True)

    st.caption(f"{len(index)} saved queries")

    for entry in index:
        qid = entry["id"]
        pinned = entry.get("pinned", False)
        pin_icon = "* " if pinned else ""
        title = entry.get("title", qid)

        with st.expander(f"{pin_icon}{title}", expanded=False):
            st.caption(entry.get("description", ""))
            st.code(entry.get("sql", ""), language="sql")
            st.caption(
                f"{entry.get('row_count', '?')} rows | {'Has chart' if entry.get('has_figure') else 'No chart'} | {storage.format_bytes(entry.get('size_bytes', 0))} | {entry.get('updated_at', '')[:10]}"
            )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("Load", key=f"load_{qid}"):
                    st.session_state.load_saved_query = qid
                    st.rerun()
            with col2:
                pin_label = "Unpin" if pinned else "Pin"
                if st.button(pin_label, key=f"pin_{qid}"):
                    storage.update_saved_query(qid, pinned=not pinned)
                    st.rerun()
            with col3:
                if st.button("Delete", key=f"del_{qid}"):
                    storage.delete_saved_query(qid)
                    st.rerun()
            with col4:
                if st.button("Copy SQL", key=f"copy_{qid}"):
                    st.code(entry.get("sql", ""), language="sql")

            # Inline rename
            new_title = st.text_input("Rename", value=title, key=f"rename_{qid}")
            if new_title != title:
                if st.button("Save name", key=f"savename_{qid}"):
                    storage.update_saved_query(qid, title=new_title)
                    st.rerun()
