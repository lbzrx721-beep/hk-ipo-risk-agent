"""工作台多页面路由的唯一位置。"""

from pathlib import Path

import streamlit as st


APP_PAGES_DIR = Path(__file__).with_name("app_pages")
ANALYSIS_PAGE_PATH = APP_PAGES_DIR / "analysis.py"
RESULTS_PAGE_PATH = APP_PAGES_DIR / "results.py"
REVIEW_PAGE_PATH = APP_PAGES_DIR / "review.py"

ANALYSIS_URL_PATH = "analysis"
RESULTS_URL_PATH = "results"
REVIEW_URL_PATH = "review"
_ANALYSIS_PAGE_STATE = "_workspace_analysis_page"
_RESULTS_PAGE_STATE = "_workspace_results_page"
_REVIEW_PAGE_STATE = "_workspace_review_page"


def bind_page_routes(*, analysis: object, results: object, review: object) -> None:
    """把当前入口注册的页面对象绑定到本次浏览器会话。"""
    st.session_state[_ANALYSIS_PAGE_STATE] = analysis
    st.session_state[_RESULTS_PAGE_STATE] = results
    st.session_state[_REVIEW_PAGE_STATE] = review


def _switch_to(state_key: str) -> None:
    page = st.session_state.get(state_key)
    if page is None:
        raise RuntimeError("workspace page route is unavailable")
    st.switch_page(page)


def switch_to_analysis() -> None:
    _switch_to(_ANALYSIS_PAGE_STATE)


def switch_to_results() -> None:
    _switch_to(_RESULTS_PAGE_STATE)


def switch_to_review() -> None:
    _switch_to(_REVIEW_PAGE_STATE)


__all__ = [
    "ANALYSIS_PAGE_PATH",
    "ANALYSIS_URL_PATH",
    "APP_PAGES_DIR",
    "bind_page_routes",
    "RESULTS_PAGE_PATH",
    "RESULTS_URL_PATH",
    "REVIEW_PAGE_PATH",
    "REVIEW_URL_PATH",
    "switch_to_analysis",
    "switch_to_results",
    "switch_to_review",
]
