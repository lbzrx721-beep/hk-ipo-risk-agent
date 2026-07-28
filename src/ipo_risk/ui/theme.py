"""Streamlit 工作台共享的轻量视觉语义。"""

from __future__ import annotations

import streamlit as st
from typing import Literal, TypeAlias


BadgeColor: TypeAlias = Literal[
    "red", "orange", "yellow", "blue", "green", "violet", "gray", "grey", "primary"
]

SEVERITY_META: dict[str, tuple[str, BadgeColor]] = {
    "low": ("低", "green"),
    "medium": ("中", "orange"),
    "high": ("高", "red"),
    "critical": ("严重", "red"),
}


def inject_theme() -> None:
    """保留统一主题入口；具体配色交给 Streamlit 原生主题配置。"""


def page_header(kicker: str, title: str, caption: str) -> None:
    """渲染紧凑的工作区标题，避免与应用级标题争抢层级。"""
    with st.container(border=True):
        st.caption(kicker)
        st.header(title)
        st.caption(caption)


def section_label(label: str) -> None:
    """渲染页面内的一级功能区标题。"""
    st.subheader(label)


__all__ = ["BadgeColor", "SEVERITY_META", "inject_theme", "page_header", "section_label"]
