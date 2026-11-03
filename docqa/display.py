"""Small helpers for showing text in Streamlit."""


def plain(text: str) -> str:
    """Escape `$` so Streamlit does not turn "$1,200 ... $2,400" into a math formula."""
    return text.replace("$", "\\$")
