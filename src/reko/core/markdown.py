from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reko.core.models import SummaryDocument


def _extract_section(lines: list[str], header: str) -> list[str]:
    marker = f"## {header}"
    try:
        start_idx = next(
            idx for idx, line in enumerate(lines) if line.strip() == marker
        )
    except StopIteration:
        return []

    content: list[str] = []
    for line in lines[start_idx + 1 :]:
        if line.strip().startswith("## "):
            break
        content.append(line)
    while content and not content[0].strip():
        content.pop(0)
    while content and not content[-1].strip():
        content.pop()
    return content


def _summary_document_to_markdown(doc: SummaryDocument) -> str:
    lines = [f"# {doc.title.strip()}"]

    if doc.summary and doc.summary.strip():
        lines.extend(["", "## Summary", "", doc.summary.strip()])

    points = [point.strip() for point in (doc.key_points or []) if point.strip()]
    if points:
        lines.extend(["", "## Key Points", ""])
        for point in points:
            lines.append(f"- {point}")

    return "\n".join(lines).strip()


def _summary_document_from_markdown(markdown: str) -> SummaryDocument:
    from reko.core.models import SummaryDocument

    lines = markdown.splitlines()
    title = ""
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    summary_lines = _extract_section(lines, "Summary")
    summary = "\n".join(summary_lines).strip() if summary_lines else None
    if summary == "":
        summary = None

    key_points_lines = _extract_section(lines, "Key Points")
    key_points: list[str] = []
    for line in key_points_lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned.startswith(("- ", "* ", "\u2022 ")):
            cleaned = cleaned[2:].strip()
        if cleaned:
            key_points.append(cleaned)
    if not key_points:
        key_points = None

    return SummaryDocument(title=title, summary=summary, key_points=key_points)
