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

    if doc.brief:
        lines.extend(["", "## TL;DR", "", doc.brief.tldr.strip()])
        lines.extend(["", "## Key Points", ""])
        for point in doc.brief.key_points:
            lines.append(f"- {point.strip()}")
        lines.extend(["", "## So What?", "", doc.brief.so_what.strip()])
        lines.extend(["", "## Takeaway", "", doc.brief.takeaway.strip()])
    else:
        if doc.summary and doc.summary.strip():
            lines.extend(["", "## Summary", "", doc.summary.strip()])

        points = [point.strip() for point in (doc.key_points or []) if point.strip()]
        if points:
            lines.extend(["", "## Key Points", ""])
            for point in points:
                lines.append(f"- {point}")

    return "\n".join(lines).strip()


def _summary_document_from_markdown(markdown: str) -> SummaryDocument:
    from reko.core.models import BriefOutput, SummaryDocument

    lines = markdown.splitlines()
    title = ""
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    is_brief = any(line.strip() == "## TL;DR" for line in lines)
    summary_lines = [] if is_brief else _extract_section(lines, "Summary")
    summary = "\n".join(summary_lines).strip() if summary_lines else None
    key_points_lines = [] if is_brief else _extract_section(lines, "Key Points")
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

    brief_tldr = "\n".join(_extract_section(lines, "TL;DR")).strip()
    brief_so_what = "\n".join(_extract_section(lines, "So What?")).strip()
    brief_takeaway = "\n".join(_extract_section(lines, "Takeaway")).strip()
    brief_points = []
    for line in _extract_section(lines, "Key Points"):
        cleaned = line.strip()
        if cleaned.startswith(("- ", "* ", "\u2022 ")):
            cleaned = cleaned[2:].strip()
        if cleaned:
            brief_points.append(cleaned)
    brief: BriefOutput | None = None
    if brief_tldr and brief_points and brief_so_what and brief_takeaway:
        brief = BriefOutput(
            tldr=brief_tldr,
            key_points=brief_points,
            so_what=brief_so_what,
            takeaway=brief_takeaway,
        )

    return SummaryDocument(
        title=title, summary=summary, key_points=key_points, brief=brief
    )
