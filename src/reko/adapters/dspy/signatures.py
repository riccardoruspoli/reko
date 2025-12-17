import dspy


class ChunkSummarySignature(dspy.Signature):
    """Summarize a transcript chunk into structured notes."""

    chunk_text: str = dspy.InputField(
        desc="Verbatim transcript chunk in chronological order."
    )
    chunk_context: str = dspy.InputField(
        desc=(
            "Metadata about chunk position, timestamps, and word count. Use it to scale coverage."
        )
    )
    summary: str = dspy.OutputField(
        desc=(
            "Paragraph-style summary for this chunk. Include concrete details sized to the chunk length."
        )
    )


class AggregateSummarySignature(dspy.Signature):
    """Merge chunk-level notes into a video-level summary."""

    mapped_chunks: str = dspy.InputField(
        desc=(
            "Structured summaries for each chunk in chronological order. Preserve ordering."
        )
    )
    reduce_context: str = dspy.InputField(
        desc=(
            "Global transcript metadata and expectations about final coverage and length."
        )
    )
    final_summary: str = dspy.OutputField(
        desc=(
            "Merged narrative that stitches the chunk summaries together with smooth transitions "
            "while preserving their details verbatim whenever possible."
        )
    )


class KeyPointsSignature(dspy.Signature):
    """Produce concise, factual key points from merged chunk summaries."""

    mapped_chunks: str = dspy.InputField(
        desc="Formatted chunk summaries with timestamps and word counts in chronological order."
    )
    final_summary: str = dspy.InputField(
        desc="Narrative summary that already merges all chunk summaries."
    )
    guidance: str = dspy.InputField(
        desc="Instructions about bullet count, chronology, and style for the key points."
    )
    key_points: list[str] = dspy.OutputField(
        desc=(
            "Bullet-friendly key points (5-10 items) in chronological order. "
            "Each entry should be a single, concrete sentence retaining names, numbers, and outcomes."
        )
    )


class TranslateSignature(dspy.Signature):
    """Translate text into the target language while preserving meaning and structure."""

    source_text: str = dspy.InputField(
        desc="Original text to translate. Can be a paragraph or bullet list."
    )
    target_language: str = dspy.InputField(desc="Target language for the translation.")
    guidance: str = dspy.InputField(
        desc="Instructions about tone and formatting to preserve."
    )
    translated_text: str = dspy.OutputField(
        desc="Accurate translation in the target language retaining formatting."
    )
