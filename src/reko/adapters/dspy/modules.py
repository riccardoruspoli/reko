import dspy
from dspy import Prediction

from reko.adapters.dspy.signatures import (
    AggregateSummarySignature,
    ChunkSummarySignature,
    KeyPointsSignature,
    TranslateSignature,
)


class ChunkSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ChunkSummarySignature)

    def forward(self, chunk_text: str, chunk_context: str) -> Prediction:
        return self.predict(chunk_text=chunk_text, chunk_context=chunk_context)


class AggregateSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(AggregateSummarySignature)

    def forward(self, mapped_chunks: str, reduce_context: str) -> Prediction:
        return self.predict(mapped_chunks=mapped_chunks, reduce_context=reduce_context)


class KeyPointsGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(KeyPointsSignature)

    def forward(
        self, mapped_chunks: str, final_summary: str, guidance: str
    ) -> Prediction:
        return self.predict(
            mapped_chunks=mapped_chunks, final_summary=final_summary, guidance=guidance
        )


class Translator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(TranslateSignature)

    def forward(
        self, source_text: str, target_language: str, guidance: str
    ) -> Prediction:
        return self.predict(
            source_text=source_text,
            target_language=target_language,
            guidance=guidance,
        )
