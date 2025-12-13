from __future__ import annotations

import argparse
import logging

from iso639 import Lang

from .app import SummaryConfig, summarize_video_url

logger = logging.getLogger(__name__)


def configure_logging(level: int) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="reko",
        description="YouTube LLM Video Summarizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("url", type=str, help="YouTube video URL.")
    parser.add_argument(
        "model",
        type=str,
        help="Language model name to use, in the llmlite format (e.g., 'openai/gpt-5-nano' or 'ollama/llama3.2:3b').",
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Language model server host URL. If Ollama is used for language model serving, and the serving host:port is different from the default, this must be provided.",
    )
    parser.add_argument(
        "--target-chunk-words",
        type=int,
        default=800,
        help="Target words per chunk; chunks flush at segment boundaries, so this is a soft limit",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="The maximum number of tokens to generate per response.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for the language model generation.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum attempts per LLM call after the first failed attempt before failing. Must be greater than or equal to 0.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Target language (ISO code) for transcript retrieval and summarization.",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable 'think' mode for the Ollama-hosted language model.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the summary even if it already exists.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress logs.",
    )

    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--summary-only",
        action="store_true",
        help="Generate the summary section only.",
    )
    output_mode.add_argument(
        "--key-points-only",
        action="store_true",
        help="Generate the key points section only.",
    )

    output_destination = parser.add_mutually_exclusive_group()
    output_destination.add_argument(
        "--print-only",
        action="store_true",
        help="Print the output to console only.",
    )
    output_destination.add_argument(
        "--save-only",
        action="store_true",
        help="Save the output to file only.",
    )

    args = parser.parse_args()

    # if neither summary-only nor key-points-only is set, generate both
    args.summary = not args.key_points_only
    args.key_points = not args.summary_only

    # default to both printing and saving
    if args.print_only:
        args.print_output = True
        args.save_output = False
    elif args.save_only:
        args.print_output = False
        args.save_output = True
    else:
        args.print_output = True
        args.save_output = True

    if args.max_retries < 0:
        parser.error("--max-retries must be greater than or equal to 0.")

    if args.verbose:
        args.log_level = logging.DEBUG
    else:
        args.log_level = logging.INFO

    return args


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    config = SummaryConfig(
        host=args.host,
        model=args.model,
        target_chunk_words=args.target_chunk_words,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        force=bool(args.force),
        include_summary=bool(args.summary),
        include_key_points=bool(args.key_points),
        max_retries=int(args.max_retries),
        print_output=bool(args.print_output),
        save_output=bool(args.save_output),
        target_language=Lang(args.language),
        think=bool(args.think),
    )

    summarize_video_url(args.url, config)


if __name__ == "__main__":
    main()
