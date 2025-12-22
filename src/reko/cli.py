import argparse
import logging

from iso639 import Lang

from reko.core.errors import RekoError
from reko.core.models import SummaryConfig
from reko.core.services import summarize

logger = logging.getLogger(__name__)


def _configure_logging(level: int) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_language(value: str) -> Lang:
    try:
        return Lang(value)
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid language code: {value!r} (expected an ISO 639 code like 'en')."
        ) from e


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="reko",
        description="YouTube LLM Video Summarizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Summarize command
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Summarize a YouTube video or playlist, or a file containing URLs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    summarize_parser.set_defaults(func=_handle_summarize)

    summarize_parser.add_argument(
        "target",
        type=str,
        help="YouTube video URL, playlist URL, or a text file path (one URL per line).",
    )
    summarize_parser.add_argument(
        "model",
        type=str,
        help="Language model name to use, in the llmlite format (e.g., 'openai/gpt-5-nano' or 'ollama/llama3.2:3b').",
    )
    summarize_parser.add_argument(
        "--host",
        type=str,
        help="Language model server host URL. If Ollama is used for language model serving, and the serving host:port is different from the default, this must be provided.",
    )
    summarize_parser.add_argument(
        "--target-chunk-words",
        type=int,
        default=800,
        help="Target words per chunk; chunks flush at segment boundaries, so this is a soft limit",
    )
    summarize_parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="The maximum number of tokens to generate per response.",
    )
    summarize_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for the language model generation.",
    )
    summarize_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum attempts per LLM call after the first failed attempt before failing. Must be greater than or equal to 0.",
    )
    summarize_parser.add_argument(
        "--language",
        type=_parse_language,
        default="en",
        help="Target language (ISO code) for transcript retrieval and summarization.",
    )
    summarize_parser.add_argument(
        "--length",
        type=str,
        choices=("short", "medium", "long"),
        default="medium",
        help="Desired summary length and key points count.",
    )
    summarize_parser.add_argument(
        "--think",
        action="store_true",
        help="Enable 'think' mode for the Ollama-hosted language model.",
    )
    summarize_parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the summary even if it already exists.",
    )
    summarize_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress logs.",
    )

    output_mode = summarize_parser.add_mutually_exclusive_group()
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

    output_destination = summarize_parser.add_mutually_exclusive_group()
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

    args = parser.parse_args(argv)
    args.prog = parser.prog

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

    args.log_level = logging.DEBUG if args.verbose else logging.INFO

    return args


def _build_config(args: argparse.Namespace) -> SummaryConfig:
    return SummaryConfig(
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
        target_language=args.language,
        length=str(args.length),
        think=bool(args.think),
    )


def _handle_summarize(args: argparse.Namespace) -> None:
    config = _build_config(args)
    summarize(args.target, config)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)

    try:
        args.func(args)
        return 0
    except RekoError as e:
        if args.verbose:
            logger.exception("%s", e)
        else:
            logger.error("%s: error: %s", args.prog, e)
        return int(getattr(e, "exit_code", 1))
    except KeyboardInterrupt:
        if args.verbose:
            logger.error("%s: interrupted", args.prog)
        return 130
    except Exception:
        if args.verbose:
            logger.exception("Unhandled error")
        else:
            logger.error(
                "%s: unexpected error; re-run with --verbose for traceback.", args.prog
            )
        return 1
