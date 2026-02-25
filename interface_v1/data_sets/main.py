"""
main.py
-------
CLI entrypoint for the eval dataset generator.

Usage:
  python main.py --kb ./my_knowledge_base --output dataset.jsonl
  python main.py --kb ./my_knowledge_base --config my_config.yaml --total 100
  python main.py --kb ./my_knowledge_base --total 30 --model openai/gpt-4o-mini

Run `python main.py --help` for full options.
"""

import sys
import yaml
import argparse
import logging
from pathlib import Path
from generator import DatasetGenerator


def setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Allow CLI flags to override config.yaml values."""
    if args.total:
        config["dataset"]["total_questions"] = args.total
    if args.model:
        config["llm"]["model"] = args.model
    if args.output:
        config["paths"]["output_file"] = args.output
    if args.examples:
        config["paths"]["examples_file"] = args.examples
    if args.temperature is not None:
        config["llm"]["temperature"] = args.temperature
    if args.no_curation:
        config["curation"]["enabled"] = False
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation JSONL datasets from a knowledge base using a 3-pass LLM pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Basic usage with a folder of markdown files
        python main.py --kb ./docs

        # Custom total, output path, and cheaper model
        python main.py --kb ./docs --total 30 --model openai/gpt-4o-mini --output my_eval.jsonl

        # Use Anthropic Claude instead of OpenAI
        python main.py --kb ./docs --model anthropic/claude-opus-4

        # Skip Pass 3 curation (faster, useful for prototyping)
        python main.py --kb ./docs --no-curation

        # Point to a custom config file
        python main.py --kb ./docs --config custom_config.yaml

        Environment variables:
        OPENAI_API_KEY        For OpenAI models
        ANTHROPIC_API_KEY     For Anthropic models
        (LiteLLM supports many more — see https://docs.litellm.ai)
        """,
    )

    parser.add_argument(
        "--kb",
        required=True,
        help="Path to knowledge base: a single file or a directory (recursively searched).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL file path (overrides config).",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        help="Total number of questions in the final dataset (overrides config).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LiteLLM model string, e.g. openai/gpt-4o or anthropic/claude-opus-4 (overrides config).",
    )
    parser.add_argument(
        "--examples",
        default=None,
        help="Path to examples JSONL file (overrides config).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM temperature (overrides config).",
    )
    parser.add_argument(
        "--no-curation",
        action="store_true",
        help="Skip Pass 3 quality curation (faster, useful for prototyping).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging.",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    # Load and merge config
    config_path = Path(args.config)
    if not config_path.exists():
        log.error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))
    config = merge_cli_overrides(config, args)

    log.info(f"Config loaded from: {config_path}")
    log.info(f"Model: {config['llm']['model']}")
    log.info(f"Target questions: {config['dataset']['total_questions']}")
    log.info(f"Knowledge base: {args.kb}")

    # Run the generator
    generator = DatasetGenerator(config)
    try:
        # generator.run(kb_path=args.kb, output_path=config["paths"].get("output_file"))
        # TODO: reactivate this after testing
        print("RAN!")
    except KeyboardInterrupt:
        log.info("\nInterrupted by user. Intermediate results saved in intermediate/ directory.")
        sys.exit(0)
    except Exception as e:
        log.error(f"Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
