"""EvalPulse CLI — command-line interface for common operations."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize EvalPulse in the current directory."""
    config_dest = Path("evalpulse.yml")
    if config_dest.exists() and not args.force:
        print("evalpulse.yml already exists. Use --force to overwrite.")
        return

    # Copy the example config
    example = Path(__file__).parent.parent / "evalpulse.yml.example"
    if example.exists():
        shutil.copy(example, config_dest)
    else:
        # Write a minimal config
        config_dest.write_text(
            "# EvalPulse Configuration\n"
            'app_name: "my-llm-app"\n'
            'storage_backend: "sqlite"\n'
            'sqlite_path: "evalpulse.db"\n'
        )

    print(f"Created {config_dest}")
    print("Edit evalpulse.yml to customize your configuration.")
    print()
    print("Quick start:")
    print("  from evalpulse import track, init")
    print("  init()")
    print('  @track(app="my-app")')
    print("  def ask_llm(query):")
    print("      return llm.generate(query)")


def cmd_regression(args: argparse.Namespace) -> None:
    """Run regression tests against a golden dataset."""
    from evalpulse.modules.golden_dataset import load_golden_dataset
    from evalpulse.modules.regression import PromptRegressionModule

    dataset_path = args.dataset
    if not Path(dataset_path).exists():
        print(f"Error: Dataset not found: {dataset_path}")
        sys.exit(1)

    dataset = load_golden_dataset(dataset_path)
    print(f"Loaded dataset: {dataset.name} ({len(dataset.examples)} examples)")

    # Create a simple echo function if no module specified
    def echo_llm(query: str) -> str:
        return f"Response to: {query}"

    module = PromptRegressionModule()
    result = module.run_regression_suite(echo_llm, dataset)

    print(f"\nResults: {result.passed}/{result.total} passed ({result.pass_rate:.0%})")

    if result.failures:
        print(f"\nFailures ({result.failed}):")
        for f in result.failures:
            print(f"  Query: {f.query}")
            for v in f.violations:
                print(f"    - {v}")

    # Save results if output path provided
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / "regression_result.json"
        with open(result_file, "w") as fp:
            json.dump(result.model_dump(), fp, indent=2, default=str)
        print(f"\nResults saved to {result_file}")

    # Exit with error code if below threshold
    if result.pass_rate < args.threshold:
        print(f"\nFAILED: Pass rate {result.pass_rate:.0%} < threshold {args.threshold:.0%}")
        sys.exit(1)


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the EvalPulse dashboard."""
    try:
        from dashboard.app import create_app
    except ImportError:
        # Fallback: try relative import for installed package
        import sys
        from pathlib import Path

        # Add the project root to sys.path if dashboard is a sibling
        project_root = Path(__file__).parent.parent
        if (project_root / "dashboard").exists():
            sys.path.insert(0, str(project_root))
            from dashboard.app import create_app
        else:
            print("Error: Dashboard module not found. Run from the project root directory.")
            sys.exit(1)

    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="evalpulse",
        description="EvalPulse — LLM Evaluation & Drift Monitor",
    )
    subparsers = parser.add_subparsers(dest="command")

    # init
    init_parser = subparsers.add_parser("init", help="Initialize EvalPulse config")
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing config",
    )

    # regression
    reg_parser = subparsers.add_parser("regression", help="Run regression tests")
    reg_sub = reg_parser.add_subparsers(dest="reg_command")
    run_parser = reg_sub.add_parser("run", help="Run regression suite")
    run_parser.add_argument("--dataset", required=True, help="Path to golden dataset JSON")
    run_parser.add_argument("--output", default=None, help="Output directory for results")
    run_parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Minimum pass rate (default: 0.9)",
    )

    # dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Launch the dashboard")
    dash_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    dash_parser.add_argument("--port", type=int, default=7860, help="Server port")
    dash_parser.add_argument("--share", action="store_true", help="Create public link")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "regression":
        if hasattr(args, "reg_command") and args.reg_command == "run":
            cmd_regression(args)
        else:
            reg_parser.print_help()
    elif args.command == "dashboard":
        cmd_dashboard(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
