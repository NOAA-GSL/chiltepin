#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""MPAS Multi-Agent Forecast - Entry Point Script

This script runs the MPAS multi-agent forecast workflow.

Usage:
    python run_mpas_forecast.py <config_file> [<config_file> ...]

Example:
    python run_mpas_forecast.py config/user_config.yaml
    python run_mpas_forecast.py config/docker_config.yaml config/user_config.yaml
"""

import asyncio
import os
import sys
from pathlib import Path

# import yaml
import uwtools.api.config as uw_config

from workflows import MPASForecastWorkflow


def load_config(config_files: list[str]) -> dict:
    """Load and merge configuration files.

    Loads configuration in this order:
    1. Default configuration (config/default_config.yaml)
    2. User configurations (provided config files in order)

    Later files override earlier ones.

    Parameters
    ----------
    config_files : list[str]
        Paths to user configuration files, applied in order

    Returns
    -------
    dict
        Merged configuration dictionary
    """
    script_dir = Path(__file__).parent
    config_dir = script_dir / "config"

    # Get default configuration
    default_config_path = config_dir / "default_config.yaml"
    if not default_config_path.exists():
        print(f"Error: Default config not found: {default_config_path}")
        sys.exit(1)

    # Get user configuration files in the order provided.
    user_config_paths = [Path(config_file) for config_file in config_files]
    for user_config_path in user_config_paths:
        if not user_config_path.exists():
            print(f"Error: User config not found: {user_config_path}")
            sys.exit(1)

    # Compose default config with user-provided config overlays.
    config = uw_config.compose(
        [default_config_path, *user_config_paths],
        True,
        Path(os.devnull),
    )

    return config.as_dict()


async def main():
    """Main entry point for MPAS forecast workflow."""
    if len(sys.argv) < 2:
        print("Usage: python run_mpas_forecast.py <config_file> [<config_file> ...]")
        print("\nExample:")
        print("  python run_mpas_forecast.py config/experiment_config.yaml")
        print(
            "  python run_mpas_forecast.py config/docker_config.yaml config/experiment_config.yaml"
        )
        sys.exit(1)

    config_files = sys.argv[1:]

    # Load configuration
    print("Loading configuration...")
    config = load_config(config_files)

    # Print configuration summary
    print("\n" + "=" * 60)
    print("MPAS Multi-Agent Forecast Workflow")
    print("=" * 60)
    print(f"Resolution: {config.get('model', {}).get('resolution', 'not specified')}")
    print(f"Experiment directory: {config.get('paths', {}).get('experiment_dir', 'not specified')}")
    print("=" * 60 + "\n")

    # Create workflow
    print("Initializing workflow...")
    workflow = MPASForecastWorkflow(config)

    # Run workflow
    print("Starting MPAS forecast workflow...\n")
    try:
        forecast_output = await workflow.run()
        print("\n" + "=" * 60)
        print("Workflow completed successfully!")
        print(f"Forecast output: {forecast_output}")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print("Workflow failed!")
        print(f"Error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
