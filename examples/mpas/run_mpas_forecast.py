#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""MPAS Multi-Agent Forecast - Entry Point Script

This script runs the MPAS multi-agent forecast workflow.

Usage:
    python run_mpas_forecast.py <config_file>

Example:
    python run_mpas_forecast.py config/user_config.yaml
"""

import asyncio
import sys
from pathlib import Path

import yaml

from workflows import MPASForecastWorkflow


def load_config(config_file: str) -> dict:
    """Load and merge configuration files.

    Loads configuration in this order:
    1. Default configuration (config/default_config.yaml)
    2. Platform-specific configuration (config/platforms/{platform}.yaml)
    3. User configuration (provided config_file)

    Later files override earlier ones.

    Parameters
    ----------
    config_file : str
        Path to user configuration file

    Returns
    -------
    dict
        Merged configuration dictionary
    """
    script_dir = Path(__file__).parent
    config_dir = script_dir / "config"

    # Load default configuration
    default_config_path = config_dir / "default_config.yaml"
    if not default_config_path.exists():
        print(f"Error: Default config not found: {default_config_path}")
        sys.exit(1)

    with open(default_config_path) as f:
        config = yaml.safe_load(f)

    # Load platform-specific configuration if specified
    platform = config.get("platform")
    if platform:
        platform_config_path = config_dir / "platforms" / f"{platform}.yaml"
        if platform_config_path.exists():
            with open(platform_config_path) as f:
                platform_config = yaml.safe_load(f)
                # Merge platform config (deep merge for nested dicts)
                config = deep_merge(config, platform_config)
        else:
            print(f"Warning: Platform config not found: {platform_config_path}")

    # Load user configuration
    user_config_path = Path(config_file)
    if not user_config_path.exists():
        print(f"Error: User config not found: {config_file}")
        print("\nCreate a user config file:")
        print(f"  cp {config_dir}/user_config.yaml.template {config_file}")
        print(f"  # Edit {config_file} with your settings")
        sys.exit(1)

    with open(user_config_path) as f:
        user_config = yaml.safe_load(f)
        # Merge user config
        config = deep_merge(config, user_config)

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries.

    Parameters
    ----------
    base : dict
        Base dictionary
    override : dict
        Dictionary with values to override

    Returns
    -------
    dict
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


async def main():
    """Main entry point for MPAS forecast workflow."""
    if len(sys.argv) != 2:
        print("Usage: python run_mpas_forecast.py <config_file>")
        print("\nExample:")
        print("  python run_mpas_forecast.py config/user_config.yaml")
        sys.exit(1)

    config_file = sys.argv[1]

    # Load configuration
    print("Loading configuration...")
    config = load_config(config_file)

    # Print configuration summary
    print("\n" + "=" * 60)
    print("MPAS Multi-Agent Forecast Workflow")
    print("=" * 60)
    print(f"Platform: {config.get('platform', 'not specified')}")
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
