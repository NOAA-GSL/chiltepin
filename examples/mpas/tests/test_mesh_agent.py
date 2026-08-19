# SPDX-License-Identifier: Apache-2.0

"""Tests for MeshAgent."""

import asyncio
from pathlib import Path

import pytest

from agents import MeshAgent


class TestMeshAgent:
    """Test suite for MeshAgent."""

    def test_init(self, mock_install_dir):
        """Test MeshAgent initialization with defaults."""
        agent = MeshAgent(work_dir=str(mock_install_dir))
        behavior = agent._behavior

        assert behavior.work_dir == mock_install_dir
        assert behavior.metis_version == "5.2.1"
        assert behavior.mpas_tools_version == "2.0.0"
        assert behavior.limited_area_version == "v2.2"
        assert behavior.log_dir == mock_install_dir / "logs"
        assert behavior.metis_downloaded is False
        assert behavior.metis_built is False
        assert behavior.metis_source_dir is None
        assert behavior.gpmetis_path is None
        assert behavior.mpas_tools_installed is False
        assert behavior.mpas_tools_dir is None
        assert behavior.hex_projection_path is None
        assert behavior.grid_rotate_path is None
        assert behavior.limited_area_installed is False
        assert behavior.limited_area_source_dir is None
        assert behavior.create_region_path is None
        assert "120km" in behavior.resolution_cells

    def test_init_custom_versions(self, mock_install_dir):
        """Test MeshAgent initialization with custom versions."""
        agent = MeshAgent(
            work_dir=str(mock_install_dir),
            metis_version="5.2.0",
            mpas_tools_version="2.1.0",
            limited_area_version="v1.0",
        )
        behavior = agent._behavior

        assert behavior.metis_version == "5.2.0"
        assert behavior.mpas_tools_version == "2.1.0"
        assert behavior.limited_area_version == "v1.0"

    @pytest.mark.asyncio
    async def test_mesh_config_from_prompt_uses_llm_parser(self, mock_install_dir, monkeypatch):
        """Prompt action should return validated config from parser helper."""
        agent = MeshAgent(work_dir=str(mock_install_dir))
        behavior = agent._behavior

        async def fake_from_prompt(prompt, model, llm_url, api_key=None):
            assert "japan" in prompt.lower()
            assert model == "qwen2.5:3b"
            return {
                "resolution": "15km",
                "name": "japan_15km",
                "regional": {
                    "project_hexes": {
                        "center_lat": 36.0,
                        "center_lon": 138.0,
                        "extent_x_km": 3000,
                        "extent_y_km": 2500,
                    }
                },
            }

        monkeypatch.setattr(behavior, "_mesh_config_from_prompt", fake_from_prompt)
        result = await behavior.mesh_config_from_prompt("Generate a 15km mesh over Japan")
        assert result["resolution"] == "15km"
        assert "regional" in result

    def test_normalize_mesh_config_accepts_top_level_project_hexes(self, mock_install_dir):
        """Normalization should accept common model output without a nested regional object."""
        agent = MeshAgent(work_dir=str(mock_install_dir))
        behavior = agent._behavior

        result = behavior._normalize_mesh_config({
            "resolution": "15km",
            "name": "japan_15km",
            "project_hexes": {
                "center_lat": 36.0,
                "center_lon": 138.0,
                "extent_x_km": 3000,
                "extent_y_km": 2500,
            },
        })

        assert result["regional"]["project_hexes"]["center_lat"] == 36.0
        assert "project_hexes" not in result

    def test_normalize_mesh_config_accepts_regional_project_hexes_fields(self, mock_install_dir):
        """Normalization should wrap raw project_hexes fields placed directly under regional."""
        agent = MeshAgent(work_dir=str(mock_install_dir))
        behavior = agent._behavior

        result = behavior._normalize_mesh_config({
            "resolution": "15km",
            "name": "japan_15km",
            "regional": {
                "center_lat": 36.0,
                "center_lon": 138.0,
                "extent_x_km": 3000,
                "extent_y_km": 2500,
            },
        })

        assert result["regional"]["project_hexes"]["center_lon"] == 138.0
        assert "center_lat" not in result["regional"]

    def test_normalize_mesh_config_wraps_flat_create_region_ellipse(self, mock_install_dir):
        """Normalization should wrap flat create_region ellipse fields under ellipse."""
        agent = MeshAgent(work_dir=str(mock_install_dir))
        behavior = agent._behavior

        result = behavior._normalize_mesh_config({
            "resolution": "15km",
            "regional": {
                "create_region": {
                    "point": "36.0, 138.0",
                    "semi-major-axis": 1800000,
                    "semi-minor-axis": 600000,
                    "orientation-angle": 35,
                }
            },
        })

        ellipse = result["regional"]["create_region"]["ellipse"]
        assert ellipse["orientation-angle"] == 35
        assert ellipse["point"] == "36.0, 138.0"

    def test_normalize_mesh_config_accepts_regional_shape_key_without_create_region(
        self, mock_install_dir,
    ):
        """Normalization should wrap direct regional shape keys under create_region."""
        agent = MeshAgent(work_dir=str(mock_install_dir))
        behavior = agent._behavior

        result = behavior._normalize_mesh_config({
            "resolution": "15km",
            "regional": {
                "ellipse": {
                    "point": "36.0, 138.0",
                    "semi-major-axis": 1800000,
                    "semi-minor-axis": 600000,
                    "orientation-angle": 35,
                }
            },
        })

        assert "create_region" in result["regional"]
        assert "ellipse" in result["regional"]["create_region"]

    def test_normalize_mesh_config_accepts_create_region_type_with_alias_keys(
        self, mock_install_dir,
    ):
        """Normalization should convert type-based shape payload and underscore aliases."""
        agent = MeshAgent(work_dir=str(mock_install_dir))
        behavior = agent._behavior

        result = behavior._normalize_mesh_config({
            "resolution": "15km",
            "regional": {
                "create_region": {
                    "type": "ellipse",
                    "point": "36.0, 138.0",
                    "semi_major_axis": 1800000,
                    "semi_minor_axis": 600000,
                    "orientation_angle": 35,
                }
            },
        })

        ellipse = result["regional"]["create_region"]["ellipse"]
        assert ellipse["semi-major-axis"] == 1800000
        assert ellipse["semi-minor-axis"] == 600000
        assert ellipse["orientation-angle"] == 35

    def test_write_create_region_spec_accepts_center_lat_lon(self, mock_install_dir):
        """create_region ellipse writer should accept center_lat/center_lon aliases."""
        agent = MeshAgent(work_dir=str(mock_install_dir))
        behavior = agent._behavior

        spec_file = behavior._write_create_region_spec(
            Path(mock_install_dir),
            {
                "ellipse": {
                    "center_lat": 36.0,
                    "center_lon": 138.0,
                    "semi-major-axis": 1800000,
                    "semi-minor-axis": 600000,
                    "orientation-angle": 35,
                }
            },
            "japan",
        )

        contents = spec_file.read_text()
        assert "Type: ellipse" in contents
        assert "Point: 36.0, 138.0" in contents

    def test_write_create_region_spec_accepts_underscore_axis_aliases(
        self, mock_install_dir,
    ):
        """create_region ellipse writer should accept underscore axis aliases."""
        agent = MeshAgent(work_dir=str(mock_install_dir))
        behavior = agent._behavior

        spec_file = behavior._write_create_region_spec(
            Path(mock_install_dir),
            {
                "ellipse": {
                    "point": "36.0, 138.0",
                    "semi_major_axis": 1800000,
                    "semi_minor_axis": 600000,
                    "orientation_angle": 35,
                }
            },
            "japan",
        )

        contents = spec_file.read_text()
        assert "Semi-major-axis: 1800000" in contents
        assert "Semi-minor-axis: 600000" in contents
        assert "Orientation-angle: 35" in contents

    @pytest.mark.asyncio
    async def test_mesh_config_from_prompt_normalizes_llm_response(
        self, mock_install_dir, monkeypatch,
    ):
        """Prompt path should normalize LLM response to valid schema."""
        agent = MeshAgent(work_dir=str(mock_install_dir))
        behavior = agent._behavior

        # Geo-lookup approach: LLM returns region names
        def fake_ollama_chat(prompt, model, llm_url, system_prompt=None, timeout_seconds=120, api_key=None):
            return {
                "resolution": "15km",
                "name": "japan_15km",
                "region_names": ["Japan"],
                "buffer_km": 50,
            }

        monkeypatch.setattr(behavior, "_ollama_chat", fake_ollama_chat)
        config = await behavior._mesh_config_from_prompt(
            "Generate a 15km mesh that covers Japan",
            "qwen2.5:3b",
            "http://localhost:11434/api/chat",
        )

        assert config["resolution"] == "15km"
        ellipse = config["regional"]["create_region"]["ellipse"]
        assert ellipse["semi-major-axis"] > 900_000
        assert ellipse["semi-minor-axis"] > 300_000
        assert 30 <= ellipse["orientation-angle"] <= 50

    @pytest.mark.asyncio
    async def test_mesh_config_from_prompt_falls_back_to_cached_on_failure(
        self, mock_install_dir, monkeypatch,
    ):
        """Prompt parsing should fall back to cached config when LLM fails."""
        agent = MeshAgent(work_dir=str(mock_install_dir))
        behavior = agent._behavior

        prompt = "Generate a 15km mesh that covers Japan"
        behavior._last_good_prompt_mesh_configs[prompt] = {
            "resolution": "15km",
            "regional": {
                "create_region": {
                    "ellipse": {
                        "point": "36.3, 138.1",
                        "semi-major-axis": 1400000,
                        "semi-minor-axis": 500000,
                        "orientation-angle": 30,
                    }
                }
            },
        }

        def fake_ollama_chat(*args, **kwargs):
            raise TimeoutError("timed out")

        monkeypatch.setattr(behavior, "_ollama_chat", fake_ollama_chat)
        config = await behavior._mesh_config_from_prompt(
            prompt,
            "qwen2.5:3b",
            "http://localhost:11434/api/chat",
        )

        assert config["resolution"] == "15km"
        ellipse = config["regional"]["create_region"]["ellipse"]
        assert ellipse["point"] == "36.3, 138.1"

    @pytest.mark.asyncio
    async def test_mesh_config_from_prompt_falls_back_to_disk_cache(
        self, mock_install_dir, monkeypatch,
    ):
        """Prompt parsing should load cached config from disk when LLM fails."""
        agent = MeshAgent(work_dir=str(mock_install_dir))
        behavior = agent._behavior

        prompt = "Generate a 15km mesh that covers Japan"
        behavior._prompt_cache_file.write_text(
            '{\n'
            f'  "{prompt}": {{\n'
            '    "resolution": "15km",\n'
            '    "regional": {\n'
            '      "create_region": {\n'
            '        "ellipse": {\n'
            '          "point": "36.1, 138.2",\n'
            '          "semi-major-axis": 1200000,\n'
            '          "semi-minor-axis": 450000,\n'
            '          "orientation-angle": 35\n'
            '        }\n'
            '      }\n'
            '    }\n'
            '  }\n'
            '}'
        )

        def fake_ollama_chat(*args, **kwargs):
            raise TimeoutError("timed out")

        monkeypatch.setattr(behavior, "_ollama_chat", fake_ollama_chat)
        config = await behavior._mesh_config_from_prompt(
            prompt,
            "qwen2.5:3b",
            "http://localhost:11434/api/chat",
        )

        assert config["resolution"] == "15km"
        ellipse = config["regional"]["create_region"]["ellipse"]
        assert ellipse["point"] == "36.1, 138.2"

    def test_write_create_region_spec_polygon_requires_vertices(self, mock_install_dir):
        """create_region polygon writer should fail fast with incomplete vertices."""
        agent = MeshAgent(work_dir=str(mock_install_dir))
        behavior = agent._behavior

        with pytest.raises(ValueError, match="at least 3 vertices"):
            behavior._write_create_region_spec(
                Path(mock_install_dir),
                {
                    "polygon": {
                        "point": "35.6, 139.6",
                        "vertices": ["35.0, 139.0", "36.0, 140.0"],
                    }
                },
                "japan",
            )

    @pytest.mark.asyncio
    async def test_prompt_queue_loop_processes_requests_sequentially(
        self, mock_install_dir, monkeypatch,
    ):
        """Queue loop should process queued prompt requests and store results."""
        agent = MeshAgent(work_dir=str(mock_install_dir))
        behavior = agent._behavior
        order = []

        async def fake_from_prompt(prompt, model, llm_url, api_key=None):
            order.append(f"parse:{prompt}")
            await asyncio.sleep(0.02)
            return {"resolution": "120km", "name": "global_test"}

        async def fake_generate_mesh(mesh_config, mesh_data_dir):
            order.append(f"generate:{mesh_data_dir}")
            await asyncio.sleep(0.02)
            return {"mesh": "dummy.nc", "graph": "dummy.graph.info", "partitions": {}}

        monkeypatch.setattr(behavior, "_mesh_config_from_prompt", fake_from_prompt)
        monkeypatch.setattr(behavior, "generate_mesh", fake_generate_mesh)

        shutdown = asyncio.Event()
        loop_task = asyncio.create_task(behavior.process_mesh_prompt_queue(shutdown))
        try:
            request_id = await behavior.submit_mesh_prompt(
                prompt="make a global mesh",
                mesh_data_dir=str(mock_install_dir / "mesh_data"),
            )

            for _ in range(50):
                status = await behavior.get_mesh_prompt_result(request_id)
                if status.get("status") == "succeeded":
                    break
                await asyncio.sleep(0.02)

            status = await behavior.get_mesh_prompt_result(request_id)
            assert status["status"] == "succeeded"
            assert status["mesh_config"]["resolution"] == "120km"
            assert status["mesh_result"]["mesh"] == "dummy.nc"
            assert order == [
                "parse:make a global mesh",
                f"generate:{str(mock_install_dir / 'mesh_data')}",
            ]
        finally:
            shutdown.set()
            await loop_task

    # TODO: Add tests for agent actions
    # These tests will require mocking the actual download, build, and partition
    # operations or setting up test fixtures with sample data
    #
    # Example tests to add:
    # - test_install_metis()
    # - test_install_limited_area()
    # - test_install()
    # - test_download_global_mesh()
    # - test_create_regional_mesh()
    # - test_create_regional_mesh_raises_without_install()
    # - test_partition_mesh()
    # - test_partition_mesh_raises_without_install()
