# SPDX-License-Identifier: Apache-2.0

"""MeshAgent - Manages mesh generation and partitioning.

This agent handles the complete mesh lifecycle for MPAS forecasts:
- Downloads and builds Metis graph partitioning library
- Downloads and builds MPAS-Tools (hex_projection, grid_rotate)
- Installs MPAS-Limited-Area for regional mesh extraction
- Downloads precomputed global mesh files
- Generates regional meshes via hex_projection + grid_rotate
- Generates regional meshes via MPAS-Limited-Area
- Partitions meshes for MPI parallel execution
"""

import asyncio
import json
import math
import os
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from agents.geo_lookup_osm import (
    lookup_region,
    geometry_to_ellipse,
    geometry_to_circle,
    geometry_to_polygon,
)
from parsl.app.errors import BashExitFailure

from chiltepin.agents import agent_action, agent_loop, chiltepin_agent
from chiltepin.tasks import bash_task, python_task


@chiltepin_agent()
class MeshAgent:
    """Agent for managing mesh generation and partitioning.

    Provides a top-level ``generate_mesh()`` action that routes to the
    correct generation path based on request configuration.  Individual
    utility actions remain callable for advanced use.
    """

    def __init__(
        self,
        work_dir: str,
        metis_version: str = "5.2.1",
        mpas_tools_version: str = "2.0.0",
        limited_area_version: str = "v2.2",
    ):
        """Initialize MeshAgent.

        Parameters
        ----------
        work_dir : str
            Directory where mesh tools will be installed and agent logs will be written
        metis_version : str, optional
            Metis version to install, by default "5.2.1"
        mpas_tools_version : str, optional
            MPAS-Tools version to clone, by default "2.0.0"
        limited_area_version : str, optional
            MPAS-Limited-Area version to clone, by default "v2.2"
        """
        self.work_dir = Path(work_dir)
        self.log_dir = self.work_dir / "logs"
        self.metis_version = metis_version
        self.mpas_tools_version = mpas_tools_version
        self.limited_area_version = limited_area_version

        # Metis state
        self.metis_downloaded = False
        self.metis_built = False
        self.metis_source_dir: Optional[Path] = None
        self.gpmetis_path: Optional[Path] = None

        # MPAS-Tools state (hex_projection + grid_rotate)
        self.mpas_tools_installed = False
        self.mpas_tools_dir: Optional[Path] = None
        self.hex_projection_path: Optional[Path] = None
        self.grid_rotate_path: Optional[Path] = None

        # MPAS-Limited-Area state
        self.limited_area_installed = False
        self.limited_area_source_dir: Optional[Path] = None
        self.create_region_path: Optional[Path] = None

        # Mesh data
        self.mesh_data_url = "https://www2.mmm.ucar.edu/projects/mpas/atmosphere_meshes"
        # Precomputed resolutions available for download from UCAR
        self.resolution_cells = {
            "480km": 2562,
            "384km": 4002,
            "240km": 10242,
            "120km": 40962,
            "60km": 163842,
            "48km": 256002,
            "30km": 655362,
            "24km": 1024002,
            "15km": 2621442,
            "12km": 4096002,
            "10km": 5898242,
            "7.5km": 10485762,
            "5km": 23592962,
            "4km": 36864002,
            "3.75km": 41943042,
            "3km": 65536002,
        }

        # Optional list-based prompt handling for background processing.
        self._pending_prompt_requests: List[Dict[str, Any]] = []
        self._prompt_results: Dict[str, Dict[str, Any]] = {}
        self._last_good_prompt_mesh_configs: Dict[str, Dict[str, Any]] = {}
        self._prompt_cache_file = self.work_dir / "prompt_mesh_config_cache.json"

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_resolution_km(resolution: str) -> float:
        """Convert resolution string like '120km' or '7.5km' to float km."""
        return float(resolution.lower().replace("km", ""))

    @staticmethod
    def _parse_coordinate(value) -> float:
        """Parse a coordinate value (string or number) to float."""
        if isinstance(value, str):
            return float(value.strip().rstrip(","))
        return float(value)

    @staticmethod
    def _resolve_mesh_data_dir(mesh_data_dir: str) -> Path:
        """Resolve and create the per-request mesh data directory."""
        resolved_dir = Path(mesh_data_dir)
        resolved_dir.mkdir(parents=True, exist_ok=True)
        return resolved_dir

    def _load_cached_prompt_config(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Load cached mesh config for a prompt from memory or disk."""
        cached = self._last_good_prompt_mesh_configs.get(prompt)
        if isinstance(cached, dict):
            return dict(cached)

        if not self._prompt_cache_file.exists():
            return None

        try:
            payload = json.loads(self._prompt_cache_file.read_text())
        except Exception:
            return None

        if not isinstance(payload, dict):
            return None

        cached = payload.get(prompt)
        if not isinstance(cached, dict):
            return None

        self._last_good_prompt_mesh_configs[prompt] = dict(cached)
        return dict(cached)

    def _store_cached_prompt_config(self, prompt: str, mesh_config: Dict[str, Any]) -> None:
        """Store successful prompt mesh config in memory and on disk."""
        self._last_good_prompt_mesh_configs[prompt] = dict(mesh_config)

        cache_payload: Dict[str, Any] = {}
        if self._prompt_cache_file.exists():
            try:
                existing = json.loads(self._prompt_cache_file.read_text())
                if isinstance(existing, dict):
                    cache_payload = existing
            except Exception:
                cache_payload = {}

        cache_payload[prompt] = mesh_config
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._prompt_cache_file.write_text(json.dumps(cache_payload, indent=2, sort_keys=True))

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any]:
        """Extract and parse a JSON object from model output text."""
        cleaned = text.strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Handle ```json ... ``` fenced code blocks
        fence_start = cleaned.find("```json")
        if fence_start != -1:
            json_start = cleaned.find("\n", fence_start) + 1
            fence_end = cleaned.find("```", json_start)
            if fence_end != -1:
                try:
                    parsed = json.loads(cleaned[json_start:fence_end].strip())
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model response did not contain a JSON object.")

        candidate = cleaned[start:end + 1]
        parsed = json.loads(candidate)
        if not isinstance(parsed, dict):
            raise ValueError("Parsed JSON payload is not an object.")
        return parsed

    def _normalize_mesh_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize and validate a model-produced mesh configuration payload."""
        if not isinstance(payload, dict):
            raise ValueError("Mesh config payload must be a JSON object.")

        config = payload.get("mesh_config") or payload.get("mesh") or payload
        if not isinstance(config, dict):
            raise ValueError("Mesh config must be a JSON object.")

        create_region_shape_keys = {"polygon", "circle", "ellipse", "channel"}
        ellipse_keys = {"point", "semi-major-axis", "semi-minor-axis", "orientation-angle"}
        circle_keys = {"point", "radius"}
        channel_keys = {"upper-lat", "lower-lat"}
        polygon_keys = {"point", "vertices"}

        if "project_hexes" in config or "create_region" in config:
            config = {
                key: value for key, value in config.items()
                if key not in {"project_hexes", "create_region"}
            } | {
                "regional": {
                    key: payload_value
                    for key, payload_value in config.items()
                    if key in {"project_hexes", "create_region"}
                }
            }

        if "regional" not in config and any(
            key in payload for key in {"project_hexes", "create_region"}
        ):
            config = dict(config)
            config["regional"] = {
                key: payload_value
                for key, payload_value in payload.items()
                if key in {"project_hexes", "create_region"}
            }

        resolution = config.get("resolution", "120km")
        if resolution not in self.resolution_cells:
            raise ValueError(
                f"Resolution '{resolution}' is invalid. "
                f"Valid values: {list(self.resolution_cells.keys())}"
            )

        regional = config.get("regional")
        if regional is not None:
            if not isinstance(regional, dict):
                raise ValueError("'regional' must be an object when provided.")
            regional = {
                key: value for key, value in regional.items()
                if value is not None and value != {}
            }

            if "create_region" not in regional and any(
                key in regional for key in create_region_shape_keys
            ):
                regional = {
                    "create_region": {
                        key: regional[key]
                        for key in create_region_shape_keys
                        if key in regional
                    }
                }

            if "create_region" in regional and isinstance(regional["create_region"], dict):
                create_region_config = {
                    key: value
                    for key, value in regional["create_region"].items()
                    if value is not None and value != {}
                }

                # Normalize common LLM key variants to MPAS-Limited-Area names.
                key_aliases = {
                    "semi_major_axis": "semi-major-axis",
                    "semi_minor_axis": "semi-minor-axis",
                    "orientation_angle": "orientation-angle",
                    "upper_lat": "upper-lat",
                    "lower_lat": "lower-lat",
                }
                create_region_config = {
                    key_aliases.get(key, key): value
                    for key, value in create_region_config.items()
                }

                # Also normalize aliases inside explicit shape payloads, e.g.
                # {"ellipse": {"semi_minor_axis": ...}}.
                for shape_name in create_region_shape_keys:
                    shape_payload = create_region_config.get(shape_name)
                    if isinstance(shape_payload, dict):
                        create_region_config[shape_name] = {
                            key_aliases.get(key, key): value
                            for key, value in shape_payload.items()
                        }

                create_region_keys = set(create_region_config.keys())
                has_shape_key = any(
                    key in create_region_shape_keys for key in create_region_keys
                )

                # Allow style: {"type": "ellipse", ...shape fields...}
                if not has_shape_key:
                    shape_type = create_region_config.get("type")
                    if not isinstance(shape_type, str):
                        shape_type = create_region_config.get("shape")
                    if isinstance(shape_type, str):
                        shape_type = shape_type.strip().lower()
                        if shape_type in create_region_shape_keys:
                            shape_payload = {
                                key: value
                                for key, value in create_region_config.items()
                                if key not in {"type", "shape"}
                            }
                            create_region_config = {shape_type: shape_payload}
                            create_region_keys = set(create_region_config.keys())
                            has_shape_key = True

                if not has_shape_key and create_region_keys:
                    if create_region_keys.issubset(ellipse_keys):
                        create_region_config = {"ellipse": create_region_config}
                    elif create_region_keys.issubset(circle_keys):
                        create_region_config = {"circle": create_region_config}
                    elif create_region_keys.issubset(channel_keys):
                        create_region_config = {"channel": create_region_config}
                    elif create_region_keys.issubset(polygon_keys):
                        create_region_config = {"polygon": create_region_config}
                regional["create_region"] = create_region_config

            project_hexes_keys = {
                "center_lat",
                "center_lon",
                "extent_x_km",
                "extent_y_km",
                "rotation_degrees",
            }
            if "project_hexes" not in regional and "create_region" not in regional:
                regional_keys = set(regional.keys())
                if regional_keys and regional_keys.issubset(project_hexes_keys):
                    regional = {"project_hexes": regional}
            config["regional"] = regional
            regional_modes = [k for k in ["project_hexes", "create_region"] if k in regional]
            if len(regional_modes) != 1:
                raise ValueError(
                    "'regional' must contain exactly one of 'project_hexes' or 'create_region'. "
                    f"Got keys: {sorted(regional.keys())}"
                )

        return config

    def _llm_chat(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        timeout_seconds: int = 120,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call an LLM via litellm and return parsed JSON object.

        The model string uses litellm provider prefixes, e.g.
        ``anthropic/claude-sonnet-5``, ``ollama_chat/qwen2.5:3b``,
        ``openai/gpt-4o``.
        """
        import litellm

        system = system_prompt or self._mesh_prompt_system_prompt()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout_seconds,
                num_retries=3,
            )
        except Exception as e:
            raise RuntimeError(f"LLM endpoint request failed: {e}") from e

        content = response.choices[0].message.content

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM response did not contain message content.")

        return self._extract_json_object(content)

    def _mesh_osm_system_prompt(self) -> str:
        """System prompt for OSM Nominatim geo-lookup."""
        resolutions = ", ".join(self.resolution_cells.keys())
        return (
            "You extract geographic entity names, mesh shape, and mesh creation "
            "method from user mesh requests. "
            "Return ONLY a JSON object (no markdown, no explanation). "
            f"Downloadable resolution values are: {resolutions}. "
            "Any resolution is valid if the user requests it explicitly. "
            "Schema: "
            '{"resolution": "STRING", "name": "STRING", '
            '"region_names": ["NAME1", "NAME2", ...], '
            '"exclude_names": ["NAME1", ...], '
            '"shape": "STRING or null", '
            '"method": "STRING or null"}. '
            "Field definitions: "
            "- region_names: a list of standard geographic entity names (countries, "
            "states, provinces, cities, or sub-national regions) whose union covers "
            "the described region. "
            "Use official English names as they appear in OpenStreetMap "
            "(e.g. 'Japan', 'United States of America', 'Texas', 'Oklahoma', "
            "'Denver', 'Manhattan'). "
            "IMPORTANT: Do NOT use continent names like 'Europe' or 'Asia'. "
            "Instead, expand them into the list of constituent countries. "
            "For example, 'Europe' should become ['France', 'Germany', 'Italy', "
            "'Spain', 'Portugal', 'United Kingdom', ...] listing all relevant "
            "European countries. "
            "- exclude_names: optional list of countries to exclude from the region. "
            "Use when the user says to exclude specific countries from a "
            "larger region (e.g. exclude Russia from Europe). "
            "- shape: the mesh shape requested by the user, or null if not "
            "specified. Valid values: 'rectangle', 'ellipse', 'circle', 'polygon'. "
            "Only set this if the user explicitly mentions a shape. "
            "- method: the mesh creation method requested by the user, or null "
            "if not specified. Valid values: 'project_hexes', 'create_region'. "
            "Only set this if the user explicitly mentions a method. "
            "Rules: "
            "- For continents, expand into ALL constituent countries. "
            "- For countries, use the country name (e.g. 'Japan', 'France'). "
            "- For countries with distant overseas territories, use only the mainland "
            "name if the user clearly means the mainland. "
            "- For vernacular regions (e.g. 'Tornado Alley', 'CONUS', 'The Rockies'), "
            "list the constituent states or provinces that make up that region. "
            "- If the user says to EXCLUDE certain countries or regions, put them in "
            "exclude_names, not in region_names. "
            "- For global meshes, set region_names to an empty list. "
            "- If the user specifies a resolution, use it exactly. "
            "- name should be a short descriptive slug (e.g. 'japan_15km')."
        )

    def _resolve_mesh_method(
        self,
        resolution: str,
        shape: Optional[str],
        method: Optional[str],
        is_regional: bool,
    ) -> str:
        """Apply decision tree to determine mesh creation method.

        Raises ValueError for unsupported configurations.
        Returns 'global', 'project_hexes', or 'create_region'.
        """
        if not is_regional:
            if resolution not in self.resolution_cells:
                raise ValueError(
                    f"Global mesh resolution '{resolution}' is not available for "
                    f"download. Available: {list(self.resolution_cells.keys())}"
                )
            return "global"

        has_downloadable_resolution = resolution in self.resolution_cells
        needs_create_region = shape in ("ellipse", "circle", "polygon")

        if method == "project_hexes":
            if needs_create_region:
                raise ValueError(
                    f"project_hexes only supports rectangular meshes, "
                    f"but shape '{shape}' was requested."
                )
            return "project_hexes"

        if method == "create_region":
            if not has_downloadable_resolution:
                raise ValueError(
                    f"create_region requires a downloadable global mesh, but "
                    f"resolution '{resolution}' is not available. "
                    f"Available: {list(self.resolution_cells.keys())}. "
                    f"Use project_hexes with a rectangular shape for custom "
                    f"resolutions."
                )
            return "create_region"

        # No method specified — infer from shape and resolution
        if needs_create_region:
            if not has_downloadable_resolution:
                raise ValueError(
                    f"Shape '{shape}' requires the create_region method, which "
                    f"needs a downloadable global mesh, but resolution "
                    f"'{resolution}' is not available. "
                    f"Available: {list(self.resolution_cells.keys())}. "
                    f"Use a rectangular shape for custom resolutions."
                )
            return "create_region"

        return "project_hexes"

    @staticmethod
    def _rectangle_to_vertices(rect: Dict[str, Any]) -> list:
        """Convert rectangle params to 4 (lat, lon) vertices.

        The equator-facing corners are pushed equatorward so the
        great-circle edge between them still covers the intended
        boundary at its midpoint.
        """
        import math
        clat = rect["center_lat"]
        clon = rect["center_lon"]
        half_x = rect["extent_x_km"] * 1000.0 / 2.0
        half_y = rect["extent_y_km"] * 1000.0 / 2.0
        rot = math.radians(rect.get("rotation_degrees", 0.0))
        meters_per_deg = 111_320.0
        cos_clat = math.cos(math.radians(clat))
        cos_r, sin_r = math.cos(rot), math.sin(rot)

        corners_local = [
            (-half_x, -half_y),
            ( half_x, -half_y),
            ( half_x,  half_y),
            (-half_x,  half_y),
        ]

        vertices = []
        for lx, ly in corners_local:
            rx = lx * cos_r - ly * sin_r
            ry = lx * sin_r + ly * cos_r
            lat = clat + ry / meters_per_deg
            lon = clon + rx / (meters_per_deg * cos_clat)
            vertices.append([lat, lon])

        # Correct equator-facing E-W edges for great-circle sag
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            lat1, lon1 = vertices[i]
            lat2, lon2 = vertices[j]
            dlon = abs(lon2 - lon1)
            dlat = abs(lat2 - lat1)
            if dlon <= dlat or dlon < 5.0:
                continue
            avg_lat = (lat1 + lat2) / 2.0
            # Skip pole-facing edges (sag increases coverage there)
            if clat >= 0 and avg_lat >= clat:
                continue
            if clat < 0 and avg_lat <= clat:
                continue
            # Push corners equatorward: gc midpoint will reach original lat
            half_dlon = math.radians(dlon / 2.0)
            cos_half = math.cos(half_dlon)
            for k in (i, j):
                lat_k = math.radians(vertices[k][0])
                vertices[k][0] = math.degrees(
                    math.atan(math.tan(lat_k) * cos_half)
                )

        return [(round(v[0], 6), round(v[1], 6)) for v in vertices]

    def _build_mesh_config_from_geometry(
        self,
        resolved_method: str,
        resolution: str,
        name: str,
        shape: Optional[str],
        geom,
        buffer_km: float,
    ) -> Dict[str, Any]:
        """Build mesh_config dict from resolved method, shape, and geometry."""
        if resolved_method == "global":
            return {"resolution": resolution, "name": name}

        # Buffer must cover 7 boundary-layer cells and be >= 10% of diagonal
        import math
        resolution_km = self._parse_resolution_km(resolution)
        boundary_thickness_km = resolution_km * 7
        hull = geom.convex_hull
        bounds = hull.bounds  # (minx, miny, maxx, maxy) in degrees
        cos_mid = math.cos(math.radians((bounds[1] + bounds[3]) / 2.0))
        dx_km = (bounds[2] - bounds[0]) * 111.32 * cos_mid
        dy_km = (bounds[3] - bounds[1]) * 111.32
        diagonal_km = math.sqrt(dx_km ** 2 + dy_km ** 2)
        buffer_km = max(buffer_km, diagonal_km * 0.10, boundary_thickness_km)

        if resolved_method == "project_hexes":
            from agents.geo_lookup_osm import geometry_to_rectangle
            rect = geometry_to_rectangle(geom, buffer_km=buffer_km)
            phex_config = {
                "center_lat": rect["center_lat"],
                "center_lon": rect["center_lon"],
                "extent_x_km": rect["extent_x_km"],
                "extent_y_km": rect["extent_y_km"],
            }
            if rect.get("rotation_degrees", 0.0) != 0.0:
                phex_config["rotation_degrees"] = rect["rotation_degrees"]
            return {
                "resolution": resolution,
                "name": name,
                "regional": {"project_hexes": phex_config},
            }

        # create_region with the requested shape (default to polygon)
        # Rectangle/square use geometry_to_rectangle → 4 polygon vertices
        if shape in ("rectangle", "square", None):
            from agents.geo_lookup_osm import geometry_to_rectangle
            rect = geometry_to_rectangle(geom, buffer_km=buffer_km)
            vertices = self._rectangle_to_vertices(rect)
            create_region = {
                "polygon": {
                    "point": f"{rect['center_lat']}, {rect['center_lon']}",
                    "vertices": vertices,
                }
            }
            return {
                "resolution": resolution,
                "name": name,
                "regional": {"create_region": create_region},
            }

        shape_funcs = {
            "ellipse": geometry_to_ellipse,
            "circle": geometry_to_circle,
            "polygon": geometry_to_polygon,
        }
        effective_shape = shape if shape in shape_funcs else "polygon"
        shape_func = shape_funcs[effective_shape]

        shape_data = shape_func(geom, buffer_km=buffer_km)
        if effective_shape == "ellipse":
            create_region = {
                "ellipse": {
                    "point": f"{shape_data['center_lat']}, {shape_data['center_lon']}",
                    "semi-major-axis": shape_data["semi_major_m"],
                    "semi-minor-axis": shape_data["semi_minor_m"],
                    "orientation-angle": shape_data["orientation_deg"],
                }
            }
        elif effective_shape == "circle":
            create_region = {
                "circle": {
                    "point": f"{shape_data['center_lat']}, {shape_data['center_lon']}",
                    "radius": shape_data["radius_m"],
                }
            }
        elif effective_shape == "polygon":
            create_region = {
                "polygon": {
                    "point": f"{shape_data['point_lat']}, {shape_data['point_lon']}",
                    "vertices": shape_data["vertices"],
                }
            }

        return {
            "resolution": resolution,
            "name": name,
            "regional": {"create_region": create_region},
        }

    async def _mesh_config_from_prompt_osm(
        self,
        prompt: str,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        buffer_km: float = 50.0,
    ) -> Dict[str, Any]:
        """LLM extracts region names, shape, and method; OSM provides geometry."""
        try:
            payload = await asyncio.to_thread(
                self._llm_chat, prompt, model,
                self._mesh_osm_system_prompt(), 300, api_key, api_base,
            )
            region_names = payload.get("region_names", [])
            if not region_names:
                return self._normalize_mesh_config(payload)

            resolution = payload.get("resolution", "120km")
            name = payload.get("name", f"mesh_{resolution}")
            exclude_names = payload.get("exclude_names", [])
            shape = payload.get("shape")
            method = payload.get("method")

            resolved_method = self._resolve_mesh_method(
                resolution, shape, method, is_regional=True,
            )

            geom = await asyncio.to_thread(
                lookup_region, region_names, exclude_names=exclude_names,
            )
            if geom is None:
                raise ValueError(
                    f"Could not resolve geographic entities: {region_names}"
                )

            mesh_config = self._build_mesh_config_from_geometry(
                resolved_method, resolution, name, shape, geom, buffer_km,
            )
            self._store_cached_prompt_config(prompt, mesh_config)
            return mesh_config
        except Exception as exc:
            cached = self._load_cached_prompt_config(prompt)
            if cached is not None:
                return cached
            raise RuntimeError(
                f"Failed to generate mesh config from prompt: {exc}"
            ) from exc

    async def _mesh_config_from_prompt(
        self,
        prompt: str,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Convert natural language into mesh config via LLM + geo-lookup."""
        return await self._mesh_config_from_prompt_osm(prompt, model, api_key, api_base)

    @staticmethod
    def _write_project_hexes_namelist(
        work_dir: Path, cell_spacing_km: float,
        extent_x_km: float, extent_y_km: float,
        center_lat: float, center_lon: float,
    ) -> None:
        """Write namelist.projections for hex_projection tool."""
        work_dir.mkdir(parents=True, exist_ok=True)
        namelist = work_dir / "namelist.projections"
        namelist.write_text(
            f"&mesh\n"
            f"  cell_spacing_km = {cell_spacing_km}\n"
            f"  mesh_length_x_km = {extent_x_km}\n"
            f"  mesh_length_y_km = {extent_y_km}\n"
            f"  earth_radius_km = 6371.229\n"
            f"/\n"
            f"&projection\n"
            f'  projection_type = "lambert_conformal"\n'
            f"/\n"
            f"&lambert_conformal\n"
            f"  reference_longitude_degrees = {center_lon}\n"
            f"  standard_longitude_degrees = {center_lon}\n"
            f"  reference_latitude_degrees = {center_lat}\n"
            f"  standard_parallel_1_degrees = {center_lat}\n"
            f"  standard_parallel_2_degrees = {center_lat}\n"
            f"/\n"
        )


    @staticmethod
    def _write_rotate_namelist(
        work_dir: Path,
        center_lat: float, center_lon: float,
        rotation_degrees: float,
        original_lat: Optional[float] = None,
        original_lon: Optional[float] = None,
    ) -> None:
        """Write namelist.input for grid_rotate tool."""
        work_dir.mkdir(parents=True, exist_ok=True)
        namelist = work_dir / "namelist.input"
        orig_lat = center_lat if original_lat is None else original_lat
        orig_lon = center_lon if original_lon is None else original_lon
        namelist.write_text(
            f"&input\n"
            f"   config_original_latitude_degrees = {orig_lat}\n"
            f"   config_original_longitude_degrees = {orig_lon}\n"
            f"\n"
            f"   config_new_latitude_degrees = {center_lat}\n"
            f"   config_new_longitude_degrees = {center_lon}\n"
            f"   config_birdseye_rotation_counter_clockwise_degrees = "
            f"{rotation_degrees}\n"
            f"/\n"
        )

    @staticmethod
    def _write_create_region_spec(
        work_dir: Path, create_region_config: Dict[str, Any], name: str,
    ) -> Path:
        """Write a region specification file for MPAS-Limited-Area create_region.

        Supports polygon, circle, ellipse, and channel region types.
        The region type is determined by which key is present in create_region_config.
        """
        work_dir.mkdir(parents=True, exist_ok=True)

        def point_string(region: Dict[str, Any], shape_name: str) -> str:
            """Return point string in 'lat, lon' format from common field variants."""
            point = region.get("point")
            if isinstance(point, str) and point.strip():
                return point.strip()
            if isinstance(point, (list, tuple)) and len(point) == 2:
                return f"{point[0]}, {point[1]}"
            if isinstance(point, dict):
                lat = point.get("lat")
                lon = point.get("lon")
                if lat is not None and lon is not None:
                    return f"{lat}, {lon}"

            center = region.get("center")
            if isinstance(center, str) and center.strip():
                return center.strip()
            if isinstance(center, (list, tuple)) and len(center) == 2:
                return f"{center[0]}, {center[1]}"
            if isinstance(center, dict):
                lat = center.get("lat")
                lon = center.get("lon")
                if lat is not None and lon is not None:
                    return f"{lat}, {lon}"

            center_lat = region.get("center_lat")
            center_lon = region.get("center_lon")
            if center_lat is not None and center_lon is not None:
                return f"{center_lat}, {center_lon}"

            raise ValueError(
                f"Missing 'point' for create_region {shape_name}. "
                "Expected one of: point, center, or center_lat/center_lon."
            )

        if "polygon" in create_region_config:
            poly = create_region_config["polygon"]
            vertices = poly.get("vertices")
            if not isinstance(vertices, list) or len(vertices) < 3:
                raise ValueError(
                    "create_region polygon requires at least 3 vertices. "
                    "Received missing or incomplete 'vertices'."
                )
            spec_file = work_dir / f"{name}.custom.pts"
            lines = [
                f"Name: {name}",
                "Type: custom",
                f"Point: {point_string(poly, 'polygon')}",
            ]
            for vertex in vertices:
                if isinstance(vertex, (list, tuple)) and len(vertex) == 2:
                    lines.append(f"{vertex[0]}, {vertex[1]}")
                else:
                    lines.append(str(vertex))
        elif "circle" in create_region_config:
            circ = create_region_config["circle"]
            spec_file = work_dir / f"{name}.circle.pts"
            lines = [
                f"Name: {name}",
                "Type: circle",
                f"Point: {point_string(circ, 'circle')}",
                f"radius: {circ['radius']}",
            ]
        elif "ellipse" in create_region_config:
            ell = create_region_config["ellipse"]
            semi_major_axis = ell.get("semi-major-axis", ell.get("semi_major_axis"))
            semi_minor_axis = ell.get("semi-minor-axis", ell.get("semi_minor_axis"))
            if semi_major_axis is None or semi_minor_axis is None:
                raise ValueError(
                    "Missing ellipse axis values for create_region. "
                    "Expected semi-major-axis/semi-minor-axis "
                    "(or semi_major_axis/semi_minor_axis)."
                )
            spec_file = work_dir / f"{name}.ellipse.pts"
            lines = [
                f"Name: {name}",
                "Type: ellipse",
                f"Point: {point_string(ell, 'ellipse')}",
                f"Semi-major-axis: {semi_major_axis}",
                f"Semi-minor-axis: {semi_minor_axis}",
                f"Orientation-angle: {ell.get('orientation-angle', ell.get('orientation_angle', 0))}",
            ]
        elif "channel" in create_region_config:
            chan = create_region_config["channel"]
            upper_lat = chan.get("upper-lat", chan.get("upper_lat"))
            lower_lat = chan.get("lower-lat", chan.get("lower_lat"))
            if upper_lat is None or lower_lat is None:
                raise ValueError(
                    "Missing channel bounds for create_region. "
                    "Expected upper-lat/lower-lat (or upper_lat/lower_lat)."
                )
            spec_file = work_dir / f"{name}.channel.pts"
            lines = [
                f"Name: {name}",
                "Type: channel",
                f"Upper-lat: {upper_lat}",
                f"Lower-lat: {lower_lat}",
            ]
        else:
            raise ValueError(
                f"No recognized region type in limited_area config. "
                f"Expected one of: polygon, circle, ellipse, channel. "
                f"Got keys: {sorted(create_region_config.keys())}"
            )

        spec_file.write_text("\n".join(lines) + "\n")
        return spec_file

    # -------------------------------------------------------------------------
    # Private bash tasks — Metis
    # -------------------------------------------------------------------------

    @bash_task
    def _download_metis(self) -> str:
        """Download Metis and GKlib source code from GitHub."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started download at $(date)"
            echo "Executing on $(hostname)"
            rm -rf {self.work_dir}/metis/build/{self.metis_version}
            mkdir -p {self.work_dir}/metis/build/{self.metis_version}
            cd {self.work_dir}/metis/build/{self.metis_version}
            # Clone GKlib tools needed by metis
            git clone https://github.com/KarypisLab/GKlib.git
            # Fetch and untar metis tarball
            wget -T 30 -t 3 https://github.com/KarypisLab/METIS/archive/refs/tags/v{self.metis_version}.tar.gz
            tar -xzf v{self.metis_version}.tar.gz
            echo "Completed download at $(date)"
            """
        )

    @bash_task
    def _build_metis(self) -> str:
        """Build GKlib and Metis, install to prefix directory."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started build at $(date)"
            echo "Executing on $(hostname)"
            cd {self.work_dir}/metis/build/{self.metis_version}/GKlib
            make config prefix={self.work_dir}/metis/{self.metis_version}
            make install
            cd {self.work_dir}/metis/build/{self.metis_version}/METIS-{self.metis_version}
            make config prefix={self.work_dir}/metis/{self.metis_version}
            make install
            # rm -rf {self.work_dir}/metis/build
            echo "Completed build at $(date)"
            """
        )

    @bash_task
    def _partition_mesh(self, mesh_path: str, num_ranks: int) -> str:
        """Run gpmetis to partition a mesh file."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started partition at $(date)"
            echo "Executing on $(hostname)"
            echo "Partitioning {mesh_path} into {num_ranks} parts"
            {self.gpmetis_path} -minconn -contig -niter=200 {mesh_path} {num_ranks}
            echo "Completed partition at $(date)"
            """
        )

    # -------------------------------------------------------------------------
    # Private bash tasks — MPAS-Tools (hex_projection + grid_rotate)
    # -------------------------------------------------------------------------

    @bash_task
    def _clone_mpas_tools(self) -> str:
        """Clone MPAS-Tools repository from GitHub."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started MPAS-Tools clone at $(date)"
            echo "Executing on $(hostname)"
            rm -rf {self.work_dir}/MPAS-Tools
            mkdir -p {self.work_dir}
            cd {self.work_dir}
            git clone https://github.com/MPAS-Dev/MPAS-Tools.git
            cd MPAS-Tools
            git checkout {self.mpas_tools_version}
            echo "Completed MPAS-Tools clone at $(date)"
            """
        )

    @bash_task
    def _build_hex_projection(self) -> str:
        """Build hex_projection (project_hexes) from MPAS-Tools."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started hex_projection build at $(date)"
            echo "Executing on $(hostname)"
            cd {self.work_dir}/MPAS-Tools/mesh_tools/hex_projection
            make clean || true
            make
            echo "Completed hex_projection build at $(date)"
            """
        )

    @bash_task
    def _build_grid_rotate(self) -> str:
        """Build grid_rotate from MPAS-Tools."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started grid_rotate build at $(date)"
            echo "Executing on $(hostname)"
            cd {self.work_dir}/MPAS-Tools/mesh_tools/grid_rotate
            make clean || true
            make
            echo "Completed grid_rotate build at $(date)"
            """
        )

    @bash_task
    def _project_hexes(self, work_dir: str) -> str:
        """Run project_hexes in the given working directory."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started hex_projection at $(date)"
            echo "Executing on $(hostname)"
            cd {work_dir}
            {self.hex_projection_path}
            echo "Completed hex_projection at $(date)"
            """
        )

    @bash_task
    def _grid_rotate(self, work_dir: str, input_file: str, output_file: str) -> str:
        """Run grid_rotate to reposition a mesh."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started grid_rotate at $(date)"
            echo "Executing on $(hostname)"
            cd {work_dir}
            {self.grid_rotate_path} {input_file} {output_file}
            echo "Completed grid_rotate at $(date)"
            """
        )

    # -------------------------------------------------------------------------
    # Private bash tasks — MPAS-Limited-Area
    # -------------------------------------------------------------------------

    @bash_task
    def _install_limited_area(self) -> str:
        """Install MPAS-Limited-Area by cloning from GitHub."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started MPAS-Limited-Area install at $(date)"
            echo "Executing on $(hostname)"
            rm -rf {self.work_dir}/MPAS-Limited-Area
            mkdir -p {self.work_dir}
            cd {self.work_dir}
            git clone https://github.com/MPAS-Dev/MPAS-Limited-Area.git
            cd MPAS-Limited-Area
            git checkout {self.limited_area_version}
            echo "Completed MPAS-Limited-Area install at $(date)"
            """
        )

    @bash_task
    def _create_region(
        self, parent_static_mesh: str, region_spec: str, output_dir: str,
    ) -> str:
        """Run create_region to cut a regional mesh from a global mesh."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started create_region at $(date)"
            echo "Executing on $(hostname)"
            mkdir -p {output_dir}
            cd {output_dir}
            {self.create_region_path} {region_spec} \
                {parent_static_mesh}
            echo "Completed create_region at $(date)"
            """
        )

    # -------------------------------------------------------------------------
    # Private python tasks — Mesh operations
    # -------------------------------------------------------------------------

    @python_task
    def _download_global_mesh(self, resolution: str, mesh_data_dir: str) -> None:
        """Download and extract a precomputed global mesh from UCAR."""
        import tarfile

        mesh_data_dir = Path(mesh_data_dir)
        mesh_data_dir.mkdir(parents=True, exist_ok=True)

        cells = self.resolution_cells[resolution]
        filename = f"x1.{cells}_static.tar.gz"
        url = f"{self.mesh_data_url}/{filename}"
        tarball = mesh_data_dir / filename

        urllib.request.urlretrieve(url, tarball)
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(path=mesh_data_dir)
        tarball.unlink()

    @python_task
    def _plot_mpas_mesh_png(self, file_path: str, resolution: Optional[str] = None) -> str:
        """Plot an MPAS mesh to a PNG image and return the output path."""
        import geoviews as gv
        import holoviews as hv
        import numpy as np
        from PIL import Image
        import uxarray as ux

        mesh_path = Path(file_path)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Could not find mesh file at: {mesh_path}")

        hv.extension("matplotlib")
        ux_ds = ux.open_dataset(
            str(mesh_path),
            str(mesh_path),
            grid_kwargs={"decode_times": False},
            decode_times=False,
        )
        has_boundary_mask = "bdyMaskCell" in ux_ds.data_vars

        if has_boundary_mask:
            from matplotlib.colors import ListedColormap
            _bdy_colors = [
                "#ffffff",  # 0 = interior
                "#d0e8ff",  # 1 = light blue
                "#80c8ff",  # 2 = medium blue
                "#40a0e0",  # 3 = deeper blue
                "#ffcc66",  # 4 = gold
                "#ff8844",  # 5 = orange
                "#e05050",  # 6 = red
                "#aa3090",  # 7 = purple
            ]
            mesh_plot = ux_ds["bdyMaskCell"].plot.polygons(
                cmap=ListedColormap(_bdy_colors),
                colorbar=True,
                clabel="Boundary Mask Layer",
            )
        else:
            mesh_plot = ux_ds.uxgrid.plot.edges()

        lon = ux_ds.uxgrid.face_lon.values
        lat = ux_ds.uxgrid.face_lat.values
        lon_min = float(np.nanmin(lon))
        lon_max = float(np.nanmax(lon))
        lat_min = float(np.nanmin(lat))
        lat_max = float(np.nanmax(lat))
        extent_deg = max(lon_max - lon_min, lat_max - lat_min)

        # Keep map overlay best-effort so mesh plotting remains robust.
        try:
            import cartopy.feature as cfeature
            import cartopy.crs as ccrs
            import cartopy.io.shapereader as shpreader

            coast_path = shpreader.natural_earth('10m', 'physical', 'coastline')
            coastlines = gv.Feature(cfeature.ShapelyFeature(
                shpreader.Reader(coast_path).geometries(), ccrs.PlateCarree(),
                facecolor="none", edgecolor="black", linewidth=1.2,
            ))

            borders_path = shpreader.natural_earth('10m', 'cultural', 'admin_0_boundary_lines_land')
            borders = gv.Feature(cfeature.ShapelyFeature(
                shpreader.Reader(borders_path).geometries(), ccrs.PlateCarree(),
                facecolor="none", edgecolor="#222222", linewidth=1.4,
            ))

            states_path = shpreader.natural_earth('10m', 'cultural', 'admin_1_states_provinces_lines')
            states = gv.Feature(cfeature.ShapelyFeature(
                shpreader.Reader(states_path).geometries(), ccrs.PlateCarree(),
                facecolor="none", edgecolor="#666666", linewidth=0.5,
            ))

            final_layout = mesh_plot * coastlines * borders

            center_lon = (lon_min + lon_max) / 2
            center_lat = (lat_min + lat_max) / 2
            states_threshold = 60.0 if (-130 <= center_lon <= -60 and 20 <= center_lat <= 55) else 30.0
            if extent_deg < states_threshold:
                final_layout = final_layout * states

            if extent_deg < 15.0:
                counties_path = shpreader.natural_earth('10m', 'cultural', 'admin_2_counties')
                boundary_lines = [g.boundary for g in shpreader.Reader(counties_path).geometries()]
                counties = gv.Feature(cfeature.ShapelyFeature(
                    boundary_lines, ccrs.PlateCarree(),
                    facecolor="none", edgecolor="#aaaaaa", linewidth=0.3,
                ))
                final_layout = final_layout * counties
        except Exception:
            final_layout = mesh_plot

        pad_frac = 0.05
        lon_pad = max((lon_max - lon_min) * pad_frac, 0.1)
        lat_pad = max((lat_max - lat_min) * pad_frac, 0.1)
        final_layout = final_layout.opts(
            xlim=(lon_min - lon_pad, lon_max + lon_pad),
            ylim=(lat_min - lat_pad, lat_max + lat_pad),
            bgcolor="white",
        )

        plot_name = mesh_path.stem.removesuffix(".static")
        title = f"MPAS Mesh: {plot_name}"
        if resolution:
            title = f"{title} ({resolution})"
        final_layout = final_layout.opts(fig_inches=9, title=title)

        output_path = mesh_path.with_name(f"{plot_name}.png")
        hv.save(final_layout, str(output_path), fmt="png", dpi=200)

        # Clamp final image size because backend/aspect settings can produce
        # unexpectedly large pixel dimensions.
        max_width_px = 2200
        max_height_px = 1400
        with Image.open(output_path) as img:
            width, height = img.size
            scale = min(max_width_px / width, max_height_px / height, 1.0)
            if scale < 1.0:
                new_size = (int(width * scale), int(height * scale))
                resized = img.resize(new_size, Image.Resampling.LANCZOS)
                resized.save(output_path)

        return str(output_path)


    # -------------------------------------------------------------------------
    # Public agent actions — Metis
    # -------------------------------------------------------------------------

    @agent_action
    async def install_metis(self) -> None:
        """Download and build Metis in one step with Parsl pipelining."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        download_future = self._download_metis(
            executor=["service"],
            stdout=str(self.log_dir / "metis_download.stdout"),
            stderr=str(self.log_dir / "metis_download.stderr"),
        )
        build_future = self._build_metis(
            executor=["compute"],
            stdout=str(self.log_dir / "metis_build.stdout"),
            stderr=str(self.log_dir / "metis_build.stderr"),
            inputs=[download_future],
        )

        try:
            await asyncio.wrap_future(download_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"Metis download failed (exit {e.exitcode}), "
                f"see {self.log_dir}/metis_download.stderr"
            )
        self.metis_source_dir = self.work_dir / "metis" / self.metis_version
        self.metis_downloaded = True

        try:
            await asyncio.wrap_future(build_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"Metis build failed (exit {e.exitcode}), "
                f"see {self.log_dir}/metis_build.stderr"
            )
        self.gpmetis_path = self.metis_source_dir / "bin" / "gpmetis"
        self.metis_built = True

    @agent_action
    async def partition_mesh(
        self, mesh_path: str, num_ranks: Union[int, List[int]],
    ) -> Union[str, Dict[int, str]]:
        """Partition MPAS mesh graph file for MPI execution.

        Parameters
        ----------
        mesh_path : str
            Path to graph.info file
        num_ranks : int or list[int]
            One rank count or multiple rank counts to partition for

        Returns
        -------
        str or dict
            If int input, returns one partition file path.
            If list input, returns {rank: partition_file_path}.
        """
        if not self.metis_built:
            raise RuntimeError(
                "Must call install_metis() or install() before partition_mesh()"
            )

        self.log_dir.mkdir(parents=True, exist_ok=True)

        is_single = isinstance(num_ranks, int)
        ranks = [num_ranks] if is_single else sorted(set(num_ranks))

        for rank in ranks:
            if rank <= 0:
                raise ValueError(f"Rank counts must be > 0, got {rank}")

        futures = [
            self._partition_mesh(
                mesh_path,
                rank,
                executor=["compute"],
                stdout=str(self.log_dir / f"partition_mesh.{rank}.stdout"),
                stderr=str(self.log_dir / f"partition_mesh.{rank}.stderr"),
            )
            for rank in ranks
        ]

        results = await asyncio.gather(
            *[asyncio.wrap_future(future) for future in futures],
            return_exceptions=True,
        )
        for rank, result in zip(ranks, results):
            if isinstance(result, BashExitFailure):
                raise RuntimeError(
                    f"Metis partition failed for {rank} ranks "
                    f"(exit {result.exitcode}), "
                    f"see {self.log_dir}/partition_mesh.{rank}.stderr"
                )
            if isinstance(result, Exception):
                raise RuntimeError(
                    f"Metis partition failed for {rank} ranks: {result}"
                )

        partition_paths = {rank: f"{mesh_path}.part.{rank}" for rank in ranks}
        if is_single:
            return partition_paths[ranks[0]]
        return partition_paths

    # -------------------------------------------------------------------------
    # Public agent actions — MPAS-Tools
    # -------------------------------------------------------------------------

    @agent_action
    async def install_mpas_tools(self) -> None:
        """Clone MPAS-Tools and build hex_projection + grid_rotate."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        clone_future = self._clone_mpas_tools(
            executor=["service"],
            stdout=str(self.log_dir / "mpas_tools_clone.stdout"),
            stderr=str(self.log_dir / "mpas_tools_clone.stderr"),
        )
        hex_future = self._build_hex_projection(
            executor=["compute"],
            stdout=str(self.log_dir / "hex_projection_build.stdout"),
            stderr=str(self.log_dir / "hex_projection_build.stderr"),
            inputs=[clone_future],
        )
        rotate_future = self._build_grid_rotate(
            executor=["compute"],
            stdout=str(self.log_dir / "grid_rotate_build.stdout"),
            stderr=str(self.log_dir / "grid_rotate_build.stderr"),
            inputs=[clone_future],
        )

        try:
            await asyncio.wrap_future(clone_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS-Tools clone failed (exit {e.exitcode}), "
                f"see {self.log_dir}/mpas_tools_clone.stderr"
            )
        self.mpas_tools_dir = self.work_dir / "MPAS-Tools"

        try:
            await asyncio.wrap_future(hex_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"hex_projection build failed (exit {e.exitcode}), "
                f"see {self.log_dir}/hex_projection_build.stderr"
            )
        self.hex_projection_path = (
            self.mpas_tools_dir / "mesh_tools" / "hex_projection" / "project_hexes"
        )

        try:
            await asyncio.wrap_future(rotate_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"grid_rotate build failed (exit {e.exitcode}), "
                f"see {self.log_dir}/grid_rotate_build.stderr"
            )
        self.grid_rotate_path = (
            self.mpas_tools_dir / "mesh_tools" / "grid_rotate" / "grid_rotate"
        )

        self.mpas_tools_installed = True

    @agent_action
    async def project_hexes(
        self,
        resolution: str,
        project_hexes_config: Dict[str, Any],
        mesh_name: str,
        mesh_data_dir: str,
    ) -> Dict[str, str]:
        """Generate a regional mesh using project_hexes (+ optional rotate).

        Parameters
        ----------
        resolution : str
            Cell spacing (e.g., "15km", "3km")
        project_hexes_config : dict
            project_hexes configuration block
        mesh_name : str
            Name used for working directory under mesh_data
        mesh_data_dir : str
            Directory where global/regional mesh files are written

        Returns
        -------
        dict
            {"mesh": path, "graph": path}
        """
        if not self.mpas_tools_installed:
            raise RuntimeError(
                "Must call install_mpas_tools() or install() before "
                "project_hexes()"
            )

        request_mesh_data_dir = self._resolve_mesh_data_dir(mesh_data_dir)
        work_dir = request_mesh_data_dir / mesh_name
        work_dir.mkdir(parents=True, exist_ok=True)

        self._write_project_hexes_namelist(
            work_dir,
            self._parse_resolution_km(resolution),
            float(project_hexes_config["extent_x_km"]),
            float(project_hexes_config["extent_y_km"]),
            self._parse_coordinate(project_hexes_config["center_lat"]),
            self._parse_coordinate(project_hexes_config["center_lon"]),
        )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        future = self._project_hexes(
            str(work_dir),
            executor=["compute"],
            stdout=str(self.log_dir / "hex_projection.stdout"),
            stderr=str(self.log_dir / "hex_projection.stderr"),
        )
        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"hex_projection failed (exit {e.exitcode}), "
                f"see {self.log_dir}/hex_projection.stderr"
            )

        mesh_path = str(work_dir / "mpas_hex_mesh.nc")
        rotation = float(project_hexes_config.get("rotation_degrees", 0.0))
        if rotation != 0.0:
            hex_mesh_path = mesh_path
            mesh_path = await self.grid_rotate(
                mesh_path,
                self._parse_coordinate(project_hexes_config["center_lat"]),
                self._parse_coordinate(project_hexes_config["center_lon"]),
                rotation,
            )
            Path(hex_mesh_path).unlink(missing_ok=True)

        # Rename outputs to use the mesh_name convention
        final_mesh_path = work_dir / f"{mesh_name}.nc"
        final_graph_path = work_dir / f"{mesh_name}.graph.info"
        Path(mesh_path).rename(final_mesh_path)
        (work_dir / "graph.info").rename(final_graph_path)

        return {
            "mesh": str(final_mesh_path),
            "graph": str(final_graph_path),
        }

    @agent_action
    async def grid_rotate(
        self,
        input_mesh: str,
        center_lat: float,
        center_lon: float,
        rotation_degrees: float = 0.0,
        original_lat: Optional[float] = None,
        original_lon: Optional[float] = None,
    ) -> str:
        """Rotate/reposition a mesh using grid_rotate.

        Parameters
        ----------
        input_mesh : str
            Path to input mesh NetCDF file
        center_lat : float
            Target center latitude (degrees)
        center_lon : float
            Target center longitude (degrees)
        rotation_degrees : float, optional
            Counter-clockwise rotation (degrees), by default 0.0
        original_lat : float, optional
            Original mesh center latitude. Defaults to center_lat.
        original_lon : float, optional
            Original mesh center longitude. Defaults to center_lon.

        Returns
        -------
        str
            Path to rotated mesh file
        """
        if not self.mpas_tools_installed:
            raise RuntimeError(
                "Must call install_mpas_tools() or install() before "
                "grid_rotate()"
            )

        work_dir = Path(input_mesh).parent
        self._write_rotate_namelist(
            work_dir,
            center_lat,
            center_lon,
            rotation_degrees,
            original_lat,
            original_lon,
        )

        output_file = str(work_dir / "rotated_mesh.nc")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        future = self._grid_rotate(
            str(work_dir),
            input_mesh,
            output_file,
            executor=["compute"],
            stdout=str(self.log_dir / "grid_rotate.stdout"),
            stderr=str(self.log_dir / "grid_rotate.stderr"),
        )
        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"grid_rotate failed (exit {e.exitcode}), "
                f"see {self.log_dir}/grid_rotate.stderr"
            )

        return output_file

    # -------------------------------------------------------------------------
    # Public agent actions — MPAS-Limited-Area
    # -------------------------------------------------------------------------

    @agent_action
    async def install_limited_area(self) -> None:
        """Install MPAS-Limited-Area by cloning from GitHub."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        future = self._install_limited_area(
            executor=["service"],
            stdout=str(self.log_dir / "limited_area_install.stdout"),
            stderr=str(self.log_dir / "limited_area_install.stderr"),
        )
        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS-Limited-Area install failed (exit {e.exitcode}), "
                f"see {self.log_dir}/limited_area_install.stderr"
            )

        self.limited_area_source_dir = self.work_dir / "MPAS-Limited-Area"
        self.create_region_path = self.limited_area_source_dir / "create_region"
        self.limited_area_installed = True

    @agent_action
    async def create_region(
        self,
        resolution: str,
        create_region_config: Dict[str, Any],
        mesh_name: str,
        mesh_data_dir: str,
    ) -> Dict[str, str]:
        """Run create_region to cut a regional mesh from a global mesh.

        Writes the region specification file from create_region_config,
        then invokes the create_region utility.

        Parameters
        ----------
        resolution : str
            Resolution of the parent global mesh (e.g., "120km")
        create_region_config : dict
            Region config containing one of: polygon, circle, ellipse, channel
        mesh_name : str
            Name used for the spec file and output directory
        mesh_data_dir : str
            Directory where global/regional mesh files are written

        Returns
        -------
        dict
            {"static": path, "graph": path}
        """
        if not self.limited_area_installed:
            raise RuntimeError(
                "Must call install_limited_area() or install() before "
                "create_region()"
            )
        if resolution not in self.resolution_cells:
            raise ValueError(
                f"Resolution '{resolution}' not available for download. "
                f"Available: {list(self.resolution_cells.keys())}"
            )
        request_mesh_data_dir = self._resolve_mesh_data_dir(mesh_data_dir)
        output_dir = request_mesh_data_dir / mesh_name
        spec_file = self._write_create_region_spec(
            output_dir, create_region_config, mesh_name,
        )

        cells = self.resolution_cells[resolution]
        parent_static_mesh = request_mesh_data_dir / f"x1.{cells}.static.nc"
        if not parent_static_mesh.exists():
            await self.download_global_mesh(resolution, mesh_data_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)

        future = self._create_region(
            str(parent_static_mesh),
            str(spec_file),
            str(output_dir),
            executor=["compute"],
            stdout=str(self.log_dir / "create_region.stdout"),
            stderr=str(self.log_dir / "create_region.stderr"),
        )
        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"create_region failed (exit {e.exitcode}), "
                f"see {self.log_dir}/create_region.stderr"
            )

        return {
            "static": str(output_dir / f"{mesh_name}.static.nc"),
            "graph": str(output_dir / f"{mesh_name}.graph.info"),
        }


    # -------------------------------------------------------------------------
    # Public agent actions — Combined install
    # -------------------------------------------------------------------------

    @agent_action
    async def install(self) -> None:
        """Install all mesh tools: Metis, MPAS-Tools, and Limited-Area.

        Runs installations concurrently where possible.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Metis: download then build (pipelined)
        metis_download_future = self._download_metis(
            executor=["service"],
            stdout=str(self.log_dir / "metis_download.stdout"),
            stderr=str(self.log_dir / "metis_download.stderr"),
        )
        metis_build_future = self._build_metis(
            executor=["compute"],
            stdout=str(self.log_dir / "metis_build.stdout"),
            stderr=str(self.log_dir / "metis_build.stderr"),
            inputs=[metis_download_future],
        )

        # MPAS-Tools: clone then build hex_projection + grid_rotate
        clone_mpas_tools_future = self._clone_mpas_tools(
            executor=["service"],
            stdout=str(self.log_dir / "mpas_tools_clone.stdout"),
            stderr=str(self.log_dir / "mpas_tools_clone.stderr"),
        )
        hex_future = self._build_hex_projection(
            executor=["compute"],
            stdout=str(self.log_dir / "hex_projection_build.stdout"),
            stderr=str(self.log_dir / "hex_projection_build.stderr"),
            inputs=[clone_mpas_tools_future],
        )
        rotate_future = self._build_grid_rotate(
            executor=["compute"],
            stdout=str(self.log_dir / "grid_rotate_build.stdout"),
            stderr=str(self.log_dir / "grid_rotate_build.stderr"),
            inputs=[clone_mpas_tools_future],
        )

        # MPAS-Limited-Area (just a clone, no build)
        limited_area_future = self._install_limited_area(
            executor=["service"],
            stdout=str(self.log_dir / "limited_area_install.stdout"),
            stderr=str(self.log_dir / "limited_area_install.stderr"),
        )

        # Await Metis
        try:
            await asyncio.wrap_future(metis_download_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"Metis download failed (exit {e.exitcode}), "
                f"see {self.log_dir}/metis_download.stderr"
            )
        self.metis_source_dir = self.work_dir / "metis" / self.metis_version
        self.metis_downloaded = True

        try:
            await asyncio.wrap_future(metis_build_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"Metis build failed (exit {e.exitcode}), "
                f"see {self.log_dir}/metis_build.stderr"
            )
        self.gpmetis_path = self.metis_source_dir / "bin" / "gpmetis"
        self.metis_built = True

        # Await MPAS-Tools
        try:
            await asyncio.wrap_future(clone_mpas_tools_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS-Tools clone failed (exit {e.exitcode}), "
                f"see {self.log_dir}/mpas_tools_clone.stderr"
            )
        self.mpas_tools_dir = self.work_dir / "MPAS-Tools"

        try:
            await asyncio.wrap_future(hex_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"hex_projection build failed (exit {e.exitcode}), "
                f"see {self.log_dir}/hex_projection_build.stderr"
            )
        self.hex_projection_path = (
            self.mpas_tools_dir / "mesh_tools" / "hex_projection" / "project_hexes"
        )

        try:
            await asyncio.wrap_future(rotate_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"grid_rotate build failed (exit {e.exitcode}), "
                f"see {self.log_dir}/grid_rotate_build.stderr"
            )
        self.grid_rotate_path = (
            self.mpas_tools_dir / "mesh_tools" / "grid_rotate" / "grid_rotate"
        )
        self.mpas_tools_installed = True

        # Await MPAS-Limited-Area
        try:
            await asyncio.wrap_future(limited_area_future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS-Limited-Area install failed (exit {e.exitcode}), "
                f"see {self.log_dir}/limited_area_install.stderr"
            )
        self.limited_area_source_dir = self.work_dir / "MPAS-Limited-Area"
        self.create_region_path = self.limited_area_source_dir / "create_region"
        self.limited_area_installed = True

    # -------------------------------------------------------------------------
    # Public agent actions — Mesh operations
    # -------------------------------------------------------------------------

    @agent_action
    async def download_global_mesh(
        self,
        resolution: str,
        mesh_data_dir: str,
    ) -> Dict[str, str]:
        """Download precomputed global mesh files for specified resolution.

        Parameters
        ----------
        resolution : str
            Mesh resolution (e.g., "120km", "60km")
        mesh_data_dir : str
            Directory where downloaded global mesh files are written

        Returns
        -------
        dict
            {"static": path, "graph": path}
        """
        if resolution not in self.resolution_cells:
            raise ValueError(
                f"Resolution '{resolution}' not available for download. "
                f"Available: {list(self.resolution_cells.keys())}"
            )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        request_mesh_data_dir = self._resolve_mesh_data_dir(mesh_data_dir)

        future = self._download_global_mesh(
            resolution,
            str(request_mesh_data_dir),
            executor=["service"],
        )
        try:
            await asyncio.wrap_future(future)
        except Exception as e:
            raise RuntimeError(
                f"Global mesh download failed: {e}"
            )

        cells = self.resolution_cells[resolution]
        return {
            "static": str(request_mesh_data_dir / f"x1.{cells}.static.nc"),
            "graph": str(request_mesh_data_dir / f"x1.{cells}.graph.info"),
        }


    @agent_action
    async def plot_mesh(
        self,
        mesh_file: str,
        output_format: str = "png",
        resolution: Optional[str] = None,
    ) -> str:
        """Generate a plot of the mesh file and return the image path.

        Parameters
        ----------
        mesh_file : str
            Path to the mesh NetCDF file to plot
        output_format : str, optional
            Output image format, by default "png" (only PNG is supported)
        resolution : str, optional
            Resolution label for the plot title (e.g., "120km")

        Returns
        -------
        str
            Path to the generated plot image
        """
        if output_format.lower() != "png":
            raise ValueError("Only PNG output is supported")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        future = self._plot_mpas_mesh_png(
            mesh_file,
            resolution,
            executor=["compute"],
        )
        try:
            return await asyncio.wrap_future(future)
        except Exception as e:
            raise RuntimeError(f"Mesh plotting failed: {e}") from e

    # -------------------------------------------------------------------------
    # Public agent actions — Top-level orchestrator
    # -------------------------------------------------------------------------

    @agent_action
    async def mesh_config_from_prompt(
        self,
        prompt: str,
        model: str = "qwen2.5:3b",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return validated mesh_config JSON inferred from natural language."""
        return await self._mesh_config_from_prompt(prompt, model, api_key, api_base)

    @agent_action
    async def create_mesh_from_prompt(
        self,
        prompt: str,
        mesh_data_dir: str,
        model: str = "ollama_chat/qwen2.5:3b",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        mesh_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a mesh from a natural language request using an LLM."""
        mesh_config = await self._mesh_config_from_prompt(prompt, model, api_key, api_base)
        if mesh_name:
            mesh_config["name"] = mesh_name
        mesh_result = await self.generate_mesh(mesh_config, mesh_data_dir)
        return {
            "mesh_config": mesh_config,
            "mesh_result": mesh_result,
        }

    @agent_action
    async def submit_mesh_prompt(
        self,
        prompt: str,
        mesh_data_dir: str,
        model: str = "ollama_chat/qwen2.5:3b",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> str:
        """Submit a prompt for background processing by the prompt queue loop."""
        request_id = uuid.uuid4().hex
        self._prompt_results[request_id] = {"status": "queued"}
        self._pending_prompt_requests.append({
            "request_id": request_id,
            "prompt": prompt,
            "mesh_data_dir": mesh_data_dir,
            "model": model,
            "api_key": api_key,
            "api_base": api_base,
        })
        return request_id

    @agent_action
    async def get_mesh_prompt_result(self, request_id: str) -> Dict[str, Any]:
        """Get the latest status or result for a queued prompt request."""
        if request_id not in self._prompt_results:
            raise KeyError(f"Unknown request_id: {request_id}")
        return self._prompt_results[request_id]

    @agent_loop
    async def process_mesh_prompt_queue(self, shutdown) -> None:
        """Process queued prompt-driven mesh requests sequentially in the background."""
        while not shutdown.is_set():
            if not self._pending_prompt_requests:
                await asyncio.sleep(0.5)
                continue

            request = self._pending_prompt_requests.pop(0)

            request_id = request["request_id"]
            self._prompt_results[request_id] = {"status": "running"}
            try:
                mesh_config = await self._mesh_config_from_prompt(
                    request["prompt"],
                    request["model"],
                    request.get("api_key"),
                    request.get("api_base"),
                )
                mesh_result = await self.generate_mesh(
                    mesh_config,
                    request["mesh_data_dir"],
                )
                self._prompt_results[request_id] = {
                    "status": "succeeded",
                    "mesh_config": mesh_config,
                    "mesh_result": mesh_result,
                }
            except Exception as e:
                self._prompt_results[request_id] = {
                    "status": "failed",
                    "error": str(e),
                }

    @agent_action
    async def ensure_tools_installed(self, mesh_config: Dict[str, Any]) -> None:
        """Install only the tools required by a mesh request.

        Can be called early to overlap tool installation with other agents.
        Also called automatically by ``generate_mesh()`` as a safety net.
        """
        regional = mesh_config.get("regional")
        install_tasks = []

        needs_partition = bool(
            mesh_config.get("init_ranks") or mesh_config.get("forecast_ranks")
        )
        if needs_partition and not self.metis_built:
            install_tasks.append(self.install_metis())

        if regional is not None:
            if "project_hexes" in regional:
                if not self.mpas_tools_installed:
                    install_tasks.append(self.install_mpas_tools())
            elif "create_region" in regional:
                if not self.limited_area_installed:
                    install_tasks.append(self.install_limited_area())

        if install_tasks:
            await asyncio.gather(*install_tasks)


    @agent_action
    async def generate_mesh(
        self,
        mesh_config: Dict[str, Any],
        mesh_data_dir: str,
    ) -> Dict[str, Any]:
        """Generate a mesh for a request configuration.

        Routes to the correct generation path and partitions the result.
        Required tools are installed automatically on demand.

        Parameters
        ----------
        mesh_config : dict
            Request-scoped mesh configuration
        mesh_data_dir : str
            Directory where this request's mesh files are written

        Returns
        -------
        dict
            - mesh: path to final mesh file (in mesh_name subdirectory)
            - graph: path to graph.info file (in mesh_name subdirectory)
            - plot: path to PNG mesh plot (if successfully generated)
            - plot_error: error message if plotting failed
            - partitions: dict of {num_ranks: partition_file_path}
        """
        resolution = mesh_config.get("resolution", "120km")
        mesh_name = mesh_config.get("name", f"mesh_{resolution}")
        init_ranks = mesh_config.get("init_ranks")
        forecast_ranks = mesh_config.get("forecast_ranks")
        regional = mesh_config.get("regional")
        request_mesh_data_dir = self._resolve_mesh_data_dir(mesh_data_dir)

        await self.ensure_tools_installed(mesh_config)

        result: Dict[str, Any] = {"partitions": {}}

        if regional is None:
            # Global mesh — download precomputed from UCAR
            if resolution not in self.resolution_cells:
                raise ValueError(
                    f"Resolution '{resolution}' not supported. "
                    f"Available: {list(self.resolution_cells.keys())}"
                )
            paths = await self.download_global_mesh(resolution, mesh_data_dir)

            # Symlink into mesh_name subdir for consistent naming
            mesh_dir = request_mesh_data_dir / mesh_name
            mesh_dir.mkdir(parents=True, exist_ok=True)
            static_link = mesh_dir / f"{mesh_name}.static.nc"
            graph_link = mesh_dir / f"{mesh_name}.graph.info"
            if not static_link.exists():
                static_link.symlink_to(Path(paths["static"]).resolve())
            if not graph_link.exists():
                graph_link.symlink_to(Path(paths["graph"]).resolve())
            result["mesh"] = str(static_link)
            result["graph"] = str(graph_link)

        elif "project_hexes" in regional:
            project_hexes_config = regional["project_hexes"]
            hex_result = await self.project_hexes(
                resolution,
                project_hexes_config,
                mesh_name,
                mesh_data_dir,
            )
            result["mesh"] = hex_result["mesh"]
            result["graph"] = hex_result["graph"]

        elif "create_region" in regional:
            create_region_config = regional["create_region"]
            create_region_result = await self.create_region(
                resolution,
                create_region_config,
                mesh_name,
                mesh_data_dir,
            )
            result["mesh"] = create_region_result["static"]
            result["graph"] = create_region_result["graph"]

        else:
            raise ValueError(
                "Unrecognized regional config. Expected 'project_hexes' "
                "or 'create_region' key under 'regional'."
            )


        # Partition for MPI if graph file available and ranks specified
        if result.get("graph"):
            ranks_list = []
            if init_ranks and init_ranks > 1:
                ranks_list.append(init_ranks)
            if forecast_ranks and forecast_ranks > 1:
                ranks_list.append(forecast_ranks)
            if ranks_list:
                partition_paths = await self.partition_mesh(
                    result["graph"], ranks_list,
                )
                result["partitions"].update(partition_paths)

        if result.get("mesh"):
            try:
                result["plot"] = await self.plot_mesh(
                    result["mesh"],
                    "png",
                    resolution,
                )
            except Exception as e:
                result["plot_error"] = str(e)

        return result
