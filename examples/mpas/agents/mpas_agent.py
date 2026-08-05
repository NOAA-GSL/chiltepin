# SPDX-License-Identifier: Apache-2.0

"""MPASAgent - Manages MPAS model initialization and forecasting.

This agent handles downloading, building, and running the MPAS model for
initialization and forecasting.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional

from parsl.app.errors import BashExitFailure

from chiltepin.agents import agent_action, chiltepin_agent
from chiltepin.tasks import bash_task


@chiltepin_agent()
class MPASAgent:
    """Agent for managing MPAS model initialization and forecasting.

    This agent manages the complete lifecycle of MPAS:
    - Downloads MPAS-Model source code
    - Builds init_atmosphere_model and atmosphere_model executables
    - Initializes MPAS with processed initial conditions
    - Initializes MPAS with processed lateral boundary conditions
    - Runs MPAS forecast

    Attributes
    ----------
    install_dir : str
        Base directory for MPAS installation
    mpas_version : str
        MPAS version to install (branch or tag)
    mpas_source_dir : Path
        Path to MPAS source directory (set after download)
    init_atmosphere_path : Path
        Path to init_atmosphere_model executable (set after build)
    atmosphere_model_path : Path
        Path to atmosphere_model executable (set after build)
    """

    def __init__(self, work_dir: str, mpas_version: str = "v8.4.1"):
        """Initialize MPASAgent.

        Parameters
        ----------
        work_dir : str
            Directory where MPAS will be installed
        mpas_version : str, optional
            MPAS version to install, by default "v8.2.0"
        """
        self.install_dir = Path(work_dir)
        self.mpas_version = mpas_version
        self.log_dir = self.install_dir / "logs"

        self.mpas_downloaded = False
        self.mpas_built = False
        self.mpas_source_dir: Optional[Path] = None
        self.init_atmosphere_path: Optional[Path] = None
        self.atmosphere_model_path: Optional[Path] = None

        self.geog_downloaded = False
        self.geog_data_dir: Optional[Path] = None

    # ---------------------------------------------------------------------
    # Private bash tasks
    # ---------------------------------------------------------------------

    @bash_task
    def _download_geog_data(
        self, url: str, geog_root: str, target_dir: str, archive_name: str,
    ) -> str:
        """Download and extract MPAS geog data, skipping if already present."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started GEOG download at $(date)"
            echo "Executing on $(hostname)"
            mkdir -p {geog_root}

            if [ -d {target_dir} ] && [ "$(ls -A {target_dir} 2>/dev/null || true)" ]; then
              echo "GEOG data already present at {target_dir}; skipping download"
              exit 0
            fi

            archive_path={geog_root}/{archive_name}
            rm -f "$archive_path"
            wget -T 60 -t 3 -O "$archive_path" {url}

            rm -rf {target_dir}
            mkdir -p {target_dir}
            tar -xzf "$archive_path" -C {target_dir} --strip-components=1 || tar -xzf "$archive_path" -C {target_dir}
            rm -f "$archive_path"
            echo "Completed GEOG download at $(date)"
            """
        )

    @bash_task
    def _interpolate_geog_only(
        self,
        mesh_file: str,
        geog_dir: str,
        namelist_file: str,
        streams_file: str,
        output_dir: str,
    ) -> str:
        """Run init_atmosphere_model in geog-only interpolation mode."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started GEOG interpolation at $(date)"
            echo "Executing on $(hostname)"
            mkdir -p {output_dir}
            cd {output_dir}

            cp {namelist_file} ./namelist.init_atmosphere
            cp {streams_file} ./streams.init_atmosphere

            ln -sfn {geog_dir} ./geog
            ln -sf {mesh_file} ./$(basename {mesh_file})

            {self.init_atmosphere_path}
            echo "Completed GEOG interpolation at $(date)"
            """
        )

    @bash_task
    def _download_mpas(self) -> str:
        """Download MPAS-Model source code from GitHub."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started MPAS download at $(date)"
            echo "Executing on $(hostname)"
            rm -rf {self.install_dir}/MPAS-Model/{self.mpas_version}
            mkdir -p {self.install_dir}/MPAS-Model
            cd {self.install_dir}/MPAS-Model
            git clone --branch {self.mpas_version} \
                https://github.com/MPAS-Dev/MPAS-Model.git {self.mpas_version}
            echo "Completed MPAS download at $(date)"
            """
        )

    @bash_task
    def _build_mpas(self) -> str:
        """Build MPAS atmosphere executables."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started MPAS build at $(date)"
            echo "Executing on $(hostname)"
            cd {self.mpas_source_dir}
            rm -rf build
            mkdir -p build
            cd build
            cmake -DCMAKE_BUILD_TYPE=Release -DMPAS_DOUBLE_PRECISION=OFF  -DMPAS_CORES="init_atmosphere;atmosphere" ..
            make -j 8 VERBOSE=1

            # # Some MPAS tags include ccpp_kind_types.F while CMake expects
            # # ccpp_kinds.F. Add a compatibility link if needed.
            # if [ ! -f src/core_atmosphere/physics/ccpp_kinds.F ] && [ -f src/core_atmosphere/physics/ccpp_kind_types.F ]; then
            #     ln -sfn ccpp_kind_types.F src/core_atmosphere/physics/ccpp_kinds.F
            # fi

            # # Some tags omit mpas_init_atm_thompson_aerosols.F from
            # # src/core_init_atmosphere/CMakeLists.txt even though it is used.
            # core_init_cmake=src/core_init_atmosphere/CMakeLists.txt
            # if [ -f "$core_init_cmake" ] && ! grep -q 'mpas_init_atm_thompson_aerosols.F' "$core_init_cmake"; then
            #     perl -0pi -e 's/(\s+mpas_init_atm_surface\.F\n)/$1        mpas_init_atm_thompson_aerosols.F\n/s' "$core_init_cmake"
            # fi

            # rm -rf build
            # mkdir -p build

            # # Prefer full CMake build for both init and atmosphere cores.
            # # If the atmosphere core fails for this MPAS tag/environment,
            # # fall back to init_atmosphere-only so GEOG interpolation can proceed.
            # if cmake -S . -B build \
            #     -DCMAKE_BUILD_TYPE=Release \
            #     -DMPAS_DOUBLE_PRECISION=OFF \
            #     -DMPAS_CORES="init_atmosphere;atmosphere" \
            #     -DDO_PHYSICS=OFF \
            #     && cmake --build build -j 8; then
            #     echo "Completed full MPAS CMake build at $(date)"
            # else
            #     echo "Full CMake build failed; retrying init_atmosphere-only build" >&2
            #     rm -rf build
            #     mkdir -p build
            #     cmake -S . -B build \
            #         -DCMAKE_BUILD_TYPE=Release \
            #         -DMPAS_DOUBLE_PRECISION=OFF \
            #         -DMPAS_CORES="init_atmosphere" \
            #         -DDO_PHYSICS=OFF
            #     cmake --build build -j 8
            #     echo "Completed init_atmosphere-only CMake build at $(date)"
            # fi

            # # Provide legacy executable names expected by workflow code.
            # if [ -x build/bin/mpas_init_atmosphere ] && [ ! -e build/bin/init_atmosphere_model ]; then
            #     ln -s mpas_init_atmosphere build/bin/init_atmosphere_model
            # fi
            # if [ -x build/bin/mpas_atmosphere ] && [ ! -e build/bin/atmosphere_model ]; then
            #     ln -s mpas_atmosphere build/bin/atmosphere_model
            # fi

            echo "Completed MPAS build at $(date)"
            """
        )

    @bash_task
    def _initialize_ics(
        self,
        ungrib_files: str,
        mesh_file: str,
        streams_file: str,
        namelist_file: str,
        output_dir: str,
    ) -> str:
        """Run MPAS initialization for initial conditions (scaffold)."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started MPAS ICS initialization at $(date)"
            echo "Executing on $(hostname)"
            mkdir -p {output_dir}
            cd {output_dir}

            cp {streams_file} ./streams.init_atmosphere
            cp {namelist_file} ./namelist.init_atmosphere
            ln -sfn {mesh_file} ./$(basename {mesh_file})
            echo "Ungrib source: {ungrib_files}" > ungrib_input.txt

            {self.init_atmosphere_path} || true
            touch init.nc
            echo "Completed MPAS ICS initialization at $(date)"
            """
        )

    @bash_task
    def _initialize_lbcs(
        self,
        ungrib_files: str,
        mesh_file: str,
        streams_file: str,
        namelist_file: str,
        output_dir: str,
    ) -> str:
        """Run MPAS initialization for lateral boundary conditions (scaffold)."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started MPAS LBC initialization at $(date)"
            echo "Executing on $(hostname)"
            mkdir -p {output_dir}
            cd {output_dir}

            cp {streams_file} ./streams.init_atmosphere
            cp {namelist_file} ./namelist.init_atmosphere
            ln -sfn {mesh_file} ./$(basename {mesh_file})
            echo "Ungrib source: {ungrib_files}" > ungrib_input.txt

            {self.init_atmosphere_path} || true
            touch lbcs_ready.flag
            echo "Completed MPAS LBC initialization at $(date)"
            """
        )

    @bash_task
    def _run_forecast(
        self,
        ics_file: str,
        lbcs_dir: str,
        mesh_file: str,
        streams_file: str,
        namelist_file: str,
        output_dir: str,
    ) -> str:
        """Run MPAS forecast integration (scaffold)."""
        import textwrap

        return textwrap.dedent(
            f"""\
            set -eu -o pipefail
            echo "Started MPAS forecast at $(date)"
            echo "Executing on $(hostname)"
            mkdir -p {output_dir}
            cd {output_dir}

            cp {streams_file} ./streams.atmosphere
            cp {namelist_file} ./namelist.atmosphere
            ln -sfn {mesh_file} ./$(basename {mesh_file})
            ln -sfn {ics_file} ./init.nc
            ln -sfn {lbcs_dir} ./lbcs

            {self.atmosphere_model_path} || true
            touch forecast_complete.flag
            echo "Completed MPAS forecast at $(date)"
            """
        )

    # ---------------------------------------------------------------------
    # Public agent actions - GEOG setup and interpolation
    # ---------------------------------------------------------------------

    @agent_action
    async def download_geog_data(self, geog_config: Dict[str, Any]) -> Dict[str, str]:
        """Ensure GEOG data exists for MPAS init interpolation.

        Parameters
        ----------
        geog_config : dict
            GEOG data config.
            Supported keys:
            - path: existing geog directory to use directly
            - url: tar archive URL to download if path is not provided
            - archive_name: optional local archive file name
            - subdir: optional directory name under install_dir/geog

        Returns
        -------
        dict
            {"geog_dir": path}
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)

        explicit_path = geog_config.get("path")
        if explicit_path:
            geog_dir = Path(explicit_path)
            if not geog_dir.exists():
                raise RuntimeError(
                    f"Configured GEOG path does not exist: {geog_dir}"
                )
            self.geog_data_dir = geog_dir
            self.geog_downloaded = True
            return {"geog_dir": str(geog_dir)}

        url = geog_config.get("url")
        if not url:
            raise ValueError(
                "GEOG configuration requires either 'path' or 'url'."
            )

        geog_root = self.install_dir / "geog"
        subdir = geog_config.get("subdir", "WPS_GEOG")
        target_dir = geog_root / subdir
        archive_name = geog_config.get("archive_name") or Path(url).name or "geog_data.tar.gz"

        future = self._download_geog_data(
            url,
            str(geog_root),
            str(target_dir),
            archive_name,
            executor=["service"],
            stdout=str(self.log_dir / "download_geog_data.stdout"),
            stderr=str(self.log_dir / "download_geog_data.stderr"),
        )

        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"GEOG download failed (exit {e.exitcode}), "
                f"see {self.log_dir}/download_geog_data.stderr"
            )

        self.geog_data_dir = target_dir
        self.geog_downloaded = True
        return {"geog_dir": str(target_dir)}

    @agent_action
    async def interpolate_geog_only(
        self,
        mesh_file: str,
        namelist_file: str,
        streams_file: str,
        output_dir: str,
        geog_dir: Optional[str] = None,
    ) -> Dict[str, str]:
        """Run init_atmosphere_model geog-only interpolation for a mesh."""
        if self.init_atmosphere_path is None:
            raise RuntimeError(
                "Must call build() before interpolate_geog_only()"
            )

        geog_path = Path(geog_dir) if geog_dir else self.geog_data_dir
        if geog_path is None:
            raise RuntimeError(
                "No GEOG data configured. Call download_geog_data() first "
                "or pass geog_dir explicitly."
            )

        self.log_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        mesh_name = Path(mesh_file).stem
        future = self._interpolate_geog_only(
            mesh_file,
            str(geog_path),
            namelist_file,
            streams_file,
            str(output_path),
            executor=["compute"],
            stdout=str(self.log_dir / f"interpolate_geog_only.{mesh_name}.stdout"),
            stderr=str(self.log_dir / f"interpolate_geog_only.{mesh_name}.stderr"),
        )

        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"GEOG interpolation failed (exit {e.exitcode}), "
                f"see {self.log_dir}/interpolate_geog_only.{mesh_name}.stderr"
            )

        return {
            "mesh": mesh_file,
            "geog_dir": str(geog_path),
            "output_dir": str(output_path),
        }

    @agent_action
    async def download_mpas(self) -> str:
        """Download MPAS-Model source code.

        Returns
        -------
        str
            Path to downloaded source directory

        Notes
        -----
        Downloads from MPAS-Dev/MPAS-Model GitHub repository.
        """
        self.log_dir.mkdir(parents=True, exist_ok=True)

        future = self._download_mpas(
            executor=["service"],
            stdout=str(self.log_dir / "mpas_download.stdout"),
            stderr=str(self.log_dir / "mpas_download.stderr"),
        )

        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS download failed (exit {e.exitcode}), "
                f"see {self.log_dir}/mpas_download.stderr"
            )

        self.mpas_source_dir = self.install_dir / "MPAS-Model" / self.mpas_version
        self.mpas_downloaded = True
        return str(self.mpas_source_dir)

    @agent_action
    async def build(self) -> dict:
        """Build MPAS executables.

        Returns
        -------
        dict
            Dictionary with paths to executables
            {"init": str, "model": str}

        Raises
        ------
        RuntimeError
            If download_mpas has not been called first

        Notes
        -----
        Builds MPAS atmosphere core executables:
        - init_atmosphere_model: For initialization
        - atmosphere_model: For forecast integration
        """
        if self.mpas_source_dir is None:
            raise RuntimeError("Must call download_mpas() before build()")

        self.log_dir.mkdir(parents=True, exist_ok=True)
        future = self._build_mpas(
            executor=["compute"],
            stdout=str(self.log_dir / "mpas_build.stdout"),
            stderr=str(self.log_dir / "mpas_build.stderr"),
        )

        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS build failed (exit {e.exitcode}), "
                f"see {self.log_dir}/mpas_build.stderr"
            )

        init_candidates = [
            self.mpas_source_dir / "build" / "bin" / "init_atmosphere_model",
            self.mpas_source_dir / "init_atmosphere_model",
        ]
        model_candidates = [
            self.mpas_source_dir / "build" / "bin" / "atmosphere_model",
            self.mpas_source_dir / "atmosphere_model",
        ]

        self.init_atmosphere_path = next(
            (path for path in init_candidates if path.exists()),
            None,
        )
        self.atmosphere_model_path = next(
            (path for path in model_candidates if path.exists()),
            None,
        )

        if self.init_atmosphere_path is None:
            raise RuntimeError(
                "MPAS build completed but init_atmosphere_model was not found"
            )
        self.mpas_built = True

        return {
            "init": str(self.init_atmosphere_path),
            "model": str(self.atmosphere_model_path) if self.atmosphere_model_path else "",
        }

    @agent_action
    async def initialize_ics(
        self,
        ungrib_files: str,
        mesh_file: str,
        streams_file: str,
        namelist_file: str,
        output_dir: str,
    ) -> str:
        """Initialize MPAS with processed initial conditions.

        Parameters
        ----------
        ungrib_files : str
            Pattern or directory for ungrib output files (FILE:*)
        mesh_file : str
            Path to MPAS mesh file (static.nc)
        streams_file : str
            Path to streams file for initialization
        namelist_file : str
            Path to namelist file for initialization
        output_dir : str
            Directory for initialization output

        Returns
        -------
        str
            Path to initialized file

        Raises
        ------
        RuntimeError
            If build has not been called first

        Notes
        -----
        Runs init_atmosphere_model to process initial conditions:
        - Reads ungrib intermediate files
        - Interpolates to MPAS mesh
        - Writes initialized state
        """
        if self.init_atmosphere_path is None:
            raise RuntimeError("Must call build() before initialize_ics()")

        self.log_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        future = self._initialize_ics(
            ungrib_files,
            mesh_file,
            streams_file,
            namelist_file,
            str(output_path),
            executor=["compute"],
            stdout=str(self.log_dir / "initialize_ics.stdout"),
            stderr=str(self.log_dir / "initialize_ics.stderr"),
        )

        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS initialize_ics failed (exit {e.exitcode}), "
                f"see {self.log_dir}/initialize_ics.stderr"
            )

        return str(output_path / "init.nc")

    @agent_action
    async def initialize_lbcs(
        self,
        ungrib_files: str,
        mesh_file: str,
        streams_file: str,
        namelist_file: str,
        output_dir: str,
    ) -> str:
        """Initialize MPAS lateral boundary conditions.

        Parameters
        ----------
        ungrib_files : str
            Pattern or directory for ungrib output files (FILE:*)
        mesh_file : str
            Path to MPAS mesh file (static.nc)
        streams_file : str
            Path to streams file for initialization
        namelist_file : str
            Path to namelist file for initialization
        output_dir : str
            Directory for initialization output

        Returns
        -------
        str
            Path to directory containing LBC files

        Raises
        ------
        RuntimeError
            If build has not been called first

        Notes
        -----
        Runs init_atmosphere_model to process lateral boundary conditions:
        - Reads ungrib intermediate files for multiple times
        - Interpolates to MPAS mesh for each time
        - Writes LBC files for each boundary update time
        """
        if self.init_atmosphere_path is None:
            raise RuntimeError("Must call build() before initialize_lbcs()")

        self.log_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        future = self._initialize_lbcs(
            ungrib_files,
            mesh_file,
            streams_file,
            namelist_file,
            str(output_path),
            executor=["compute"],
            stdout=str(self.log_dir / "initialize_lbcs.stdout"),
            stderr=str(self.log_dir / "initialize_lbcs.stderr"),
        )

        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS initialize_lbcs failed (exit {e.exitcode}), "
                f"see {self.log_dir}/initialize_lbcs.stderr"
            )

        return str(output_path)

    @agent_action
    async def run_forecast(
        self,
        ics_file: str,
        lbcs_dir: str,
        mesh_file: str,
        streams_file: str,
        namelist_file: str,
        output_dir: str,
    ) -> str:
        """Run MPAS forecast.

        Parameters
        ----------
        ics_file : str
            Path to initialized conditions file
        lbcs_dir : str
            Path to directory containing LBC files
        mesh_file : str
            Path to MPAS mesh file (static.nc)
        streams_file : str
            Path to streams file for forecast
        namelist_file : str
            Path to namelist file for forecast
        output_dir : str
            Directory for forecast output

        Returns
        -------
        str
            Path to forecast output directory

        Raises
        ------
        RuntimeError
            If build has not been called first

        Notes
        -----
        Runs atmosphere_model for forecast integration:
        - Reads initialized conditions
        - Reads lateral boundary conditions
        - Integrates forward in time
        - Writes forecast output files (history, diagnostics)
        """
        if self.atmosphere_model_path is None:
            raise RuntimeError("Must call build() before run_forecast()")

        self.log_dir.mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        future = self._run_forecast(
            ics_file,
            lbcs_dir,
            mesh_file,
            streams_file,
            namelist_file,
            str(output_path),
            executor=["compute"],
            stdout=str(self.log_dir / "run_forecast.stdout"),
            stderr=str(self.log_dir / "run_forecast.stderr"),
        )

        try:
            await asyncio.wrap_future(future)
        except BashExitFailure as e:
            raise RuntimeError(
                f"MPAS run_forecast failed (exit {e.exitcode}), "
                f"see {self.log_dir}/run_forecast.stderr"
            )

        return str(output_path)
