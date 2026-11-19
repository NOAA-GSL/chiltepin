# SPDX-License-Identifier: Apache-2.0

import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--config", action="store")


@pytest.fixture(scope="session")
def config_file(request):
    config_value = request.config.option.config
    if config_value is None:
        pytest.skip()
    return config_value


@pytest.fixture(scope="module")
def mesh_test_directory(config_file, request):
    """Provide a directory for a real MPAS mesh agent test module.

    It is under ``tests/test_output``, which the platform's compute nodes can
    also see, and is named after the test module. It is removed afterwards
    unless a test in the module failed, so the logs remain available for
    debugging.
    """
    failed_before = request.session.testsfailed
    output_dir = Path(__file__).parent.resolve() / "test_output"
    output_dir.mkdir(exist_ok=True)
    module = request.module.__name__.rsplit(".", 1)[-1]
    prefix = module.removeprefix("test_").replace("_", "-") + "-"
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=output_dir))
    yield path
    if request.session.testsfailed == failed_before:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="module")
def launch_mesh_agent(config_file, mesh_test_directory):
    """Return a function that launches a MeshAgent on the platform's resources.

    The agent runs locally and sends its tasks to the platform's ``service`` and
    ``compute`` resources. Each launch has its own run directories, named after
    its ``tag``, and all launches in a module share ``work`` in
    ``mesh_test_directory``, so a later one finds what an earlier one installed.
    Extra keyword arguments are passed to the agent, such as tool versions.
    """
    import chiltepin.configure
    from chiltepin import AgentRuntime, Workflow
    from chiltepin.agents.mpas import MeshAgent

    directory = mesh_test_directory

    @asynccontextmanager
    async def launch(tag, **agent_kwargs):
        platform = chiltepin.configure.parse_file(config_file)
        src = Path(__file__).parent.parent.resolve() / "src"
        agent_config = {}
        for name in ("service", "compute"):
            resource = dict(platform[name])
            resource["environment"] = list(resource.get("environment", [])) + [
                f"export PYTHONPATH={src}${{PYTHONPATH:+:$PYTHONPATH}}"
            ]
            agent_config[name] = resource
        controller_config = {"agent-executor": {"provider": "localhost"}}
        workflow = Workflow(
            controller_config, run_dir=str(directory / f"runinfo-{tag}")
        )
        with workflow:
            runtime = AgentRuntime(workflow)
            async with await runtime.manager() as manager:
                yield await manager.launch(
                    MeshAgent,
                    executor="agent-executor",
                    kwargs={"work_dir": str(directory / "work"), **agent_kwargs},
                    agent_workflow_config=agent_config,
                    agent_workflow_run_dir=str(directory / f"agent-runinfo-{tag}"),
                )

    return launch
