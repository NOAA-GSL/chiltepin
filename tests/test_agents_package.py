# SPDX-License-Identifier: Apache-2.0

"""Import and serialization compatibility for the agents package."""

import asyncio
import importlib
import pickle

import pytest

import chiltepin
import chiltepin.agents as agents


@pytest.mark.parametrize("name", ["agent_action", "agent_loop", "chiltepin_agent"])
def test_public_decorator_exports(name):
    decorator = getattr(agents, name)

    assert name in agents.__all__
    assert callable(decorator)
    assert getattr(chiltepin, name) is decorator
    assert pickle.loads(pickle.dumps(decorator)) is decorator


@pytest.mark.parametrize("name", ["agent_action", "agent_loop", "chiltepin_agent"])
def test_legacy_decorator_pickle(name):
    # Previously serialized references resolve through the public package.
    reference = f"cchiltepin.agents\n{name}\n.".encode("ascii")

    assert pickle.loads(reference) is getattr(agents, name)


def test_agents_is_package():
    assert agents.__spec__.submodule_search_locations is not None
    assert importlib.import_module("chiltepin.agents") is agents


@agents.chiltepin_agent()
class SerializableAgent:
    @agents.agent_action
    def get_value(self):
        return 42

    @agents.agent_action
    async def get_async_value(self):
        return 43

    @agents.agent_loop
    async def wait_for_shutdown(self, shutdown):
        await shutdown.wait()
        self.stopped = True


def test_decorated_agent_pickle():
    assert SerializableAgent.__module__ == __name__
    assert SerializableAgent.__qualname__ == "SerializableAgent"
    assert pickle.loads(pickle.dumps(SerializableAgent)) is SerializableAgent


async def test_reexported_decorators_preserve_behavior():
    agent = SerializableAgent()

    assert await agent.get_value() == 42
    assert await agent.get_async_value() == 43

    shutdown = asyncio.Event()
    shutdown.set()
    await agent.wait_for_shutdown(shutdown)
    assert agent._behavior.stopped is True
