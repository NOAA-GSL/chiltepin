# SPDX-License-Identifier: Apache-2.0
"""Test the generator and, when available, Compose's resolved configuration."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

DOCKER_DIR = Path(__file__).resolve().parent
SCRIPT = DOCKER_DIR / "generate-compose.py"


class GeneratorTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.output = self.directory / "cluster.yml"

    def generate(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=self.directory,
            capture_output=True,
            text=True,
        )


class GenerateComposeTests(GeneratorTestCase):
    def test_default_matches_checked_in_config(self):
        result = self.generate()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            (self.directory / "docker-compose.yml").read_text(),
            (DOCKER_DIR / "docker-compose.yml").read_text(),
        )
        config = yaml.safe_load((self.directory / "docker-compose.yml").read_text())
        self.assertEqual(
            {name for name in config["services"] if name.startswith("slurmnode")},
            {f"slurmnode{i}" for i in range(1, 6)},
        )

    def test_all_supported_node_counts(self):
        for count in range(1, 11):
            with self.subTest(nodes=count):
                result = self.generate(
                    "--nodes", str(count), "--output", str(self.output)
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                config = yaml.safe_load(self.output.read_text())
                services = config["services"]
                self.assertEqual(
                    set(services),
                    {"slurmfrontend", "slurmmaster"}
                    | {f"slurmnode{i}" for i in range(1, count + 1)},
                )
                self.assertEqual(set(config["volumes"]), {"home-vol", "opt-vol"})
                for name, service in services.items():
                    role = name.removeprefix("slurm")
                    self.assertEqual(
                        service["container_name"],
                        f"${{CLUSTER_NAME:-chiltepin}}-{role}",
                    )
                    self.assertEqual(service["hostname"], name)
                    self.assertEqual(service["user"], "admin")
                    self.assertEqual(service["shm_size"], "4g")
                    self.assertNotIn("ports", service)
                    self.assertNotIn("expose", service)
                    self.assertNotIn("build", service)
                    opt = "opt-vol:/opt" + ("" if role == "frontend" else ":ro")
                    self.assertEqual(
                        service["volumes"],
                        ["home-vol:/home/admin", opt, "..:/home/admin/chiltepin"],
                    )
                    image_role = "node" if role.startswith("node") else role
                    self.assertEqual(
                        service["image"],
                        "ghcr.io/noaa-gsl/dockerspackstackslurmcluster/"
                        f"slurm-spack-stack-{image_role}:"
                        "ubuntu-${UBUNTU_VERSION:-26.04}"
                        "-spack-stack-${SPACK_STACK_VERSION:-2.1.0}",
                    )
                    if role.startswith("node"):
                        self.assertEqual(
                            service["environment"],
                            [f"SLURM_NODENAME={name}", "SLURM_CPUS_ON_NODE=8"],
                        )
                        self.assertEqual(
                            service["depends_on"],
                            {
                                "slurmmaster": {"condition": "service_started"},
                                "slurmfrontend": {"condition": "service_healthy"},
                            },
                        )
                        self.assertEqual(
                            service["healthcheck"],
                            config["x-node-common"]["healthcheck"],
                        )
                self.assertEqual(
                    services["slurmmaster"]["depends_on"],
                    {"slurmfrontend": {"condition": "service_healthy"}},
                )
                self.assertEqual(
                    services["slurmmaster"]["environment"], ["SLURM_CPUS_ON_NODE=8"]
                )
                self.assertEqual(
                    services["slurmfrontend"]["healthcheck"]["test"],
                    ["CMD", "test", "-f", "/tmp/ssh-ready"],
                )

    def test_invalid_counts_do_not_create_or_overwrite_output(self):
        for count in ("0", "-1", "11", "100", "1.5", "invalid"):
            for existing in (False, True):
                with self.subTest(nodes=count, existing=existing):
                    self.output.unlink(missing_ok=True)
                    if existing:
                        self.output.write_text("keep this configuration")
                    result = self.generate(
                        "--nodes", count, "--output", str(self.output)
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("error:", result.stderr)
                    self.assertNotIn("Wrote", result.stdout)
                    if existing:
                        self.assertEqual(
                            self.output.read_text(), "keep this configuration"
                        )
                    else:
                        self.assertFalse(self.output.exists())

    def test_output_errors_are_reported(self):
        result = self.generate("--output", str(self.directory / "missing" / "file.yml"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FileNotFoundError", result.stderr)
        self.assertNotIn("Wrote", result.stdout)


class ComposeConfigTests(GeneratorTestCase):
    @classmethod
    def setUpClass(cls):
        docker = shutil.which("docker")
        standalone = shutil.which("docker-compose")
        if docker:
            command = [docker, "compose"]
            result = subprocess.run(command + ["version"], capture_output=True)
            if result.returncode == 0:
                cls.compose = command
                return
        if standalone:
            cls.compose = [standalone]
            return
        raise unittest.SkipTest("Docker Compose is not available")

    def resolve(self, file, cluster=None, project=None, **overrides):
        env = os.environ.copy()
        for key in (
            "CLUSTER_NAME",
            "COMPOSE_PROJECT_NAME",
            "UBUNTU_VERSION",
            "SPACK_STACK_VERSION",
        ):
            env.pop(key, None)
        if cluster is not None:
            env["CLUSTER_NAME"] = cluster
        env.update(overrides)
        command = self.compose + [
            "--env-file",
            str(DOCKER_DIR / ".env"),
            "-f",
            str(file),
        ]
        if project is not None:
            command += ["-p", project]
        result = subprocess.run(
            command + ["config", "--format", "json"],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_compose_accepts_all_node_counts(self):
        for count in range(1, 11):
            with self.subTest(nodes=count):
                result = self.generate(
                    "--nodes", str(count), "--output", str(self.output)
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                config = self.resolve(self.output)
                self.assertEqual(len(config["services"]), count + 2)

    def test_compose_resolves_repository_bind_mounts(self):
        config = self.resolve(DOCKER_DIR / "docker-compose.yml")
        for service in config["services"].values():
            repository_mount = [
                mount
                for mount in service["volumes"]
                if mount["target"] == "/home/admin/chiltepin"
            ]
            self.assertEqual(len(repository_mount), 1)
            self.assertEqual(repository_mount[0]["type"], "bind")
            self.assertEqual(repository_mount[0]["source"], str(DOCKER_DIR.parent))
            self.assertFalse(repository_mount[0].get("read_only", False))

    def test_default_names_and_image_overrides(self):
        config = self.resolve(DOCKER_DIR / "docker-compose.yml")
        self.assertEqual(config["name"], "chiltepin")
        self.assertEqual(
            config["services"]["slurmfrontend"]["container_name"], "chiltepin-frontend"
        )
        overridden = self.resolve(
            DOCKER_DIR / "docker-compose.yml",
            UBUNTU_VERSION="24.04",
            SPACK_STACK_VERSION="custom",
        )
        for service in overridden["services"].values():
            self.assertTrue(
                service["image"].endswith("ubuntu-24.04-spack-stack-custom")
            )

    def test_two_clusters_isolate_runtime_resources_and_share_checkout(self):
        configs = []
        for cluster, count in (("cluster-a", 3), ("cluster-b", 5)):
            file = self.directory / f"{cluster}.yml"
            result = self.generate("--nodes", str(count), "--output", str(file))
            self.assertEqual(result.returncode, 0, result.stderr)
            config = self.resolve(file, cluster=cluster, project=cluster)
            self.assertEqual(config["name"], cluster)
            self.assertEqual(len(config["services"]), count + 2)
            for service in config["services"].values():
                self.assertTrue(service["container_name"].startswith(cluster + "-"))
                self.assertFalse(service.get("ports"))
                for volume in service["volumes"]:
                    if volume["target"] == "/home/admin/chiltepin":
                        self.assertEqual(volume["type"], "bind")
                        self.assertEqual(volume["source"], str(self.directory.parent))
                    else:
                        self.assertEqual(volume["type"], "volume")
                self.assertEqual(set(service["networks"]), {"default"})
            configs.append(config)
        for section in ("volumes", "networks"):
            names = [
                {resource["name"] for resource in config[section].values()}
                for config in configs
            ]
            self.assertTrue(names[0].isdisjoint(names[1]))
        containers = [
            {service["container_name"] for service in config["services"].values()}
            for config in configs
        ]
        self.assertTrue(containers[0].isdisjoint(containers[1]))


if __name__ == "__main__":
    unittest.main()
