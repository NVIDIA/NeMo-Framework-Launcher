# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import functools
import inspect
import os
import random
import re
import shlex
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

import nemo_launcher.utils.job_utils as job_utils
import yaml
from hera.workflows import Workflow
from nemo_launcher.core.logger import logger
from omegaconf import DictConfig, OmegaConf

NEMO_LAUNCHER_CI = os.getenv("NEMO_LAUNCHER_CI", "False").lower() in ("true", "t", "1")
NEMO_LAUNCHER_DEBUG = os.getenv("NEMO_LAUNCHER_DEBUG", "False").lower() in (
    "true",
    "t",
    "1",
)
NEMO_LAUNCHER_MEMORY_MEASURE = os.getenv(
    "NEMO_LAUNCHER_MEMORY_MEASURE", "False"
).lower() in ("true", "t", "1")


class AutoLauncher:
    """
    Automatic launcher class. It will create a launcher based on input cluster name.
    """

    def __init__(
        self,
        folder: Union[str, Path],
        job_name: str,
        cluster: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.cluster = cluster or self.which()
        self.cluster = self.cluster.lower()

        launchers = self.get_launchers()
        if self.cluster not in launchers:
            raise ValueError(
                f"AutoLauncher doesn't know any cluster named {self.cluster}"
            )

        self._launcher = launchers[self.cluster](folder, job_name, **kwargs)

    def launch(self, command_groups: List[List[str]]) -> str:
        """
        Use the launcher to launch the command groups.

        :param List[List[str]] command_groups: Command groups to launch with
        :return: job id on slurm based system otherwise empty string
        :rtype: str
        """
        job_id = self._launcher.launch(command_groups)
        return job_id

    @staticmethod
    def which() -> str:
        """Returns what the detected cluster is"""
        raise NotImplementedError

    @staticmethod
    def get_launchers():
        """Returns supported launchers as a dictionary from launcher name to launcher class"""
        return {
            "bcm": SlurmLauncher,
            "bcp": BCPLauncher,
            "interactive": InteractiveLauncher,
            "k8s": K8SLauncher,
        }


class Launcher:
    """Base launcher class"""

    def __init__(self, folder: Union[Path, str], job_name: str):
        self.folder = folder
        self.job_name = job_name

    def launch(self, command_groups: List[List[str]]) -> str:
        """
        Use the launcher to launch the command groups.

        :param List[List[str]] command_groups: Command groups to launch with
        :return: job id on slurm based system otherwise empty string
        :rtype: str
        """
        submission_file_path = self._make_submission_file(command_groups)
        logger.info(
            f"Job {self.job_name} submission file created at '{submission_file_path}'"
        )

        job_id = ""
        if not NEMO_LAUNCHER_DEBUG:
            job_id = self._submit_command(submission_file_path)
            if job_id:
                logger.info(f"Job {self.job_name} submitted with Job ID {job_id}")
                with open(self.folder / "launcher.log", "w") as f:
                    f.write(f"Submitted batch job {job_id}")
        else:
            job_id = str(random.randint(10000, 99999))
            logger.info(
                f"[DEBUG] Job {self.job_name} submitted with FAKE Job ID {job_id}"
            )

        return job_id

    def _submit_command(self, submission_file_path: Path) -> str:
        """Submits a set of command groups to the cluster"""
        raise NotImplementedError

    def _make_submission_file(self, command_groups: List[List[str]]) -> Path:
        """
        Make a submission script file, as following
            on interactive cluster, it's a bash file, trigger with bash.
            on slurm cluster, it's a slurm script file, trigger with sbatch.
            on BCP cluster, it's a BCP script file, trigger with bash.
            on k8s cluster, it's a Helm chart, triggered with helm.

        :param List[List[str]] command_groups: Command groups to launch with
        :return: job id on slurm based system otherwise empty string
        :rtype: str
        """
        job_paths = job_utils.JobPaths(folder=self.folder, job_name=self.job_name)
        folder = job_paths.folder
        folder.mkdir(parents=True, exist_ok=True)

        submission_file_path = job_paths.submission_file
        with submission_file_path.open("w") as f:
            f.write(self._make_submission_file_text(command_groups))
        return submission_file_path


class InteractiveLauncher(Launcher):
    """
    Interactive job launcher
    This class is used to hold the parameters to run a job on an interactive node (single node only).
    In practice, it will create a batch file in the specified directory for the job and
    trigger the job with `bash` command.

    :param Union[Path, str] folder: folder for storing job submission/output and logs.
    :param str job_name: Name of the job, used as job folder name
    :param Any **kwargs: Parse other cluster parameters required for interactive running
    """

    def __init__(self, folder: Union[Path, str], job_name: str, **kwargs: Any) -> None:
        super().__init__(folder, job_name)
        self.parameters = kwargs

    def _submit_command(self, submission_file_path: Path) -> str:
        """Launch the submission command"""
        command_list = self._make_submission_command(submission_file_path)
        # run
        job_utils.CommandFunction(
            command_list, ret_stdout=False, verbose=False
        )()  # explicit errors
        return ""

    @staticmethod
    def _make_submission_command(submission_file_path: Path) -> List[str]:
        """Make a command to trigger submission script. On interactive cluster, the script is triggerred with bash"""
        return ["bash", str(submission_file_path)]

    def _make_submission_file_text(self, command_groups: List[List[str]]) -> str:
        """
        Given the command groups, generate submission script file's text.
        Command groups is a list of command group. A command group is defined as:
              0. Command group is a list of command strings
              1. Each command group occupies one bcprun, srun or bash
              2. Each command group eventually has multiple commands connected by ";"
        On interactive cluster, multi-gpu python scripts are launched with `torchrun --nproc_per_node=??`

        :param List[List[str]] command_groups: Command groups to launch with
        :return: submission script file's text
        :rtype: str
        """
        nodes = self.parameters.get("nodes", 1)
        ntasks_per_node = self.parameters.get("ntasks_per_node", 1)
        assert nodes == 1, "Multi-node is not supported in interactive mode."

        paths = job_utils.JobPaths(folder=self.folder, job_name=self.job_name)
        time_tag = datetime.datetime.now().strftime("%m%d_%H%M%S")
        stdout = str(paths.stdout).replace("_%j", f"_{time_tag}")

        # now create
        lines = ["#!/bin/bash", ""]

        # environment setup:
        setup = self.parameters.get("setup", None)
        if setup is not None:
            lines += ["", "# setup"] + setup

        for group_ind, command_group in enumerate(command_groups):
            command = ";\n  ".join(command_group)
            command = command.replace(
                "python3 -u", f"torchrun --nproc_per_node={ntasks_per_node}"
            )

            lines += [
                "",
                f"# command {group_ind + 1}",
                f'bash -c "',
                f'  {command} " 2>&1 | tee -a {stdout}',
                "",
            ]
        return "\n".join(lines)


class BCPLauncher(Launcher):
    """
    BCP job launcher
    This class is used to hold the parameters to run a job on BCP platform.
    In practice, it will create a batch file in the specified directory for the job
    and trigger the job with `bash` command.

    :param Union[Path, str] folder: folder for storing job submission/output and logs.
    :param str job_name: Name of the job, used as job folder name
    :param Any **kwargs: Parse other cluster parameters required for BCP running,
        including `nodes`, `ntasks_pernode`, `bcp_launcher`, etc.
    """

    def __init__(self, folder: Union[Path, str], job_name: str, **kwargs: Any) -> None:
        super().__init__(folder, job_name)
        self.parameters = kwargs
        self.parameters = self._convert_parameters(self.parameters)

    @classmethod
    def _equivalence_dict(cls):
        return {
            "name": "job_name",
            "nodes": "nnodes",
            "tasks_per_node": "npernode",
            "ntasks_per_node": "npernode",
            "bcp_launcher": "launcher",
        }

    def _convert_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """translate bcp parameter names"""
        # replace type in some cases
        eq_dict = self._equivalence_dict()
        if eq_dict is not None:
            params = {eq_dict.get(k, k): v for k, v in params.items()}
        return params

    def _submit_command(self, submission_file_path: Path) -> str:
        """Launch the submission command"""
        command_list = self._make_submission_command(submission_file_path)
        # run
        job_utils.CommandFunction(
            command_list, ret_stdout=False, verbose=False
        )()  # explicit errors

        return ""

    @staticmethod
    def _make_submission_command(submission_file_path: Path) -> List[str]:
        """Make a command to trigger submission script. On BCP cluster, the script is triggerred with bash"""
        return ["bash", str(submission_file_path)]

    def _make_submission_file_text(self, command_groups: List[List[str]]) -> str:
        """
        Given the command groups, generate submission script file's text.
        Command groups is a list of command group. A command group is defined as:
              0. Command group is a list of command strings
              1. Each command group occupies one bcprun, srun or bash
              2. Each command group eventually has multiple commands connected by ";"
        On BCP cluster, multi-gpu python scripts are launched with `bcprun --nnodes ? --npernode ?`

        :param List[List[str]] command_groups: Command groups to launch with
        :return: submission script file's text
        :rtype: str
        """
        paths = job_utils.JobPaths(folder=self.folder, job_name=self.job_name)
        time_tag = datetime.datetime.now().strftime("%m%d_%H%M%S")
        stdout = str(paths.stdout).replace("_%j", f"_{time_tag}")

        nnodes = self.parameters.get("nnodes", 1)
        npernode = self.parameters.get("npernode", 1)
        launcher = self.parameters.get("launcher")
        launcher_flags = ""
        if launcher is not None:
            launcher_flags = f"--launcher {launcher}"
        env_vars = self.parameters.get("env_vars")
        env_flags = ""
        if env_vars is not None:
            env_flags = [f"--env '{k}={v}'" for k, v in env_vars.items()]
            env_flags = " ".join(env_flags)

        # now create
        lines = ["#!/bin/bash", ""]

        # environment setup:
        setup = self.parameters.get("setup", None)
        if setup is not None:
            lines += ["", "# setup"] + setup

        # Add pause_and_prime_dns_connection to command groups on BCP
        launcher_scripts_path = Path(
            "/opt/NeMo-Framework-Launcher/launcher_scripts"
        )  # Hard code path on BCP
        pause_and_prime_dns_connection_command = f"python3 -u {launcher_scripts_path / 'nemo_launcher/collections/pause_and_prime_dns_connections.py'}"
        _nemo_code_path = "/opt/NeMo"
        for ind in range(len(command_groups)):
            # TODO: Find a better way to insert pause_and_prime_dns_connection_command
            if _nemo_code_path in command_groups[ind]:
                command_groups[ind] = [
                    pause_and_prime_dns_connection_command
                ] + command_groups[ind]

        for group_ind, command_group in enumerate(command_groups):
            command = ";\n  ".join(command_group)
            redirect = ""

            if self.parameters["no_redirect"]:
                redirect = "-no_redirect"

            if group_ind + 1 == len(command_groups):
                bcprun_cmd = (
                    f"bcprun --nnodes {nnodes} --npernode {npernode} {redirect}"
                )
            else:
                bcprun_cmd = f"bcprun --nnodes 1 --npernode 1 {redirect}"
            lines += [
                "",
                f"# command {group_ind + 1}",
                f"{bcprun_cmd} " f'{launcher_flags} {env_flags} --cmd "',
                f'  {command} " 2>&1 | tee -a {stdout}',
                "",
            ]
        return "\n".join(lines)


class SlurmLauncher(Launcher):
    """
    Slurm job launcher
    This class is used to hold the parameters to run a job on slurm.
    In practice, it will create a batch file in the specified directory for the job,
    trigger the job with `sbatch` command and return a job id.

    :param Union[Path, str] folder: folder for storing job submission/output and logs.
    :param str job_name: Name of the job, used as job folder name
    :param bool use_fault_tolerance: Use fault tolerance launcher to run the job
    :param Any **kwargs: See slurm documentation for most parameters.
            Most useful parameters are: time, mem, gpus_per_node, cpus_per_task, partition
            Below are the parameters that differ from slurm documentation:
                setup: a list of command to run in sbatch before running srun
    """

    def __init__(
        self,
        folder: Union[Path, str],
        job_name: str,
        use_fault_tolerance: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(folder, job_name)
        self.parameters = {}
        self.use_fault_tolerance = use_fault_tolerance
        self._update_parameters(job_name=job_name, **kwargs)

        if shutil.which("srun") is None and not NEMO_LAUNCHER_DEBUG:
            raise RuntimeError(
                'Could not detect "srun", are you indeed on a slurm cluster?'
            )

    @classmethod
    def _equivalence_dict(cls):
        return {
            "name": "job_name",
            "timeout_min": "time",
            "mem_gb": "mem",
            "nodes": "nodes",
            "cpus_per_task": "cpus_per_task",
            "gpus_per_node": "gpus_per_node",
            "tasks_per_node": "ntasks_per_node",
        }

    @classmethod
    def _valid_parameters(cls, use_fault_tolerance) -> Set[str]:
        """Parameters that can be set through update_parameters"""
        if use_fault_tolerance:
            return set(_get_default_parameters(_make_sbatch_string_ft_launcher))
        else:
            return set(_get_default_parameters(_make_sbatch_string))

    def _convert_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """translate slurm parameter names"""
        # replace type in some cases
        eq_dict = self._equivalence_dict()
        if eq_dict is not None:
            params = {eq_dict.get(k, k): v for k, v in params.items()}
        if "mem" in params:
            params["mem"] = _convert_mem(params["mem"])
        return params

    def _update_parameters(self, **kwargs: Any) -> None:
        """
        Updates sbatch submission file parameters
        Raises ValueError:
            In case an erroneous keyword argument is added, a list of all eligible parameters
            is printed, with their default values

        :param Any **kwargs: See slurm documentation for most parameters.
            Most useful parameters are: time, mem, gpus_per_node, cpus_per_task, partition
            Below are the parameters that differ from slurm documentation:
                setup: a list of command to run in sbatch before running srun
        """

        if self.use_fault_tolerance:
            defaults = _get_default_parameters_ft_launcher()
        else:
            defaults = _get_default_parameters()
        in_valid_parameters = sorted(set(kwargs) - set(defaults))
        if in_valid_parameters:
            string = "\n  - ".join(
                f"{x} (default: {repr(y)})" for x, y in sorted(defaults.items())
            )
            logger.warning(
                f"Unrecognized sbatch parameter(s): {in_valid_parameters}. Use at your own risk.\n\nValid parameters are:\n  - {string}"
            )

        self.parameters.update(
            {k: v for k, v in kwargs.items() if k not in in_valid_parameters}
        )
        self.parameters.update(
            {"additional_parameters": {k: kwargs[k] for k in in_valid_parameters}},
        )
        self.parameters = self._convert_parameters(self.parameters)

    def _submit_command(self, submission_file_path: Path) -> str:
        """Launch the submission command"""
        command_list = self._make_submission_command(submission_file_path)
        # run
        output = job_utils.CommandFunction(
            command_list, verbose=False
        )()  # explicit errors

        job_id = ""
        if output:
            job_id = self._get_job_id_from_submission_command(output)
        return job_id

    def _make_submission_file_text(self, command_groups: List[List[str]]) -> str:
        """
        Given the command groups, generate submission script file's text.
        Command groups is a list of command group. A command group is defined as:
              0. Command group is a list of command strings
              1. Each command group occupies one bcprun, srun or bash
              2. Each command group eventually has multiple commands connected by ";"

        :param List[List[str]] command_groups: Command groups to launch with
        :return: submission script file's text
        :rtype: str
        """
        if self.use_fault_tolerance:
            return _make_sbatch_string_ft_launcher(
                command_groups=command_groups, folder=self.folder, **self.parameters
            )
        else:
            return _make_sbatch_string(
                command_groups=command_groups, folder=self.folder, **self.parameters
            )

    @staticmethod
    def _make_submission_command(submission_file_path: Path) -> List[str]:
        """Make a command to trigger submission script. On slurm cluster, the script is triggerred with sbatch"""
        return ["sbatch", str(submission_file_path)]

    @staticmethod
    def _get_job_id_from_submission_command(string: Union[bytes, str]) -> str:
        """Returns the job ID from the output of sbatch string"""
        if not isinstance(string, str):
            string = string.decode()
        output = re.search(r"job (?P<id>[0-9]+)", string)
        if output is None:
            raise utils.FailedSubmissionError(
                f'Could not make sense of sbatch output "{string}"\n'
                "Job instance will not be able to fetch status\n"
                "(you may however set the job job_id manually if needed)"
            )
        return output.group("id")


class K8SLauncher(Launcher):
    """
    K8s job launcher
    This class is used to hold the parameters to run a job on kubernetes.
    In practice, it will create a Helm chart in the specified directory for the job
    and trigger the job with `bash` command.

    :param Union[Path, str] folder: folder for storing job submission/output and logs.
    :param str job_name: Name of the job, used as job folder name
    :param Any **kwargs: Parse other cluster parameters required for k8s running,
        including `nodes`, `ntasks_pernode`, `bcp_launcher`, etc.
    """

    def __init__(self, folder: Union[Path, str], job_name: str, **kwargs: Any) -> None:
        super().__init__(folder, job_name)
        self.parameters = kwargs
        self.parameters = self._convert_parameters(self.parameters)

    @classmethod
    def _equivalence_dict(cls):
        return {
            "name": "job_name",
            "nodes": "nnodes",
            "tasks_per_node": "npernode",
            "ntasks_per_node": "npernode",
        }

    def _convert_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """translate k8s parameter names"""
        # replace type in some cases
        eq_dict = self._equivalence_dict()
        if eq_dict is not None:
            params = {eq_dict.get(k, k): v for k, v in params.items()}
        return params

    def _submit_command(self, submission_file_path: Path) -> str:
        """Launch the submission command"""
        command_list = self._make_submission_command(submission_file_path)
        # run
        job_utils.CommandFunction(
            command_list, ret_stdout=False, verbose=False
        )()  # explicit errors
        return ""

    @staticmethod
    def _make_submission_command(submission_file_path: Path) -> List[str]:
        """Make a command to trigger submission script. On a k8s cluster, the script is triggerred with Helm"""
        return ["bash", str(submission_file_path)]

    def _make_submission_file_text(self, command_groups: List[List[str]]) -> str:
        """
        Generate the script to launch the Helm chart.
        A very simple bash script is generated which runs `helm install` for the
        Helm chart that was generated.

        :param List[List[str]] command_groups: Command groups to launch with
        :return: submission script file's text
        :rtype: str
        """
        paths = job_utils.JobPaths(folder=self.folder, job_name=self.job_name)
        helm_charts = paths.folder / "k8s_template"
        job_name = self.job_name.replace("_", "-")

        if NEMO_LAUNCHER_DEBUG:
            # If NEMO_LAUNCHER_DEBUG is set, we just print the template.
            # The submission script will also have this command so a user is
            # expected to drop this env-var if they want to subsequently run.
            sub_command = "template"
        else:
            sub_command = "install"

        extra_helm_args = ""
        if self.parameters.get("namespace", None):
            extra_helm_args += f" --namespace {self.parameters['namespace']}"

        # Apply a timeout of 15min in case images take a long time to bring up
        # or pre-install hooks take a while
        return f"#!/bin/bash\nhelm {sub_command} --timeout=15m --wait {extra_helm_args} {job_name} {helm_charts}\n"


class K8SLauncherV2:
    """
    K8s V2 job launcher
    This class simply creates 

    """

    def __init__(self, job_path: job_utils.JobPaths):
        self.job_path = job_path

    def launch(self, workflow: Workflow):
        """
        Three steps of launching:
        1. make workflow manifest file
        2. make submission script
        3. run submission script
        """

        # Step 1
        workflow_path = workflow.to_file(output_directory=self.job_path.folder)

        # Step 2
        extra_args = ""
        if NEMO_LAUNCHER_DEBUG:
            # If NEMO_LAUNCHER_DEBUG is set, we just print the template.
            # The submission script will also have this command so a user is
            # expected to drop this env-var if they want to subsequently run.
            extra_args = "--dry-run=client"
        submission_cmd = (
            f"#!/bin/bash\nkubectl create {extra_args} -f {workflow_path}\n"
        )

        submission_file_path = self.job_path.submission_file
        with submission_file_path.open("w") as f:
            f.write(submission_cmd)

        # Step 3
        job_utils.CommandFunction(
            command=["bash", str(submission_file_path)],
            ret_stdout=False,
            verbose=False,
        )()


@functools.lru_cache()
def _get_default_parameters() -> Dict[str, Any]:
    """Parameters that can be set through update_parameters"""
    specs = inspect.getfullargspec(_make_sbatch_string)
    zipped = zip(specs.args[-len(specs.defaults) :], specs.defaults)  # type: ignore
    return {key: val for key, val in zipped if key not in {"command_groups", "folder"}}


# pylint: disable=too-many-arguments,unused-argument, too-many-locals
def _make_sbatch_string(
    command_groups: List[List[str]],
    folder: Union[str, Path],
    job_name: str = "nemo_launcher",
    partition: Optional[str] = None,
    time: int = 5,
    nodes: Union[int, List[int]] = 1,
    ntasks_per_node: Optional[Union[int, List[int]]] = None,
    cpus_per_task: Optional[int] = None,
    cpus_per_gpu: Optional[int] = None,
    num_gpus: Optional[int] = None,  # legacy
    gpus_per_node: Optional[int] = None,
    gpus_per_task: Optional[int] = None,
    qos: Optional[str] = None,  # quality of service
    setup: Optional[List[str]] = None,
    mem: Optional[str] = None,
    mem_per_gpu: Optional[str] = None,
    mem_per_cpu: Optional[str] = None,
    dependency: Optional[str] = None,
    comment: Optional[str] = None,
    constraint: Optional[str] = None,
    exclude: Optional[str] = None,
    account: Optional[str] = None,
    gres: Optional[str] = None,
    exclusive: Optional[Union[bool, str]] = None,
    array: Optional[str] = None,
    stderr_to_stdout: bool = False,
    container_image: Optional[str] = None,
    container_mounts: Optional[str] = None,
    additional_parameters: Optional[Dict[str, Any]] = None,
    srun_args: Optional[Iterable[str]] = None,
    heterogeneous: bool = False,
) -> str:
    """Creates the content of an sbatch file with provided parameters

    Parameters
    ----------
    See slurm sbatch documentation for most parameters:
    https://slurm.schedmd.com/sbatch.html

    Below are the parameters that differ from slurm documentation:

    command_groups:
        each command group will be assigned one srun
    folder: str/Path
        folder where print logs and error logs will be written
    setup: list
        a list of command to run in sbatch before running srun
    additional_parameters: dict
        Forces any parameter to a given value in sbatch. This can be useful
        to add parameters which are not currently available in nemo_launcher.
        Eg: {"mail-user": "blublu@nvidia.com", "mail-type": "BEGIN"}
    srun_args: List[str]
        Add each argument in the list to the srun call

    Raises
    ------
    ValueError
        In case an erroneous keyword argument is added, a list of all eligible parameters
        is printed, with their default values
    """
    nonslurm = [
        "nonslurm",
        "folder",
        "command_groups",
        "additional_parameters",
        "setup",
        "stderr_to_stdout",
        "container_image",
        "container_mounts",
        "srun_args",
        "heterogeneous",
    ]
    parameters = {
        k: v for k, v in locals().items() if v is not None and k not in nonslurm
    }
    # rename and reformat parameters

    if num_gpus is not None:
        warnings.warn(
            '"num_gpus" is deprecated, please use "gpus_per_node" instead (overwritting with num_gpus)'
        )
        parameters["gpus_per_node"] = parameters.pop("num_gpus", 0)
    if "cpus_per_gpu" in parameters and "gpus_per_task" not in parameters:
        warnings.warn(
            '"cpus_per_gpu" requires to set "gpus_per_task" to work (and not "gpus_per_node")'
        )
    # add necessary parameters
    job_name = parameters.get("job_name")
    paths = job_utils.JobPaths(folder=folder, job_name=job_name)
    stdout = str(paths.stdout)
    stderr = str(paths.stderr)

    if array is not None:
        stdout = stdout.replace("%j", "%A_%a")
        stderr = stderr.replace("%j", "%A_%a")
    parameters["output"] = stdout.replace("%t", "0")

    if not stderr_to_stdout:
        parameters["error"] = stderr.replace("%t", "0")

    if NEMO_LAUNCHER_CI:  # Override output file for slurm
        parameters["output"] = parameters["error"] = str(paths.folder / "slurm_%j.out")
        stdout = stderr = parameters["output"]

    if additional_parameters is not None:
        parameters.update(additional_parameters)
    # now create
    lines = ["#!/bin/bash", "", "# Parameters"]
    if heterogeneous:
        for i in range(len(nodes)):
            het_parameters = parameters.copy()
            het_parameters["output"] = parameters["output"].replace("_%j", f"_{i}_%j")
            if "error" in parameters:
                het_parameters["error"] = parameters["error"].replace("_%j", f"_{i}_%j")
            het_parameters.update(
                {
                    "job_name": f"{job_name}_{i}",
                    "nodes": nodes[i],
                    "ntasks_per_node": ntasks_per_node[i],
                }
            )
            for k in sorted(parameters):
                lines.append(_as_sbatch_flag(k, het_parameters[k]))
            if i != len(nodes) - 1:
                lines.append(f"#SBATCH hetjob")
    else:
        for k in sorted(parameters):
            lines.append(_as_sbatch_flag(k, parameters[k]))
    # environment setup:
    if setup is not None:
        lines += ["", "# setup"] + setup

    # commandline (this will run the function and args specified in the file provided as argument)
    # We pass --output and --error here, because the SBATCH command doesn't work as expected with a filename pattern
    stderr_flags = [] if stderr_to_stdout else ["--error", stderr]
    container_flags = ["--container-image", container_image] if container_image else []
    container_flags += (
        ["--container-mounts", container_mounts] if container_mounts else []
    )
    if srun_args is None:
        srun_args = []

    if NEMO_LAUNCHER_MEMORY_MEASURE:
        srun_args += ["--overlap"]

        mem_stdout = stdout.replace("_%j", "_mem_%j")
        mem_stdout = mem_stdout.replace("_%A_%a", "_mem_%A_%a")
        mem_srun_cmd = shlex.join(
            [
                "srun",
                "--ntasks=1",
                "--ntasks-per-node=1",
                "--output",
                mem_stdout,
                *container_flags,
                *srun_args,
            ]
        )
        lines += [
            "",
            "# run memory measure",
            f"{mem_srun_cmd} \\",
            f"  nvidia-smi --query-gpu=timestamp,index,,memory.total,memory.free,memory.used --format=csv -l 1 & ",
            "",
        ]

    for group_ind, command_group in enumerate(command_groups):
        if heterogeneous:
            het_group = f"--het-group={group_ind}"
            het_stdout = stdout.replace("_%j", f"_{group_ind}_%j")
            het_stderr = stderr_flags.copy()
            if het_stderr:
                het_stderr[-1] = het_stderr[-1].replace("_%j", f"_{group_ind}_%j")
            srun_cmd = shlex.join(
                [
                    "srun",
                    "--output",
                    het_stdout,
                    *het_stderr,
                    *container_flags,
                    *srun_args,
                    het_group,
                ]
            )
            command = ";\n  ".join(command_group)
            lines += [
                "",
                f"# command {group_ind + 1}",
                f'{srun_cmd} bash -c "',
                f'  {command} " &',
                "",
            ]
            if group_ind == len(nodes) - 1:
                lines += ["wait"]
            else:
                lines += ["sleep 30"]
        else:
            srun_cmd = shlex.join(
                [
                    "srun",
                    "--output",
                    stdout,
                    *stderr_flags,
                    *container_flags,
                    *srun_args,
                ]
            )
            command = ";\n  ".join(command_group)
            lines += [
                "",
                f"# command {group_ind + 1}",
                f'{srun_cmd} bash -c "',
                f'  {command} "',
                "",
            ]
    return "\n".join(lines)


@functools.lru_cache()
def _get_default_parameters_ft_launcher() -> Dict[str, Any]:
    """Parameters that can be set through update_parameters"""
    specs = inspect.getfullargspec(_make_sbatch_string_ft_launcher)
    zipped = zip(specs.args[-len(specs.defaults) :], specs.defaults)  # type: ignore
    return {key: val for key, val in zipped if key not in {"command_groups", "folder"}}


# pylint: disable=too-many-arguments,unused-argument, too-many-locals
def _make_sbatch_string_ft_launcher(
    command_groups: List[List[str]],
    folder: Union[str, Path],
    job_name: str = "nemo_launcher",
    partition: Optional[str] = None,
    time: int = 5,
    nodes: Union[int, List[int]] = 1,
    ntasks_per_node: Optional[Union[int, List[int]]] = None,
    cpus_per_task: Optional[int] = None,
    cpus_per_gpu: Optional[int] = None,
    num_gpus: Optional[int] = None,  # legacy
    gpus_per_node: Optional[int] = None,
    gpus_per_task: Optional[int] = None,
    qos: Optional[str] = None,  # quality of service
    setup: Optional[List[str]] = None,
    mem: Optional[str] = None,
    mem_per_gpu: Optional[str] = None,
    mem_per_cpu: Optional[str] = None,
    dependency: Optional[str] = None,
    comment: Optional[str] = None,
    constraint: Optional[str] = None,
    exclude: Optional[str] = None,
    account: Optional[str] = None,
    gres: Optional[str] = None,
    exclusive: Optional[Union[bool, str]] = None,
    array: Optional[str] = None,
    stderr_to_stdout: bool = False,
    container_image: Optional[str] = None,
    container_mounts: Optional[str] = None,
    additional_parameters: Optional[Dict[str, Any]] = None,
    srun_args: Optional[Iterable[str]] = None,
    heterogeneous: bool = False,
    max_subsequent_job_failures: int = 0,
    max_rank_restarts: int = 0,
    additional_ft_launcher_args: str = "",
) -> str:

    """Creates the content of an sbatch file with provided parameters

    Parameters
    ----------
    See slurm sbatch documentation for most parameters:
    https://slurm.schedmd.com/sbatch.html

    Below are the parameters that differ from slurm documentation:

    command_groups:
        each command group will be assigned one srun
    folder: str/Path
        folder where print logs and error logs will be written
    setup: list
        a list of command to run in sbatch before running srun
    additional_parameters: dict
        Forces any parameter to a given value in sbatch. This can be useful
        to add parameters which are not currently available in nemo_launcher.
        Eg: {"mail-user": "blublu@nvidia.com", "mail-type": "BEGIN"}
    srun_args: List[str]
        Add each argument in the list to the srun call

    Raises
    ------
    ValueError
        In case an erroneous keyword argument is added, a list of all eligible parameters
        is printed, with their default values
    """
    nonslurm = [
        "nonslurm",
        "folder",
        "command_groups",
        "additional_parameters",
        "setup",
        "stderr_to_stdout",
        "container_image",
        "container_mounts",
        "srun_args",
        "heterogeneous",
        "max_subsequent_job_failures",
        "max_rank_restarts",
        "additional_ft_launcher_args",
    ]
    parameters = {
        k: v for k, v in locals().items() if v is not None and k not in nonslurm
    }
    # rename and reformat parameters

    if num_gpus is not None:
        warnings.warn(
            '"num_gpus" is deprecated, please use "gpus_per_node" instead (overwritting with num_gpus)'
        )
        parameters["gpus_per_node"] = parameters.pop("num_gpus", 0)
    if "cpus_per_gpu" in parameters and "gpus_per_task" not in parameters:
        warnings.warn(
            '"cpus_per_gpu" requires to set "gpus_per_task" to work (and not "gpus_per_node")'
        )
    # add necessary parameters
    job_name = parameters.get("job_name")
    paths = job_utils.JobPaths(folder=folder, job_name=job_name)
    stdout = str(paths.stdout)
    stderr = str(paths.stderr)

    if array is not None:
        stdout = stdout.replace("%j", "%A_%a")
        stderr = stderr.replace("%j", "%A_%a")
    parameters["output"] = stdout.replace("%t", "0")

    if not stderr_to_stdout:
        parameters["error"] = stderr.replace("%t", "0")

    if NEMO_LAUNCHER_CI:  # Override output file for slurm
        parameters["output"] = parameters["error"] = str(paths.folder / "slurm_%j.out")
        stdout = stderr = parameters["output"]

    if additional_parameters is not None:
        parameters.update(additional_parameters)
    # now create
    lines = ["#!/bin/bash", "", "# Parameters"]
    if heterogeneous:
        raise ValueError("Fault tolerance is not supported with heterogeneous jobs.")
    else:
        # run 1 FT launcher per node, it will spawn the actual tasks
        parameters["ntasks_per_node"] = 1
        for k in sorted(parameters):
            lines.append(_as_sbatch_flag(k, parameters[k]))
        parameters["ntasks_per_node"] = ntasks_per_node

    # environment setup:
    if setup is not None:
        lines += ["", "# setup"] + setup

    if srun_args is None:
        srun_args = []

    # A safety measures:
    # let SLURM immediately kill all tasks if any FT launcher returns with a failure.
    # let SLURM kill the job, 1h after any task ended without a failure.
    srun_args += ["--kill-on-bad-exit=1", "--wait=3600"]

    lines += [
        "",
        "# Fault tolerance related items",
        f'export FAULT_TOL_CFG_PATH="{str(paths.config_file)}"',
        f'export FAULT_TOL_FINISHED_FLAG_FILE="{str(paths.folder / "_finished_flag")}"',
        "RDZV_HOST=$(hostname)",
        "ANY_JOB_STEP_FAILED=0",
    ]

    if max_subsequent_job_failures > 0:
        lines += [
            "",
            "# Automatic job resubmission related items",
            f'JOB_RESULTS_FILE="{str(paths.folder / "_job_results")}"',
            f"MAX_JOB_FAILURES={max_subsequent_job_failures}",
            "is_job_failures_limit_reached() {",
            '    tail -n $MAX_JOB_FAILURES "$JOB_RESULTS_FILE" | \\',
            '       awk "/^[[:alnum:]]+[[:space:]]+[XF]$/{f++} END{exit !(f>=$MAX_JOB_FAILURES)}"',
            "}",
            "is_training_finished() {",
            '    test -f "$FAULT_TOL_FINISHED_FLAG_FILE"',
            "}",
            "# Exit immediately if finished flag file exists and this job is a continuation",
            'if [ "$FT_RESUMED" = "1" ] ; then',
            '    if is_training_finished ; then echo "Training is finished" ; exit 0 ; fi',
            '    if is_job_failures_limit_reached ; then echo "Job failures limit reached ($MAX_JOB_FAILURES)" ; exit 1 ; fi',
            "else",
            '    rm -f "$FAULT_TOL_FINISHED_FLAG_FILE" "$JOB_RESULTS_FILE"',
            "fi",
            "# Pre-schedule continuation job",
            'CONT_SBATCH_OUT=$(FT_RESUMED=1 sbatch --parsable --dependency=afterany:"$SLURM_JOB_ID" "$0")',
            'if [ $? -ne 0 ] ; then echo "Couldnt schedule continuation job. Check stderr for details." ; exit 1 ; fi',
            'CONT_SLURM_JOB_ID=$(echo $CONT_SBATCH_OUT | cut -f1 -d",")',
            "# Write unknown job status to the job log, we will fix it at the end",
            'echo "$SLURM_JOB_ID X" >> "$JOB_RESULTS_FILE"',
        ]

    # commandline (this will run the function and args specified in the file provided as argument)
    # We pass --output and --error here, because the SBATCH command doesn't work as expected with a filename pattern
    stderr_flags = [] if stderr_to_stdout else ["--error", stderr]
    container_flags = ["--container-image", container_image] if container_image else []
    container_flags += (
        ["--container-mounts", container_mounts] if container_mounts else []
    )

    if NEMO_LAUNCHER_MEMORY_MEASURE:
        srun_args += ["--overlap"]

        mem_stdout = stdout.replace("_%j", "_mem_%j")
        mem_stdout = mem_stdout.replace("_%A_%a", "_mem_%A_%a")
        mem_srun_cmd = shlex.join(
            [
                "srun",
                "--ntasks=1",
                "--ntasks-per-node=1",
                "--output",
                mem_stdout,
                *container_flags,
                *srun_args,
            ]
        )
        lines += [
            "",
            "# run memory measure",
            f"{mem_srun_cmd} \\",
            f"  nvidia-smi --query-gpu=timestamp,index,,memory.total,memory.free,memory.used --format=csv -l 1 & ",
            "",
        ]

    ft_launcher_cmd_part = (
        "ft_launcher "
        + f"--fault-tol-cfg-path=$FAULT_TOL_CFG_PATH --ignore-missing-fault-tol-cfg {additional_ft_launcher_args} "
        + "--rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$RDZV_HOST "
        + f"--nnodes={nodes} --nproc_per_node={ntasks_per_node} --max-restarts={max_rank_restarts}"
    )

    for group_ind, command_group in enumerate(command_groups):
        if heterogeneous:
            raise ValueError("This PoC does not support heterogeneous jobs")
        else:
            srun_cmd = shlex.join(
                [
                    "srun",
                    "--output",
                    stdout,
                    *stderr_flags,
                    *container_flags,
                    *srun_args,
                ]
            )
            command = ";\n  ".join(command_group)
            assert "python3 -u" in command
            command = command.replace("python3 -u", ft_launcher_cmd_part,)
            lines += [
                "",
                f"# command {group_ind + 1}",
                f'{srun_cmd} bash -c "',
                f'  {command} "',
                "",
            ]
            lines += ["if [ $? -ne 0 ]; then ANY_JOB_STEP_FAILED=1 ; fi"]

    if max_subsequent_job_failures > 0:
        lines += [
            "",
            '# Fix the job log entry ("JOB_ID X" -> "JOB_ID S/F"), depending on the job result',
            'if [ "$ANY_JOB_STEP_FAILED" = "0" ] ; then',
            '   sed -i "s/$SLURM_JOB_ID X/$SLURM_JOB_ID S/" "$JOB_RESULTS_FILE"',
            "else",
            '   sed -i "s/$SLURM_JOB_ID X/$SLURM_JOB_ID F/" "$JOB_RESULTS_FILE"',
            "fi",
            "# Check if the continuation job can be cancelled",
            "if is_training_finished ; then",
            '    echo "Training is finished" ; scancel $CONT_SLURM_JOB_ID ; exit 0',
            "fi",
            "if is_job_failures_limit_reached ; then",
            '    echo "Job failures limit reached ($MAX_JOB_FAILURES)" ; scancel $CONT_SLURM_JOB_ID ; exit 1',
            "fi",
        ]

    return "\n".join(lines)


def _convert_mem(mem_gb: float) -> str:
    """Convert non-integer mem_gb to unit MB"""
    if mem_gb == int(mem_gb):
        if int(mem_gb) == 0:
            return "0"
        return f"{int(mem_gb)}GB"
    return f"{int(mem_gb * 1024)}MB"


def _as_sbatch_flag(key: str, value: Any) -> str:
    """Convert key value pairs to `#SBATCH --{key}={value}` flags"""
    key = key.replace("_", "-")
    if value is True:
        return f"#SBATCH --{key}"

    value = shlex.quote(str(value))
    return f"#SBATCH --{key}={value}"
