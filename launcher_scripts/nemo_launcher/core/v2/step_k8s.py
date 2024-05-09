from hera.workflows import (
    Container,
    Parameter,
    Step,
    Steps,
    Workflow,
    Resource,
    script,
)
from kubeflow.training import (
    TrainingClient,
    KubeflowOrgV1ReplicaSpec,
    KubeflowOrgV1PyTorchJob,
    KubeflowOrgV1PyTorchJobSpec,
    KubeflowOrgV1RunPolicy,
    KubeflowOrgV1SchedulingPolicy,
    KubeflowOrgV1ElasticPolicy,
    KubeflowOrgV1MPIJob,
    KubeflowOrgV1MPIJobSpec,
)
from kubeflow.training.constants import constants
from kubernetes.client import (
    V1PodTemplateSpec,
    V1ObjectMeta,
    V1PodSpec,
    V1Container,
    V1ContainerPort,
    V1ResourceRequirements,
    V1VolumeMount,
    V1EnvVar,
    V1LocalObjectReference,
    V1SecurityContext,
    V1Capabilities,
    ApiClient,
)
from pydantic import BaseModel
import yaml
from typing import Callable, Any, Optional
from nemo_launcher.core.v2.config_k8s import (
    K8sVolume,
    K8sNetworkInterfaces,
    adapt_volume_to,
)
from textwrap import dedent
import os

_unset = object()

# TODO: Plumb this value from the K8sClusterConfig
# The value is set to a relatively small value to avoid
# using resources if a pod is continually failing.
BACKOFF_LIMIT = int(os.environ.get("K8S_BACKOFF_LIMIT", 10))


def prune_tree(tree: dict, is_prunable: Callable[[Any], bool] = None):
    """
    Recursively prunes a nested dictionary or list based on a given predicate function is_prunable.

    :param tree: The dictionary or list to prune.
    :param is_prunable: A lambda function that takes a value and returns True if it should be pruned. Default prunes None/{}/[]
    :return: The pruned dictionary or list.
    """
    if not is_prunable:
        is_prunable = lambda x: (x is None or x == {} or x == [])

    def prune_helper(tree: Any) -> bool:
        # Post-order recursion guarantees pruning is propagated from leaf up to root
        if isinstance(tree, dict):
            for key in list(tree.keys()):
                if prune_helper(tree[key]):
                    del tree[key]

        elif isinstance(tree, list):
            for i in range(len(tree) - 1, -1, -1):
                if prune_helper(tree[i]):
                    # At worst quadratic, but keeps code simpler
                    del tree[i]

        # Returning True if we should prune this leaf
        if is_prunable(tree):
            return True

    # Root will never be pruned
    prune_helper(tree)

    return tree


def to_env_list(env: dict[str, Any]) -> list[V1EnvVar]:
    if not env:
        return []
    return [
        V1EnvVar(name=name, value=str(value))
        for name, value in env.items()
        if value is not None
    ]


def create_pytorchjob_resource(
    generate_name: str,
    namespace: str,
    image: str,
    n_workers: int,
    gpus_per_worker: int,
    env: Optional[dict] = None,
    command: Optional[list[str]] = None,
    args: Optional[list[str]] = None,
    image_pull_secret: Optional[str] = None,
    volumes: Optional[dict[str, K8sVolume]] = None,
    network_interfaces: Optional[K8sNetworkInterfaces] = None,
    ports: Optional[list[int]] = None,
    success_condition: Optional[str] = _unset,
    resource_inputs: Optional[list[Parameter]] = None,
    capabilities: Optional[list[str]] = None,
) -> Resource:
    if success_condition == _unset:
        success_condition = f"status.replicaStatuses.Worker.succeeded = {n_workers}"
    pod_annotations = {
        constants.ISTIO_SIDECAR_INJECTION: "false",
    }
    worker_resources = {}
    if gpus_per_worker:
        worker_resources |= {"nvidia.com/gpu": str(gpus_per_worker)}
    if network_interfaces:
        pod_annotations["k8s.v1.cni.cncf.io/networks"] = network_interfaces.annotation
        worker_resources |= network_interfaces.resources
    if volumes:
        vols, vol_mounts = adapt_volume_to(volumes, to_format="k8s")
    else:
        vols, vol_mounts = None, None
    if ports:
        ports = [V1ContainerPort(container_port=p) for p in ports]
    if capabilities:
        security_context = V1SecurityContext(
            capabilities=V1Capabilities(add=capabilities)
        )
    else:
        security_context = None

    container = V1Container(
        name="pytorch",
        image=image,
        command=command,
        args=args,
        env=to_env_list(env),
        resources=V1ResourceRequirements(limits=worker_resources,),
        volume_mounts=vol_mounts,
        ports=ports,
        security_context=security_context,
    )
    worker = KubeflowOrgV1ReplicaSpec(
        replicas=n_workers,
        restart_policy="OnFailure",
        template=V1PodTemplateSpec(
            metadata=V1ObjectMeta(
                annotations=pod_annotations,
                # This label allows you to query the logs thru argo CLI
                # Ex: argo logs -c pytorch -l replica-index=0
                labels={"workflows.argoproj.io/workflow": "{{workflow.name}}",},
            ),
            spec=V1PodSpec(
                containers=[container],
                volumes=vols,
                image_pull_secrets=[V1LocalObjectReference(name=image_pull_secret)],
            ),
        ),
    )
    pytorch_job = KubeflowOrgV1PyTorchJob(
        api_version=f"{constants.KUBEFLOW_GROUP}/{constants.OPERATOR_VERSION}",
        kind=constants.PYTORCHJOB_KIND,
        metadata=V1ObjectMeta(generate_name=generate_name, namespace=namespace),
        spec=KubeflowOrgV1PyTorchJobSpec(
            run_policy=KubeflowOrgV1RunPolicy(
                clean_pod_policy="None", backoff_limit=BACKOFF_LIMIT
            ),
            elastic_policy=KubeflowOrgV1ElasticPolicy(
                rdzv_backend="c10d",
                min_replicas=n_workers,
                max_replicas=n_workers,
                max_restarts=10,
                n_proc_per_node=gpus_per_worker,
            ),
            pytorch_replica_specs={"Worker": worker},
        ),
    )
    # Create client just to convert snake-case to camel-case json obj
    client = ApiClient()
    pytorch_obj = client.sanitize_for_serialization(pytorch_job)
    # Pruning is cosmetic and keeps manifest more readible and smaller
    pytorch_job_manifest = prune_tree(pytorch_obj)

    return Resource(
        # TODO: valid, but leads a slightly strange name of the launcher pod:
        # $wf-$hash-$generate_name-$hash = peft-af432-peft--412fe
        name=generate_name,
        action="create",
        success_condition=success_condition,
        # failure_condition="status.replicaStatuses.Worker.failed > 0",
        manifest=yaml.dump(pytorch_job_manifest, indent=2),
        set_owner_reference=True,
        # Use dependency_name to chain jobs and reference the resources from other jobs
        # at the moment the dependency_name is limited to a single resource
        inputs=resource_inputs,
        outputs=[
            Parameter(
                name="metadata_name", value_from={"jsonPath": "{.metadata.name}"}
            ),
            Parameter(
                name="metadata_namespace",
                value_from={"jsonPath": "{.metadata.namespace}"},
            ),
        ],
    )


def create_mpijob_resource(
    generate_name: str,
    namespace: str,
    image: str,
    n_workers: int,
    env: Optional[dict] = None,
    command: Optional[list[str]] = None,
    args: Optional[list[str]] = None,
    image_pull_secret: Optional[str] = None,
    volumes: Optional[dict[str, K8sVolume]] = None,
    network_interfaces: Optional[K8sNetworkInterfaces] = None,
    success_condition: Optional[str] = _unset,
    resource_inputs: Optional[list[Parameter]] = None,
    capabilities: Optional[list[str]] = None,
) -> Resource:
    if success_condition == _unset:
        success_condition = f"status.replicaStatuses.Launcher.succeeded = 1"
    pod_annotations = {
        constants.ISTIO_SIDECAR_INJECTION: "false",
    }
    worker_resources = {}
    if network_interfaces:
        pod_annotations["k8s.v1.cni.cncf.io/networks"] = network_interfaces.annotation
        worker_resources |= network_interfaces.resources
    if volumes:
        vols, vol_mounts = adapt_volume_to(volumes, to_format="k8s")
    else:
        vols, vol_mounts = None, None
    if capabilities:
        security_context = V1SecurityContext(
            capabilities=V1Capabilities(add=capabilities)
        )
    else:
        security_context = None

    launch_container = V1Container(
        name="mpi-launcher",
        image=image,
        command=command,
        args=args,
        env=to_env_list(env),
        volume_mounts=vol_mounts,
        security_context=security_context,
    )
    worker_container = V1Container(
        name="mpi-worker",
        image=image,
        command=["/usr/sbin/sshd"],
        args=["-De"],
        env=to_env_list(env),
        resources=V1ResourceRequirements(limits=worker_resources,),
        volume_mounts=vol_mounts,
        security_context=security_context,
    )

    def replica_template(n_replicas: int, container: V1Container):
        return KubeflowOrgV1ReplicaSpec(
            replicas=n_replicas,
            restart_policy="OnFailure",
            template=V1PodTemplateSpec(
                metadata=V1ObjectMeta(
                    annotations=pod_annotations,
                    # This label allows you to query the logs thru argo CLI
                    # Ex: argo logs -c pytorch -l replica-index=0
                    labels={"workflows.argoproj.io/workflow": "{{workflow.name}}",},
                ),
                spec=V1PodSpec(
                    containers=[container],
                    volumes=vols,
                    image_pull_secrets=[V1LocalObjectReference(name=image_pull_secret)],
                ),
            ),
        )

    launcher = replica_template(n_replicas=1, container=launch_container,)
    worker = replica_template(n_replicas=n_workers, container=worker_container,)
    mpijob = KubeflowOrgV1MPIJob(
        api_version=f"{constants.KUBEFLOW_GROUP}/{constants.OPERATOR_VERSION}",
        kind=constants.MPIJOB_KIND,
        metadata=V1ObjectMeta(generate_name=generate_name, namespace=namespace),
        spec=KubeflowOrgV1MPIJobSpec(
            # Clean running pods since the workers will hang around even after the launcher finishes
            run_policy=KubeflowOrgV1RunPolicy(
                clean_pod_policy="Running", backoff_limit=BACKOFF_LIMIT
            ),
            mpi_replica_specs={"Launcher": launcher, "Worker": worker,},
        ),
    )
    # Create client just to convert snake-case to camel-case json obj
    client = ApiClient()
    mpijob_obj = client.sanitize_for_serialization(mpijob)
    # Pruning is cosmetic and keeps manifest more readible and smaller
    mpijob_manifest = prune_tree(mpijob_obj)

    return Resource(
        # TODO: valid, but leads a slightly strange name of the launcher pod:
        # $wf-$hash-$generate_name-$hash = peft-af432-peft--412fe
        name=generate_name,
        action="create",
        success_condition=success_condition,
        # failure_condition="status.replicaStatuses.Worker.failed > 0",
        manifest=yaml.dump(mpijob_manifest, indent=2),
        set_owner_reference=True,
        # Use dependency_name to chain jobs and reference the resources from other jobs
        # at the moment the dependency_name is limited to a single resource
        inputs=resource_inputs,
        outputs=[
            Parameter(
                name="metadata_name", value_from={"jsonPath": "{.metadata.name}"}
            ),
            Parameter(
                name="metadata_namespace",
                value_from={"jsonPath": "{.metadata.namespace}"},
            ),
        ],
    )


def delete_pytorchjob(name: str = "delete-pytorchjob"):
    manifest = dedent(
        f"""
        apiVersion: {constants.KUBEFLOW_GROUP}/{constants.OPERATOR_VERSION}
        kind: {constants.PYTORCHJOB_KIND}
        metadata:
          name: {{{{inputs.parameters.metadata_name}}}}
        """
    )
    return Resource(
        name=name,
        action="delete",
        inputs=[Parameter(name="metadata_name")],
        manifest=manifest,
    )
