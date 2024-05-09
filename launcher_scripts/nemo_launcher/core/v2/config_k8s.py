from pydantic import BaseModel, validator, Field, computed_field, model_validator
from kubernetes.client import (
    V1VolumeMount,
    V1Volume,
    V1NFSVolumeSource,
    V1HostPathVolumeSource,
    V1PersistentVolumeClaimVolumeSource,
    V1EmptyDirVolumeSource,
)
from hera.workflows.models import (
    Volume,
    VolumeMount,
    # TODO: Use these BaseModels when solution we have solution to: https://github.com/argoproj-labs/hera/issues/984
    # NFSVolumeSource,
    # HostPathVolumeSource,
    # PersistentVolumeClaimVolumeSource,
)
from omegaconf import OmegaConf
from hydra.utils import get_class
from typing import overload, Optional, Union
from enum import Enum

# TODO: Use these BaseModels until a solution to this exists: https://github.com/argoproj-labs/hera/issues/984
class NFSVolumeSource(BaseModel):
    server: str  # Hostname or IP address for the NFS server where data is stored.
    path: str  # Path to store data in the NFS server.


# Only suitable for 1 node clusters where all workers have this path mounted/available
class HostPathVolumeSource(BaseModel):
    # Path on the host
    path: str
    # https://kubernetes.io/docs/concepts/storage/volumes/#hostpath-volume-types
    # Directory = errors if does not exist
    type: str = "Directory"


# This claim should be created before running
class PersistentVolumeClaimVolumeSource(BaseModel):
    claim_name: str


class EmptyDirVolumeSource(BaseModel):
    medium: str
    size_limit: str


@overload
def convert_pydantic_vol_to_openapi(vol: NFSVolumeSource) -> V1NFSVolumeSource:
    ...


@overload
def convert_pydantic_vol_to_openapi(
    vol: HostPathVolumeSource,
) -> V1HostPathVolumeSource:
    ...


@overload
def convert_pydantic_vol_to_openapi(
    vol: PersistentVolumeClaimVolumeSource,
) -> V1PersistentVolumeClaimVolumeSource:
    ...


@overload
def convert_pydantic_vol_to_openapi(
    vol: EmptyDirVolumeSource,
) -> V1EmptyDirVolumeSource:
    ...


@overload
def convert_pydantic_vol_to_openapi(vol: None):
    ...


def convert_pydantic_vol_to_openapi(
    vol: Optional[
        Union[
            NFSVolumeSource,
            HostPathVolumeSource,
            PersistentVolumeClaimVolumeSource,
            EmptyDirVolumeSource,
        ]
    ]
) -> Optional[
    Union[
        V1NFSVolumeSource,
        V1HostPathVolumeSource,
        V1PersistentVolumeClaimVolumeSource,
        V1EmptyDirVolumeSource,
    ]
]:
    if vol is None:
        return vol
    elif isinstance(vol, NFSVolumeSource):
        return V1NFSVolumeSource(**vol.model_dump())
    elif isinstance(vol, HostPathVolumeSource):
        return V1HostPathVolumeSource(**vol.model_dump())
    elif isinstance(vol, PersistentVolumeClaimVolumeSource):
        return V1PersistentVolumeClaimVolumeSource(**vol.model_dump())
    elif isinstance(vol, EmptyDirVolumeSource):
        return V1EmptyDirVolumeSource(**vol.model_dump())
    else:
        raise NotImplementedError


# A convenience class that contains all info needed to define V1VolumeMount/V1Volumes
class K8sVolume(BaseModel):
    # By default, the path is mirrored into containers, use mount_path to specify a different path
    mount_path: Optional[str] = None
    # Path within the volume from which the container's volume should be mounted. default = /. Useful for PVC
    sub_path: Optional[str] = None
    read_only: bool = False

    nfs: Optional[NFSVolumeSource] = None
    persistent_volume_claim: Optional[PersistentVolumeClaimVolumeSource] = None
    host_path: Optional[HostPathVolumeSource] = None
    empty_dir: Optional[EmptyDirVolumeSource] = None

    @validator("sub_path")
    def sub_path_should_not_start_with_slash(cls, v: str) -> str:
        if v and v.startswith("/"):
            raise ValueError(
                f"{cls.__name__}.sub_path={v} should be an absolute path except it should not lead with /"
            )
        return v

    @model_validator(mode="after")
    def mutually_exclusive(self) -> "K8sVolume":
        num_defined = sum(
            bool(getattr(self, v_type))
            for v_type in ("nfs", "persistent_volume_claim", "host_path", "empty_dir")
        )
        if num_defined != 1:
            raise ValueError(
                f"Only one of nfs, persistent_volume_claim, host_path, empty_dir can be defined: {self}"
            )
        return self

    @computed_field
    @property
    def _mount_path(self) -> str:
        # This is the derived mount_path that accounts for the one passed in and the volume given.
        path = (
            self.mount_path
            or getattr(self.nfs, "path", None)
            or getattr(self.host_path, "path", None)
            or (
                f"/{self.sub_path}" if self.sub_path else None
            )  # sub_path would normally be set if PVC is used
        )
        if not path:
            raise ValueError(
                "Could not set _mount_path. Check that one of the following is set: mount_path|nfs.path|host_path.path|sub_path"
            )
        return path


class K8sNetworkInterfaces(BaseModel):
    # Specify the networks as comma separated values
    annotation: str = ""
    # Specify the resource name (as dict key) for IB devices according to kubernetes, such as "nvidia.com/hostdev" for Mellanox IB adapters, and the value is the count.
    resources: dict[str, int] = Field(default_factory=dict)


class K8sClusterConfig(BaseModel):
    # These volumes are mounted to all containers. Mapping of name to K8sVolume
    volumes: dict[str, K8sVolume]
    # Default is to use the current context's namespace
    namespace: Optional[str] = None

    ib_interfaces: Optional[K8sNetworkInterfaces] = None
    # dns_policy: str | None = None # Specify a dnsPolicy to use in all pods, if necessary
    pull_secret: str = "ngc-registry"  # Kubernetes secret for the container registry to pull private containers.
    shm_size: str = "512Gi"  # Amount of system memory to allocate in Pods. Should end in "Gi" for gigabytes.
    capabilities: Optional[list[str]] = [
        "IPC_LOCK"
    ]  # capabilities to add to all containers (useful for debugging), ex. ["IPC_LOCK", "SYS_PTRACE"]

    def check_path_in_volumes(self, path: str):
        # This is a helper method to help make sure users configure their k8s paths correctly.
        # For example if someone mounts a PVC under path=/foobar-workspace, stages can use this method
        # to make sure paths they expect are prefixed by a volume that will be mounted in.
        if not any(path.startswith(v._mount_path) for v in self.volumes.values()):
            raise ValueError(f"{path} needs to appear in one of volumes={self.volumes}")


class VolumeFormat(Enum):
    HERA = "hera"
    K8S = "k8s"
    OBJECT = "object"


def adapt_volume_to(
    volumes: dict[str, K8sVolume],
    to_format: Union[VolumeFormat, str] = VolumeFormat.HERA,
) -> Union[
    tuple[list[V1Volume], list[V1VolumeMount]], tuple[list[Volume], list[VolumeMount]]
]:
    if VolumeFormat(to_format) == VolumeFormat.HERA:
        vol_cls = Volume
        vol_mount_cls = VolumeMount
        convert_fn = lambda x: x
    elif VolumeFormat(to_format) in (VolumeFormat.K8S, VolumeFormat.OBJECT):
        # If to_format == 'object', convert to K8s types, then use client to convert to camel-case below
        vol_cls = V1Volume
        vol_mount_cls = V1VolumeMount
        convert_fn = convert_pydantic_vol_to_openapi
    else:
        raise NotImplementedError

    vols, vol_mounts = [], []
    for name, v in volumes.items():
        vols.append(
            vol_cls(
                name=name,
                nfs=convert_fn(v.nfs),
                host_path=convert_fn(v.host_path),
                persistent_volume_claim=convert_fn(v.persistent_volume_claim),
                empty_dir=convert_fn(v.empty_dir),
            )
        )
        vol_mounts.append(
            vol_mount_cls(
                name=name,
                mount_path=v._mount_path,
                read_only=v.read_only,
                sub_path=v.sub_path,
            )
        )
    if VolumeFormat(to_format) == VolumeFormat.OBJECT:
        # Create client just to convert snake-case to camel-case json obj
        from kubernetes.client import ApiClient
        from nemo_launcher.core.v2.step_k8s import prune_tree

        client = ApiClient()
        vols = [client.sanitize_for_serialization(v) for v in vols]
        vol_mounts = [client.sanitize_for_serialization(vm) for vm in vol_mounts]
        # Pruning is cosmetic and keeps manifest more readible and smaller
        vols = prune_tree(vols)
        vol_mounts = prune_tree(vol_mounts)

    return vols, vol_mounts


def instantiate_model_from_omegaconf(cfg: OmegaConf) -> BaseModel:
    kwargs = OmegaConf.to_object(cfg)
    _target_ = kwargs.pop("_target_")
    cls = get_class(_target_)
    if not issubclass(cls, BaseModel):
        raise ValueError(
            f"Expected _target_={_target_} to be a subclass of pydantic.BaseModel"
        )
    return cls.model_validate(kwargs, strict=True)
