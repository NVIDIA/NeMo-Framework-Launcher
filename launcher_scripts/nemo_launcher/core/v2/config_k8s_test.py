from nemo_launcher.core.v2.config_k8s import (
    K8sClusterConfig,
    instantiate_model_from_omegaconf,
)
from omegaconf import OmegaConf
import pytest


def test_k8s_cluster_config_with_pvc():
    cluster_cfg = OmegaConf.create(
        """
_target_: nemo_launcher.core.v2.config_k8s.K8sClusterConfig
volumes:
  workspace:
    sub_path: mnt/nemo/workspace
    persistent_volume_claim:
      claim_name: foobar
        """
    )
    if "_target_" in cluster_cfg:
        k8s_config: K8sClusterConfig = instantiate_model_from_omegaconf(cluster_cfg)

    assert k8s_config.model_dump()["volumes"] == {
        "workspace": {
            "sub_path": "mnt/nemo/workspace",
            "persistent_volume_claim": {"claim_name": "foobar",},
            "host_path": None,
            "nfs": None,
            "empty_dir": None,
            "_mount_path": "/mnt/nemo/workspace",
            "mount_path": None,
            "read_only": False,
        }
    }


def test_mutually_exclusive_k8s_volumes():
    cluster_cfg = OmegaConf.create(
        """
_target_: nemo_launcher.core.v2.config_k8s.K8sClusterConfig
volumes:
  workspace:
    persistent_volume_claim:
      claim_name: foobar
    nfs:
      server: localhost
      path: /mnt/nemo/workspace
        """
    )

    with pytest.raises(ValueError):
        instantiate_model_from_omegaconf(cluster_cfg)


@pytest.mark.parametrize(
    "volume,expected_mount_path",
    [
        ({"nfs": {"server": "", "path": "/a/b"}}, "/a/b"),
        ({"host_path": {"path": "/c/d"}}, "/c/d"),
        ({"persistent_volume_claim": {"claim_name": ""}, "sub_path": "e/f"}, "/e/f"),
    ],
)
def test_mount_path_selection(volume, expected_mount_path):
    cluster_cfg = {
        "_target_": "nemo_launcher.core.v2.config_k8s.K8sClusterConfig",
        "volumes": {"workspace": volume,},
    }
    k8s_cfg: K8sClusterConfig = instantiate_model_from_omegaconf(
        OmegaConf.create(cluster_cfg)
    )
    assert k8s_cfg.volumes["workspace"]._mount_path == expected_mount_path

    # Test override is respected
    cluster_cfg["volumes"]["workspace"]["mount_path"] = "/mnt/override"
    k8s_cfg: K8sClusterConfig = instantiate_model_from_omegaconf(
        OmegaConf.create(cluster_cfg)
    )
    assert k8s_cfg.volumes["workspace"]._mount_path == "/mnt/override"
