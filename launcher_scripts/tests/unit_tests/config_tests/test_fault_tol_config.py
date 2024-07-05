import math
import os

import nemo_launcher.core.launchers
import omegaconf
import pytest
from nemo_launcher.core.stages import Training
from omegaconf import OmegaConf

# Setup NEMO_LAUNCHER_DEBUG=True, so no 'srun' or 'sbatch' is required
nemo_launcher.core.launchers.NEMO_LAUNCHER_DEBUG = True

omegaconf.OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)

omegaconf.OmegaConf.register_new_resolver(
    "divide_ceil", lambda x, y: int(math.ceil(x / y)), replace=True
)

omegaconf.OmegaConf.register_new_resolver(
    "divide_floor", lambda x, y: int(math.floor(x / y)), replace=True
)

LAUNCHER_SCRIPTS_PATH = "."
TEST_RESULTS_DIR = "test_folder_ft"


@pytest.fixture(autouse=True)
def _setup_and_teardown():
    yield
    os.system(f"rm -rf {TEST_RESULTS_DIR}")


def test_fault_tol_config_fault_tol_disabled_bcm():
    """ No fault tolerance, BCM cluster, should be fine """
    cfg = OmegaConf.load("conf/config.yaml")
    cfg.stages = ["training"]
    cfg.launcher_scripts_path = LAUNCHER_SCRIPTS_PATH
    cfg.base_results_dir = TEST_RESULTS_DIR
    cfg.cluster_type = "bcm"
    cfg.cluster = OmegaConf.load("conf/cluster/bcm.yaml")
    cfg.training_config = "gpt3/126m"
    cfg.training = OmegaConf.load("conf/training/gpt3/126m.yaml")
    assert cfg.training.exp_manager.get("create_fault_tolerance_callback", None) is None
    assert cfg.training.exp_manager.get("fault_toleranace", None) is None
    stage = Training(cfg)
    _ = stage.run()


def test_fault_tol_config_fault_tol_disabled_bcp():
    """ No fault tolerance, BCP cluster, should be fine """
    cfg = OmegaConf.load("conf/config.yaml")
    cfg.stages = ["training"]
    cfg.launcher_scripts_path = LAUNCHER_SCRIPTS_PATH
    cfg.base_results_dir = TEST_RESULTS_DIR
    cfg.cluster_type = "bcp"
    cfg.cluster = dict()
    cfg.training_config = "gpt3/126m"
    cfg.training = OmegaConf.load("conf/training/gpt3/126m.yaml")
    assert cfg.training.exp_manager.get("create_fault_tolerance_callback", None) is None
    assert cfg.training.exp_manager.get("fault_toleranace", None) is None
    stage = Training(cfg)
    _ = stage.run()


def test_fault_tol_config_with_bcm():
    """ Fault tolerance + BCM cluster, should be fine """
    cfg = OmegaConf.load("conf/config.yaml")
    cfg.stages = ["training"]
    cfg.launcher_scripts_path = LAUNCHER_SCRIPTS_PATH
    cfg.base_results_dir = TEST_RESULTS_DIR
    cfg.cluster_type = "bcm"
    cfg.cluster = OmegaConf.load("conf/cluster/bcm.yaml")
    cfg.training_config = "gpt3/126m"
    cfg.training = OmegaConf.load("conf/training/gpt3/126m.yaml")
    cfg.training.exp_manager.create_fault_tolerance_callback = True
    cfg.training.exp_manager.fault_tolerance = OmegaConf.create(
        {"max_subsequent_job_failures": 1}
    )
    stage = Training(cfg)
    _ = stage.run()


def test_fault_tol_config_with_bcm_no_ft_section():
    """ Fault tolerance + BCM cluster, no "fault_tolerance" section in cfg, should be fine """
    cfg = OmegaConf.load("conf/config.yaml")
    cfg.stages = ["training"]
    cfg.launcher_scripts_path = LAUNCHER_SCRIPTS_PATH
    cfg.base_results_dir = TEST_RESULTS_DIR
    cfg.cluster_type = "bcm"
    cfg.cluster = OmegaConf.load("conf/cluster/bcm.yaml")
    cfg.training_config = "gpt3/126m"
    cfg.training = OmegaConf.load("conf/training/gpt3/126m.yaml")
    cfg.training.exp_manager.create_fault_tolerance_callback = True
    stage = Training(cfg)
    _ = stage.run()


def test_fault_tol_config_with_bcp():
    """ Fault tolerance + BCP cluster, BCP is not supported """
    cfg = OmegaConf.load("conf/config.yaml")
    cfg.stages = ["training"]
    cfg.launcher_scripts_path = LAUNCHER_SCRIPTS_PATH
    cfg.base_results_dir = TEST_RESULTS_DIR
    cfg.cluster_type = "bcp"
    cfg.cluster = dict()
    cfg.training_config = "gpt3/126m"
    cfg.training = OmegaConf.load("conf/training/gpt3/126m.yaml")
    cfg.training.exp_manager.create_fault_tolerance_callback = True
    with pytest.raises(ValueError):
        stage = Training(cfg)
        _ = stage.run()
