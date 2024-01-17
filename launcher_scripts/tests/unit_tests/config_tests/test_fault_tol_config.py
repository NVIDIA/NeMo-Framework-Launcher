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


def test_fault_tol_config_no_fault_tol_section_bcm():
    """ No fault tolerance section in config, BCM cluster, should be fine """
    cfg = OmegaConf.load("conf/config.yaml")
    cfg.stages = ["training"]
    cfg.launcher_scripts_path = LAUNCHER_SCRIPTS_PATH
    cfg.base_results_dir = TEST_RESULTS_DIR
    cfg.cluster_type = "bcm"
    cfg.cluster = OmegaConf.load("conf/cluster/bcm.yaml")
    cfg.training_config = "gpt3/126m"
    cfg.training = OmegaConf.load("conf/training/gpt3/126m.yaml")
    assert cfg.training.exp_manager.get("fault_tolernace", None) is None
    stage = Training(cfg)
    _ = stage.run()


def test_fault_tol_config_no_fault_tol_section_bcp():
    """ No fault tolerance section in config, BCP cluster, should be fine """
    cfg = OmegaConf.load("conf/config.yaml")
    cfg.stages = ["training"]
    cfg.launcher_scripts_path = LAUNCHER_SCRIPTS_PATH
    cfg.base_results_dir = TEST_RESULTS_DIR
    cfg.cluster_type = "bcp"
    cfg.cluster = dict()
    cfg.training_config = "gpt3/126m"
    cfg.training = OmegaConf.load("conf/training/gpt3/126m.yaml")
    assert cfg.training.exp_manager.get("fault_tolernace", None) is None
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
    cfg.training.exp_manager.fault_tolerance = OmegaConf.create(
        {"max_subsequent_job_failures": 1}
    )
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
    cfg.training.exp_manager.fault_tolerance = OmegaConf.create(
        {"max_subsequent_job_failures": 1}
    )
    with pytest.raises(ValueError):
        stage = Training(cfg)
        _ = stage.run()

