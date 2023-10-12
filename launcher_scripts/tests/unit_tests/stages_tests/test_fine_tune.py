import os
import subprocess

ERROR = 'RuntimeError: Could not detect "srun", are you indeed on a slurm cluster?'


def fine_tune(model_type, task_type):
    cmd = (
        "python3 main.py "
        "stages=[fine_tuning] "
        f"fine_tuning={model_type}/{task_type} "
        "launcher_scripts_path=. "
        "base_results_dir=test_folder"
    )

    command = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, errors = command.communicate()

    return errors.decode()


class TestFineTune:
    def test_t5(self):

        output = fine_tune("t5", "squad")
        assert ERROR in output

    def test_mt5(self):

        output = fine_tune("mt5", "xquad")
        assert ERROR in output

    def test_remove_folders(self):
        os.system("rm -rf test_folder")
