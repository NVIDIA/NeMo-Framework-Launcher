import os
import subprocess

ERROR = 'RuntimeError: Could not detect "srun", are you indeed on a slurm cluster?'


def adapter_learning(model_type):
    cmd = (
        "python3 main.py "
        "stages=[adapter_learning] "
        f"adapter_learning={model_type}/squad "
        "launcher_scripts_path=. "
        "base_results_dir=test_folder"
    )

    command = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, errors = command.communicate()

    return errors.decode()


class TestAdapterLearn:
    def test_gpt3(self):

        output = adapter_learning("gpt3")
        assert ERROR in output

    def test_t5(self):

        output = adapter_learning("t5")
        assert ERROR in output

    def test_remove_folders(self):
        os.system("rm -rf test_folder")
