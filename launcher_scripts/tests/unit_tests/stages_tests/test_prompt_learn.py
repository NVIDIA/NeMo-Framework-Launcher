import os
import subprocess

ERROR = 'RuntimeError: Could not detect "srun", are you indeed on a slurm cluster?'


def prompt_learn(model_type):
    cmd = (
        "python3 main.py "
        "stages=[prompt_learning] "
        f"prompt_learning={model_type}/squad "
        "launcher_scripts_path=. "
        "base_results_dir=test_folder"
    )

    command = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, errors = command.communicate()

    return errors.decode()


class TestPromptLearn:
    def test_gpt3(self):

        output = prompt_learn("gpt3")
        assert ERROR in output

    def test_t5(self):

        output = prompt_learn("t5")
        assert ERROR in output

    def test_mt5(self):

        output = prompt_learn("mt5")
        assert ERROR in output

    def test_remove_folders(self):
        os.system("rm -rf test_folder")
        os.system("rm -rf data")
