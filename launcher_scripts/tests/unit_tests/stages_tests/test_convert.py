import os
import subprocess

ERROR = 'RuntimeError: Could not detect "srun", are you indeed on a slurm cluster?'


def convert(model_type):
    cmd = (
        "python3 main.py "
        "stages=[conversion] "
        f"conversion={model_type}/convert_{model_type} "
        "launcher_scripts_path=. "
        "base_results_dir=test_folder"
    )

    command = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, errors = command.communicate()

    return errors.decode()


class TestConvert:
    def test_gpt3(self):

        output = convert("gpt3")
        assert ERROR in output

    def test_prompt_t5(self):

        output = convert("t5")
        assert ERROR in output

    def test_prompt_mt5(self):

        output = convert("mt5")
        assert ERROR in output

    def test_remove_folders(self):
        os.system("rm -rf test_folder")
