import os
import subprocess

ERROR = 'RuntimeError: Could not detect "srun", are you indeed on a slurm cluster?'


def export(model_type):
    cmd = (
        "python3 main.py "
        "stages=[export] "
        f"export={model_type}/export_{model_type} "
        "launcher_scripts_path=. "
        "base_results_dir=test_folder"
    )

    command = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, errors = command.communicate()

    return errors.decode()


class TestExport:
    def test_gpt3(self):

        output = export("gpt3")
        assert ERROR in output

    def test_t5(self):

        output = export("t5")
        assert ERROR in output

    def test_mt5(self):

        output = export("mt5")
        assert ERROR in output

    def test_remove_folders(self):
        os.system("rm -rf test_folder")
