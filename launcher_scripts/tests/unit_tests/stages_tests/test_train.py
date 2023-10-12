import os
import subprocess

ERROR = 'RuntimeError: Could not detect "srun", are you indeed on a slurm cluster?'


def train(model_type, model_size):
    cmd = (
        "python3 main.py "
        "stages=[training] "
        f"training={model_type}/{model_size} "
        "launcher_scripts_path=. "
        "base_results_dir=test_folder"
    )

    command = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, errors = command.communicate()

    return errors.decode()


class TestTrain:
    def test_gpt3(self):
        sizes = [
            "126m",
            "5b",
            "20b",
            "40b",
            "175b",
            "400m_improved",
            "1b_improved",
            "7b_improved",
            "40b_improved",
        ]

        for size in sizes:
            output = train("gpt3", size)
            assert ERROR in output

    def test_t5(self):
        sizes = ["220m", "3b", "11b", "23b", "41b"]

        for size in sizes:
            output = train("t5", size)
            assert ERROR in output

    def test_mt5(self):
        sizes = ["170m", "390m", "3b", "11b", "23b"]

        for size in sizes:
            output = train("mt5", size)
            assert ERROR in output

    def test_bert(self):
        sizes = ["110m", "4b", "20b", "100b"]

        for size in sizes:
            output = train("bert", size)
            assert ERROR in output

    def test_remove_folders(self):
        os.system("rm -rf test_folder")
