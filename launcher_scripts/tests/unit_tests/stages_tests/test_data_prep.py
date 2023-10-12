import os
import subprocess

ERROR = 'RuntimeError: Could not detect "srun", are you indeed on a slurm cluster?'


def data_prep(model_type, data_type):
    cmd = (
        "python3 main.py "
        "stages=[data_preparation] "
        f"data_preparation={model_type}/{data_type} "
        "launcher_scripts_path=. "
        "base_results_dir=test_folder"
    )

    command = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, errors = command.communicate()

    return errors.decode()


class TestDataPrep:
    def test_gpt3(self):

        output = data_prep("gpt3", "download_gpt3_pile")
        assert ERROR in output

    def test_t5(self):

        output = data_prep("t5", "download_t5_pile")
        assert ERROR in output

    def test_mt5(self):

        output = data_prep("mt5", "download_mc4")
        assert ERROR in output

    def test_bert(self):

        output = data_prep("bert", "download_bert_pile")
        assert ERROR in output

    def test_remove_folders(self):
        os.system("rm -rf test_folder")
        os.system("rm -rf data")
