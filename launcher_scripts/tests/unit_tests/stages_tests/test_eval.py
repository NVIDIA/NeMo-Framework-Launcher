import os
import subprocess

ERROR = 'RuntimeError: Could not detect "srun", are you indeed on a slurm cluster?'


def eval(model_type, eval_type):
    cmd = (
        "python3 main.py "
        "stages=[evaluation] "
        f"evaluation={model_type}/{eval_type} "
        "launcher_scripts_path=. "
        "base_results_dir=test_folder"
    )

    command = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    output, errors = command.communicate()

    return errors.decode()


class TestEval:
    def test_gpt3(self):
        types = ["evaluate_all.yaml", "evaluate_lambada.yaml"]

        for typee in types:
            output = eval("gpt3", typee)
            assert ERROR in output

    def test_prompt_gpt3(self):
        types = ["squad.yaml"]

        for typee in types:
            output = eval("prompt_gpt3", typee)
            assert ERROR in output

    def test_adapter_gpt3(self):
        types = ["squad.yaml"]

        for typee in types:
            output = eval("adapter_gpt3", typee)
            assert ERROR in output

    def test_t5(self):
        types = ["squad.yaml"]

        for typee in types:
            output = eval("t5", typee)
            assert ERROR in output

    def test_prompt_t5(self):
        types = ["squad.yaml"]

        for typee in types:
            output = eval("prompt_t5", typee)
            assert ERROR in output

    def test_adapter_t5(self):
        types = ["squad.yaml"]

        for typee in types:
            output = eval("adapter_t5", typee)
            assert ERROR in output

    def test_mt5(self):
        types = ["xquad.yaml"]

        for typee in types:
            output = eval("mt5", typee)
            assert ERROR in output

    def test_prompt_mt5(self):
        types = ["squad.yaml"]

        for typee in types:
            output = eval("prompt_mt5", typee)
            assert ERROR in output

    def test_remove_folders(self):
        os.system("rm -rf test_folder")
