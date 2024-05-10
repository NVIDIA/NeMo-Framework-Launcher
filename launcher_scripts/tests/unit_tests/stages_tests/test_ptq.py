import os
import subprocess

ERROR = 'RuntimeError: Could not detect "srun", are you indeed on a slurm cluster?'
NEMO_LAUNCHER_DEBUG_MSG = "submitted with FAKE Job ID"


def ptq(model_type, task_type, nemo_launcher_debug=False):
    cmd = (
        f"NEMO_LAUNCHER_DEBUG={nemo_launcher_debug} python3 main.py "
        "stages=[ptq] "
        f"ptq={model_type}/{task_type} "
        "launcher_scripts_path=. "
        "base_results_dir=test_folder "
        "ptq.run.model_train_name=llama2_70b "
        "ptq.model_file=llama2_70b.nemo "
        "ptq.model_save=llama2_70b_qnemo "
        "ptq.tensor_model_parallel_size=8 "
        "ptq.export.inference_tensor_parallel=1"
    )

    command = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )

    output, errors = command.communicate()
    return output.decode(), errors.decode()


class TestPTQ:
    def test_llama2(self):

        _, output = ptq("llama2", "quantization")
        assert ERROR in output

    def test_llama2_debug(self):

        output, _ = ptq("llama2", "quantization", True)
        assert NEMO_LAUNCHER_DEBUG_MSG in output

    def test_remove_folders(self):
        os.system("rm -rf test_folder")
