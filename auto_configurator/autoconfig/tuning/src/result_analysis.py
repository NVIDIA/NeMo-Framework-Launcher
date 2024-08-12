# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import hydra
import numpy as np
import omegaconf
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_run_summary(result_dir: str) -> pd.DataFrame:
    """
    Processes fine-tuning hyperparameter search experiments, outputs a table with aggregated results.
    The results are sorted in ascending order based on validation loss for the first validation dataset.
    :param str result_dir: path with run logs generated by fine-tuning autoconfigurator.
    :return: pandas dataframe, containing aggregated run hyperparameters and metrics.
    :rtype: pd.DataFrame
    """
    log_dir = os.path.join(result_dir, "ft_logs")
    out_df = []
    for run_dir in os.listdir(log_dir):
        try:
            run_path = os.path.join(log_dir, run_dir)
            yaml_files = [file for file in os.listdir(run_path) if file.endswith(".yaml")]
            assert len(yaml_files) == 1
            run_config = omegaconf.OmegaConf.load(os.path.join(run_path, yaml_files[0]))
            run_overrides = run_config.overrides

            results_dir = os.path.join(run_path, "results")
            event_files = [file for file in os.listdir(results_dir) if file.startswith("events.")]
            assert len(event_files) == 1
            event_file = event_files[0]
            event_acc = EventAccumulator(os.path.join(results_dir, event_file))
            event_acc.Reload()

            run_data = dict(run_overrides)
            # Extract averaged metrics
            for stat in ["train_step_timing in s"]:
                run_data[stat] = np.mean(list(scalar.value for scalar in event_acc.Scalars(stat)))
            # Extract most recent metric
            for stat in ["reduced_train_loss"]:
                run_data[stat] = [scalar.value for scalar in event_acc.Scalars(stat)][-1]
            # Extract validation losses
            for ind, _ in enumerate(run_config.model.data.validation_ds.file_names):
                run_data[f"validation_loss_{ind}"] = event_acc.Scalars(f"validation_loss_dataloader{ind}")[-1].value
            run_data["total_steps"] = event_acc.Scalars("validation_loss_dataloader0")[-1].step
            out_df.append(run_data)
        except Exception as e:
            print(f"Warning: Could not parse logs from {run_dir}, skipping ({e})")

    return pd.DataFrame(out_df).sort_values("validation_loss_0").reset_index(drop=True)


@hydra.main(config_path="../conf", config_name="config")
def main(cfg: omegaconf.dictconfig.DictConfig) -> None:
    """
    Entry point for the fine-tuning autoconfigurator result analysis. Reads the config with
      hydra, Slurm result analysis jobs, saves a .csv file with results.
    :param omegaconf.dictconfig.DictConfig cfg: OmegaConf object, read using
      the @hydra.main decorator.
    :return: None
    """
    result_dir = os.path.join(cfg.base_results_dir, cfg.search_config.model_name, cfg.search_config.start_time)
    out_dir = os.path.join(result_dir, "final_result")

    summary_df = extract_run_summary(result_dir)
    summary_df.to_csv(os.path.join(out_dir, "results.csv"), index=None)

    print(f"Grid search concluded, best loss {summary_df['validation_loss_0'][0]:.4f}, best hyperparameters:")
    for key in cfg.search_config.param_grid:
        print(f"{key}: {summary_df[key][0]}")


if __name__ == "__main__":
    main()
