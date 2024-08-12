# NeMo Framework Launcher Fine-Tuning Autoconfigurator

The fine-tuning autoconfigurator allows to conduct hyperparameter search for fine-tuning jobs on Slurm cluster. Similarly to usual grid search, it starts a sequence of jobs for different hyperparameter configurations, and analyses their results to find the best-performing one in terms of validation loss.

## Usage
1. Specify Slurm cluster parameters in `conf/cluster/bcm.yaml`
2. Fill all required values in `conf/config.yaml`. The `search_config.param_grid` field corresponds a set of hyperparamter values to use for grid search. Hyperparameter names should be specified in Hydra dot notation. The values should be lists of hyperparameter values to choose from.
3. Run hyperparameter search with `python3 main.py`

The following results will be stored in `base_results_dir` for each hyperparameter search:
- `candidate_configs` - .yaml config files, used for different experiments
- `ft_logs` - logs of NeMo fine-tuning jobs
- `final_result` - folder, containing result analysis logs and experiment summary in results.csv