# thesis_nlp
Repository with experimental code for my Master's thesis.

Data including generated text, plots and checkpoints is saved during training runs. All data required to reproduce the plots and tables included  in the thesis has been saved in the repository under runs/attacker and runs/  static_victim_retraining. The plots and LaTeX tables themselves are generated  with code saved under src/ pretty_plots_and_stats_for_thesis, with only minor manual modifications afterwards.

A log of attacker training runs is automatically maintained (`runs/attacker/log`). This includes information about the Git commit on which a given training run was executed, as  well as the parameters of the training script (`src/training_scripts/train_attacker.py`) and a summary of the run. Note that due to randomness in the training, rerunning with the same configuration (Git commit and script parameters) is _not_ guaranteed to yield the same results.

