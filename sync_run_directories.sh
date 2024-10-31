# Downloading run results (plots and generated sentences) from the university machine on which the trainings are run.
rsync -r --exclude '*/checkpoints/' r11922182@pepper.csie.ntu.edu.tw:/disks/local/p.4t_6/antoni/thesis_nlp/runs/attacker ./runs/
rsync -r --exclude '*.bin' r11922182@pepper.csie.ntu.edu.tw:/disks/local/p.4t_6/antoni/thesis_nlp/runs/static_victim_retraining ./runs/
