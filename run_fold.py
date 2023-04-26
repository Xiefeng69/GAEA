import os
import argparse
'''
    this file is aiming to automatically run the 5-fold cross validation:
    python run_fold.py
'''
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default='all', choices=['all', 'zh_en', 'fr_en', 'ja_en', 'en_fr_15k', 'en_de_15k', 'd_w_15k', 'd_y_15k', 'en_fr_100k', 'en_de_100k', 'd_w_100k', 'd_y_100k'])
args = parser.parse_args()

if args.task == "all":
    task_list = ['en_fr_15k', 'en_de_15k', 'd_w_15k', 'd_y_15k']
else:
    task_list = [args.task]

for task in task_list:
    for aug_balance in [100]:
        for pr in [0.1]: # for pr in [0.05, 0.15, 0.1]:
            for eval_metric in ["euclidean"]:
                for fold in [1, 2, 3, 4, 5]:
                    cmd = f"python train.py --task {task} --fold {fold} --pr {pr} --aug_balance {aug_balance} --eval_metric {eval_metric}"
                    print(cmd)
                    os.system(cmd)