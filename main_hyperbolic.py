import argparse
import sys
sys.path.append("..")

import pandas as pd
import json
import numpy as np

from torch.utils.data import DataLoader
from model_zoo import get_model
from dataset_zoo import VG_Relation

parser = argparse.ArgumentParser()
parser.add_argument('--root', help='root of the project', default='', type=str)
parser.add_argument('--model_name', help='Specify MERU/CLIP and S/B/L_', default='meru_vit_b', type=str)
args = parser.parse_args()

root_dir = args.root + '/datasets/VG_Relation'

model, preprocess = get_model(model_name=args.model_name, device="cuda", root=args.root, root_dir=root_dir)

# Get the VG-R dataset
vgr_dataset = VG_Relation(image_preprocess=preprocess, download=True, root_dir=root_dir)
vgr_loader = DataLoader(vgr_dataset, batch_size=16, shuffle=False)

# Compute the scores for each test case
vgr_scores = model.get_retrieval_scores_batched(vgr_loader)
vgr_records = vgr_dataset.evaluate_scores(vgr_scores)

#Bootstrap scores
scores = np.asarray(vgr_scores)
rng = np.random.RandomState(seed=12345)
idx = np.arange(vgr_scores.shape[0])
test_accuracies = []
for i in range(500):
    pred_idx = rng.choice(idx, size = vgr_scores.shape[0], replace=True)
    selected_scores = scores[pred_idx]
    test_output = vgr_dataset.evaluate_scores_bootstrap(pred_idx, selected_scores)
    df = pd.DataFrame(test_output)
    test_accuracy = df.Accuracy.mean()
    test_accuracies.append(test_accuracy*100)

filename='scores' + args.model_name + '.txt'
with open (filename, "w") as filehandle:
    json.dump(test_accuracies, filehandle)
bootstrap_test_mean = np.mean(test_accuracies)
print('bootstrap test mean', bootstrap_test_mean)
ci_lower = np.percentile(test_accuracies, 2.5)
ci_upper = np.percentile(test_accuracies, 97.5)

print(ci_lower, ci_upper)
print('std', np.std(test_accuracies))
print('var', np.var(test_accuracies))


df = pd.DataFrame(vgr_records)
print(f"VG-Relation Macro Accuracy: {df.Accuracy.mean()}")
df.to_csv(args.model_name +'_vgr_acc.csv')
