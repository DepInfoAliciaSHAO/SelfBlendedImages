import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
from datasets import *
from datasets import CROP_DIR
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

import pickle

DATASETS = ['akool', 'ibeta', 'vidnoz', 'ffcm_subset', 'gitw', 'FF', 'CDF']

def main(args):
    device = torch.device('cuda')
    print(f"Testing model {os.path.basename(args.weight_name)}")
    model=Detector()
    model=model.to(device)
    cnn_sd=torch.load(args.weight_name)["model"]
    model.load_state_dict(cnn_sd)
    model.eval()
    output_list=[]
    target_list_all = []
    for dataset in DATASETS:
        _, target_list, video_root = init_dataset(dataset)
        target_list_all += target_list
        data_path = os.path.join(CROP_DIR, video_root, 'video_data.pkl')
        assert(os.path.exists(data_path))
        print(f"------Inference on {dataset}------")

        with open(data_path, 'rb') as f:
            video_data = pickle.load(f)
        for filename in tqdm(video_data.keys()):
            try:
                face_list = video_data[filename]['face_list'] 
                idx_list = video_data[filename]['idx_list']

                with torch.no_grad():
                    img=torch.tensor(face_list).to(device).float()/255
                    pred=model(img).softmax(1)[:,1]


                pred_list=[]
                idx_img=-1
                for i in range(len(pred)):
                    if idx_list[i]!=idx_img:
                        pred_list.append([])
                        idx_img=idx_list[i]
                    pred_list[-1].append(pred[i].item())
                pred_res=np.zeros(len(pred_list))
                for i in range(len(pred_res)):
                    pred_res[i]=max(pred_list[i])
                pred=pred_res.mean()
            except Exception as e:
                print(e)
                pred=0.5
            output_list.append(pred)


    you_auc = True
    try :
        auc=roc_auc_score(target_list_all,output_list)
    except: 
        auc = None
        you_auc = False
    # Convert output probabilities to binary predictions using a threshold (e.g., 0.5)
    binary_predictions = [1 if p >= 0.5 else 0 for p in output_list]

    # Calculate Accuracy
    accuracy = accuracy_score(target_list_all, binary_predictions)

    # Calculate Average Precision
    avg_precision = average_precision_score(target_list_all, output_list)

    # Calculate Average Recall
    avg_recall = recall_score(target_list_all, binary_predictions)

    if you_auc:
        print(f'{DATASETS}| AUC: {auc:.4f}, Accuracy: {accuracy:.4f}, Avg Precision: {avg_precision:.4f}, Avg Recall: {avg_recall:.4f}')
    else: 
        print(f'{DATASETS}| AUC: N/A (only one label), Accuracy: {accuracy:.4f}, Avg Precision:  N/A (only one label), Avg Recall:  N/A (only one label)')
    
    if (args.plot and you_auc):
        plot_dir = "figures"
        # --- ROC Curve Plot ---
        fpr, tpr, thresholds = roc_curve(target_list_all, output_list)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve for {DATASETS}')
        plt.legend(loc="lower right")
        roc_plot_filename = f'ROC_Curve_{DATASETS}_{os.path.basename(args.weight_name)}.png'
        plt.savefig(os.path.join(plot_dir, 'ROC', roc_plot_filename))
        plt.close()
        print(f"ROC curve saved to {roc_plot_filename}")

        # --- BPCER vs. APCER Plot ---
        bpcer_values = fpr  # BPCER is FPR
        apcer_values = 1 - tpr # APCER is FNR (1 - TPR)
        plt.figure(figsize=(8, 6))
        # Plot BPCER on x-axis and APCER on y-axis
        plt.plot(bpcer_values, apcer_values, color='red', lw=2, label='APCER vs. BPCER')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('BPCER (False Positive Rate)') # X-axis is BPCER
        plt.ylabel('APCER (False Negative Rate)') # Y-axis is APCER
        plt.title(f'APCER vs. BPCER for {DATASETS}') # Title reflects order
        plt.legend(loc="upper right") # Adjust legend location as needed, often top-right for error curves
        bpcer_apcer_plot_filename = f'APCER_BPCER_Plot_{DATASETS}_{os.path.basename(args.weight_name)}.png' # Consistent filename
        plt.savefig(os.path.join(plot_dir, 'bpcer_apcer', bpcer_apcer_plot_filename))
        plt.close()
        print(f"APCER vs. BPCER plot saved to {bpcer_apcer_plot_filename}")
    return auc, accuracy, avg_precision, avg_recall, target_list_all, output_list


if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-p', dest='plot', default = False)
    args=parser.parse_args()

    main(args)
    #raw ['akool', 'ibeta', 'vidnoz', 'ffcm_subset', 'gitw', 'FF', 'CDF']| AUC: 0.8387, Accuracy: 0.7353, Avg Precision: 0.9288, Avg Recall: 0.6939
    #c23 ['akool', 'ibeta', 'vidnoz', 'ffcm_subset', 'gitw', 'FF', 'CDF']| AUC: 0.8175, Accuracy: 0.7253, Avg Precision: 0.9179, Avg Recall: 0.6939
    #77  ['akool', 'ibeta', 'vidnoz', 'ffcm_subset', 'gitw', 'FF', 'CDF']| AUC: 0.8061, Accuracy: 0.7132, Avg Precision: 0.9125, Avg Recall: 0.6857