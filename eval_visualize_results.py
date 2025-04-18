import sys
import os
from trimesh import PointCloud

sys.path.append(os.getcwd())
import glob
import gen_utils as gu
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import copy
import argparse

parser = argparse.ArgumentParser(description='inference models')
parser.add_argument('--mesh_path', type=str)
parser.add_argument('--gt_json_path', type=str)
parser.add_argument('--pred_json_path', type=str)
args = parser.parse_args()


def cal_metric(gt_labels, pred_sem_labels, pred_ins_labels, is_half=None, vertices=None):
    ins_label_names = np.unique(pred_ins_labels)
    ins_label_names = ins_label_names[ins_label_names != 0]
    IOU = 0
    F1 = 0
    ACC = 0
    SEM_ACC = 0
    IOU_arr = []
    for ins_label_name in ins_label_names:
        # instance iou
        ins_label_name = int(ins_label_name)
        ins_mask = pred_ins_labels == ins_label_name
        gt_label_uniqs, gt_label_counts = np.unique(gt_labels[ins_mask], return_counts=True)
        gt_label_name = gt_label_uniqs[np.argmax(gt_label_counts)]
        gt_mask = gt_labels == gt_label_name

        # 关注预测标签11
        # 真阳性，预测和实际都是11
        TP = np.count_nonzero(gt_mask * ins_mask)
        # 假阳性，预测11，实际非（错误识别为这颗牙的点（别的牙被误认成它））
        FN = np.count_nonzero(gt_mask * np.invert(ins_mask))
        # 假阴性，实际11，预测非（模型漏掉的该牙点（没识别出来））
        FP = np.count_nonzero(np.invert(gt_mask) * ins_mask)
        # 真阴性，预测和实际都不是11（无关区域都预测正确（背景、其他牙没误判））
        TN = np.count_nonzero(np.invert(gt_mask) * np.invert(ins_mask))

        ACC += (TP + TN) / (FP + TP + FN + TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 += 2 * (precision * recall) / (precision + recall)
        IOU += TP / (FP + TP + FN)
        IOU_arr.append(TP / (FP + TP + FN))
        # segmentation accuracy
        pred_sem_label_uniqs, pred_sem_label_counts = np.unique(pred_sem_labels[ins_mask], return_counts=True)
        sem_label_name = pred_sem_label_uniqs[np.argmax(pred_sem_label_counts)]
        if is_half:
            if sem_label_name == gt_label_name or sem_label_name + 8 == gt_label_name:
                SEM_ACC += 1
        else:
            if sem_label_name == gt_label_name:
                SEM_ACC += 1
        # print("gt is", gt_label_name, "pred is", sem_label_name, sem_label_name == gt_label_name)
    return IOU / len(ins_label_names), F1 / len(ins_label_names), ACC / len(ins_label_names), SEM_ACC / len(
        ins_label_names), IOU_arr


# # 原读取单个文件部分
# gt_loaded_json = gu.load_json(args.gt_json_path)
# gt_labels = np.array(gt_loaded_json['labels']).reshape(-1)
#
# pred_loaded_json = gu.load_json(args.pred_json_path)
# pred_labels = np.array(pred_loaded_json['labels']).reshape(-1)
#
# IoU, F1, Acc, SEM_ACC, _ = cal_metric(gt_labels, pred_labels, pred_labels) # F1 -> TSA, SEM_ACC -> TIR
# print("IoU", IoU, "F1(TSA)", F1, "SEM_ACC(TIR)", SEM_ACC)
# _, mesh = gu.read_txt_obj_ls(args.mesh_path, ret_mesh=True, use_tri_mesh=True)
# gu.print_3d(gu.get_colored_mesh(mesh, gt_labels)) # color is random
# gu.print_3d(gu.get_colored_mesh(mesh, pred_labels)) # color is random


# 读取所有文件求平均IoU等参数
inference_path = "run/inference/"
obj_path = "../../dataset/data_obj_parent_directory/"
truth_path = "../../dataset/data_json_parent_directory/"

files = glob.glob(f"{inference_path}/*")
total_IoU = []
total_F1 = []
total_TIR = []
total_Acc = []

for file in files:
    if os.path.isfile(file):
        file_name = os.path.basename(file)
        file_stem = os.path.splitext(file_name)[0]
        # 整合单个文件的IoU
        # print(inference_path + file_name)
        first_name = file_name.split("_")[0]
        # 原标注部分
        gt_loaded_json = gu.load_json(f"{truth_path}/{first_name}/{file_name}")
        gt_labels = np.array(gt_loaded_json['labels']).reshape(-1)
        # 推理部分
        pred_loaded_json = gu.load_json(f"{inference_path}/{file_name}")
        pred_labels = np.array(pred_loaded_json['labels']).reshape(-1)

        IoU, F1, Acc, SEM_ACC, _ = cal_metric(gt_labels, pred_labels, pred_labels)  # F1 -> TSA, SEM_ACC -> TIR
        total_IoU.append(IoU)
        total_F1.append(F1)
        total_Acc.append(Acc)
        total_TIR.append(SEM_ACC)

        print(f"文件{file_name}的IoU: {IoU},F1: {F1},ACC: {Acc}, SEM_ACC: {SEM_ACC}")

# 输出平均值
print()
print(f"IoU: {np.mean(total_IoU) * 100:.2f}%")
print(f"F1: {np.mean(total_F1) * 100:.2f}%")
print(f"SEM_ACC(TIR): {np.mean(total_TIR) * 100:.2f}%")
print(f"Acc: {np.mean(total_Acc) * 100:.2f}%")
