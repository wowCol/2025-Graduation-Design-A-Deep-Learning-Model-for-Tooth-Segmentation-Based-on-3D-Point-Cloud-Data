import sys
import os
from trimesh import PointCloud

sys.path.append(os.getcwd())
import glob
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import copy
import argparse
# 读入文件
import gen_utils as gu
import trimesh
from pathlib import Path

parser = argparse.ArgumentParser(description='inference models')
parser.add_argument('--mesh_path',
                    default="G:/tooth_seg/main/all_datas/chl/3D_scans_per_patient_obj_files/013FHA7K/013FHA7K_lower.obj",
                    type=str)
parser.add_argument('--gt_json_path',
                    default="G:/tooth_seg/main/all_datas/chl/ground-truth_labels_instances/013FHA7K/013FHA7K_lower.json",
                    type=str)
parser.add_argument('--pred_json_path', type=str, default="test_results/013FHA7K_lower.json")
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

        TP = np.count_nonzero(gt_mask * ins_mask)
        FN = np.count_nonzero(gt_mask * np.invert(ins_mask))
        FP = np.count_nonzero(np.invert(gt_mask) * ins_mask)
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


def np_to_pcd_local(arr, color=[1, 0, 0]):
    # print("Enter pcd change")
    arr = np.array(arr)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:, :3])
    if arr.shape[1] >= 6:
        pcd.normals = o3d.utility.Vector3dVector(arr[:, 3:6])
    pcd.colors = o3d.utility.Vector3dVector([color] * len(pcd.points))
    return pcd


def print_3d_local(*data_3d_ls):
    # print("Enter print")
    data_3d_ls = [item for item in data_3d_ls]
    for idx, item in enumerate(data_3d_ls):
        if type(item) == np.ndarray:
            data_3d_ls[idx] = np_to_pcd_local(item)
    o3d.visualization.draw_geometries(data_3d_ls, mesh_show_wireframe=True, mesh_show_back_face=True)


def read_txt_obj_ls_local(path, ret_mesh=False, use_tri_mesh=False):
    # print("Enter read")
    # In some cases, trimesh can change vertex order
    if use_tri_mesh:
        tri_mesh_loaded_mesh = trimesh.load_mesh(path, process=False)
        vertex_ls = np.array(tri_mesh_loaded_mesh.vertices)
        tri_ls = np.array(tri_mesh_loaded_mesh.faces) + 1
    else:
        f = open(path, 'r')
        vertex_ls = []
        tri_ls = []
        # vertex_color_ls = []
        while True:
            line = f.readline().split()
            if not line: break
            if line[0] == 'v':
                vertex_ls.append(list(map(float, line[1:4])))
                # vertex_color_ls.append(list(map(float,line[4:7])))
            elif line[0] == 'f':
                tri_verts_idxes = list(map(str, line[1:4]))
                if "//" in tri_verts_idxes[0]:
                    for i in range(len(tri_verts_idxes)):
                        tri_verts_idxes[i] = tri_verts_idxes[i].split("//")[0]
                tri_verts_idxes = list(map(int, tri_verts_idxes))
                tri_ls.append(tri_verts_idxes)
            else:
                continue
        f.close()

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_ls) - 1)
    mesh.compute_vertex_normals()

    norms = np.array(mesh.vertex_normals)

    vertex_ls = np.array(vertex_ls)
    output = [np.concatenate([vertex_ls, norms], axis=1)]

    if ret_mesh:
        output.append(mesh)
    return output


def get_colored_mesh_local(mesh, label_arr):
    # print("Enter draw color")
    palte = np.array([
        [255, 153, 153],  # 浅红色,牙龈的颜色

        [153, 76, 0],  # 棕色
        [255, 20, 147],  # 深粉色
        [76, 153, 0],  # 绿色
        [0, 153, 153],  # 青色
        [0, 0, 153],  # 蓝色
        [153, 0, 153],  # 紫色
        [255, 255, 0],  # 黄色
        [64, 64, 64],  # 深灰色

        [0, 255, 127],  # 浅绿色
        [139, 0, 0],  # 深红色
        [0, 104, 139],  # 深天蓝色
        [104, 34, 139],  # 深兰花色（深紫色）
        [205, 133, 0],  # 橙色
        [131, 111, 255],  # 板条蓝
        [255, 174, 185],  # 浅粉色
        [255, 106, 106],  # 印度红
    ]) / 255
    # [153, 0, 153],
    # [255, 255, 0],
    # [64, 64, 64],
    # palte[9:] *= 0.4
    label_arr = label_arr.copy()
    # 根据标签值依次分配颜色:
    # 板条蓝 橙色 深紫色 深天蓝色 深红色 浅绿色 深灰色 | 棕色 深粉色 绿色 青色 蓝色 紫色 黄色
    # 右上 | 左上 （不记录智齿）
    # (18) 17 16 15 14 13 12 11 21 22 23 24 25 26 27 (28)
    # 右下 | 左下
    # (48) 47 46 45 44 43 42 41 31 32 33 34 35 36 37 (48)
    # 设置数列
    label_colors = [-1] * 50
    # 设置牙龈为浅红色
    label_colors[0] = palte[0]
    # 左上
    for i in range(11, 18):
        label_colors[i] = palte[i-10]
    # 右上
    for i in range(21, 28):
        label_colors[i] = palte[i-20+7]
    # 左下
    for i in range(31, 38):
        label_colors[i] = palte[i-30]
    # 左下
    for i in range(41, 48):
        label_colors[i] = palte[i-40+7]

    # label_arr %= palte.shape[0]
    label_arr_uni = np.unique(label_arr)
    # mapping = {val: palte[idx] for idx, val in enumerate(label_arr_uni)}
    label_colors_list = np.zeros((label_arr.shape[0], 3))
    count = 1
    for num in label_arr_uni:
        # print(f"当前索引为:{num}, 对应颜色RGB为:{label_colors[num]*255} --- {count}")
        label_colors_list[label_arr == num] = label_colors[num]
        count += 1
    # print(len(label_colors))
    # print(label_colors[0])
    # exit()
    mesh.vertex_colors = o3d.utility.Vector3dVector(label_colors_list)
    return mesh


def save_colored_mesh_ply(mesh, filename="output.ply"):
    # 将得到的网格输出为ply文件
    # 确保 mesh 具有颜色信息
    if not hasattr(mesh, "vertex_colors") or len(mesh.vertex_colors) == 0:
        print("网格没有颜色，可能会导致颜色信息丢失！")

    # 写入 PLY 文件
    o3d.io.write_triangle_mesh(filename, mesh)
    # print(f"已保存带颜色的网格为 {filename}")

def read_output_results(path: str, output: str,fullname: str ,type = 0):
    if type == 0:
        if fullname:
            print("### 单个文件真实标注的ply文件展示 ###")
            # 读取单个文件
            name = fullname.split('_')[0]

            # 读取所有文件求平均IoU等参数
            inference_path_local = f"run/inference/{fullname}.json"
            obj_path = f"../../dataset/data_obj_parent_directory/{name}/{fullname}.obj"
            truth_path = f"../../dataset/data_json_parent_directory/{name}/{fullname}.json"

            # 原读取单个文件部分
            gt_loaded_json = gu.load_json(truth_path)
            gt_labels = np.array(gt_loaded_json['labels']).reshape(-1)
            # print(gt_labels)
            print(f"真实标注的点数为：{len(gt_labels)}")

            pred_loaded_json = gu.load_json(inference_path_local)
            pred_labels = np.array(pred_loaded_json['labels']).reshape(-1)
            # print(pred_labels.shape)
            print(f"推理出的点数为：{len(pred_labels)}")
            # unique_arr = np.unique(pred_labels)
            # print(f"共有{len(unique_arr)}个分组")
            # for i in range(len(unique_arr)):
            #     print(unique_arr[i])
            # IoU, F1, Acc, SEM_ACC, _ = cal_metric(gt_labels, pred_labels, pred_labels) # F1 -> TSA, SEM_ACC -> TIR
            # print("IoU", IoU, "F1(TSA)", F1, "SEM_ACC(TIR)", SEM_ACC)

            input_nparray, mesh = read_txt_obj_ls_local(obj_path, ret_mesh=True, use_tri_mesh=True)

            # get_colored_mesh_local(mesh, pred_labels)
            print_3d_local(get_colored_mesh_local(mesh, gt_labels))
            print_3d_local(get_colored_mesh_local(mesh, pred_labels))

            exit(0)
        else:
            print("选择了读取单个文件模式但是没有输入文件名！")
            exit(0)
    else:
        # 读取所有文件
        folder_path = Path(path)
        all_file_names = [f.stem for f in folder_path.iterdir() if f.is_file()]
        if type == 1:  # 真实标注
            print("### 真实标注的ply文件输出 ###")
            count = 1
            for file_name in all_file_names:
                # 输出到指定目录
                ply_output_path = f"{output}/{file_name}.ply"
                name = file_name.split("_")[0]
                obj_path = f"../../dataset/data_obj_parent_directory/{name}/{file_name}.obj"

                _, mesh = read_txt_obj_ls_local(obj_path, ret_mesh=True, use_tri_mesh=True)

                label_path_local = f"../../dataset/data_json_parent_directory/{name}/{file_name}.json"

                pred_loaded_json = gu.load_json(label_path_local)
                pred_labels = np.array(pred_loaded_json['labels']).reshape(-1)

                # 将上色后的mesh输出为ply文件
                get_colored_mesh_local(mesh, pred_labels)
                save_colored_mesh_ply(mesh, ply_output_path)
                print(f"完成{file_name}.ply输出 -------- {count}")
                count += 1
        else:  # 模型推理标注
            print("### 推理模型标注的ply文件输出 ###")
            count = 1
            for file_name in all_file_names:
                # 输出到指定目录
                ply_output_path = f"{output}/{file_name}.ply"
                name = file_name.split("_")[0]
                obj_path = f"../../dataset/data_obj_parent_directory/{name}/{file_name}.obj"

                _, mesh = read_txt_obj_ls_local(obj_path, ret_mesh=True, use_tri_mesh=True)

                label_path_local = f"run/inference/{file_name}.json"

                pred_loaded_json = gu.load_json(label_path_local)
                pred_labels = np.array(pred_loaded_json['labels']).reshape(-1)

                # 将上色后的mesh输出为ply文件
                get_colored_mesh_local(mesh, pred_labels)
                save_colored_mesh_ply(mesh, ply_output_path)
                print(f"完成{file_name}.ply输出 -------- {count}")
                count += 1

        print(f"完成共计{len(all_file_names)}个文件的PLY文件输出")


if __name__ == "__main__":
    # 0AAQ6BO3_lower, 0EJBIPTC_lower, 01A6GW4A_upper, 0165W7J4_lower
    fullname = "0EJBIPTC_lower"
    inference_path = "run/inference"
    output_path = "../../backup/objFiles/02"

    read_output_results(inference_path, output_path, fullname, 2)

