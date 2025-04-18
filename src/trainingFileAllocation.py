import os
import random

# 设置文件夹路径
folder_path = "../../../dataset/data_json_parent_directory"

# 获取文件夹中所有的子文件夹名
subfolder_names = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

# 打乱文件夹名列表
random.shuffle(subfolder_names)

# 计算各部分的文件夹数
total_folders = len(subfolder_names)
train_count = int(total_folders * 0.7)
val_count = int(total_folders * 0.2)
test_count = total_folders - train_count - val_count  # 剩下的为测试集

# 将文件夹名分割成 训练集、验证集和测试集
train_folders = subfolder_names[:train_count]
val_folders = subfolder_names[train_count:train_count + val_count]
test_folders = subfolder_names[train_count + val_count:]

# 将文件夹名写入三个不同的 txt 文件
with open("base_name_train_fold.txt", "w") as f_train:
    f_train.write("\n".join(train_folders))

with open("base_name_val_fold.txt", "w") as f_val:
    f_val.write("\n".join(val_folders))

with open("base_name_test_fold.txt", "w") as f_test:
    f_test.write("\n".join(test_folders))

print("文件夹名已成功写入到三个txt文件中：base_name_train_fold.txt, base_name_val_fold.txt, base_name_test_fold.txt")