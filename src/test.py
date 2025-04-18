# // torch测试部分 //
# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.version.cuda)
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_capability())
# // 训练结束邮件通知部分 //
# import os
# import torch
#
# # 获取当前脚本所在的目录
# current_path = os.path.dirname(os.path.abspath(__file__))
#
# # 获取当前脚本所在的项目根目录
# root_path = os.path.dirname(current_path)
#
# print("项目根目录路径：", root_path)
# print(torch.__version__)
# print(torch.cuda.is_available())
#
# for i in range(100):
#     if (i+1) % 20 != 0:
#         print(f"{i}\t", end="")
#     else:
#         print(f"{i}\t")
#
# os.system('shutdown -h now')
# import datetime
# import smtplib
# beginClock = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# print(beginClock)
# from email.mime.text import MIMEText
# runtimes = 200
# # 运行完关机
# print(f"Train success, shutdown computer.")
# # 使用QQ邮箱提醒完成
# sender = user = 'wqy2693699654@qq.com'
# passwd = 'rxegdqksnrlydefe'
#
# receiver = 'wqy2693699654@qq.com'
#
# # msg = MIMEText(f'服务器上运行的训练程序已经完成，时间为: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n总计训练次数为: {runtimes}\n服务器即将关机。~o( =∩ω∩= )m', 'plain', 'utf-8')
# msg = MIMEText(f'服务器上运行的训练程序已经完成，时间为: '
#                f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
#                f'总计训练次数为: {runtimes}\n'
#                f'服务器即将关机。~o( =∩ω∩= )m', 'plain', 'utf-8')
# msg['From'] = f'Myself<wqy2693699654@qq.com>'
# msg['To'] = receiver
# msg['Subject'] = '服务器完成训练邮件通知'
#
# try:
#     smtp = smtplib.SMTP_SSL('smtp.qq.com', 465)
#
#     smtp.login(user, passwd)
#
#     smtp.sendmail(sender, receiver, msg.as_string())
#     print("完成邮件发送")
#     smtp.quit()
# except Exception as e:
#     print(e)
#     print("发送邮件失败")
# // 文件检测部分 //
# from pathlib import Path
# folder_path = Path("../run/inference")
#
# files = []
# for f in folder_path.iterdir():
#     if f.is_file():
#         files.append(f.stem)
# print(files)
# // 3D网格上色部分 //
# import open3d as o3d
# import numpy as np
#
# # 创建一个简单的三角形网格
# mesh = o3d.geometry.TriangleMesh()
# mesh.vertices = o3d.utility.Vector3dVector(np.array([
#     [0, 0, 0],  # 顶点 1
#     [1, 0, 0],  # 顶点 2
#     [0, 1, 0]   # 顶点 3
# ]))
# mesh.triangles = o3d.utility.Vector3iVector(np.array([
#     [0, 1, 2]   # 三角形
# ]))
#
# # 生成颜色数据 (3 个顶点, 每个颜色为 RGB)
# label_colors = np.array([
#     [1.0, 0.0, 0.0],  # 红色
#     [0.0, 1.0, 0.0],  # 绿色
#     [0.0, 0.0, 1.0]   # 蓝色
# ])
#
# # 将颜色应用到网格
# mesh.vertex_colors = o3d.utility.Vector3dVector(label_colors)
#
# # 可视化
# o3d.visualization.draw_geometries([mesh])
# // npy文件读取部分 //
# import numpy as np
#
# # 读取 .npy 文件
# data = np.load("../run/preprocessed_data/0EJBIPTC_upper_upper_sampled_points.npy")
#
# # 输出数据内容和形状
# print(data)
# print(data.shape)

