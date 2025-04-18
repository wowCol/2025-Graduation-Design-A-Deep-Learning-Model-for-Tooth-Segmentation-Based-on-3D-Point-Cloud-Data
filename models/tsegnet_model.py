import torch
from . import tsg_loss
from models.base_model import BaseModel
from loss_meter import LossMap
from external_libs.pointnet2_utils.pointnet2_utils import square_distance
import ops_utils as ou

class TSegNetModel(BaseModel):
    # 继承自 BaseModel，说明模型结构、优化器、调度器等是在基类里统一封装好的。
    def __init__(self, config, module):
        super().__init__(config, module)
        # config 是否启用预训练模型、是否启用分割模块等。module 实际的神经网络模块
        if self.config.get("pretrained_centroid_model_path", None) not in [None,""]:
            self.module.load_state_dict(torch.load(self.config["pretrained_centroid_model_path"] +".h5"), strict=False)

    def get_loss(self, outputs, gt):
        losses = {} # 计算多种损失，输出一个字典。

        # // 质心模型的训练 //
        # offset_result: 模型预测的中心偏移量, l3_xyz: 第3层的点坐标,
        # dist_result: 距离分数, gt["centroid_coords"]: 真实的中心点坐标。
        dist_loss, cent_loss, chamf_loss = tsg_loss.centroid_loss(
            outputs["offset_result"], outputs["l3_xyz"], outputs["dist_result"], gt["centroid_coords"]
        )
        # dist_loss: 距离损失（点是否靠近中心）, cent_loss: 中心偏移损失, chamf_loss: Chamfer 距离约束（点云之间形状距离）
        # 加权方式：Chamfer 损失权重为 0.1，其他为 1。
        losses.update({
            "dist_loss": (dist_loss, 1),
            "cent_loss": (cent_loss, 1),
            "chamf_loss": (chamf_loss, 0.1),
        })
        # 如果配置里不启用牙齿分割模块，直接返回中心点损失。
        if self.config["run_tooth_segmentation_module"] is False: return losses
        # // 牙齿模型分割部分 //
        # 将预测中心点与最近的真实中心点对齐，匹配标签。
        # 计算每个中心点与 GT 的距离
        sqd = square_distance(torch.from_numpy(outputs["center_points"]).cuda(),gt["centroid_coords"].permute(0,2,1))  # 1, N, 3 X 1, M, 3 => 1, N, M
        # # 最近 GT 中心的索引, 设置为1维数组
        sqd_argmin =  sqd.argmin(axis=2).reshape(-1)
        pred_centerpoint_gt_label_ls = gt["centroid_labels"][:, sqd_argmin] # 1, N, M => 1, N

        # 创建一个 GT 的二值标签图
        cluster_gt_seg_bin_label_ls = torch.zeros_like(outputs["cluster_gt_seg_label"]).cuda()
        for i in range(outputs["cluster_gt_seg_label"].shape[0]):
            cluster_gt_seg_bin_label_ls[i, 0, pred_centerpoint_gt_label_ls[0][i]==outputs["cluster_gt_seg_label"][i][0]+1] = 1

        # (seg_1_loss 第一阶段牙齿二分类损失, 二值：当前聚类点是否属于目标牙齿, 用于训练聚类模块判断某个点属于哪个牙齿类别。)
        # (seq_2_loss 第二阶段精细分割损失, 细粒度地判断点属于哪个牙齿区域, 在第一阶段基础上进一步细化。)
        # (id_pred_loss 牙齿编号预测损失, 类似于分类任务，让模型预测每个牙齿聚类所属的编号, 用于训练模型给每个牙齿一个实际编号，提升结果解释性。)
        # 分别计算三个分割损失
        seg_1_loss, seg_2_loss, id_pred_loss = tsg_loss.segmentation_loss(outputs["pd_1"], outputs["weight_1"], outputs["pd_2"], outputs["id_pred"],
        pred_centerpoint_gt_label_ls, cluster_gt_seg_bin_label_ls)
        losses.update({
            "seg_1_loss":(seg_1_loss,1), 
            "seg_2_loss":(seg_2_loss,1), 
            "id_pred_loss":(id_pred_loss,1)
        })
        
        return losses

    # 单个训练过程
    def step(self, batch_idx, batch_item, phase):
        # 设置模型状态
        self._set_model(phase)
        # B, 一次训练中有多少个样本（点云）。
        # C, 每个点的通道数（特征维度）
        # N, 每个样本中的点数
        B, C, N = batch_item["feat"].shape

        print("Get: ", B, " ", C, " ", N)

        # 准备标签
        # 把 GT 分割标签转化为“真实中心点坐标”和是否存在中心, 标签范围是 1～16
        gt_centroid_coords, gt_centroid_exists = ou.seg_label_to_cent(batch_item["feat"][:,:3,:], batch_item["gt_seg_label"])

        # 根据 gt_centroid_exists 筛选实际存在的中心点
        # 修改range(0, 15)为arange(0, 16)
        gt_centroids_label = torch.arange(0, 16).view(1,-1).cuda() + 1
        gt_centroid_exists = gt_centroid_exists.view(1, -1)
        gt_centroid_coords = gt_centroid_coords.permute(0,2,1)
        gt_centroid_coords = gt_centroid_coords[gt_centroid_exists>0, :]
        gt_centroids_label = gt_centroids_label[gt_centroid_exists>0]
        gt_centroid_coords = gt_centroid_coords.unsqueeze(dim=0)
        gt_centroids_label = gt_centroids_label.unsqueeze(dim=0)
        gt_centroid_coords = gt_centroid_coords.permute(0,2,1)
        gt_centroid_coords = gt_centroid_coords.cuda() # B, 3, 14
        gt_centroids_label = gt_centroids_label.cuda() # B, 14

        points = batch_item["feat"].cuda()
        l0_xyz = batch_item["feat"][:,:3,:].cuda()
        
        seg_label = batch_item["gt_seg_label"].cuda()
        
        inputs = [points, seg_label]

        # 输入网络, 得到输出
        # 前向传播	输入 → 模型 → 输出	得到模型的预测结果
        if phase == "train":
            output = self.module(inputs)
        else:
            with torch.no_grad():
                output = self.module(inputs)
        loss_meter = LossMap()

        #  损失计算并统计
        loss_meter.add_loss_by_dict(self.get_loss(
            output,
            {
                "seg_label": seg_label,  # 点到预测中心点的距离损失
                "centroid_coords": gt_centroid_coords,  # 中心点存在与否的置信度损失
                "centroid_labels": gt_centroids_label,  # 预测中心点集和真实中心点集的 Chamfer 距离
            } 
            )
        )

        # 训练时反向传播
        # 反向传播	损失 → 梯度传播	更新模型参数，减少预测误差
        if phase == "train":
            loss_sum = loss_meter.get_sum()
            self.optimizer.zero_grad()
            loss_sum.backward()
            self.optimizer.step()

        return loss_meter
    

    def infer(self, batch_idx, batch_item, **options):
        pass