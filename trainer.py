import torch
import wandb
from loss_meter import LossMeter
from math import inf


class Trainer:
    def __init__(self, config=None, model=None, gen_set=None):
        self.gen_set = gen_set
        self.config = config
        self.model = model

        self.val_count = 0
        self.train_count = 0
        self.step_count = 0
        if config["wandb"]["wandb_on"]:
            wandb.init(
                entity=self.config["wandb"]["entity"],
                project=self.config["wandb"]["project"],
                notes=self.config["wandb"]["notes"],
                tags=self.config["wandb"]["tags"],
                name=self.config["wandb"]["name"],
                config=self.config,
            )

        self.best_val_loss = inf
        # 设置早停机制
        # 设置耐心值，即在验证集损失没有提升的情况下，最多等待 20 个 epoch
        self.patience = 20
        # 记录没有提升的 epoch 数
        self.epochs_without_improvement = 0
        # 是否提前停止训练
        self.early_stop = False
        # 记录保存的最佳模型序号
        self.best_model_num = 0

    def train(self, epoch, data_loader):
        # 记录整个 epoch 所有 batch 的平均损失, 所有子损失加权后的总 loss
        # 　LossMeter 是自定义类，用于累积和平均损失值。
        total_loss_meter = LossMeter()
        # 　记录当前 scheduler step 内的平均损失
        step_loss_meter = LossMeter()
        pre_step = self.step_count

        # // 测试用 //
        count = 0
        # 记录当前的训练步数，用于判断是否需要调度学习率。
        for batch_idx, batch_item in enumerate(data_loader):
            loss = self.model.step(batch_idx, batch_item, "train")
            # 　清空内存与缓存
            torch.cuda.empty_cache()
            total_loss_meter.aggr(loss.get_loss_dict_for_print("train"))
            step_loss_meter.aggr(loss.get_loss_dict_for_print("step"))
            print(loss.get_loss_dict_for_print("step"))
            # 　判断是否需要进行一个调度 step（每 N 次 batch 或最后一批）
            if ((batch_idx + 1) % self.config["tr_set"]["scheduler"]["schedueler_step"] == 0) or (
                    self.step_count == pre_step and batch_idx == len(data_loader) - 1):
                if self.config["wandb"]["wandb_on"]:
                    wandb.log(step_loss_meter.get_avg_results(), step=self.step_count)
                    wandb.log({"step_lr": self.model.scheduler.get_last_lr()[0]}, step=self.step_count)
                self.step_count += 1
                self.model.scheduler.step(self.step_count)
                step_loss_meter.init()

            count += 1
            # 测试运行只运行8次
            if count > 8 :
                break

        if self.config["wandb"]["wandb_on"]:
            wandb.log(total_loss_meter.get_avg_results(), step=self.step_count)
            self.train_count += 1
        # 保存当前模型为train
        # self.model.save("train")

    def test(self, epoch, data_loader, save_best_model):
        # 所有子损失加权后的总 loss
        total_loss_meter = LossMeter()
        for batch_idx, batch_item in enumerate(data_loader):
            loss = self.model.step(batch_idx, batch_item, "test")
            total_loss_meter.aggr(loss.get_loss_dict_for_print("val"))

        # 验证集上所有 batch 的平均损失
        avg_total_loss = total_loss_meter.get_avg_results()
        if self.config["wandb"]["wandb_on"]:
            wandb.log(avg_total_loss, step=self.step_count)
            self.val_count += 1

        if self.best_val_loss > avg_total_loss["total_val"]:
            self.best_val_loss = avg_total_loss["total_val"]
            self.model.save("val")

            # 记录最佳模型的次序
            self.best_model_num = self.step_count
            if save_best_model:
                # 如果有提升，重置计数器
                self.epochs_without_improvement = 0
            else:
                # 如果没有提升，增加计数器
                self.epochs_without_improvement += 1

            # 如果验证损失没有提升超过耐心值，提前停止训练
            if self.epochs_without_improvement >= self.patience:
                print("Early stopping due to no improvement in validation loss for recent 20 times.")
                self.early_stop = True

    def train_depr(self):
        total_loss = 0

        step_loss = 0
        for batch_idx, batch_item in enumerate(self.train_loader):
            loss = self.model.step(batch_idx, batch_item, "train")
            total_loss += loss
            step_loss += loss
            if (batch_idx + 1) % self.config["tr_set"]["schedueler_step"] == 0:
                self.model.scheduler.step()
                step_loss /= self.config["tr_set"]["schedueler_step"]
                if self.config["wandb"]["wandb_on"]:
                    wandb.log({"step_train_loss": step_loss})
                step_loss = 0
        total_loss /= len(self.train_loader)
        if self.config["wandb"]["wandb_on"]:
            wandb.log({"train_loss": total_loss})
        self.model.save("train")

    def test_depr(self):
        total_loss = 0
        for batch_idx, batch_item in enumerate(self.val_loader):
            loss = self.model.step(batch_idx, batch_item, "test")
            total_loss += loss
        total_loss /= len(self.val_loader)
        if self.config["wandb"]["wandb_on"]:
            wandb.log({"val_loss": total_loss})

        if self.best_val_loss > total_loss:
            self.best_val_loss = total_loss
            self.model.save("val")

    def run(self):
        # Mini-Batch 梯度下降 + 每个 epoch 后验证集评估 + 保存最佳模型 + 支持早停（Early Stopping）
        train_data_loader = self.gen_set[0][0]
        val_data_loader = self.gen_set[0][1]
        # Batch（批次） 指的是在训练神经网络时，我们不是一次用整个训练集去训练模型，而是每次取出一小部分数据样本（一个 batch）来训练一次
        # 修改卷积次数为200, 原为100000
        for epoch in range(1):
            # 如果检测到早停信号，停止训练
            if self.early_stop:
                print("Training stopped early.")
                return [self.step_count, self.early_stop, self.best_model_num]
            self.train(epoch, train_data_loader)
            # self.test(epoch, val_data_loader, False)

        print(self.step_count, ";", self.early_stop, ";", self.best_model_num)
        return [self.step_count, self.early_stop, self.best_model_num]
