import smtplib

from runner import runner
from train_configs import train_config_maker
import argparse
# 导入关机与提醒文件
from src.shutdown_alert import alert_sd
import datetime

parser = argparse.ArgumentParser(description='inference models')
# parser.add_argument('--model_name', default="tsegnet", type=str, help = "model name. list: tsegnet | tgnet_fps/tgnet_bdl | pointnet | pointnetpp | dgcnn | pointtransformer")
# parser.add_argument('--config_path', default="train_configs/tsegnet.py", type=str, help = "train config file path.")
# parser.add_argument('--experiment_name', default="tsegnet_0620", type=str, help = "experiment name.")
# parser.add_argument('--input_data_dir_path', default="data_preprocessed_path", type=str, help = "input data dir path.")
# parser.add_argument('--train_data_split_txt_path', default="base_name_train_fold.txt", type=str, help = "train cases list file path.")
# parser.add_argument('--val_data_split_txt_path', default="base_name_val_fold.txt", type=str, help = "val cases list file path.")
parser.add_argument('--model_name', required=True, type=str, help = "model name. list: tsegnet | tgnet_fps/tgnet_bdl | pointnet | pointnetpp | dgcnn | pointtransformer")
parser.add_argument('--config_path', required=True, type=str, help = "train config file path.")
parser.add_argument('--experiment_name', required=True, type=str, help = "experiment name.")
parser.add_argument('--input_data_dir_path', required=True, type=str, help = "input data dir path.")
parser.add_argument('--train_data_split_txt_path', required=True, type=str, help = "train cases list file path.")
parser.add_argument('--val_data_split_txt_path', required=True, type=str, help = "val cases list file path.")

# 记录开始训练的时间
beginClock = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

args = parser.parse_args()

config = train_config_maker.get_train_config(
    args.config_path,
    args.experiment_name,
    args.input_data_dir_path,
    args.train_data_split_txt_path,
    args.val_data_split_txt_path,
)

if args.model_name == "tgnet_fps":
    from models.fps_grouping_network_model import FpsGroupingNetworkModel
    from models.modules.grouping_network_module import GroupingNetworkModule
    model = FpsGroupingNetworkModel(config, GroupingNetworkModule)
if args.model_name == "tsegnet":
    from models.tsegnet_model import TSegNetModel
    from models.modules.tsegnet import TSegNetModule
    model = TSegNetModel(config, TSegNetModule)
elif args.model_name == "dgcnn":
    from models.dgcnn_model import DGCnnModel
    from models.modules.dgcnn import DGCnnModule
    model = DGCnnModel(config, DGCnnModule)
elif args.model_name == "pointnet":
    from models.pointnet_model import PointFirstModel
    from models.modules.pointnet import PointFirstModule
    model = PointFirstModel(config, PointFirstModule)
elif args.model_name == "pointnetpp":
    from models.pointnet_pp_model import PointPpFirstModel
    from models.modules.pointnet_pp import PointPpFirstModule
    model = PointPpFirstModel(config, PointPpFirstModule)
elif args.model_name == "pointtransformer":
    from models.transformer_model import TransformerModel
    from models.modules.point_transformer import PointTransformerModule
    model = TransformerModel(config, PointTransformerModule)
elif args.model_name == "tgnet_bdl":
    from models.bdl_grouping_netowrk_model import BdlGroupingNetworkModel
    from models.modules.grouping_network_module import GroupingNetworkModule
    model = BdlGroupingNetworkModel(config, GroupingNetworkModule)

# 添加卷积次数统计
data_list = runner(config, model)

# 关机并提示
# alert_sd(beginClock, data_list)
