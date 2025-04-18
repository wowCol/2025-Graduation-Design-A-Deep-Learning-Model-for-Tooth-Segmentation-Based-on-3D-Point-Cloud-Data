import shutil

source_folder = "../../../autodl-fs/dataset"
destination_folder  = "../../../autodl-tmp/dataset"

shutil.copytree(source_folder, destination_folder)