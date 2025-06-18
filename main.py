from logger import get_logger
from model.main_model import train

if __name__ == '__main__':
    logger = get_logger("main")
    dataset_path = "/home/Datasets/bionet/Dataset/lsj_MICCAI_BraTS2020_TrainingData/"
    trainer = train(dataset_path, devices_numbers=[0])
    logger.info("Training completed successfully")


