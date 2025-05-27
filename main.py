from model.main_model import train
from logger import get_logger

if __name__ == '__main__':
    logger = get_logger("main")
    try:
        trainer = train(1, ".\\checkpoints")
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
