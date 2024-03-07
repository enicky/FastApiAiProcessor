from datetime import datetime
import logging, coloredlogs
import matplotlib.pyplot as plt
from business.trainers.lstm import LstmTrainer
import os

logger = logging.getLogger(__name__)
coloredlogs.install("DEBUG", logger = logger)

class Ai(object):
    def __init__(self) -> None:
        self.trainer= LstmTrainer()
        pass
    
    def start_training(self, id: str, data_path=None):
        formatted_date = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
        MODELS_PATH = os.getenv('AI_MODELS_PATH', 'models/')
        if data_path is None:
            data_path = os.getenv('DATA_PATH', 'data')
        logger.debug(f'[{id}] Start training a model on data path "{data_path}" and use "{formatted_date}" as model name')
        logger.debug(f'[{id}] MODELS_PATH = {MODELS_PATH}')
            

        
        model_file_name = f"{MODELS_PATH}model_epoch_{formatted_date}.pt"
        model_image_file_name =  f"model_x{formatted_date}"
            
        x, train_loss_list, test_loss_list = self.trainer.process(id, model_file_name)
        plt.figure(figsize=(8, 6))
            
        # # Plotting the values after the loop ends
        #plt.figure(figsize=(8, 6))
        plt.plot(train_loss_list, label='Train Loss')
        plt.plot(test_loss_list, label='Test Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss Comparison - Stage Ivan Groffils')
        plt.legend()
        plt.grid(True)
        #plt.show()
        plt.savefig(f"{MODELS_PATH}{model_image_file_name}.png")
            