from business.trainers.datacontroller import DataController
from business.trainers.sequencedataset import SequenceDataset
from business.trainers.shallowregressionlstm import ShallowRegressionLSTM
from business.trainers.modelcontroller import ModelController
from torch.utils.data import DataLoader
from torch import nn
import logging, coloredlogs
import torch
import os

logger = logging.getLogger(__name__)
coloredlogs.install('DEBUG', logger=logger)

target_sensor = "energy_power" 
forecast_follow = 21
period_between_each_row = "30min"
forecast_col = "Model forecast"
columns_features = ["bme_temp", "bme_hum", "hour", "day_of_week", "day_of_year"]
folder_path  = os.getenv('MODELS_PATH', "/files")
learning_rate = 0.001#5e-5
num_hidden_units = 16
model_train_epoch_count = 530
train_test_split_factor = 0.80
batch_size = 64
sequence_length = 30

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LstmTrainer():
    def __init__(self) -> None:
        self.data_controller = DataController(
             target_sensor,
            forecast_follow,
            period_between_each_row,
            forecast_col,
            folder_path,
            learning_rate,
            num_hidden_units,
            model_train_epoch_count,
            train_test_split_factor,
            batch_size,
            sequence_length,
            columns_features,
        )
    
    def process(self, id: str, model_filename: str):
        logger.debug(f'Start training model {model_filename}')
        data= self.data_controller.read_data()
        data = self.data_controller.prepare_data(data)
        data = self.data_controller.create_features_data(data)
        features = list(data.columns.difference([target_sensor]))
        logger.debug(f'[update] features : {features}')
        shiftFollowColumn = f"{target_sensor}_follow{forecast_follow}"
        #logger.debug(f'[main] shiftFollowColumn {shiftFollowColumn}')
        #logger.debug(f'[main] len(data)= {len(data)}')
        data = self.data_controller.create_shift_column(data, shiftFollowColumn)
        #logger.debug(f'[main] len(data)= {len(data)}')
        data_train, data_test = self.data_controller.split_data(data)
        logger.debug(f'[main] data_train {len(data_train)} and data_test {len(data_test)}' )
        self.data_controller.scale_transform_data(data_train, shiftFollowColumn, "fit")
        
        data_train = self.data_controller.scale_transform_data(
             data_train, shiftFollowColumn, "transform"
         )
        data_test = self.data_controller.scale_transform_data(
             data_test, shiftFollowColumn, "transform"
         )

        torch.manual_seed(101)
        
        SeqData_train = SequenceDataset(
            data_train,
            target=shiftFollowColumn,
            features=features,
            sequence_length=sequence_length,
        )
        SeqData_test = SequenceDataset(
            data_test,
            target=shiftFollowColumn,
            features=features,
            sequence_length=sequence_length,
        )

        logger.debug('[main] train_loader')
        train_loader = DataLoader(SeqData_train, batch_size=batch_size, shuffle=True, pin_memory=True)
        logger.debug('[main] test_loader')
        self.test_loader = DataLoader(SeqData_test, batch_size=batch_size, shuffle=False, pin_memory=True)
        logger.debug('[main] train_eval_loader')
        train_eval_loader = DataLoader(SeqData_train, batch_size=batch_size, shuffle=False, pin_memory = True)
        logger.debug('[main] finished dataloaders ')
        self.model = ShallowRegressionLSTM(
            num_sensors=len(features), hidden_units=num_hidden_units
        )
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self. model.parameters(), lr=learning_rate)

        modelController = ModelController(
            self.model, loss_function, optimizer, model_train_epoch_count
        )
        logger.debug(f'Start training model on {model_filename}')
        train_loss_list, test_loss_list = modelController.train_test_model(
            train_loader, self.test_loader, model_filename=model_filename
        )        
        return model_filename, train_loss_list, test_loss_list