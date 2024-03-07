from sklearn.preprocessing import StandardScaler
import glob
import pandas as pd
import numpy
import os

import logging, coloredlogs
logger = logging.getLogger(__name__)
coloredlogs.install('DEBUG', logger=logger)

class DataController:
    def __init__(
        self,
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
    ):
        self.target_sensor = target_sensor
        self.forecast_follow = forecast_follow
        self.period_between_each_row = period_between_each_row
        self.forecast_col = forecast_col
        self.folder_path = folder_path
        self.learning_rate = learning_rate
        self.num_hidden_units = num_hidden_units
        self.model_train_epoch_count = model_train_epoch_count
        self.train_test_split_factor = train_test_split_factor
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        # Instantiate the StandardScaler
        self.scaler_features = StandardScaler()
        self.scaler_energy = StandardScaler()
        self.columns_features = columns_features

    def printDebugInfo(self, status, step_message, input_df, output_df):
        if status is True:
            logger.debug(f'------------------------[OK]------------------------')
            logger.debug(f'[DEBUG] - {step_message}')
            logger.debug(f'[DEBUG] - IN shape {input_df.shape} / OUT shape {output_df.shape}')
            logger.debug(f'\r\n')
        if status is False:
            logger.debug(f'------------------------[ERROR]------------------------')
            logger.debug(f'[DEBUG] - {step_message}')
            logger.debug(f'[DEBUG] - IN shape {input_df.shape} / OUT shape {output_df.shape}')
            logger.debug(f'\r\n')
            return

    def read_data(self):
        # Get a list of CSV files in the folder and sort it
        full_path = f'{self.folder_path}/*.csv'
        logger.debug(f'full path: {full_path}')
        csv_files = sorted(glob.glob(self.folder_path + "/*.csv"), reverse=True)
        logger.debug(f'CSV FILES {csv_files}')

        # Initialize an empty list to store DataFrames
        data_frames = []

        # Drop features
        drop_features_list = [
            "epoch",
            "energy_vref",
            "energy_ct1_current",
            "bme_pres",
            "bme_alt",
            "energy_ct2_current",
            "energy_ct1_power",
            "energy_ct2_power",
        ]

        # Iterate through each CSV file and read it into a DataFrame, then append to the list
        for file in csv_files:
            if os.stat(file).st_size > 0:
                logger.debug(f'[datacontroller:readData] processing file {file}')
                data = pd.read_csv(file, sep=",", decimal=".", index_col="time", dtype={'bme_hum': numpy.float64, 'energy_power': numpy.float64})
                #logger.debug(f'[readData] data = {data}')
                data.drop(drop_features_list, axis='columns', inplace=True)
                
                # Load data from csv using pandas read_csv method
                # data = pd.read_csv(file, sep=",", decimal=".", index_col="time")
                # logger.debug(f'[readData] data = {data}')
                # data.drop(drop_features_list, axis='columns', inplace=True)
                # data['tijd'] = data.index
                # logger.debug(f'[readData] data after first drop')
                # logger.debug(data)
                # logger.debug(f'[readData] type  : {type(data)}')
                # logger.debug(f'[] test = {test}')
                # #logger.debug(f'[] data energy power: {data[data["energy_power"] > 5000].index}')
                # #TODO fix following line
                data = data.drop(data[data["energy_power"] > 5000].index)

                data_frames.append(data)
        #logger.debug(f'[readData] data_frames : {data_frames}')
        csv_df = pd.concat(data_frames)[::-1]
        #logger.debug(f'[readData] csv_df : {csv_df}')

        condition = csv_df.index.is_monotonic_increasing 
        logger.debug(f'[readData] condition {condition}')
        self.printDebugInfo(condition, "Data Read", pd.DataFrame(), csv_df)

        return csv_df if condition else pd.DataFrame()

    def prepare_data(self, f_df):
        data = f_df.copy()
        # Convert index to datetime
        #logger.debug('[prepareData] pre to_datetime')
        data.index = pd.to_datetime(data.index)
        #data['date'] = data.index
        #logger.debug(f'[prepareData] data index : {data.index}')

        # Round the index timestamps to the nearest minute
        data.index = data.index.round(self.period_between_each_row)
        #logger.debug(f'[prepareData] na round on time {data.index}')

        # Group by the rounded index timestamps and calculate the mean for each group
        data = data.groupby(level=0).mean()

        condition = data.shape[0] < f_df.shape[0] and data.shape[1] == f_df.shape[1]
        self.printDebugInfo(condition, "Data Prepare", f_df, data)
        #logger.debug(f'[prepareData] condition : {condition} ==> {data.shape[0]} < {f_df.shape[0]} and {data.shape[1]} == {f_df.shape[1]}')
        return data if condition else pd.DataFrame()

    def create_features_data(self, f_df):
        #logger.debug(f'[createFeaturesData] f_df : {f_df}')
        data = f_df.copy()
        #logger.debug(f'[createFeaturesData] index : {data}')
        data["hour"] = data.index.hour
        data["day_of_week"] = data.index.dayofweek
        data["day_of_year"] = data.index.dayofyear
        #logger.debug(f'[createFeaturesData] data : {data}')
        # If you want to include additional time-related features, you can uncomment the lines below

        # data['quarter'] = data.index.quarter
        # data['month'] = data.index.month
        # data['year'] = data.index.year
        # data['day_of_month'] = data.index.day
        # data['week_of_year'] = data.index.isocalendar().week
        condition = data.shape[1] > f_df.shape[1] and data.shape[0] == f_df.shape[0]
        self.printDebugInfo(
            condition,
            f"Data Create Features ({data.shape[1] - f_df.shape[1]})",
            f_df,
            data,
        )

        return data if condition else pd.DataFrame()

    def create_shift_column(self, f_df, newColumnName):
        data = f_df.copy()
        data[newColumnName] = data[self.target_sensor].shift(self.forecast_follow)
        data = data.iloc[self.forecast_follow :]

        condition = (
            data.shape[1] > f_df.shape[1]
            and data.shape[0] == f_df.shape[0] - self.forecast_follow
        )
        self.printDebugInfo(
            condition, f"Data Create Shift Column ({newColumnName})", f_df, data
        )

        return data if condition else pd.DataFrame()

    def split_data(self, f_df : pd.DataFrame):
        data = f_df.copy()
        #logger.debug(f'[splitData] index : {data.index}')
        start_index = int(len(data) * self.train_test_split_factor)
        test_start = data.index[start_index]
        test_end = data.index[len(data)-1]
        #logger.debug(f'[splitData]  len(data) = {len(data)}  start_index = {start_index} end_index = {test_end}')
        
        #logger.debug(f'[splitData] typeof test_start {type(test_start)}')
        #logger.debug(f'[splitData] test_start : {test_start}')
        
        data_train = data.loc[data.index[0]:test_start].copy()
        #logger.debug(f'[splitData] data_train : {data_train} => ')
        data_test = data.loc[test_start:test_end].copy()

        condition = (
            len(data_train) > len(data_test)
            and data_train.shape[1] == data_test.shape[1]
        )
        self.printDebugInfo(
            condition,
            f"Data Split (Train: {(len(data_train) / len(data))*100:.1f}%  | Test: {(len(data_test) / len(data))*100:.1f}%)",
            f_df,
            pd.concat((data_train, data_test)),
        )
        return (
            (data_train, data_test) if condition else (pd.DataFrame(), pd.DataFrame())
        )

    def scale_transform_data(self, f_df, shiftFollowColumn, action=""):
        logger.debug(f'[scaleTransformData] calling action {action}')
        data = f_df.copy()
        data_features = data[self.columns_features].copy()
        logger.debug(f'[scaleTransformData] data_features = {data_features.columns}')
        data_others = data.drop(columns=self.columns_features).copy()
        logger.debug(f'[scaleTransformData] data_others = {data_others.columns}')

        data_out = pd.DataFrame()

        if action == "transform":
            data_out = pd.DataFrame(
                self.scaler_features.transform(data_features.values),
                columns=self.columns_features,
                index=data.index,
            )
        elif action == "inverse_transform":
            data_out = pd.DataFrame(
                self.scaler_features.inverse_transform(data_features.values),
                columns=self.columns_features,
                index=data.index,
            )
        elif action == "fit":
            #logger.debug(f'[scaleTransformData] calling fit on {data_features.values}')
            self.scaler_features.fit(data_features.values)
        else:
            # Handling for other actions or raise an exception for unsupported actions
            raise ValueError("Unsupported action")
        logger.debug(f'========= FIT HAS BEEN PERFORMED ON SCALAR FEATURES ============')
        first_fit_done = False
        for column in data_others.columns:
            logger.debug(f'[scaleTransformData] column {column} / {action}')
            if action == "transform":
                data_out[column] = self.scaler_energy.transform(
                    data_others[[column]].values
                )
            elif action == "inverse_transform":
                data_out[column] = self.scaler_energy.inverse_transform(
                    data_others[[column]].values
                )
            elif action == "fit":
                if not first_fit_done:
                    logger.debug(f'[scaleTransformData] not first fit done ... so ... ')
                    self.scaler_energy.fit(data_others[[column]].values)
                    first_fit_done = True

        condition = (
            data_out.shape[0] == f_df.shape[0] and data_out.shape[1] == f_df.shape[1]
        )
        logger.debug(f'[scaleTransformData] condition = {condition}')
        logger.debug(f'[scaleTransformData] data_out.shape[0] == f_df.shape[0] => {data_out.shape[0]} == {f_df.shape[0]}')
        logger.debug(f'[scaleTransformData] data_out.shape[1] == f_df.shape[1] => {data_out.shape[1]} == {f_df.shape[1]}')
        
        self.printDebugInfo(condition, f"Data scale {action}", f_df, data_out)
        return data_out if condition else pd.DataFrame()
