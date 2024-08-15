import warnings

import dask.dataframe as dd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

warnings.filterwarnings("ignore")

HEARTRATE_ID = [200]        # heartrate
SYSBP_ID = [100]            # Invasive systolic arterial pressure
DIABP_ID = [120]            # Invasive diastolic arterial pressure
MAP_ID = [110]                # Invasive mean arterial pressure
SATURATION_ID = [4000, 8280]        # Peripheral oxygen saturation
ST_ID = [210, 211, 212]        # ST elevation
CVP_ID = [700, 15001441, 960]    # CVP

class Extractor:
    def __init__(self, index_df, path="data/hirid/raw_stage/observation_tables/csv"):
        self.index_df = index_df
        self.path = path
    
    def get_patient(self, patient_id, mode='dask'):
        assert patient_id in self.index_df['patientid'].values, "patient_id not found"
        row = self.index_df[self.index_df['patientid'] == patient_id]
        part = int(row['part'])
        if mode == 'pandas':
            df = pd.read_csv(f"{self.path}/part-{part}.csv")
        elif mode == 'dask':
            df = dd.read_csv(f"{self.path}/part-{part}.csv", dtype={'stringvalue': 'object', 'type': 'object'})
        df = df[df['patientid'] == patient_id]
        return df
    
    def get_feature(self, patient_id, variable_id_arr, mode='dask'):
        patient_df = self.get_patient(patient_id, mode=mode)
        feature_df = patient_df[patient_df['variableid'].isin(variable_id_arr)]
        
        if isinstance(feature_df, pd.DataFrame):
            return feature_df
        elif isinstance(feature_df, dd.DataFrame):
            return feature_df.compute()

def extract_data(extractor, timesteps, num_patients, variable_id_arr, patient_id, suffix) -> np.ndarray:
    samples = []
    # pbar = tqdm(total=num_patients, desc=f'Extracting patients with {timesteps} timesteps...')
    for i, patient in tqdm(enumerate(list(patient_id)), desc="Extracting (may take a long time)..."):          # for every patient
        feature_samples = []
        
        for variable_id in variable_id_arr:             # extract every needed feature and append
            df = extractor.get_feature(patient, variable_id)

            if len(df) >= timesteps:
                feature_samples.append(list(df['value'].astype(float))[:timesteps])
                # pbar.update(1)
        
        if len(feature_samples) != len(variable_id_arr):        # if this patient does not have all the features, skip
            continue
        
        samples.append(feature_samples)
        if len(samples) % 10 == 0:
            print("Patient {} done".format(len(samples)))
            np.save(f"data/hirid-multiple-checkpoint-{suffix}.npy", samples)          # check point
        
        if len(samples) == num_patients:
            break
        
    return samples

def visualize(data, subplots, figsize=(20, 10), linewidth=1, markersize=5, marker='s') -> None:
    fig, axs = plt.subplots(subplots[0], subplots[1], figsize=figsize)
    data_iterator = iter(data)
    
    for i in range(subplots[0]):
        for j in range(subplots[1]):
            timeseries = next(data_iterator)
            sns.lineplot(x=np.arange(1, len(timeseries)+1), y=timeseries, ax=axs[i, j], linewidth=linewidth, marker=marker, markersize=markersize)
            axs[i, j].set_xlabel("Time (every 2 mins)")
            axs[i, j].set_ylabel("Value")
    
    plt.tight_layout()
    plt.show()        

if __name__ == "__main__":
    path = "data/hirid/raw_stage/observation_tables"
    index_df = pd.read_csv(f"{path}/observation_tables_index.csv")
    general_df = pd.read_csv("data/hirid/reference_data/general_table.csv")

    alive_id = general_df[general_df["discharge_status"] == 'alive']['patientid']
    dead_id = general_df[general_df["discharge_status"] == 'dead']['patientid']
    print(f"There are {len(alive_id)} alive patients\nThere are {len(dead_id)} dead patients\nMortality Rate: {100*len(dead_id) / (len(alive_id) + len(dead_id)):.2f}%")

    extractor = Extractor(index_df)
    
    alive_id = list(pd.read_csv('data/hirid-extract/hirid-alive-id.csv')['patientid'])            # load randomly shuffle patient_id
    dead_id = list(pd.read_csv('data/hirid-extract/hirid-dead-id.csv')['patientid'])          # load randomly shuffle patient_id
    print(len(alive_id), len(dead_id))
    # alive_samples = extract_data(extractor, timesteps=100, num_patients=10000, variable_id_arr=[HEARTRATE_ID, SYSBP_ID, DIABP_ID, MAP_ID, SATURATION_ID, ST_ID, CVP_ID], patient_id=alive_id, suffix="alive")
    print("Complete alive samples! Start extracting expired patients...")
    dead_samples = extract_data(extractor, timesteps=100, num_patients=10000, variable_id_arr=[HEARTRATE_ID, SYSBP_ID, DIABP_ID, MAP_ID, SATURATION_ID, ST_ID, CVP_ID], patient_id=dead_id, suffix="dead")
    # np.save("data/hirid-multiple-checkpoint-alive.npy", alive_samples)
    np.save("data/hirid-multiple-checkpoint-dead.npy", dead_samples)
    
    print("=== COMPLETE ===")
    

