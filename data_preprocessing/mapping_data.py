import os
import pandas as pd
import json

import numpy as np
from ast import literal_eval

def findElements(snp, bin_indices):
    return [snp[i] for i in bin_indices]

def map_json_to_bins(read_path, write_path, trait):
    """
    create a new csv with partitioned snp data into bins
    Args:
        read_path (str): Path to the original json file
        write_path (str): Path where the new csv should be created
        trait (str): e.g "emergence" or "emergence adjusted"
    """
    data_df = convert_json_to_dataframe(read_path)
    snp_ordering = "data/Position_GeneticData_ordered.txt"
    bins_df = pd.read_csv(snp_ordering,delimiter='\t')
    bins_df['bin'] = bins_df['Chromosome'] + '_' + bins_df['Position'].map(str)
    bins_group_indices_df = bins_df.groupby('bin')['Index'].apply(list)
    for i in range(len(bins_group_indices_df)):
        col = str(bins_group_indices_df.index[i])
        data_df[col] = data_df['snpSelf'].apply(lambda x: ''.join(findElements(x,bins_group_indices_df[i])) if x!=None else None)
    new_data_df = data_df.copy()
    new_data_df = new_data_df[new_data_df['trait']==trait]
    new_data_df.to_csv(write_path, index=False)
    
def map_json_to_chromosomes(read_path, write_path, trait):
    """
    create a new csv with partitioned snp data into chromosome
    Args:
        read_path (str): Path to the original json file
        write_path (str): Path where the new csv should be created
        trait (str): e.g "emergence" or "emergence adjusted"
    """
    data_df = convert_json_to_dataframe(read_path)
    snp_ordering = "data/Position_GeneticData_ordered.txt"
    bins_df = pd.read_csv(snp_ordering,delimiter='\t')
    bins_group_indices_df = bins_df.groupby('Chromosome')['Index'].apply(list)
    for i in range(len(bins_group_indices_df)):
        col = str(bins_group_indices_df.index[i])
        data_df[col] = data_df['snpSelf'].apply(lambda x: ''.join(findElements(x,bins_group_indices_df[i])) if x!=None else None)
    new_data_df = data_df.copy()
    new_data_df = new_data_df[new_data_df['trait']==trait]
    new_data_df.to_csv(write_path, index=False)

def combine_csv_bin_and_image_data():
    """
    create a new csv with images modalities and bins data combined
    Args:
        read_path (str): Path to the csv file that contain snp partitioned into bins
        write_path (str): Path where the new csv should be created
        trait (str): e.g "emergence" or "emergence adjusted"
    """
    pass

def combine_csv_chromosome_and_image_data(read_path, write_path, trait):
    """
    create a new csv with images modalities and chromosome data combined
    Args:
        read_path (str): Path to the csv file that contain snp partitioned into chromosomes
        write_path (str): Path where the new csv should be created
        trait (str): e.g "emergence" or "emergence adjusted"
    """
    df_gen = pd.read_csv(read_path)
    df_gen_non_adjust_init = df_gen.loc[df_gen['trait']==trait]
    df_gen_non_adjust_init['droneImages'] = df_gen_non_adjust_init.droneImages.apply(literal_eval)
    df = pd.concat([pd.DataFrame(x) for x in df_gen_non_adjust_init['droneImages']], 
                   keys=df_gen_non_adjust_init['plotCode']).reset_index(level=1, drop=True).reset_index()
    df=df[df['processingStatus']=='Uncropped']
    df.loc[df['cam']=='RGB', 'waveLength'] = '0nm'
    df.loc[df['waveLength']=='530', 'waveLength'] = '530nm'
    df.loc[df['waveLength']=='570', 'waveLength'] = '570nm'
    df.loc[df['waveLength']=='670', 'waveLength'] = '670nm'
    df.loc[df['waveLength']=='700', 'waveLength'] = '700nm'
    df.loc[df['waveLength']=='730', 'waveLength'] = '730nm'
    df.loc[df['waveLength']=='780', 'waveLength'] = '780nm'
    df.loc[df['waveLength']=='900', 'waveLength'] = '900nm'
    df.drop_duplicates(subset=['plotCode','date'], inplace=True)
    df_pivot = df.pivot(index=['plotCode','date'],columns='waveLength',values='imageCode')
    df_pivot = df_pivot.reset_index()
    df_pivot.index = np.arange(len(df_pivot))
    final_df_merge = df_pivot.merge(df_gen_non_adjust_init, how='left', on='plotCode')
    final_df_merge = final_df_merge.drop(columns=['crop','droneImages','snpFather','snpMother','sowingYear'])
    final_df_merge.to_csv(write_path, index=False)
    
    

def read_json_file(Input_Data_Path):
    return [json.loads(line) for line in open(Input_Data_Path, 'r')]

def convert_json_to_dataframe(Input_Data_Path):
    json_data = read_json_file(Input_Data_Path)
    df = pd.json_normalize(data = json_data, record_path = 'phenotypes', meta = ['plotCode', 'crop', 'droneImages',
                                                                              'harvestYear', 'locationNumber', 'plotType', 
                                                                              'snpFather', 'snpMother', 'snpSelf',
                                                                              'sowingYear', 'trialNumber'])
    df['observation'] = df['observation'].replace(',', '.', regex=True).apply(pd.to_numeric, errors='raise')
    #print(df.columns)
    return df



if __name__ == '__main__':
    data_root = "/data/varshneya/clean_data_di/traits_csv"
    # # map json to chromosome
    # read_path = os.path.join(data_root,"begin_of_flowering/updates/BeginOfFlowering_Clean_Updates.txt")
    # write_path = os.path.join(data_root,"begin_of_flowering/updates/BeginOfFlowering_Clean_mapped_chromosome_non-adjusted.csv")
    # map_json_to_chromosomes(read_path, write_path, trait="begin of flowering")
    
    # combine chromosome and image modaltities  
    read_path = os.path.join(data_root,"begin_of_flowering/updates/BeginOfFlowering_Clean_mapped_chromosome_non-adjusted.csv")
    write_path = os.path.join(data_root,"begin_of_flowering/updates/BeginOfFlowering_Clean_non-adjusted_mapped_chromosome_images.csv")
    combine_csv_chromosome_and_image_data(read_path, write_path, trait="begin of flowering")
    
    