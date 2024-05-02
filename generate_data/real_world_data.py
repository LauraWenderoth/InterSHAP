import os
import sys
from pathlib import Path
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pandas as pd
from sklearn.model_selection import train_test_split
import umap
from generate_data import visualize_umap, create_split

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from utils.utils import save_data

def load_cell_omic(root_folder,donor = 31800,target='cell_type',dict_classes = {'BP': 0, 'EryP': 1, 'MoP': 2, 'NeuP': 3},columns_to_drop = ['cell_id', 'day', 'donor', 'technology']):
    root_folder= Path(root_folder)
    meta_df = pd.read_csv(root_folder / 'meta_data_train.csv')

    df_protein = pd.read_csv(root_folder / 'protein_data_train.csv')
    df_rna = pd.read_csv(root_folder / 'rna_rand_data.csv')
    dfs = [df_protein, df_rna]
    id_p = df_protein['cell_id'].values
    id_r = df_rna['cell_id'].values
    ids = list(set(id_p).intersection(set(id_r)))

    mods = []
    for df in dfs:
        merged_df = pd.merge(meta_df, df, on='cell_id', how='inner')
        merged_df = merged_df.drop_duplicates(subset='cell_id', keep='first')
        split = merged_df[merged_df['cell_id'].isin(ids)]
        split = split.copy()
        split.drop(columns=columns_to_drop, inplace=True)
        y = split[target].map(dict_classes).values
        X = split.drop(columns=[target], inplace=False)
        X = X.values
        mods.append(X)

    X_transformed = [
        [modality[i] for modality in mods]
        for i in range(len(mods[0]))]
    y = y[:, np.newaxis]
    return X_transformed, y






def load_omic(dataset, n_bins, dataset_path, subset,
              eps: float = 1e-6
              ) -> pd.DataFrame:
    """
    adapted from https://github.com/konst-int-i/healnet
    Loads in omic data and returns a dataframe and filters depending on which whole slide images
    are available, such that only samples with both omic and WSI data are kept.
    Also calculates the discretised survival time for each sample.
    Args:
        eps (float): Epsilon value to add to min and max survival time to ensure all samples are included

    Returns:
        pd.DataFrame: Dataframe with omic data and discretised survival time (target)
    """
    data_path = Path(dataset_path).joinpath(f"omic/tcga_{dataset}_all_clean.csv.zip")
    df = pd.read_csv(data_path, compression="zip", header=0, index_col=0, low_memory=False)
    valid_subsets = ["all", "uncensored", "censored"]
    assert subset in valid_subsets, "Invalid cut specified. Must be one of 'all', 'uncensored', 'censored'"

    # handle missing values
    num_nans = df.isna().sum().sum()
    nan_counts = df.isna().sum()[df.isna().sum() > 0]
    df = df.fillna(df.mean(numeric_only=True))
    print(f"Filled {num_nans} missing values with mean")
    print(f"Missing values per feature: \n {nan_counts}")


    # assign target column (high vs. low risk in equal parts of survival)
    label_col = "survival_months"
    if subset == "all":
        df["y_disc"] = pd.qcut(df[label_col], q=n_bins, labels=False).values
    else:
        if subset == "censored":
            subset_df = df[df["censorship"] == 1]
        elif subset == "uncensored":
            subset_df = df[df["censorship"] == 0]
        # take q_bins from uncensored patients
        disc_labels, q_bins = pd.qcut(subset_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = df[label_col].max() + eps
        q_bins[0] = df[label_col].min() - eps
        # use bin cuts to discretize all patients
        df["y_disc"] = pd.cut(df[label_col], bins=q_bins, retbins=False, labels=False, right=False,
                              include_lowest=True).values

    df["y_disc"] = df["y_disc"].astype(int)
    return df

def load_slides(omic_df,dataset_path,subset_name,level):
    prep_path=Path(dataset_path).joinpath(f"wsi/{subset_name}_preprocessed_level{level}")
    data = dict()
    for slide_id_org in tqdm(omic_df["slide_id"]):
        slide_id = slide_id_org.rsplit(".", 1)[0]
        load_path = prep_path.joinpath(f"patch_features/{slide_id}.pt")
        try:
            with open(load_path, "rb") as file:
                patch_features = torch.load(file, weights_only=True)
                slide_tensor = torch.flatten(patch_features)
                slide_tensor = torch.flatten(slide_tensor)
                data[slide_id_org] = (slide_tensor)
        except:
            print(f'Could not load slide with id {slide_id}')
    return data

def split_omics( df):
    substrings = ['rnaseq', 'cnv', 'mut']
    dfs = [df.filter(like=sub) for sub in substrings]
    return {"omic": df, "rna-sequence": dfs[0], "mutation": dfs[2], "copy-number": dfs[1]}

def load_brca(args):
    print(args.dataset, args.setting, args.n_bins)
    org_omic_df = load_omic(dataset=args.dataset, n_bins=args.n_bins, dataset_path=args.data_path, subset='all')
    omic_df = org_omic_df.copy()
    X = []
    if 'slides' in args.setting:
        images = load_slides(omic_df, dataset_path=args.data_path, subset_name=args.dataset, level=2)
        list_of_slide_ids = list(images.keys())
        omic_df = filtered_df = omic_df[omic_df['slide_id'].isin(list_of_slide_ids)]
        org_omic_df = omic_df.copy()
        X.append(list(images.values()))

    omic_df = omic_df.drop(
        ["site", "oncotree_code", "case_id", "slide_id", "train", "censorship", "survival_months", "y_disc"],
        axis=1)
    omic_features = split_omics(omic_df)

    if "rna-sequence" in args.setting:
        X.append(omic_features["rna-sequence"].values)
    if "mutation" in args.setting:
        X.append(omic_features["mutation"].values)
    if "copy-number" in args.setting:
        X.append(omic_features["copy-number"].values)
    X_transformed = [
        [modality[i] for modality in X]
        for i in range(len(X[0]))]

    y = org_omic_df["y_disc"].values
    y = y[:, np.newaxis]
    return X_transformed, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str,
                        default="/home/lw754/masterproject/real_world_data/",
                        help="Path to save the file")
    parser.add_argument("--data_path", type=str,
                        default="/net/archive/export/tcga/tcga",
                        help="Path where data is stored")
    parser.add_argument("--dataset", type=str, default="single-cell",help='brca,single-cell')
    parser.add_argument("--setting", type=str, default= ["rna-sequence","mutation","copy-number"], help='["omic", "slides", "rna-sequence", "mutation", "copy-number"]')
    parser.add_argument("--n_bins", type=int, default=2,   help="How many output classes")




    args = parser.parse_args()

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'single-cell':
        print('create singel-cell')
        X_transformed, y = load_cell_omic('/home/lw754/R255/data/singlecell')
        mods = 2
        labels = ['B-Cell Progenitor','Erythrocyte Progenitor','Monocyte Progenitor','Neutrophil Progenitor' ]
        save_name = f"{args.dataset}_DATA_all"

    elif args.dataset == 'brca':
        X_transformed, y =  load_brca(args)
        mods = len(args.setting)
        labels = [str(i) for i in range(args.n_bins)]
        name_settings = ''.join(setting[0] for setting in args.setting)
        save_name = f"{args.dataset}{len(args.setting)}_DATA_{args.n_bins}_{name_settings}"

    data = create_split(X_transformed, y, mods)



    visualize_umap(data,mods ,save_path/save_name,legend_labels = labels)
    # Save the data
    filename = save_path/f"{save_name}.pickle"  # Using setting variable directly
    save_data(data, filename)