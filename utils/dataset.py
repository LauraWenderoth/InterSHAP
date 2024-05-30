import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torch
import copy
import numpy as np
from pathlib import Path
from torchvision import transforms
from sklearn.model_selection import train_test_split
import itertools
import random
from PIL import Image
import einops
from torch.utils.data import Dataset
from torchvision import transforms
from openslide import OpenSlide
import os
from multiprocessing import Lock
from multiprocessing import Manager
import h5py
import torch
import pprint
from einops import rearrange, repeat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import *
from box import Box

class MMDataset(Dataset):
    def __init__(self, X,y=None,concat='early', device = 'cuda:0' if torch.cuda.is_available() else 'cpu',length=None):
        assert isinstance(X, list) or isinstance(X, dict) or isinstance(X, np.ndarray), "X must be a list or dict or np.array with modalities"
        self.X = X
        self.y = y
        if y is None:
            try:
                self.y = X['label']
                self.X = [value for key, value in X.items() if key != 'label']
            except KeyError:
                raise ValueError("'label' key is not present in the input dictionary X. Store y in 'label'")
        assert len(self.X) >= 2, "X must have at least 2 modalities"

        if length is not None:
            self.set_length(length)
        self.X_org = copy.deepcopy(self.X)
        try:
            self.modality_shape = [mod.shape[1] for mod in self.X]
        except:
            self.modality_shape = [mod[1].shape for mod in self.X]
        self.device = device
        self.concat = True if concat == 'early' else False

        ## preprocess data
        self.y = np.squeeze(self.y)
        if self.concat:
            self.X = np.concatenate(self.X,axis=1)

        assert len(self.X) == len(self.y) or all((isinstance(sublist, list) or isinstance(sublist, np.ndarray)) and len(sublist) == len(self.y) for sublist in self.X), \
            "Inconsistent data: X must be either a single list with the same length as y or a list of lists with each sublist having the same length as y"


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        label = torch.tensor(self.y[idx])
        if self.concat:
            features = torch.tensor(self.X[idx], dtype=torch.float)
        else:
            features = [torch.tensor(modality[idx], dtype=torch.float) for modality in self.X]
        return features, label

    def get_data(self):
        # get unconcatinated data
        return self.X_org, self.y

    def set_length(self,length):
        if length <=  len(self.y):
            self.y = self.y[:length]
            try:
                self.X = [mod[:length] for mod in self.X]
            except:
                self.X = self.X[:length]
    def get_modality_shapes(self):
        return self.modality_shape

    def get_number_classes(self):
        return len(np.unique(self.y))

    def get_concat(self):
        return self.concat

class TCGADataset(Dataset):
    def __init__(self,dataset_path, subset_name,modality_names, concat='early', device = 'cuda:0' if torch.cuda.is_available() else 'cpu',n_bins=4,level=2, model = 'healnet', omic_attention=False,survival_subset ='uncensored',image_resize_dims =None,length=None):
        super(TCGADataset, self).__init__()

        #comments
        # resize dims should be tuple (width, height)

        self.concat = True if concat == 'early' else False
        self.device = device
        self.subset= survival_subset # subset used to calculate survival bin cutoffs, valid: all, censored, uncensored
        self.dataset = subset_name
        self.sources = modality_names
        self.model = model
        self.omic_attention = omic_attention
        self.filter_overlap = True
        self.n_bins = n_bins
        self.dataset_path = dataset_path
        self.resize_dims = image_resize_dims
        self.raw_path = Path(dataset_path).joinpath(f"wsi/{subset_name}")
        prep_path = Path(dataset_path).joinpath(f"wsi/{subset_name}_preprocessed_level{level}")
        self.prep_path = prep_path
        # create patch feature directory for first-time run
        os.makedirs(self.prep_path.joinpath("patch_features"), exist_ok=True)
        self.slide_ids = [slide_id.rsplit(".", 1)[0] for slide_id in os.listdir(prep_path.joinpath("patches"))]

        valid_sources = ["omic", "slides", "rna-sequence", "mutation", "copy-number"]
        assert all([source in valid_sources for source in
                    modality_names]), f"Invalid source specified. Valid sources are {valid_sources}"
        assert not ("omic" in list(modality_names) and any(option in list(modality_names) for option in
                                                                ["rna-sequence", "mutation",
                                                                 "copy-number"])), f'Choose only "omic" or  "rna-sequence", "mutation", "copy-number"'

        self.wsi_paths: dict = self._get_slide_dict()  # {slide_id: path}
        self.sample_slide_id = self.slide_ids[0] + ".svs"
        self.sample_slide = OpenSlide(self.wsi_paths[self.sample_slide_id])
        # pre-load and transform omic data
        omic_df = self.load_omic()
        self.omic_df = omic_df.copy()
        omic_df = omic_df.drop(
            ["site", "oncotree_code", "case_id", "slide_id", "train", "censorship", "survival_months", "y_disc"],
            axis=1)
        self.omic_features = self.split_omics(omic_df)

        self.omic_tensor = {key: torch.Tensor(self.omic_features[key].values) for key in self.omic_features.keys()}
        if 'healnet' in self.model:
            # Healnet expects inputs of the shape (batch_size, input_dim, channels)
            # TODO Not tested for rna mut ect
            if self.omic_attention:
                self.omic_tensor.update(
                    {key: einops.repeat(tensor, "n feat -> n feat channels", channels=1) for key, tensor in
                     self.omic_tensor.items()})
            else:
                self.omic_tensor.update(
                    {key: einops.repeat(tensor, "n feat -> n channels feat", channels=1) for key, tensor in
                     self.omic_tensor.items()})

        self.level = level
        self.slide_idx: dict = self._get_slide_idx()  # {idx (molecular_df): slide_id}
        override = True if self.resize_dims is not None else False
        self.wsi_width, self.wsi_height = self.get_resize_dims(level=self.level, override=override)
        self.censorship = self.omic_df["censorship"].values
        self.survival_months = self.omic_df["survival_months"].values
        self.y_disc = self.omic_df["y_disc"].values

        manager = Manager()
        self.patch_cache = manager.dict()
        # self.patch_cache = SharedLRUCache(capacity=256) # capacity should be multiple of num_workers
        print(f"Dataloader initialised for {subset_name} dataset")
        self.get_info(full_detail=False)

        #caculate data dims:
        self.modality_shapes = []
        if "omic" in self.sources:
            omic_tensor = self.omic_tensor['omic'][0]
            self.modality_shapes.append(omic_tensor.shape[1])
        else:
            if "rna-sequence" in self.sources:
                omic_tensor = self.omic_tensor['rna-sequence'][0]
                self.modality_shapes.append(omic_tensor.shape[1])
            if "mutation" in self.sources:
                omic_tensor = self.omic_tensor['mutation'][0]
                self.modality_shapes.append(omic_tensor.shape[1])
            if "copy-number" in self.sources:
                omic_tensor = self.omic_tensor['copy-number'][0]
                self.modality_shapes.append(omic_tensor.shape[1])
        if "slides" in self.sources:
            slide_id = self.omic_df.iloc[0]["slide_id"].rsplit(".", 1)[0]
            slide_tensor = self.load_patch_features(slide_id)
            self.modality_shapes.append(list(slide_tensor.shape))
    def __getitem__(self, index):
        y_disc = self.y_disc[index]
        censorship = self.censorship[index]
        event_time = self.survival_months[index]
        data = []

        if "omic" in self.sources:
            if any(source in ["rna-sequence", "mutation", "copy-number"] for source in self.sources):
                print(
                    f'Only "omic" is used. Any of the following source will NOT be used: "rna-sequence", "mutation", "copy-number".')
            omic_tensor = self.omic_tensor['omic'][index]
            if self.concat:
                omic_tensor = torch.flatten(omic_tensor)
            data.append(omic_tensor)
            # return [omic_tensor], censorship, event_time, y_disc
        else:
            if "rna-sequence" in self.sources:
                omic_tensor = self.omic_tensor['rna-sequence'][index]
                if self.concat:
                    omic_tensor = torch.flatten(omic_tensor)
                data.append(omic_tensor)
            if "mutation" in self.sources:
                omic_tensor = self.omic_tensor['mutation'][index]
                if self.concat:
                    omic_tensor = torch.flatten(omic_tensor)
                data.append(omic_tensor)
            if "copy-number" in self.sources:
                omic_tensor = self.omic_tensor['copy-number'][index]
                if self.concat:
                    omic_tensor = torch.flatten(omic_tensor)
                data.append(omic_tensor)

        if "slides" in self.sources:
            slide_id = self.omic_df.iloc[index]["slide_id"].rsplit(".", 1)[0]

            # if self.config.model == "mm_prognosis": # raw WSI
            #     slide, slide_tensor = self.load_wsi(slide_id, level=self.level)
            #     return [slide_tensor], censorship, event_time, y_disc

            if index not in self.patch_cache:
                slide_tensor = self.load_patch_features(slide_id)
                # self.patch_cache.set(index, slide_tensor)
                self.patch_cache[index] = slide_tensor

            else:
                slide_tensor = self.patch_cache[index]
                # slide_tensor = self.patch_cache.get(index)
            if self.model == "fcnn":  # for fcnn baseline
                slide_tensor = torch.flatten(slide_tensor)
            if self.concat:
                slide_tensor = torch.flatten(slide_tensor)
            data.append(slide_tensor)
            # return [slide_tensor], censorship, event_time, y_disc

        assert data, "You must select at least one source."

        if self.concat:  # for early fusion baseline
            concat_tensor = torch.cat(data, dim=0)
            if self.model == "healnet_early":
                concat_tensor = concat_tensor.unsqueeze(0)
            return [concat_tensor], y_disc
        else:  # keep separate for HEALNet
            return data, y_disc

    def get_resize_dims(self, level: int, patch_height: int = 128, patch_width: int = 128, override=False):
        if override is False:
            width = self.sample_slide.level_dimensions[level][0]
            height = self.sample_slide.level_dimensions[level][1]
            # take nearest multiple of 128 of height and width (for patches)
            width = round(width / patch_width) * patch_width
            height = round(height / patch_height) * patch_height
        else:
            width = self.resize_dims[0]
            height = self.resize_dims[1]
        return width, height

    def _get_slide_idx(self):
        # filter slide index to only include samples with WSIs availables
        filter_keys = [slide_id + ".svs" for slide_id in self.slide_ids]
        tmp_df = self.omic_df[self.omic_df.slide_id.isin(filter_keys)]
        return dict(zip(tmp_df.index, tmp_df["slide_id"]))

    def __len__(self):
        if self.sources == ["omic"]:
            # use all omic samples when running single modality
            return self.omic_df.shape[0]
        else:
            # only use overlap otherwise
            return len(self.slide_ids)

    def _get_slide_dict(self):
        """
        Given the download structure of the gdc-client, each slide is stored in a folder
        with a non-meaningful name. This function returns a dictionary of slide_id to
        the path of the slide.
        Returns:
            svs_dict (dict): Dictionary of slide_id to path of slide
        """
        slide_path = Path(self.dataset_path).joinpath(f"wsi/{self.dataset}")
        svs_files = list(slide_path.glob("**/*.svs"))
        svs_dict = {path.name: path for path in svs_files}
        return svs_dict

    def _load_patch_coords(self):
        """
        Loads all patch coordinates for the dataset and level specified in the config and writes it to a dictionary
        with key: slide_id and value: patch coordinates (where each coordinate is a x,y tupe)
        """
        coords = {}
        for slide_id in self.slide_ids:
            patch_path = self.prep_path.joinpath(f"patches/{slide_id}.h5")
            h5_file = h5py.File(patch_path, "r")
            patch_coords = h5_file["coords"][:]
            coords[slide_id] = patch_coords
        return coords

    def get_info(self, full_detail: bool = False):
        """
        Logging util to print some basic dataset information. Normally called at the start of a pipeline run
        Args:
            full_detail (bool): Print all slide properties

        Returns:
            None
        """
        slide_path = Path(self.dataset_path).joinpath(f"wsi/{self.dataset}/")
        print(f"Dataset: {self.dataset.upper()}")
        print(f"Molecular data shape: {self.omic_df.shape}")
        sample_overlap = (set(self.omic_df["slide_id"]) & set(self.wsi_paths.keys()))
        print(f"Molecular/Slide match: {len(sample_overlap)}/{len(self.omic_df)}")
        # TODO change print if subset of omics used
        print(f"Slide level count: {self.sample_slide.level_count}")
        print(f"Slide level dimensions: {self.sample_slide.level_dimensions}")
        print(f"Slide resize dimensions: w: {self.wsi_width}, h: {self.wsi_height}")
        print(f"Sources selected: {self.sources}")
        print(
            f"Censored share: {np.round(len(self.omic_df[self.omic_df['censorship'] == 1]) / len(self.omic_df), 3)}")
        print(f"Survival_bin_sizes: {dict(self.omic_df['y_disc'].value_counts().sort_values())}")

        if full_detail:
            pprint(dict(self.sample_slide.properties))

    def show_samples(self, n=1):
        """
        Logging util to show some detailed sample stats and render the whole slide image (e.g., in a notebook)
        Args:
            n (int): Number of samples to show

        Returns:
            None
        """
        # sample_df = self.omic_df.sample(n=n)
        sample_df = self.omic_df[self.omic_df["slide_id"].isin(self.wsi_paths.keys())].sample(n=n)
        for idx, row in sample_df.iterrows():
            print(f"Case ID: {row['case_id']}")
            print(f"Patient age: {row['age']}")
            print(f"Gender: {'female' if row['is_female'] else 'male'}")
            print(f"Survival months: {row['survival_months']}")
            print(f"Survival years:  {np.round(row['survival_months'] / 12, 1)}")
            print(f"Censored (survived follow-up period): {'yes' if row['censorship'] else 'no'}")
            # print(f"Risk: {'high' if row['high_risk'] else 'low'}")
            # plot wsi
            slide, slide_tensor = self.load_wsi(row["slide_id"], level=self.level)
            print(f"Shape:", slide_tensor.shape)
            plt.figure(figsize=(10, 10))
            plt.imshow(slide_tensor)
            plt.show()

    def load_omic(self,
                  eps: float = 1e-6
                  ) -> pd.DataFrame:
        """
        Loads in omic data and returns a dataframe and filters depending on which whole slide images
        are available, such that only samples with both omic and WSI data are kept.
        Also calculates the discretised survival time for each sample.
        Args:
            eps (float): Epsilon value to add to min and max survival time to ensure all samples are included

        Returns:
            pd.DataFrame: Dataframe with omic data and discretised survival time (target)
        """
        data_path = Path(self.dataset_path).joinpath(f"omic/tcga_{self.dataset}_all_clean.csv.zip")
        df = pd.read_csv(data_path, compression="zip", header=0, index_col=0, low_memory=False)
        valid_subsets = ["all", "uncensored", "censored"]
        assert self.subset in valid_subsets, "Invalid cut specified. Must be one of 'all', 'uncensored', 'censored'"

        # handle missing values
        num_nans = df.isna().sum().sum()
        nan_counts = df.isna().sum()[df.isna().sum() > 0]
        df = df.fillna(df.mean(numeric_only=True))
        print(f"Filled {num_nans} missing values with mean")
        print(f"Missing values per feature: \n {nan_counts}")

        # filter samples for which there are no slides available
        if self.filter_overlap:
            slides_available = self.slide_ids
            omic_available = [id[:-4] for id in df["slide_id"]]
            overlap = set(slides_available) & set(omic_available)
            print(f"Slides available: {len(slides_available)}")
            print(f"Omic available: {len(omic_available)}")
            print(f"Overlap: {len(overlap)}")
            if len(slides_available) < len(omic_available):
                print(
                    f"Filtering out {len(omic_available) - len(slides_available)} samples for which there are no omic data available")
                overlap_filter = [id + ".svs" for id in overlap]
                df = df[df["slide_id"].isin(overlap_filter)]
            elif len(slides_available) > len(omic_available):
                print(
                    f"Filtering out {len(slides_available) - len(omic_available)} samples for which there are no slides available")
                self.slide_ids = overlap
            else:
                print("100% modality overlap, no samples filtered out")

        # assign target column (high vs. low risk in equal parts of survival)
        label_col = "survival_months"
        if self.subset == "all":
            df["y_disc"] = pd.qcut(df[label_col], q=self.n_bins, labels=False).values
        else:
            if self.subset == "censored":
                subset_df = df[df["censorship"] == 1]
            elif self.subset == "uncensored":
                subset_df = df[df["censorship"] == 0]
            # take q_bins from uncensored patients
            disc_labels, q_bins = pd.qcut(subset_df[label_col], q=self.n_bins, retbins=True, labels=False)
            q_bins[-1] = df[label_col].max() + eps
            q_bins[0] = df[label_col].min() - eps
            # use bin cuts to discretize all patients
            df["y_disc"] = pd.cut(df[label_col], bins=q_bins, retbins=False, labels=False, right=False,
                                  include_lowest=True).values

        df["y_disc"] = df["y_disc"].astype(int)

        return df

    def load_wsi(self, slide_id: str, level: int = None) -> Tuple:
        """
        Load in single slide and get region at specified resolution level
        Args:
            slide_id:
            level:
            resolution:

        Returns:
            Tuple (openslide object, tensor of region)
        """

        # load in openslide object
        # slide_path = self.wsi_paths[slide_id]
        # slide = OpenSlide(slide_path + ".svs")
        slide = OpenSlide(self.raw_path.joinpath(f"{slide_id}.svs"))

        # specify resolution level
        if level is None:
            level = slide.level_count  # lowest resolution by default
        if level > slide.level_count - 1:
            level = slide.level_count - 1
        # load in region
        size = slide.level_dimensions[level]
        region = slide.read_region((0, 0), level, size)
        # add transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3, :, :]),  # remove alpha channel
            transforms.Resize((self.wsi_height, self.wsi_width)),
            RearrangeTransform("c h w -> h w c")  # rearrange for Healnet architecture
        ])
        region_tensor = transform(region)
        return slide, region_tensor

    def load_patch_features(self, slide_id: str) -> torch.Tensor:
        """
        Loads patch features for a single slide from torch.pt file
        Args:
            slide_id (str): Slide ID

        Returns:
            torch.Tensor: Patch features
        """
        load_path = self.prep_path.joinpath(f"patch_features/{slide_id}.pt")
        with open(load_path, "rb") as file:
            patch_features = torch.load(file, weights_only=True)
        return patch_features

    def split_omics(self, df):
        substrings = ['rnaseq', 'cnv', 'mut']
        dfs = [df.filter(like=sub) for sub in substrings]

        return {"omic": df, "rna-sequence": dfs[0], "mutation": dfs[2], "copy-number": dfs[1]}



    def get_modality_shapes(self):
        return self.modality_shapes

    def get_number_classes(self):
        return self.n_bins

    def get_concat(self):
        return self.concat

class SharedLRUCache:
    """
    Shared LRU cache for multiprocessing
    """
    def __init__(self, capacity: int):
        """

        Args:
            capacity (int): Number of items to be stored in the cache
        """
        manager = Manager()
        self.capacity = capacity
        self.cache = manager.dict()
        self.order = manager.list()
        self.lock = Lock()
    def get(self, key: int):
        with self.lock:
            if key in self.cache:
                # Move key to end to show it was recently used.
                self.order.remove(key)
                self.order.append(key)
                return self.cache[key]
            else:
                return None

    def set(self, key: int, value):
        with self.lock:
            if key in self.cache:
                self.order.remove(key)
            else:
                if len(self.order) >= self.capacity:
                    removed_key = self.order.pop(0)  # Remove the first (least recently used) item.
                    del self.cache[removed_key]

            self.order.append(key)
            self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache


class RearrangeTransform(object):
    """
    Wrapper for einops.rearrange to pass into torchvision.transforms.Compose
    """
    def __init__(self, pattern):
        self.pattern = pattern

    def __call__(self, img):
        img = rearrange(img, self.pattern)
        return img