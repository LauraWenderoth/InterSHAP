# InterSHAP - Quantifying Cross-modal Interactions in Multimodal Machine Learning 

We present InterSHAP, a cross-modal interaction score, that can be used for an arbitrary number of modalities, is local, performance agnostic, and operates on unlabelled data. We are the first, to the best of our knowledge, to use the Shapley interaction index [(Grabisch and Roubens, 1999)](https://econpapers.repec.org/article/sprjogath/v_3a28_3ay_3a1999_3ai_3a4_3ap_3a547-565.htm) to effectively separate cross-modal interactions from modality contributions. 
<p align="center">
  <img src="visualization/Overview InterSHAP.png" alt="Image 1" width="800"/>
</p>

This repository supports the following cross-modal interaction scores: 

- InterSHAP (Ours)
- [PID score](https://papers.nips.cc/paper_files/paper/2023/hash/575286a73f238b6516ce0467d67eadb2-Abstract-Conference.html)
- [EMAP](https://aclanthology.org/2020.emnlp-main.62.pdf) 
- [SHAPE](https://www.ijcai.org/proceedings/2022/0425.pdf)

Few approaches, such as [PID score](https://papers.nips.cc/paper_files/paper/2023/hash/575286a73f238b6516ce0467d67eadb2-Abstract-Conference.html), [EMAP](https://aclanthology.org/2020.emnlp-main.62.pdf) , and [SHAPE](https://www.ijcai.org/proceedings/2022/0425.pdf) have been developed to explore whether a model learns cross-modal interactions. While PID quantifies synergy within a dataset without considering the model, methods like EMAP calculate how a model would perform without cross-modal interactions. Additionally, SHAPE uses Shapley values to approximate cross-modal interactions. The table shows the limitations of the four approaches:

| Score            | Modalities $> 2$ | Local | Unsupervised | Performance Agnostic | Controlled Testing |
|------------------|-------------------|-------|--------------|----------------------|---------------------|
| [PID score](https://papers.nips.cc/paper_files/paper/2023/hash/575286a73f238b6516ce0467d67eadb2-Abstract-Conference.html)              | :x:               | :x:   | :x:          | <font color="green">:heavy_check_mark:</font>   | <font color="green">:heavy_check_mark:</font>  |
| [EMAP](https://aclanthology.org/2020.emnlp-main.62.pdf)              | :x:               | :x:   | :x:          | :x:                  | <font color="green">:heavy_check_mark:</font>  |
| [SHAPE](https://www.ijcai.org/proceedings/2022/0425.pdf)            | <font color="green">:heavy_check_mark:</font>| :x:   | :x:          | :x:                  | :x:                 |
| InterSHAP (Ours)| <font color="green">:heavy_check_mark:</font>| <font color="green">:heavy_check_mark:</font>| <font color="green">:heavy_check_mark:</font>| <font color="green">:heavy_check_mark:</font> | <font color="green">:heavy_check_mark:</font> |


- *Modalities $> 2$*: The score is effective for two or more modalities.
- *Local*: Besides its global application at the dataset level, it is also applicable at the data point level.
- *Unsupervised*: Ground truth labels are unnecessary to calculate the score.
- *Performance agnostic*: The degree of cross-modality interactions is not influenced by the model's performance, ensuring robustness.
- *Controlled testing*: The metric underwent rigorous testing for effectiveness using a synthetic dataset.
  
## Setup

1. **Create Conda Environment:**
   ```bash
   conda create --name interactions python=3.9
   ```

2. **Activate the Environment:**
   ```bash
   conda activate interactions
   ```

3. **Install Dependencies:**
   ```bash
   conda install --file requirements.txt
   ```

## Generate Synthetic Data (HD-XOR)
To generate the synthetic datasets (settings: uniqueness 1, uniqueness 2, synergy, random and redundancy) with the HD-XOR method for the following scripts:
```
bash generate_data/generate_VEC2.sh $PATH/TO/SAVE/$
bash generate_data/generate_VEC3.sh $PATH/TO/SAVE/$
bash generate_data/generate_VEC5.sh $PATH/TO/SAVE/$
```

##  Application to one's own models. 

```
import torch
from synergy_evaluation.evaluation import eval_synergy
from utils.dataset import MMDataset

synergy_eval_metrics = ['SHAPE','SRI','Interaction','PID','EMAP'] #choose the one you want
save_path = $PATH
number_of_classes = $INT

val_dataset, test_dataset = $ LOAD DATA in MMDataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = torch.load(PATH)
model.to(device)

synergy_results = eval_synergy(model, val_dataset, test_dataset, device = device, eval_metrics=synergy_eval_metrics, batch_size=100, save_path=save_path, use_wandb=False, n_samples_for_interaction=10, classes = number_of_classes)
print(synergy_results)
```

# Notes
```
 bash generate_VEC4.sh
 
bash generate_data/generate_VEC2.sh /home/lw754/masterproject/synthetic_data
python main.py --seeds 1 42 113 --n_samples_for_interaction 1000 --epochs 300 --label VEC4_ --synergy_metrics SHAPE SRI Interaction  --settings synergy uniqueness0 redundancy uniqueness1 uniqueness2 uniqueness3
python main.py --seeds 1 42 113 --n_samples_for_interaction 2000 --epochs 200 --label VEC2XOR_ --synergy_metrics SHAPE SRI Interaction EMAP PID --settings synergy uniqueness0 redundancy uniqueness1 --concat early --train_uni_model
python main.py --seeds 1 42 113 --n_samples_for_interaction 2000 --epochs 200 --label VEC3XOR_ --synergy_metrics SHAPE SRI Interaction --settings synergy uniqueness0 redundancy uniqueness1 uniqueness2 --concat early --train_uni_model
python main.py --seeds 1 42 113 --n_samples_for_interaction 2000 --epochs 250 --label VEC4XOR_ --synergy_metrics SHAPE SRI Interaction --settings synergy uniqueness0 redundancy uniqueness1 uniqueness2 uniqueness3 --concat early --train_uni_model
python main.py --seeds 1 42 113 --n_samples_for_interaction 2000 --epochs 250 --label VEC5XOR_ --synergy_metrics SHAPE SRI Interaction --settings synergy uniqueness0 redundancy uniqueness1 uniqueness2 uniqueness3 uniqueness4 --concat early --train_uni_model

% eval with XOR funktion:
python main.py --seeds 1  --n_samples_for_interaction 500 --epochs 200 --label VEC2XOR_org_ --synergy_metrics SHAPE SRI Interaction EMAP  --settings synergy uniqueness0 redundancy uniqueness1 --concat function 

conda list -e > requirements.txt

% real world
python main.py --seeds 1 42 113 --n_samples_for_interaction 2000 --epochs 200 --label brca2_ --synergy_metrics SHAPE SRI Interaction EMAP PID --settings 2_mc 2_rc 2_rm --concat early --train_uni_model
python main.py --seeds 1 42 113 --n_samples_for_interaction 200 --epochs 200 --label single-cell_ --synergy_metrics SHAPE SRI Interaction EMAP PID --settings all --concat late --train_uni_model --data_path /home/lw754/masterproject/real_world_data --number_of_classes 4

 


% with mimic
mfm
python main_mimic.py   \
    --n_samples_for_interaction 2000   \
    --label mimic_  \
    --synergy_metrics SHAPE SRI Interaction EMAP PID  \
    --settings task7_2D  \
    --concat mvae   \
    --data_path /home/lw754/masterproject/real_world_data  \
    --number_of_classes 2  \
    --pretrained_paths '/home/lw754/masterproject/results_real_world/MFM_task7_seed1.pt' '/home/lw754/masterproject/results_real_world/MFM_task7_seed42.pt' '/home/lw754/masterproject/results_real_world/MFM_task7_seed113.pt'

python main_mimic.py   \
    --n_samples_for_interaction 2000   \
    --label mimic_  \
    --synergy_metrics SHAPE SRI Interaction EMAP PID  \
    --settings taskmortality_2D  \
    --concat mvae   \
    --data_path /home/lw754/masterproject/real_world_data  \
    --number_of_classes 2  \
    --pretrained_paths '/home/lw754/masterproject/results_real_world/MFM_taskmortality_seed1.pt' '/home/lw754/masterproject/results_real_world/MFM_taskmortality_seed42.pt' '/home/lw754/masterproject/results_real_world/MFM_taskmortality_seed113.pt'

python main_mimic.py   \
    --n_samples_for_interaction 2000   \
    --label mimic_  \
    --synergy_metrics SHAPE SRI Interaction EMAP PID  \
    --settings task1_2D  \
    --concat mvae   \
    --data_path /home/lw754/masterproject/real_world_data  \
    --number_of_classes 2  \
     --pretrained_paths '/home/lw754/masterproject/results_real_world/MFM_task1_seed1.pt' '/home/lw754/masterproject/results_real_world/MFM_task1_seed42.pt' '/home/lw754/masterproject/results_real_world/MFM_task1_seed113.pt'
 

baseline
python main_mimic.py   \
    --n_samples_for_interaction 2000   \
    --label mimic_  \
    --synergy_metrics  Interaction PID  \
    --settings task7_2D  \
    --concat mvae   \
    --data_path /home/lw754/masterproject/real_world_data  \
    --number_of_classes 2  \
    --pretrained_paths '/home/lw754/masterproject/results_real_world/baseline_task7_seed1.pt' '/home/lw754/masterproject/results_real_world/baseline_task7_seed42.pt' '/home/lw754/masterproject/results_real_world/baseline_task7_seed113.pt'

python main_mimic.py   \
    --n_samples_for_interaction 2000   \
    --label mimic_  \
    --synergy_metrics   Interaction  PID  \
    --settings taskmortality_2D  \
    --concat mvae   \
    --data_path /home/lw754/masterproject/real_world_data  \
    --number_of_classes 6  \
    --pretrained_paths '/home/lw754/masterproject/results_real_world/baseline_taskmortality_seed1.pt' '/home/lw754/masterproject/results_real_world/baseline_taskmortality_seed42.pt' '/home/lw754/masterproject/results_real_world/baseline_taskmortality_seed113.pt'

python main_mimic.py   \
    --n_samples_for_interaction 2000   \
    --label mimic_  \
    --synergy_metrics   Interaction  PID  \
    --settings task1_2D  \
    --concat mvae   \
    --data_path /home/lw754/masterproject/real_world_data  \
    --number_of_classes 2  \
     --pretrained_paths '/home/lw754/masterproject/results_real_world/baseline_task1_seed1.pt' '/home/lw754/masterproject/results_real_world/baseline_task1_seed42.pt' '/home/lw754/masterproject/results_real_world/baseline_task1_seed113.pt'
    
  
MVAE
python main_mimic.py   \
    --n_samples_for_interaction 2000   \
    --label mimic_  \
    --synergy_metrics  Interaction  PID  \
    --settings task7  \
    --concat mvae   \
    --data_path /home/lw754/masterproject/real_world_data  \
    --number_of_classes 2  \
    --pretrained_paths '/home/lw754/masterproject/results_real_world/MVAE_task7_seed1.pt' '/home/lw754/masterproject/results_real_world/MVAE_task7_seed42.pt' '/home/lw754/masterproject/results_real_world/MVAE_task7_seed113.pt'
  
      
python main_mimic.py   \
    --n_samples_for_interaction 2000   \
    --label mimic_  \
    --synergy_metrics Interaction PID \
    --settings taskmortality  \
    --concat mvae   \
    --data_path /home/lw754/masterproject/real_world_data  \
    --number_of_classes 6  \
    --pretrained_paths '/home/lw754/masterproject/results_real_world/MVAE_taskmortality_seed1.pt' '/home/lw754/masterproject/results_real_world/MVAE_taskmortality_seed42.pt' '/home/lw754/masterproject/results_real_world/MVAE_taskmortality_seed113.pt'

    

python main_mimic.py   \
    --n_samples_for_interaction 2000   \
    --label mimic_  \
    --synergy_metrics Interaction  PID  \
    --settings task1  \
    --concat mvae   \
    --data_path /home/lw754/masterproject/real_world_data  \
    --number_of_classes 2  \
    --pretrained_paths '/home/lw754/masterproject/results_real_world/MVAE_task1_seed1.pt' '/home/lw754/masterproject/results_real_world/MVAE_task1_seed42.pt' '/home/lw754/masterproject/results_real_world/MVAE_task1_seed113.pt'
     
   


%%%%
cd cross-modal-interaction/synergy_evaluation
python interaction_index_dataset.py --results_path '/home/lw754/masterproject/cross-modal-interaction/results/' \
    --dataset_label 'VEC2XOR_' 'VEC2XOR_' 'VEC2XOR_' \
    --settings 'uniqueness0' 'synergy' 'uniqueness1' 'redundancy' \
    --seeds 1 42 113 \
    --concat 'early' 'intermediate' 'late' \
    --epochs 200 200 200

python interaction_index_dataset.py --results_path '/home/lw754/masterproject/cross-modal-interaction/results/' \
    --dataset_label 'VEC2XOR_org_' \
    --settings 'uniqueness0' 'synergy' 'uniqueness1' 'redundancy' \
    --seeds 1 \
    --concat 'function' \
    --epochs 200
 
 python interaction_index_dataset.py --results_path '/home/lw754/masterproject/cross-modal-interaction/results/' \
    --dataset_label ''  \
    --settings 'uniqueness0' 'synergy' 'uniqueness1' 'redundancy' 'syn_mix5-10-0' 'syn_mix10-5-0' \
    --seeds 1 42 113 \
    --concat 'early'  \
    --epochs 200 


cd cross-modal-interaction/synergy_evaluation
python interaction_index_dataset.py --results_path '/home/lw754/masterproject/cross-modal-interaction/results/' \
    --dataset_label 'VEC3XOR_' 'VEC4XOR_' 'VEC5XOR_' \
    --settings 'uniqueness0' 'synergy' 'uniqueness1' 'redundancy' \
    --seeds 1 42 113 \
    --concat 'early' 'early' 'early'  \
    --epochs 200 250 250

%% create SUM dataset
python synthetic/generate_data.py --num-data 20000 --setting synergy --out-path synthetic/experiments --num-classes 2 --transform-dim 50


%%SUM
cd cross-modal-interaction/synergy_evaluation
python interaction_index_dataset.py --results_path '/home/lw754/masterproject/cross-modal-interaction/results/' \
    --dataset_label ''  \
    --settings 'uniqueness0' 'uniqueness1' 'syn_mix5-10-0' 'syn_mix10-5-0' 'synergy'   'redundancy'   \
    --seeds 1 42 113 \
    --concat 'early'  \
    --epochs 200 
    
%% VEC3

python interaction_index_dataset.py --results_path '/home/lw754/masterproject/cross-modal-interaction/results/' \
    --dataset_label 'VEC3XOR_org' \
    --settings 'uniqueness0' 'synergy' 'uniqueness1' 'redundancy' \
    --seeds 1 \
    --concat 'function' \
    --epochs 0
    
python interaction_index_dataset.py --results_path '/home/lw754/masterproject/cross-modal-interaction/results/' \
    --dataset_label 'single-cell_' 'single-cell_' 'single-cell_' \
    --settings 'all' \
    --seeds 1 42 113 \
    --concat 'early' 'intermediate' 'late' \
    --epochs 200 200 200
```
/home/lw754/masterproject/cross-modal-interaction/results/VEC2XOR_uniqueness0_epochs_200_concat_early
/home/lw754/masterproject/cross-modal-interaction/results/VEC2XOR_uniqueness0_epochs_200_concat_early/seed_1/interaction_index_best_with_ii.csv'

/auto/archive/tcga/other_data/MIMIC-IV
/auto/archive/tcga/other_data/MIMIC-III/MultiBench_preprocessed
