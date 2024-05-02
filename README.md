#

## Cross modal interaction metrics

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
python main.py --seeds 1 42 113 --n_samples_for_interaction 200 --epochs 200 --label mimic_ --synergy_metrics SHAPE SRI Interaction EMAP PID --settings task7 --concat early --train_uni_model --data_path /home/lw754/masterproject/real_world_data --number_of_classes 2


%%%%
cd cross-modal-interaction/synergy_evaluation
python interaction_index_dataset.py --results_path '/home/lw754/masterproject/cross-modal-interaction/results/' \
    --dataset_label 'VEC2XOR' 'VEC2XOR' 'VEC2XOR' \
    --settings 'uniqueness0' 'synergy' 'uniqueness1' 'redundancy' \
    --seeds 1 42 113 \
    --concat 'early' 'intermediate' 'late' \
    --epochs 200 200 200

python interaction_index_dataset.py --results_path '/home/lw754/masterproject/cross-modal-interaction/results/' \
    --dataset_label 'VEC2XOR_org' \
    --settings 'uniqueness0' 'synergy' 'uniqueness1' 'redundancy' \
    --seeds 1 \
    --concat 'function' \
    --epochs 200

%% VEC3

python interaction_index_dataset.py --results_path '/home/lw754/masterproject/cross-modal-interaction/results/' \
    --dataset_label 'VEC3XOR_org' \
    --settings 'uniqueness0' 'synergy' 'uniqueness1' 'redundancy' \
    --seeds 1 \
    --concat 'function' \
    --epochs 0
```
/home/lw754/masterproject/cross-modal-interaction/results/VEC2XOR_uniqueness0_epochs_200_concat_early
/home/lw754/masterproject/cross-modal-interaction/results/VEC2XOR_uniqueness0_epochs_200_concat_early/seed_1/interaction_index_best_with_ii.csv'

/auto/archive/tcga/other_data/MIMIC-IV
