 bash generate_VEC4.sh
 
bash generate_data/generate_VEC2.sh /home/lw754/masterproject/synthetic_data
python main.py --seeds 1 42 113 --n_samples_for_interaction 1000 --epochs 300 --label VEC4_ --synergy_metrics SHAPE SRI Interaction  --settings synergy uniqueness0 redundancy uniqueness1 uniqueness2 uniqueness3
python main.py --seeds 1 42 113 --n_samples_for_interaction 2000 --epochs 200 --label VEC2XOR_ --synergy_metrics SHAPE SRI Interaction EMAP PID --settings synergy uniqueness0 redundancy uniqueness1 --concat early --train_uni_model
python main.py --seeds 1 42 113 --n_samples_for_interaction 2000 --epochs 200 --label VEC3XOR_ --synergy_metrics SHAPE SRI Interaction --settings synergy uniqueness0 redundancy uniqueness1 uniqueness2 --concat early --train_uni_model
python main.py --seeds 1 42 113 --n_samples_for_interaction 2000 --epochs 250 --label VEC4XOR_ --synergy_metrics SHAPE SRI Interaction --settings synergy uniqueness0 redundancy uniqueness1 uniqueness2 uniqueness3 --concat early --train_uni_model
python main.py --seeds 1 42 113 --n_samples_for_interaction 2000 --epochs 250 --label VEC5XOR_ --synergy_metrics SHAPE SRI Interaction --settings synergy uniqueness0 redundancy uniqueness1 uniqueness2 uniqueness3 uniqueness4 --concat early --train_uni_model
