#!/bin/bash

# Train First model h1
python3 main_mnist.py --logdir ./h1_pgd_mnist.pt --epochs 16 --adv_epochs 14

# Create P2 dataset containing half samples where h1 has adversarial examples and half samples of the original dataset
python3 create_dataset.py --saved_model ./h1_pgd_mnist.pt  --logdir ./P2_dataset.pt

# Train Second model h2 on P2
python3 main_mnist.py --logdir ./h2_pgd_mnist.pt --epochs 16 --adv_epochs 14 --custom_dataset ./P2_dataset.pt

# Create P3 dataset containing samples where there exists a pertrubation where h1 and h2 are not equal
python3 create_dataset.py --saved_model ./h1_pgd_mnist.pt --set_second_model ./h2_pgd_mnist.pt --logdir ./P3_dataset.pt

# Train third model h3 on P3
python3 main_mnist.py --logdir ./h3_pgd_mnist.pt --epochs 16 --adv_epochs 14 --custom_dataset ./P3_dataset.pt
