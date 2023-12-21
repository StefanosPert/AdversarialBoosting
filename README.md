## Adversarial Robustness in Model Ensemble
This repository provides the code for the following technical report "Adversarial Robustness in Model Ensembles" which can be found [here](https://stefanospert.github.io/data/AdversarialBoosting.pdf)

### Training the robust ensemble
In order to train the ensemble of model to MNIST run the following bash script
```
./training.sh
```

This will create the 3 trained models h1_pgd_mnist.pt, h2_pgd_mnist.pt, h3_pgd_mnist.pt.

<br />

Then in order to test the overall ensemble on the test set of MNIST run the following
```
python3 adversarial_test.py --saved_model h1_pgd_mnist.pt --saved_model2 h2_pgd_mnist.pt --saved_model3 h3_pgd_mnist.pt
```

<br />

To test the individual models on the test set of MNIST you can run adversarial_test.py using only one of the models as arguments.
For example to test h1 classifier run
```
python3 adversarial_test.py --saved_model h1_pgd_mnist.pt
```
