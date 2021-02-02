# Lung abnormalities classification from Chest X-rays

## What is Chexpert?
CheXpert is a large dataset of chest X-rays and competition for automated chest x-ray interpretation, which features uncertainty labels and radiologist-labeled reference standard evaluation sets.
## Why Chexpert?
Chest radiography is the most common imaging examination globally, critical for screening, diagnosis, and management of many life threatening diseases. Automated chest radiograph interpretation at the level of practicing radiologists could provide substantial benefit in many medical settings, from improved workflow prioritization and clinical decision support to large-scale screening and global population health initiatives. For progress in both development and validation of automated algorithms, we realized there was a need for a labeled dataset that (1) was large, (2) had strong reference standards, and (3) provided expert human performance metrics for comparison.
## How to take part in?
CheXpert uses a hidden test set for official evaluation of models. Teams submit their executable code on Codalab, which is then run on a test set that is not publicly readable. Such a setup preserves the integrity of the test results.

Here's a tutorial walking you through official evaluation of your model. Once your model has been evaluated officially, your scores will be added to the leaderboard.**Please refer to the** [https://stanfordmlgroup.github.io/competitions/chexpert/](https://stanfordmlgroup.github.io/competitions/chexpert/)
## What the code include?
* If you want to train yourself from scratch, we provide training and test the footwork code. In addition, we provide complete training courses
* If you want to use our model in your method, we provide **a best single network pre-training model,** and you can get the network code in the code

### Train the model by yourself

* Data preparation
> We gave you the example file, which is in the folder `config/train.csv`
> You can follow it and write its path to `config/example.json`

* If you want to train the model,please run the command (you can change the configuration in config file, which is in the folder `config/example`):

```shell
pip install -r requirements.txt
python train.py
```

* If you want to train your model, please run the command:

```shell
python test.py
```

### Contact
* If you have any quesions, please post it on github issues or contact with us at [https://vnopenai.org/contact/](https://vnopenai.org/contact/)

### Reference
* [https://stanfordmlgroup.github.io/competitions/chexpert/](https://stanfordmlgroup.github.io/competitions/chexpert/)
* [https://github.com/jfhealthcare/Chexpert](https://github.com/jfhealthcare/Chexpert)
* [https://arxiv.org/abs/1911.06475](https://arxiv.org/abs/1911.06475)
* [https://www.researchgate.net/publication/316805811_Hierarchical_Multi-Label_Classification_using_Fully_Associative_Ensemble_Learning](https://www.researchgate.net/publication/316805811_Hierarchical_Multi-Label_Classification_using_Fully_Associative_Ensemble_Learning)