#!bin/bash
# conda update -n base conda

conda create -n pytorch-scoring

source activate pytorch-scoring
pip install torch torchvision numpy matplotlib pandas scikit-learn seaborn pandas progressbar2

# echo "INFO: exporting conda requirements"
# conda list --explicit > requirements-conda.txt

# pip install progressbar2
echo "INFO: exporting pip requirements"
pip freeze > requirements-pip.txt
# pip freeze | egrep "progressbar2" > requirements-pip.txt