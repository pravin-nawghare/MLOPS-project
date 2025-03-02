#create a virtual environment
conda create -n myenv python=3.9 -y

#activate the virtual environment
conda activate myenv

#install the libraries
pip install -r requirements_dev.txt

#run in bash mode
#bash init_setup.sh