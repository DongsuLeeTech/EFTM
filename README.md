# EFTM 

## NeurIPS 2024
#### Episodic Future Thinking Mechanism for Multi-agent Reinforcement Learning
[[Webpage]](https://sites.google.com/d/11amsN-ZSTpUNJOtbpBRLTGi2xCWvN2Ca/p/1ZVpJwzQAQ_fSC1tBAt91CUoZl0IIt79V/edit)

## 1. FLOW Framework
See https://flow-project.github.io/ for Detail information of this framework

### Installation
#### A. Anaconda with Python 3
1. Install prequisites: `sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6` 
2. Download the Anaconda installation file for Linux in [Anaconda](https://anaconda.com/), and unzip the file
3. Install Anaconda `bash ~/Downloads/Anaconda3-2023.03-1-Linux-x86_64.sh`
> **_NOTE:_**  we recommend you to running conda init '**yes**'.

#### B. FLOW installation
Following the below scripts in your terminal.
```
# Download FLOW github repo'.
git clone https://github.com/flow-project/flow.git
cd flow

# Create a conda env and install the FLOW
conda env create -f environment.yml
conda activate flow
python setup.py develop

# install flow on previoulsy created environment 
pip install -e .
```

###### B-1. SUMO installation
Install driving simulator (SUMO) 
```
bash scripts/setup_sumo_ubuntu1804.sh
which sumo
sumo --version
sumo-gui
```
Testing the connection between FLOW and SUMO
```
conda activate flow
python examples/simulate.py ring
```

###### B-2. Pytorch installation
Install torch: `conda install pytorch torchvision cudatoolkit=10.2 -c pytorch`
> **_NOTE:_**  Should install at least 1.6.0 version of pytorch (Recommend torch = 1.11.0 & cudatoolkit=10.2).\
> Check the [Pytorch Documents](https://pytorch.org/get-started/previous-versions/).

###### B-3. Ray RLlib installation
Install Ray: `pip install -U ray==0.8.7`
> **_NOTE:_**  Should install at least 0.8.6 version of Ray. (Recommend 0.8.7).