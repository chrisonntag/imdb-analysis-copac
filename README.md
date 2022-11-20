# Exploratory Analysis on the IMDB Binary Dataset and COPAC implementation

Exploratory analysis part of the Data Mining course. All the utility functions are used from the 
[ICML 2020 Workshop on Graph Representation Learning and Beyond](https://github.com/chrsmrrs/tudataset) repository. 

We are also comparing our COPAC implementation with the [ELKI](https://elki-project.github.io) implementation which is 
why the \*.jar is part of this repository. 

## Setup
Just clone this repository and install packages with pip from the ```requirements.txt``` file. I altered some 
of the code from the TUDataset repo so that we don't have to deal with the correct version numnber of torch 
since we don't need torch for our uses anyway. 

Clone the repo.

```
git clone https://github.com/chrisonntag/imdb-analysis-copac.git
```
(or use some UI client)

Create a virtual environement. This creates a directory ```env/``` where all the dependencies will be installed.
```
python3 -m venv env
```

Choose the environment.
```
source env/bin/activate
```

Install all requirements.
```
pip install -r requirements.txt
```

Open Jupyter lab.
```
jupyter lab
```

Everything should work from hereon since all needed artifacts are part of this repo as well. 
You can find the data analysis notebook for the exploratory part and our COPAC implementation 
in ```/analysis/eda.ipynb```. 

Feel free to move the COPAC implementation into its own module. 


