# ontology-project-2023

This repository contains all the files done by our group: 
- Yingqing Chen
- Robin Pourtaud

## Structure
In this project, we have 3 main folders: 
- `data`: contains processed or generated data
- `rawData`: contains raw data (only the pdf : EU.pdf)
- `results`: contains the results of the project, can be CSV, Pictures, Pickle, etc.
- `tools`: contains the libraries used in the project
- `reportsPDF`: contains the reports of the project (step1, step2...)

In the `tools` folder, here is the description of each file:
- `init.py`: contains the imports of the libraries used in the project (it is then possible to use each file like this for example: `from lib import learning`)
- ...

In the main folder, here is the description of each file:
- `main.ipynb`: contains the main code of the project
- `.gitignore`: contains the files that are not pushed on the repository
- `README.md`: the file that you are reading right now
- `requirements.txt`: contains the libraries used in the project

## Requirements
We use python 3.10 and Jupyter Notebook to run the project. 

To install the libraries, you can use the command `pip install -r requirements.txt` in the terminal. If you use Anaconda, you can use `conda install --file requirements.txt` instead.

If there is a problem with the libraries, you can probably install it usually with `pip install <library_name>`, or `conda install <library_name>` if you use Anaconda.

## How to run the project
To use this project, just open the `main.ipynb` file with Jupyter Notebook.
