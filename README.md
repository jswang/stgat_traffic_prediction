# Spatial-Temporal Graph Attention Networks:A Deep Learning Approach for Traffic Forecasting

This repository provides an open source implementation of the Spatio-Temporal GAT introduced by Zhang et al in "Spatial-Temporal Graph Attention Networks:A Deep Learning Approach for Traffic Forecasting" https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8903252

## Running the code

We recommend creating and activating a virtual environment to run this repo. Use the following steps:
```
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
```

To start training, simple run:
```
python3 main.py
```

We also provide `.launch` files in the `.vscode` folder for launching main through the python debugger.

## Folder structure

Dataloading and preprocessed datasets are available in `data_loader` and `dataset`. The model is available in `models/st_gat.py`. Training functions are provided in `models/trainer.py`.
```
├── data_loader
│   ├── dataloader.py
│   └── __init__.py
├── dataset
│   ├── PeMSD7_V_228.csv
│   └── PeMSD7_W_228.csv
├── models
│   ├── __init__.py
│   ├── st_gat.py
│   └── trainer.py
├── runs
│   ├── model_final_200epochs.pt
│   └── model_final_60epochs.pt
├── utils
│   ├── __init__.py
|   └── math_utils.py
├── main.py
├── requirements.txt
└── README.md
```


## Citation
This repository was based on that provided by Bing Yu*, Haoteng Yin*, Zhanxing Zhu. [Spatio-temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting](https://www.ijcai.org/proceedings/2018/0505). In *Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI)*, 2018

    @inproceedings{yu2018spatio,
        title={Spatio-temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting},
        author={Yu, Bing and Yin, Haoteng and Zhu, Zhanxing},
        booktitle={Proceedings of the 27th International Joint Conference on Artificial Intelligence (IJCAI)},
        year={2018}
    }

The model architecture came from C. Zhang, J. J. Q. Yu and Y. Liu, “Spatial-Temporal Graph Attention Networks: A Deep Learning Approach for Traffic Forecasting,” in IEEE Access, vol. 7, pp. 166246–166256, 2019, doi: 10.1109/ACCESS.2019.2953888
