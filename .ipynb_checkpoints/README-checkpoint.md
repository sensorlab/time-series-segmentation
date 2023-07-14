# time-series-segmentation

# Content
## Rutgers
The Rutgers.ipynb notebook is designed to convert time series data from the Rutgers dataset into graphs. These graphs are categorized into 5 groups, each representing a specific anomaly type. The notebook includes a GINE model for classifying the graphs based on the anomaly types and a GAT model for identifying the nodes where the anomalies occurred within the time series.

## Rutgers Importance
Rutgers_importance.ipynb is a notebook that builds on Rutgers.ipynb. It calculates importance scores for each node using the gradient method. The importance score can be calculated when the model is trained by sending the test loader through the model and inspecting the gradients. Nodes with higher scores are considered more important. To assess the accuracy of the importance scores, the notebook includes a modified test step where the most important nodes are masked, and the prediction changes are observed. You can choose between the importance calculation method provided by GraphXAI and the one implemented in this notebook.

## TSSB
TSSB.ipynb is a notebook that converts time series from the TSSB dataset into graphs. The dataset consists of 75 different time series, each with varying lengths and segments. This notebook transforms the time series into graphs, selects labels based on the segments, and determines which nodes will be used for training and testing by applying masks.
The TSSB datasets can be acquired from this github repository: [TSSB](https://github.com/ermshaua/time-series-segmentation-benchmark/tree/main/tssb/datasets)

# Results
Rutgers has a trained model and an Rutgers.xlsx excell table using other combination of graph creation.

TSSB has a TSSB_results folder containing the main results using only training and testing masks an TSSB.xlsx excell table with some more results of models using training, validation and testing masks.

For the Rutgers importance there is a folder named importance_for_VG that shows plots that evaluate the Gradient method presented in this notebook and the GraphXAI method.

# How to install and run
Each of the three main notebooks contains cells at the beginning that install the required versions of the packages.
In each notebook, the cells, with the exception of pip install, Config, and the last main cell, are functions that must be executed before the last cell is executed, which runs the entire program.
To change the parameters, you can edit the Config Cell located before the last cell in each notebook.

# Citations
@inproceedings{bertalanivc2023graph,
  title={Graph Isomorphism Networks for Wireless Link Layer Anomaly Classification},
  author={Bertalani{\v{c}}, Bla{\v{z}} and Fortuna, Carolina},
  booktitle={2023 IEEE Wireless Communications and Networking Conference (WCNC)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}

@INPROCEEDINGS{10167910,
  author={Bertalanič, Blaž and Vnučec, Matej and Fortuna, Carolina},
  booktitle={2023 International Balkan Conference on Communications and Networking (BalkanCom)}, 
  title={Graph Neural Networks Based Anomalous RSSI Detection}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/BalkanCom58402.2023.10167910}}
  
GraphXAI project:
[Github] (https://github.com/mims-harvard/GraphXAI/tree/main)
@article{agarwal2023evaluating,
  title={Evaluating Explainability for Graph Neural Networks},
  author={Agarwal, Chirag and Queen, Owen and Lakkaraju, Himabindu and Zitnik, Marinka},
  journal={Scientific Data},
  volume={10},
  number={144},
  url={https://www.nature.com/articles/s41597-023-01974-x},
  year={2023},
  publisher={Nature Publishing Group}
}