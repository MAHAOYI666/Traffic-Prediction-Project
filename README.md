# High-Speed Rail Network Traffic Prediction

This project implements and evaluates several deep learning models for predicting network traffic in high-speed rail scenarios. The primary model is a CNN-Transformer hybrid, with comparisons against baseline models such as RNN, LSTM, Bi-LSTM, CNN, and a standard Transformer.

traffic_prediction_project/
├── data/
│   └── HSR.csv
├── results/
│   ├── plots/
│   │   ├── .gitkeep
│   │   └── README.md
│   └── saved_models/
│       ├── .gitkeep
│       └── README.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_processing.py
│   ├── trainer.py
│   ├── utils.py
│   └── models/
│       ├── __init__.py
│       ├── base_layers.py
│       ├── cnn_transformer.py
│       └── baselines.py
├── train.py
├── evaluate.py
├── tune_hyperparams.py
└── run_comparison.py
