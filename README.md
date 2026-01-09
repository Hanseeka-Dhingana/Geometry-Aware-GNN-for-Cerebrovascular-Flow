# Geometry-Aware-GNN-for-Cerebrovascular-Flow

## Overview
This project explores the use of **Graph Neural Networks (GNNs)** as a fast surrogate for estimating **cerebrovascular blood flow patterns** from vascular geometry.
The model is trained on **precomputed CFD-derived velocity fields** and operates directly on **graph representations of blood vessels**.

The goal is to reduce the **computational time and complexity** of running full CFD simulations for each new case.

---

## Key Idea

* Blood vessels are represented as **graphs**
* CFD velocity fields are used as **training targets**
* A GNN learns flow patterns from geometry and connectivity
* Once trained, the model provides **fast flow estimation without rerunning CFD**

---

## What This Project Does
* Uses vascular geometry as graph input
* Trains a GNN on CFD-generated blood flow data
* Estimates relative blood flow patterns efficiently

---

## Project Structure

```
├── src/        # Data processing, training, evaluation
├── outputs/     # Outputs and visualizations
├── data/         # Dataset
├── requirements.txt
└── README.md
```

---

## Tools & Technologies

* Python
* PyTorch / PyTorch Geometric
* Jupyter Notebook

