## Overview
**Data Shapley** applies the Data Shapley framework to assess the value of individual training data points by treating each datum as a player in a cooperative game. Data quality is a critical factor in the performance of supervised learning algorithms. This project provides an equitable measure of a data point's contribution to a model's success, which is useful for diagnosing performance, removing noisy data, and fairly evaluating datasets.

### Report of the Project
[Link to Report](https://github.com/echo85/data-shapley/blob/main/report.pdf)

## Methodology
* **Theoretical Foundation**: The project uses Cooperative Game Theory to assign a value to each data point based on its marginal contribution.
* **Algorithm**: Computations are performed using the Truncated Monte Carlo Shapley (TMC-Shapley) algorithm, which efficiently approximates Shapley values by sampling random permutations of data points.
* **Learning Algorithm**: The framework uses Logistic Regression as the baseline learning algorithm to calculate the performance metric.

## Datasets Evaluated
* **Bank Campaign Dataset**: A real-world dataset focused on predicting whether a bank client will subscribe to a term deposit.
* **Synthetic Dataset**: Generated from a Gaussian distribution with intentionally injected noise to provide a reliable ground truth for testing data valuation.

## Repository Structure
* `data_shapley.ipynb`: Main Jupyter Notebook containing the estimation and analysis code.
* `classes/`: Core Python modules implementing the TMC-Shapley logic.
* `dataset/`: Directory storing the Bank Marketing and Synthetic datasets.
* `charts/`: Generated plots and visualizations (e.g., convergence charts, performance comparisons).

### Usage
Launch the notebook:
`jupyter notebook`

Open `data_shapley.ipynb` and run the cells sequentially to calculate the Shapley values, reproduce the convergence evaluations, and generate the charts.
