### Cost Efficient Stock Forecasting with Enhanced LightGBM 📈💡

### Table of Contents
1. [Introduction](https://github.com/aditya-saxena-7/Cost-Efficient-Stock-Forecasting-with-Enhanced-LightGBM/blob/master/Introduction.md)
2. [Model and Methodology](https://github.com/aditya-saxena-7/Cost-Efficient-Stock-Forecasting-with-Enhanced-LightGBM/blob/master/Model%20and%20Methodology.md)
3. [Data Descriptions and Feature Engineering](https://github.com/aditya-saxena-7/Cost-Efficient-Stock-Forecasting-with-Enhanced-LightGBM/blob/master/Data%20Descriptions%20and%20Feature%20Engineering.md)
4. [Hyperparameter Optimization](https://github.com/aditya-saxena-7/Cost-Efficient-Stock-Forecasting-with-Enhanced-LightGBM/blob/master/Hyperparameter%20Optimization.md)
5. [Cost Awareness](https://github.com/aditya-saxena-7/Cost-Efficient-Stock-Forecasting-with-Enhanced-LightGBM/blob/master/Cost%20Awareness.md)
6. [Performance and Measurement](https://github.com/aditya-saxena-7/Cost-Efficient-Stock-Forecasting-with-Enhanced-LightGBM/blob/master/Performance%20and%20Measurement.md)
7. [Conclusion](https://github.com/aditya-saxena-7/Cost-Efficient-Stock-Forecasting-with-Enhanced-LightGBM/blob/master/Conclusion.md)

#### Objective:

This paper focuses on using machine learning techniques to predict stock prices while being mindful of costs involved in trading. Cost awareness focuses on minimizing false-positive errors, which are incorrect predictions of stock price increases. 🛑📊

#### Four Main Stages of the Study:

1. **Feature Engineering**:
   - This involves selecting and processing the right data features that will be used to train the model. 📊🔧
   - In this study, data from the Shanghai Stock Exchange, including over 1,500 stocks from 2010 to 2019, was used. 📅🏦

2. **Hyper-Parameter Optimization**:
   - This stage focuses on fine-tuning the model's parameters to improve its performance. 🎛️
   - The Optuna framework was used for this purpose, which helps in finding the best set of parameters through various trials. 🔍

3. **Cost Awareness Adjustment**:
   - Here, the model is adjusted to be more sensitive to false-positive errors to reduce potential investment losses. 💡
   - By optimizing certain parameters, the model becomes better at distinguishing real investment opportunities from fake ones. 📈📉

4. **Model Effect Evaluation**:
   - The final stage involves evaluating the model's performance in terms of accuracy, profitability, and risk control. 🏆
   - The model's predictions are compared with those from other popular models like XGBoost and Random Forest. ⚖️

#### Model Used: LightGBM (Light Gradient Boosting Machine)

**LightGBM** is a powerful machine learning framework that implements the Gradient Boosting Decision Tree (GBDT) algorithm. It's known for its efficiency and high performance in handling large datasets.

- **Gradient Boosting**: This technique builds multiple decision trees sequentially, where each new tree corrects errors made by the previous ones. 

(Add link here)

### How LightGBM Works 🔧

1. **Initialization**: LightGBM starts with an initial prediction, often the mean of the target variable.
2. **Gradient Computation**: For each iteration, gradients of the loss function with respect to the current prediction are computed.
3. **GOSS**: Instances are sampled using GOSS.
4. **Tree Construction**: Trees are grown leaf-wise using the sampled data. Each split is chosen to maximize the reduction in the loss function.
5. **Model Update**: Predictions are updated by adding the new tree's predictions multiplied by a learning rate.
6. **Iteration**: Steps 2-5 are repeated for a specified number of iterations or until convergence.

### Parameters in LightGBM ⚙️

Some important parameters to tune in LightGBM include:

- `num_leaves`: The maximum number of leaves in one tree. Higher values can improve accuracy but may lead to overfitting.
- `max_depth`: Maximum depth of the tree. Helps in controlling overfitting.
- `learning_rate`: Controls the contribution of each tree. Lower values lead to slower training but potentially better accuracy.
- `n_estimators`: The number of trees (boosting rounds).
- `feature_fraction`: The fraction of features to consider when building each tree. Helps in reducing overfitting.
- `bagging_fraction`: The fraction of data to use for each tree. Combined with `bagging_freq`, it helps in reducing overfitting.
- `lambda_l1` and `lambda_l2`: L1 and L2 regularization terms. Helps in controlling overfitting.

#### Feature Engineering

1. **Data Selection**:
   - The data was sourced from the main board trading market of the Shanghai Stock Exchange, including over 1,500 stocks from 2010 to 2019. 📅🏦
   
2. **Features**:
   - **Time Series Indicators**: These include variables like week and month, helping the model understand temporal patterns. 📆
   - **Technical Indicators**: Metrics derived from historical price and volume data, like moving averages and relative strength index (RSI). These indicators help in identifying trends and potential reversal points. 📊 (ADD LINKS HERE)
   - **OHLC Indicators**: Daily open, high, low, and close prices that summarize daily transactions. 📈📉

3. **Feature Selection**:
   - **Removing Missing Values**: Features with over 90% missing data were eliminated to ensure data quality. 🗑️
   - **Unique Value Elimination**: Features with a single unique value were removed as they do not provide useful information. ❌
   - **Correlation Filtering**: Highly correlated features (correlation > 90%) were removed to prevent redundancy. Think of this as removing duplicate ingredients in a recipe. 🍲➡️🌟
   - **Importance Ranking**: Features were ranked based on their importance to the model, and less important ones were discarded. 📋➡️🚀

#### Hyper-Parameter Optimization

Hyper-parameter optimization fine-tunes the parameters of the model to enhance its performance. This research uses the Optuna framework for this purpose.

- **Optuna**: An open-source framework that automates the search for the best hyper-parameters using different sampling methods like GridSampler, RandomSampler, and TPESampler. 🎛️🔍
- **Parameters Adjusted**:
  - **lambda l1 & lambda l2**: Regularization parameters to prevent overfitting, ensuring the model generalizes well to new data. 🛡️
  - **num leaves**: Number of leaves in one tree, controlling the complexity of the model. 🍃
  - **feature fraction**: Fraction of features used in each iteration, promoting diversity in learning. 📊
  - **bagging fraction & frequency**: Control the sampling of data to ensure robustness. 📦
  - **min child samples**: Minimum number of samples required on a leaf node to ensure the tree is well-formed. 🌱

#### Key Features of Optuna 🚀

1. **Define-by-Run**
2. **Efficient Sampling**
3. **Pruning Mechanism**
4. **Visualizations**
5. **Integration with Popular Libraries**










