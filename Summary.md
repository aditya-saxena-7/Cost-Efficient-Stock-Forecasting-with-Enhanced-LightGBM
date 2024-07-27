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

#### Feature Engineering

1. **Data Selection**:
   - The data was sourced from the main board trading market of the Shanghai Stock Exchange, including over 1,500 stocks from 2010 to 2019. 📅🏦
   
2. **Features**:
   - **Time Series Indicators**: These include variables like week and month, helping the model understand temporal patterns. 📆
   - **Technical Indicators**: Metrics derived from historical price and volume data, like moving averages, relative strength index (RSI), Bolinger Bands, Chande Momentum Oscillator, Commodity Channel Index, Momentum. These indicators help in identifying trends and potential reversal points. 📊 (ADD LINKS HERE)
   - **OHLC Indicators**: Daily open, high, low, and close prices that summarize daily transactions. 📈📉

3. **Feature Selection**:

**Handling Missing Values**:
- Variables with more than 90% missing data were removed. This ensures the data used is complete and reliable. 🗑️

**Eliminating Unique Values**:
- Variables with a single unique value were discarded as they don't provide useful information for predictions. ❌

**Removing Highly Correlated Variables**:
- Using the Pearson correlation coefficient, variables with correlations above 90% were removed to avoid redundancy.
   
- **Example**: If height and weight are highly correlated, using both doesn't add much value. By removing one, the model becomes more efficient. 📏⚖️

**Ranking Importance**:
- Variables were ranked based on their importance to the model. Less important variables were removed to focus on the most relevant data. 📋➡️🚀

#### Labeling Methods

**Labeling**:
- The research changes the experiment into a binary classification problem, where the actions are "buy" or "sell." 📥📤
- **Label Definition**:
  - If the next day's closing price is higher than today's closing price, the label is 1 (indicating a buy signal). If not, the label is 0 (indicating a sell signal).
  - **Example**: If today's closing price is $100 and tomorrow's is $105, the label is 1 (buy). If tomorrow's is $95, the label is 0 (sell). 💸➡️💵

#### Hyper-Parameter Optimization

Hyperparameters are settings or configurations that control the learning process of a machine learning model. Unlike model parameters (like weights in a neural network), hyperparameters are not learned from the data during training. They need to be set before the training process begins.

Hyper-parameter optimization fine-tunes the parameters of the model to enhance its performance. The research utilizes the Optuna framework, an open-source tool designed for optimizing hyperparameters efficiently. Optuna automates the search for the best hyperparameters by using different sampling methods like GridSampler, RandomSampler, and TPESampler.

- **Optuna**: An open-source framework that automates the search for the best hyper-parameters using different sampling methods like GridSampler, RandomSampler, and TPESampler. 🎛️🔍  
- **Parameters Adjusted**:
  - **lambda l1 & lambda l2**: Regularization parameters to prevent overfitting, ensuring the model generalizes well to new data. 🛡️
  - **num leaves**: Number of leaves in one tree, controlling the complexity of the model. 🍃
  - **feature fraction**: Fraction of features used in each iteration, promoting diversity in learning. 📊
  - **bagging fraction & frequency**: Control the sampling of data to ensure robustness. 📦
  - **min child samples**: Minimum number of samples required on a leaf node to ensure the tree is well-formed. 🌱

#### Key Features of Optuna 🚀

(Add link here)

1. **Define-by-Run**
2. **Efficient Sampling**
3. **Pruning Mechanism**
4. **Visualizations**
5. **Integration with Popular Libraries**

#### Cross-Validation with Time Series Data ⏳

**Cross-validation** is a technique used to evaluate the performance of a model by partitioning the data into training and testing sets multiple times. In time series data, an 'n-year sliding window' approach is often used, but selecting the right 'n' can be challenging.

- **Example**: If you want to test the durability of a product, you would use it in different conditions over time to ensure it performs well consistently. Similarly, cross-validation tests the model's performance over different time periods to ensure reliability.

For this study, three training sets were considered:
1. 2010-2012
2. 2010-2015
3. 2010-2018

The test sets were the subsequent years: 2013, 2016, and 2019, respectively.

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

### Cost Awareness: Optimizing Stock Predictions 📊💡

#### What is Cost Awareness? 🤔
Cost awareness in stock prediction refers to the model's sensitivity to different types of prediction errors, particularly focusing on minimizing false-positive errors. A false-positive error in stock trading is when the model predicts a stock price will go up (a "buy" signal), but it doesn't, leading to a potential loss.

In stock prediction, not all errors have the same cost. A false-positive error (FP) can lead to direct financial loss, while a false-negative error (FN) results in missed profit opportunities but doesn't incur a direct loss. Hence, it's crucial to design the model to minimize false-positive errors.

- **Example**: If a model incorrectly signals a buy (FP) and the stock doesn't rise, the investor loses money. However, if the model misses a buy signal (FN), the investor only misses a profit opportunity but doesn't lose money. 🏦➡️🚫📉

#### Traditional Evaluation Metrics 📏

Traditional stock prediction models are evaluated using standard binary classification metrics:
1. **Accuracy**: The ratio of correctly predicted observations to the total observations.
   - **Formula**: Accuracy = TP + TN / TP + TN + FP + FN
   - **Example**: If you correctly predict 80 out of 100 stock movements, your accuracy is 80%. 🎯

2. **Recall (Sensitivity)**: The ratio of true positive predictions to the sum of true positive and false negative predictions.
   - **Formula**: Recall = TP / TP + FN
   - **Example**: If you correctly predict 70 out of 100 actual positive movements, your recall is 70%. 🔍

3. **Precision**: The ratio of true positive predictions to the sum of true positive and false positive predictions.
   - **Formula**: Precision = TP / TP + FP
   - **Example**: If 60 out of 100 predicted positive movements are actually positive, your precision is 60%. 🧐

4. **F1 Score**: The harmonic mean of precision and recall.
   - **Formula**: F1 = 2 * Precision * Recall / (Precision + Recall)
   - **Example**: If both precision and recall are 75%, the F1 score is also 75%. 🔄

#### Confusion Matrix 📊

The confusion matrix is a table used to describe the performance of a classification model. It has four components:
- **True Positive (TP)**: Correctly predicted positive cases.
- **True Negative (TN)**: Correctly predicted negative cases.
- **False Positive (FP)**: Incorrectly predicted positive cases (Type I error).
- **False Negative (FN)**: Incorrectly predicted negative cases (Type II error).

**Table: Confusion Matrix**
|                | Predicted Negative | Predicted Positive |
|----------------|--------------------|--------------------|
| Actual Negative| True Negative (TN) | False Positive (FP)|
| Actual Positive| False Negative (FN)| True Positive (TP) |

- **True Positive (TP) and True Negative (TN) costs** are considered zero since they don't incur any loss.
- **False Positive (FP) cost (\(fp\_Amt\))**: The financial loss incurred due to a false buy signal.
- **False Negative (FN) cost (\(fn\_Amt\))**: The missed profit opportunity due to a false sell signal.

**Algorithm: Calculating Cost Matrix**
1. Initialize investment amount.
2. Calculate transaction amounts for each prediction.
3. Compute service charges and stamp duties.
4. Determine false-positive and false-negative costs.
5. Aggregate individual costs to get the total cost.

### Calculate Cost Matrix 📊💸

```plaintext
Algorithm 2: Calculate Cost Matrix
Input: test_df : array of shape = [n_samples]
Input: money_init: amount invested in each stock
Output: cost_mat: array-like of shape = [n_samples, 4]

init buy_rate, sell_rate, stamp duty
money = money_init
cost_df = test_df
for all (i, row) ∈ cost_df do
  fp_rate = fabs(row[buy_price] − row[sell_price])
  fn_rate = fp_rate
  tran_num = (money / row[buy_price]) // 100
  buy_money = tran_num * row[buy_price]
  sell_money = tran_num * row[sell_price]
  service_change = buy_money * buy_rate + sell_money * sell_rate
  stamp_duty = stamp_duty * sell_money
  fp_Amt[i] = fp_rate * tran_num + service_change + stamp_duty
  fn_Amt[i] = fn_rate * tran_num − service_change − stamp_duty
end for
cost_mat[:, 0] = fp_Amt
cost_mat[:, 1] = fn_Amt
cost_mat[:, 2] = 0.0
cost_mat[:, 3] = 0.0
return cost_mat
```
#### Variables:
- `buy_rate`: The rate applied when buying stocks.
- `sell_rate`: The rate applied when selling stocks.
- `stamp_duty`: The tax or duty applied on stock transactions.
- `money`: The initial amount of money available for investment.
- `cost_df`: A copy of `test_df` used for calculations.
- `fp_rate`: The rate for calculating the cost of false-positive errors.
- `fn_rate`: The rate for calculating the cost of false-negative errors.
- `tran_num`: The number of transactions based on the available money and stock price.
- `buy_money`: The total amount spent on buying stocks.
- `sell_money`: The total amount received from selling stocks.
- `service_change`: The total service charge for the transactions.
- `fp_Amt`: The array storing the calculated false-positive costs.
- `fn_Amt`: The array storing the calculated false-negative costs.
- `cost_mat`: The final cost matrix containing all the costs.

### Step-by-Step Explanation:

1. **Initialization**:
   ```python
   init buy_rate, sell_rate, stamp duty
   money = money_init
   cost_df = test_df
   ```
   - Initialize the buy rate, sell rate, and stamp duty.
   - Set the initial amount of money (`money`) to `money_init`.
   - Copy the test data (`test_df`) to `cost_df` for further processing.

2. **Loop Through Each Row of `cost_df`**:
   ```python
   for all (i, row) ∈ cost_df do
   ```
   - Iterate through each row of the `cost_df` to calculate the costs for each stock.

3. **Calculate Rates**:
   ```python
   fp_rate = fabs(row[buy_price] − row[sell_price])
   fn_rate = fp_rate
   ```
   - Calculate the false-positive rate (`fp_rate`) as the absolute difference between the buy price and the sell price.
   - Set the false-negative rate (`fn_rate`) to be equal to the false-positive rate.

4. **Calculate Transaction Numbers and Amounts**:
   ```python
   tran_num = (money / row[buy_price]) // 100
   buy_money = tran_num * row[buy_price]
   sell_money = tran_num * row[sell_price]
   ```
   - Determine the number of transactions (`tran_num`) that can be made with the available money.
   - Calculate the total amount spent on buying stocks (`buy_money`).
   - Calculate the total amount received from selling stocks (`sell_money`).
  
5. **Calculate Service Charges and Stamp Duty**:
   ```python
   service_change = buy_money * buy_rate + sell_money * sell_rate
   stamp_duty = stamp_duty * sell_money
   ```
   - Calculate the total service charge (`service_change`) based on the buy and sell amounts and their respective rates.
   - Calculate the stamp duty (`stamp_duty`) based on the sell amount.

6. **Calculate False-Positive and False-Negative Costs**:
   ```python
   fp_Amt[i] = fp_rate * tran_num + service_change + stamp_duty
   fn_Amt[i] = fn_rate * tran_num − service_change − stamp_duty
   ```
   - Compute the false-positive cost (`fp_Amt`) for the current row.
   - Compute the false-negative cost (`fn_Amt`) for the current row.

7. **Assign Calculated Costs to Cost Matrix**:
   ```python
   cost_mat[:, 0] = fp_Amt
   cost_mat[:, 1] = fn_Amt
   cost_mat[:, 2] = 0.0
   cost_mat[:, 3] = 0.0
   ```
   - Assign the calculated false-positive costs to the first column of the cost matrix.
   - Assign the calculated false-negative costs to the second column of the cost matrix.
   - Set the third and fourth columns of the cost matrix to zero.

8. **Return the Cost Matrix**:
   ```python
   return cost_mat
   ```
   - Return the final cost matrix containing the calculated costs.
  
#### Implementing the Cost Matrix in LightGBM

**Step:** Adjusting LightGBM Parameters for Cost Sensitivity

To integrate cost awareness, the algorithm adjusts the `scale_pos_weight` parameter. This parameter helps balance the class distribution and makes the model more sensitive to the class with higher cost (false positives in this case).

- The `scale_pos_weight` parameter in LightGBM is tuned using Optuna to make the model more sensitive to false-positive errors.
- The cost matrix guides this adjustment by assigning a higher cost to false-positive errors, making the model more cautious about generating "buy" signals without strong evidence.

**Example:**
- If `scale_pos_weight` is set to a high value, LightGBM will weigh false positives more heavily during training, reducing their occurrence in the final model.

**Step:** Model Performance Evaluation
- Performance is measured using precision, F0.5 score (which prioritizes precision over recall), and risk indicators like Sharpe ratio and Sortino ratio. (ADD LINK HERE)

**Example:**
- If the model achieves a precision of 58.65% and an F0.5 score of 57.94%, it indicates a high accuracy in predicting profitable trades while minimizing false positives.

### Detailed Example

Let's consider a practical example to illustrate how minimizing false positives works:

1. **Data Input:**
   - Daily stock prices for the past 10 years.
   - Technical indicators such as moving averages, momentum, and volatility.

2. **Training the Model:**
   - LightGBM is trained on this data with features such as the 10-day moving average, 14-day RSI (Relative Strength Index), and daily closing prices.
   - Hyperparameters are optimized using Optuna.

3. **Cost Matrix Implementation:**
   - Define the cost of false positives (`fp_Amt`) and false negatives (`fn_Amt`).
   - `fp_Amt` could be set to $1000 (cost of wrong "buy" signals), and `fn_Amt` to $500 (cost of missed "buy" signals).

4. **Adjusting LightGBM:**
   - The `scale_pos_weight` parameter is adjusted to make the model more sensitive to false positives. For instance, if initially set to 1 (no bias), it might be increased to 5 to reduce false positives significantly.
  
5. **Model Prediction:**
   - The model predicts stock prices for the next day. It only issues a "buy" signal if the predicted probability of price increase is very high, reducing the chances of false positives.
   - If the model predicts a 70% probability of price increase, it evaluates the potential cost.
If the cost of a false positive (buying and the price doesn’t rise) is high, the model might avoid issuing a "buy" signal unless the probability is higher, say 85%.






