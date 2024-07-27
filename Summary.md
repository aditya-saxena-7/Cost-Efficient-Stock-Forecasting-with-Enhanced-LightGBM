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

#### How Are `fp_Amt` and `fn_Amt` Calculated?

1. **Data Inputs:**
   - **Buy Price:** Price at which you would buy the stock.
   - **Sell Price:** Price at which you would sell the stock.
   - **Number of Shares:** Number of shares you are trading.
   - **Buy Rate:** Transaction fee for buying the stock.
   - **Sell Rate:** Transaction fee for selling the stock.
   - **Stamp Duty:** Tax applied on stock transactions.

2. **Calculate Amounts for False Positive:**
   - When you buy a stock at the buy price expecting it to go up, but it doesn't and instead, you sell it at a lower price.
   - Example: Buy 100 shares at $50 each, sell them at $45 each.
   - **Cost Calculation:**
     - Loss per share: $50 (buy price) - $45 (sell price) = $5
     - Total Loss: 100 shares * $5 = $500
     - Add transaction fees and taxes to this loss to get `fp_Amt`.

3. **Calculate Amounts for False Negative:**
   - When you don't buy a stock at the buy price, missing out on the chance to sell it at a higher price.
   - Example: Buy 100 shares at $50 each, sell them at $55 each.
   - **Cost Calculation:**
     - Missed profit per share: $55 (sell price) - $50 (buy price) = $5
     - Total Missed Profit: 100 shares * $5 = $500
     - Subtract transaction fees and taxes to get `fn_Amt`.
  
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

### Understanding `scale_pos_weight`

The `scale_pos_weight` parameter in LightGBM adjusts the weight of positive samples to balance the classes. By increasing this weight, the algorithm penalizes errors on the positive class more heavily, which in turn reduces the likelihood of false positives.

### Cost Matrix and `scale_pos_weight`

A cost matrix assigns different penalties to different types of classification errors. For stock prediction:
- **False Positive (FP)**: Predicting a price increase (buy signal) when the price does not increase.
- **False Negative (FN)**: Not predicting a price increase (missed buy signal).

### Example Cost Matrix
Consider the following cost matrix:

     ```python
     cost_matrix = [[0, fp_Amt],
                    [fn_Amt, 0]]
     ```
### Calculating `scale_pos_weight`

To adjust `scale_pos_weight` using the cost matrix, you need to derive a weight that reflects the relative costs of false positives and false negatives. The idea is to increase the weight of the positive class based on the cost ratio.

**Step-by-Step Calculation:**

1. **Define Costs:**
   - \( fp\_Amt \) = Cost of a false positive.
   - \( fn\_Amt \) = Cost of a false negative.

2. **Calculate Cost Ratio:**
   - The ratio of false positive cost to false negative cost:
   \[
   \text{Cost Ratio} = \frac{fp\_Amt}{fn\_Amt}
   \]

3. **Set `scale_pos_weight`:**
   - The `scale_pos_weight` parameter is then adjusted according to this cost ratio to ensure the model penalizes false positives more heavily.
   \[
   \text{scale\_pos\_weight} = \text{Cost Ratio}
   \]

### Example Calculation

Let's assume:
- \( fp\_Amt = 1000 \) (Cost of a false positive)
- \( fn\_Amt = 200 \) (Cost of a false negative)

1. **Calculate Cost Ratio:**
   \[
   \text{Cost Ratio} = \frac{1000}{200} = 5
   \]

2. **Set `scale_pos_weight`:**
   \[
   \text{scale\_pos\_weight} = 5
   \]

This means that the model will treat false positives as being 5 times more costly than false negatives, thereby increasing the weight for the positive class (buy signal).

### Implementation in LightGBM

When setting up the LightGBM model, you incorporate this calculated `scale_pos_weight`:

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(scale_pos_weight=5,  # Adjusted weight
                       num_leaves=31,
                       feature_fraction=0.8,
                       bagging_fraction=0.8,
                       bagging_freq=5)

model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=10)
```

### Understanding the Loss Function in LightGBM

LightGBM, like other gradient boosting frameworks, optimizes a loss function during training. For binary classification, a common loss function is the binary cross-entropy (logistic loss).

#### Binary Cross-Entropy Loss Function:

The binary cross-entropy loss for a single prediction is given by:

\[ \text{Loss} = -y \log(p) - (1 - y) \log(1 - p) \]

where:
- \( y \) is the actual label (1 for positive, 0 for negative).
- \( p \) is the predicted probability of the positive class.

#### Modified Loss Function:

With `scale_pos_weight`, the loss function for each instance becomes:

\[ \text{Loss} = -w_y \cdot y \log(p) - w_{1-y} \cdot (1 - y) \log(1 - p) \]

where:
- \( w_y \) is the weight for the positive class.
- \( w_{1-y} \) is the weight for the negative class.

For binary classification with `scale_pos_weight`, we typically adjust the weight for the positive class. Let's denote `scale_pos_weight` as \( w_p \).

- For positive class (y=1): \( w_y = w_p \)
- For negative class (y=0): \( w_{1-y} = 1 \)

Thus, the modified loss function becomes:

\[ \text{Loss} = -w_p \cdot y \log(p) - (1 - y) \log(1 - p) \]

### Example Calculation

#### Without `scale_pos_weight`:

Consider the following example:

- Actual label (\( y \)): 1 (positive class)
- Predicted probability (\( p \)): 0.7

The binary cross-entropy loss without any weights is:

\[ \text{Loss} = -1 \cdot \log(0.7) - (1 - 1) \cdot \log(1 - 0.7) \]
\[ \text{Loss} = -\log(0.7) \approx 0.3567 \]

#### With `scale_pos_weight`:

Suppose we set `scale_pos_weight` \( w_p = 5 \):

The modified loss function is:

\[ \text{Loss} = -5 \cdot 1 \cdot \log(0.7) - (1 - 1) \cdot \log(1 - 0.7) \]
\[ \text{Loss} = -5 \cdot \log(0.7) \approx 5 \cdot 0.3567 = 1.7835 \]

### Detailed Example with `scale_pos_weight`

Let's go through a more detailed example with multiple predictions to see the impact of `scale_pos_weight`.

#### Data:

| Instance | Actual Label (y) | Predicted Probability (p) | Log Loss Without Weight | Log Loss With Weight (scale_pos_weight = 5) |
|----------|------------------|---------------------------|-------------------------|--------------------------------------------|
| 1        | 1                | 0.9                       | -log(0.9) ≈ 0.105       | -5 * log(0.9) ≈ 0.525                      |
| 2        | 0                | 0.1                       | -log(0.9) ≈ 0.105       | -log(0.9) ≈ 0.105                          |
| 3        | 1                | 0.7                       | -log(0.7) ≈ 0.357       | -5 * log(0.7) ≈ 1.785                      |
| 4        | 0                | 0.2                       | -log(0.8) ≈ 0.223       | -log(0.8) ≈ 0.223                          |
| 5        | 1                | 0.4                       | -log(0.4) ≈ 0.916       | -5 * log(0.4) ≈ 4.580                      |

#### Summarizing the Losses:

**Without Weight:**

\[ \text{Total Loss} = 0.105 + 0.105 + 0.357 + 0.223 + 0.916 = 1.706 \]

**With Weight (scale_pos_weight = 5):**

\[ \text{Total Loss} = 0.525 + 0.105 + 1.785 + 0.223 + 4.580 = 7.218 \]

### Analysis

- **Without `scale_pos_weight`:**
  - The model treats all errors equally.
  - The total loss is lower because false positives are not penalized heavily.

- **With `scale_pos_weight`:**
  - The model places more importance on minimizing false positives.
  - The total loss is higher for false positives due to the increased weight.

### Range of Entropy in Weighted Cases

In non-weighted cases, the binary cross-entropy loss typically ranges between 0 and 1 for individual predictions. However, in weighted cases, the range of the entropy loss is affected by the weighting factor.

#### Non-Weighted Binary Cross-Entropy:

\[ \text{Loss} = -y \log(p) - (1 - y) \log(1 - p) \]

For a single prediction:
- If \( y = 1 \) and \( p = 0.5 \), the loss is \(-\log(0.5) = 0.693\).
- If \( y = 0 \) and \( p = 0.5 \), the loss is also \(-\log(0.5) = 0.693\).

The loss for individual predictions ranges between 0 (for perfect predictions) and \(\infty\) (for very confident but wrong predictions), though in practical applications, it is usually between 0 and a small number greater than 1.

#### Weighted Binary Cross-Entropy:

\[ \text{Loss} = -w_y \cdot y \log(p) - w_{1-y} \cdot (1 - y) \log(1 - p) \]

Here, the weights \( w_y \) and \( w_{1-y} \) modify the loss values:

- When \( y = 1 \) and the weight \( w_y = w_p \):
  \[ \text{Loss} = -w_p \log(p) \]
- When \( y = 0 \) and the weight \( w_{1-y} = 1 \):
  \[ \text{Loss} = -\log(1 - p) \]

The range of the weighted loss depends on the value of \( w_p \):
- The minimum loss is still 0 (for perfect predictions).
- The maximum loss can be much higher than 1, depending on the value of \( w_p \).

For example, if \( w_p = 5 \) and the prediction is incorrect with \( p = 0.1 \) when \( y = 1 \):
\[ \text{Loss} = -5 \log(0.1) \approx 11.51 \]

#### High Entropy Loss:

A high entropy loss indicates that the model's predictions are not aligning well with the actual labels, particularly when the model is confident in its incorrect predictions. In the context of stock predictions with weighted loss:

- **High Weighted Loss:** Indicates the model is making costly mistakes (e.g., predicting a "buy" signal when it should not have, leading to a significant financial loss).
- **Focus on Reducing False Positives:** The weighted loss penalizes false positives more heavily, making the model cautious in issuing "buy" signals.

### Impact on Model Training

By increasing the weight for the positive class, the loss for false positives becomes larger. This higher penalty influences the model during training:

1. **Gradient Boosting Process:**
   - LightGBM builds trees to minimize the overall loss function.
   - With a higher penalty for false positives, the model will focus more on correctly predicting the positive class to minimize the loss.

2. **Decision Boundary Adjustment:**
   - The model adjusts its decision boundary to reduce the number of false positives.
   - In practice, this means the model will require stronger evidence to classify an instance as positive (i.e., "buy" signal).

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






