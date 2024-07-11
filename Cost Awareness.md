### Cost Awareness: Optimizing Stock Predictions 📊💡

#### What is Cost Awareness? 🤔
Cost awareness in stock prediction refers to the model's sensitivity to different types of prediction errors, particularly focusing on minimizing false-positive errors. A false-positive error in stock trading is when the model predicts a stock price will go up (a "buy" signal), but it doesn't, leading to a potential loss.

- **Example**: Imagine you receive a false alarm for a fire in your house. You evacuate and call the fire department, only to find out there was no fire. This false alarm wasted resources and time. Similarly, a false-positive prediction in stock trading leads to unnecessary investments and potential losses. 🚫🔥

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

#### Cost Awareness in Model Evaluation 💸

In stock prediction, not all errors have the same cost. A false-positive error (FP) can lead to direct financial loss, while a false-negative error (FN) results in missed profit opportunities but doesn't incur a direct loss. Hence, it's crucial to design the model to minimize false-positive errors.

- **Example**: If a model incorrectly signals a buy (FP) and the stock doesn't rise, the investor loses money. However, if the model misses a buy signal (FN), the investor only misses a profit opportunity but doesn't lose money. 🏦➡️🚫📉

#### Cost Matrix 📊➡️💰

A cost matrix quantifies the financial implications of different prediction errors. The cost matrix for a binary classification model in this context can be structured as:

**Table: Cost Matrix**
|                | Predicted Negative | Predicted Positive |
|----------------|--------------------|--------------------|
| Actual Negative| 0                  | \(fp\_Amt\)        |
| Actual Positive| \(fn\_Amt\)        | 0                  |

- **True Positive (TP) and True Negative (TN) costs** are considered zero since they don't incur any loss.
- **False Positive (FP) cost (\(fp\_Amt\))**: The financial loss incurred due to a false buy signal.
- **False Negative (FN) cost (\(fn\_Amt\))**: The missed profit opportunity due to a false sell signal.

**Algorithm: Calculating Cost Matrix**
1. Initialize investment amount.
2. Calculate transaction amounts for each prediction.
3. Compute service charges and stamp duties.
4. Determine false-positive and false-negative costs.
5. Aggregate individual costs to get the total cost.

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
### Explanation of Algorithm 2: Calculate Cost Matrix 📊💸

This algorithm is designed to calculate the cost matrix for evaluating the financial implications of false-positive and false-negative predictions in a stock trading model. Let's break down the algorithm step-by-step, explaining the purpose and each part of the process.

#### Purpose:
The algorithm calculates the financial costs associated with false-positive (FP) and false-negative (FN) predictions for a set of stock transactions. It helps in understanding the potential losses due to incorrect predictions.

#### Inputs and Outputs:
- **Inputs**:
  - `test_df`: A data frame or array with the shape `[n_samples]` containing the test data for stocks.
  - `money_init`: The initial amount of money invested in each stock.
- **Output**:
  - `cost_mat`: An array-like structure with the shape `[n_samples, 4]`, containing the calculated costs.

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

### Understanding LightGBM Algorithm for Minimizing False-Positive Errors in Stock Prediction

The paper "Stock Prediction Using Optimized LightGBM Based on Cost Awareness" introduces an innovative method for improving the reliability of stock price predictions by incorporating cost awareness, which particularly focuses on minimizing false-positive errors. Let's break down how this is achieved in the LightGBM algorithm step-by-step, using examples to illustrate key points.

#### 1. Feature Engineering and Data Preparation
**Step:** Feature Engineering
- The dataset includes stocks from the Shanghai Stock Exchange between 2010 and 2019, with 1,500 stocks and 49 features derived from technical indicators, OHLC (Open, High, Low, Close) prices, and time-series data.
- Features with high missing values, unique values, high correlation, or low importance are removed.

**Example:**
- Variables like daily closing prices, moving averages, and momentum indicators are selected and cleaned.

#### 2. Hyperparameter Optimization with Optuna
**Step:** Hyperparameter Optimization
- Optuna, a hyperparameter optimization framework, is used to fine-tune LightGBM parameters such as `num_leaves`, `feature_fraction`, and `bagging_fraction`.
- Time series split cross-validation ensures that the model generalizes well to unseen data without overfitting.

**Example:**
- If `num_leaves` is set to 143, it determines the complexity of the tree structure used in LightGBM.

#### 3. Introducing Cost Awareness
**Step:** Cost Awareness Adjustment
- The core innovation is integrating cost awareness to reduce false-positive errors (predicting a "buy" when the price won't rise).
- A cost matrix is developed to assign higher penalties to false-positive errors compared to false-negative errors.

**Example:**
- If the model predicts a stock will rise and it doesn’t (false positive), the financial cost is calculated and heavily penalized compared to missing a rise (false negative).

#### 4. Implementing the Cost Matrix in LightGBM
**Step:** Adjusting LightGBM Parameters for Cost Sensitivity
- The `scale_pos_weight` parameter in LightGBM is tuned using Optuna to make the model more sensitive to false-positive errors.
- The cost matrix guides this adjustment by assigning a higher cost to false-positive errors, making the model more cautious about generating "buy" signals without strong evidence.

**Example:**
- If `scale_pos_weight` is set to a high value, LightGBM will weigh false positives more heavily during training, reducing their occurrence in the final model.

#### 5. Model Evaluation
**Step:** Model Performance Evaluation
- Performance is measured using precision, F0.5 score (which prioritizes precision over recall), rate of return, annualized return, and risk indicators like Sharpe ratio and Sortino ratio.

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
   - If the model predicts a 70% probability of price increase, it evaluates the potential cost. If the cost of a false positive (buying and the price doesn’t rise) is high, the model might avoid issuing a "buy" signal unless the probability is higher, say 85%.

Let's dive deeper into how the cost matrix works, how `fp_Amt`, `fn_Amt`, and `scale_pos_weight` are calculated, and the flow of the algorithm to understand how it adjusts to minimize false positives.

### Cost Matrix and Algorithm Flow

#### 1. Defining Cost Matrix Values (`fp_Amt` and `fn_Amt`)

The cost matrix is central to making the LightGBM model cost-aware. The cost matrix helps the model focus more on avoiding false positives by assigning higher penalties to them. Here's a step-by-step process:

1. **Data Preparation and Labeling:**
   - The stock data is labeled for binary classification (buy/sell).
   - If the next day's closing price is higher than today's, the label is 1 (buy); otherwise, it's 0 (sell).

2. **Calculating Costs:**
   - The cost of a false positive (`fp_Amt`) is the financial loss incurred if the model incorrectly predicts a price increase (buy signal).
   - The cost of a false negative (`fn_Amt`) is the opportunity cost of not buying a stock that increases in price.

**Example Calculation:**
   - **False Positive (`fp_Amt`):** Assume you buy 100 shares at $50 each, expecting the price to go up. If the price drops to $45, the loss per share is $5. So, `fp_Amt = 100 * $5 = $500`.
   - **False Negative (`fn_Amt`):** If you don't buy 100 shares at $50 each, and the price goes up to $55, the missed gain per share is $5. So, `fn_Amt = 100 * $5 = $500`.

#### 2. Integrating Cost Awareness into LightGBM

To integrate cost awareness, the algorithm adjusts the `scale_pos_weight` parameter. This parameter helps balance the class distribution and makes the model more sensitive to the class with higher cost (false positives in this case).

### Algorithm Flow

1. **Feature Engineering:**
   - Prepare the features (technical indicators, OHLC prices, etc.) and label the data.

2. **Hyperparameter Optimization with Optuna:**
   - Optimize hyperparameters using time-series cross-validation.
   - Initially, `scale_pos_weight` might be set to balance the classes without considering cost.

3. **Cost Awareness Adjustment:**
   - **Initialize Cost Matrix:**
     ```python
     cost_matrix = [[0, fp_Amt],
                    [fn_Amt, 0]]
     ```

   - **Adjusting `scale_pos_weight`:**
     - Use the cost matrix to adjust `scale_pos_weight`. This weight increases the importance of reducing false positives by increasing the weight for positive class (buy signal).

Certainly! Let's delve into the mathematical rationale behind adjusting the `scale_pos_weight` using a cost matrix to reduce false positives in the LightGBM model.

### Understanding `scale_pos_weight`

The `scale_pos_weight` parameter in LightGBM adjusts the weight of positive samples to balance the classes. By increasing this weight, the algorithm penalizes errors on the positive class more heavily, which in turn reduces the likelihood of false positives.

### Cost Matrix and `scale_pos_weight`

A cost matrix assigns different penalties to different types of classification errors. For stock prediction:
- **False Positive (FP)**: Predicting a price increase (buy signal) when the price does not increase.
- **False Negative (FN)**: Not predicting a price increase (missed buy signal).

### Example Cost Matrix
Consider the following cost matrix:

\[
\begin{matrix}
 & \text{Predict Negative} & \text{Predict Positive} \\
\text{Actual Negative} & 0 & fp\_Amt \\
\text{Actual Positive} & fn\_Amt & 0 \\
\end{matrix}
\]

Here, `fp_Amt` is the financial loss due to a false positive, and `fn_Amt` is the opportunity cost due to a false negative.

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

### Summary

1. **Define the costs** of false positives and false negatives based on the cost matrix.
2. **Calculate the cost ratio** to understand the relative importance of penalizing false positives.
3. **Set `scale_pos_weight`** to this ratio, increasing the model's sensitivity to false positives.
4. **Train the LightGBM model** with the adjusted `scale_pos_weight`, leading to fewer false positive predictions.

By following these mathematical steps, you ensure that the LightGBM model is optimized to minimize financial losses associated with false positives in stock predictions.
