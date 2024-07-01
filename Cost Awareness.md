### Cost Awareness: Optimizing Stock Predictions 📊💡

In this section, we explore the concept of cost awareness and its application in optimizing the LightGBM model for stock predictions. Let's break down the key concepts, terminologies, and methodologies involved, with detailed explanations and real-world examples.

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

#### Optimization with Cost Awareness 📉➡️📈

To reduce false-positive errors, the model's sensitivity is adjusted using the Optuna framework, optimizing the 'scale_pos_weight' parameter to place more emphasis on minimizing these costly errors. 

- **Real-World Example**: Imagine you are fine-tuning a smoke detector to be less sensitive to steam (false alarms) but still sensitive to actual smoke. Similarly, the model is adjusted to be less prone to false-positive errors in stock predictions. 🚨➡️🔥
