### Data Descriptions and Feature Engineering 📊💡

This section focuses on how data was selected, processed, and prepared for the stock prediction model. We'll cover data selection, input variables, and labeling methods, with detailed explanations and real-world examples.

#### A. Data Selection and Process

**Data Selection**:
- The study focuses on data from the Shanghai Stock Exchange's main board, selecting 1,560 stocks from 2010 to 2019. 📅🏦
- **Z-score Normalization**: LightGBM does not require any scaling/normalization/standardization method

#### B. Input Variables and Variable Selection

**Input Variables**:
1. **Time Series Features**:
   - Variables that describe time, such as week and month. These help the model understand temporal patterns. 📆
   - **Example**: Identifying if a stock typically performs better at the start or end of the month. 🗓️

2. **OHLC Variables**:
   - Daily Open, High, Low, and Close prices summarize daily stock transactions. 📈📉
   - **Example**: Tracking how much a stock's price fluctuates within a single day. 🔍

3. **Technical Indicators**:
   - Metrics derived from historical price and volume data to predict future price movements. These help identify trends and potential reversal points. 📊
   - **Example**: Moving averages (MA) smooth out price data to highlight trends, while the relative strength index (RSI) measures the speed and change of price movements. 🧮

**Variable Selection**:
1. **Handling Missing Values**:
   - Variables with more than 90% missing data were removed. This ensures the data used is complete and reliable. 🗑️

2. **Eliminating Unique Values**:
   - Variables with a single unique value were discarded as they don't provide useful information for predictions. ❌

3. **Removing Highly Correlated Variables**:
   - Using the Pearson correlation coefficient, variables with correlations above 90% were removed to avoid redundancy.
   
   - **Example**: If height and weight are highly correlated, using both doesn't add much value. By removing one, the model becomes more efficient. 📏⚖️

4. **Ranking Importance**:
   - Variables were ranked based on their importance to the model. Less important variables were removed to focus on the most relevant data. 📋➡️🚀

#### C. Labeling Methods

**Labeling**:
- The research changes the experiment into a binary classification problem, where the actions are "buy" or "sell." 📥📤
- **Label Definition**:
  - If the next day's closing price is higher than today's closing price, the label is 1 (indicating a buy signal). If not, the label is 0 (indicating a sell signal).
  - **Example**: If today's closing price is $100 and tomorrow's is $105, the label is 1 (buy). If tomorrow's is $95, the label is 0 (sell). 💸➡️💵

**Technical Indicators Explained**:
1. **CMO (Chande Momentum Oscillator)**: Measures the difference between the sum of recent gains and losses. Helps identify overbought or oversold conditions.
   - **Formula**: 

     CMO = 100 * ((Su - Sd)/(Su + Sd))

     where Su = Sum of gains, Sd = Sum of losses.
   - **Example**: Tracking if a stock is frequently gaining more than losing. 📈📉

2. **CCI (Commodity Channel Index)**: Measures the difference between the current price and its historical average. Helps identify price reversals.
   - **Formula**: 

     CCI = (Typical Price - SMATP) / (0.015 * Mean Deviation)

     where SMATP = Simple Moving Average of Typical Price.
   - **Example**: Identifying if a stock is trading above or below its average price. 📊

3. **SAR (Stop and Reversal)**: Detects the price direction of an asset, helping to determine when to switch from buying to selling.
   - **Example**: Acting like a traffic signal, indicating when to stop buying and start selling. 🚦

4. **KAMA (Kaufman's Adaptive Moving Average)**: A moving average that adapts to market volatility, smoothing out noise.
   - **Example**: Filtering out the daily fluctuations to see the broader trend. 📉➡️📈

5. **ADX (Average Directional Index)**: Measures the strength of a price trend.
   - **Example**: Determining if a trend is strong enough to trade. 📊💪

6. **MOM (Momentum)**: Measures the speed of price changes over a period.
   - **Formula**: 

     MOM = Price - Price of n periods ago

   - **Example**: Identifying stocks that are gaining or losing momentum. 🏃💨

7. **ROC (Rate of Change)**: Measures the percentage change in price over a given period.
   - **Formula**: 

     ROC = ((Closing Price_p - Closing Price_p-n)/ Closing Price_p-n) * 100

   - **Example**: Calculating how much a stock's price has changed compared to its price a month ago. 🔄📈


---

### Chande Momentum Oscillator (CMO) 💪📉📈

**Purpose**:
- The Chande Momentum Oscillator (CMO) is used to identify the strength and direction of a stock's momentum. It helps traders understand if a stock is overbought or oversold.

**How It Works**:
- The CMO calculates the difference between the sum of all recent gains and the sum of all recent losses over a specific period. The result is then divided by the sum of all price movements (both gains and losses) over the same period.

**Interpretation**:
- The CMO ranges from -100 to +100.
- A CMO above +50 indicates that the stock is overbought.
- A CMO below -50 indicates that the stock is oversold.

**Real-World Example**:
- Imagine you are looking at a stock over the past 20 days. If the stock has had more days with gains than losses, the CMO will be positive. If the gains are significantly higher than the losses, the CMO might be above +50, indicating that the stock is overbought and might be due for a price correction. Conversely, if there are more losses than gains, the CMO will be negative, and a value below -50 suggests the stock is oversold and might be due for a price increase.

### Commodity Channel Index (CCI) 📈📉📊

**Purpose**:
- The Commodity Channel Index (CCI) is used to identify cyclical trends in a stock's price. It helps traders determine if a stock is trending or moving sideways and if it's reaching extreme conditions (overbought or oversold).

**How It Works**:
- The CCI measures the difference between the current price and the average price over a specific period. This difference is then divided by the mean deviation (average absolute deviation from the average price) and scaled by a constant (0.015).

- **Typical Price**: The average of the high, low, and close prices for a given period.
- **SMATP (Simple Moving Average of Typical Price)**: The average of the typical prices over a specific period.
- **Mean Deviation**: The average of the absolute differences between the typical prices and the SMATP.

**Interpretation**:
- The CCI typically ranges from -100 to +100.
- A CCI above +100 indicates that the stock is overbought.
- A CCI below -100 indicates that the stock is oversold.

**Real-World Example**:
- Suppose you are analyzing a stock and its CCI value is +120. This suggests that the stock's price is significantly higher than its average price, indicating an overbought condition. It might be a good time to consider selling. Conversely, if the CCI value is -120, the stock's price is significantly lower than its average price, indicating an oversold condition. It might be a good time to consider buying.

### Summary in Simple Terms
- **CMO (Chande Momentum Oscillator)**: Measures the momentum of a stock by comparing recent gains to recent losses. If there are more gains, the CMO is high; if there are more losses, the CMO is low. High values suggest the stock is overbought, and low values suggest it is oversold. 💪📉📈
- **CCI (Commodity Channel Index)**: Measures how far the current price is from the average price over a specific period. High values (above +100) suggest the stock is overbought, and low values (below -100) suggest it is oversold. 📊📉📈

Both indicators help traders make informed decisions about buying or selling stocks based on market conditions. 📈📉📊

---

### Table of Contents
1. [Introduction](https://github.com/aditya-saxena-7/Cost-Efficient-Stock-Forecasting-with-Enhanced-LightGBM/blob/master/Introduction.md)
2. [Model and Methodology](https://github.com/aditya-saxena-7/Cost-Efficient-Stock-Forecasting-with-Enhanced-LightGBM/blob/master/Model%20and%20Methodology.md)
3. [Data Descriptions and Feature Engineering](https://github.com/aditya-saxena-7/Cost-Efficient-Stock-Forecasting-with-Enhanced-LightGBM/blob/master/Data%20Descriptions%20and%20Feature%20Engineering.md)
4. [Hyperparameter Optimization](https://github.com/aditya-saxena-7/Cost-Efficient-Stock-Forecasting-with-Enhanced-LightGBM/blob/master/Hyperparameter%20Optimization.md)
5. [Cost Awareness](https://github.com/aditya-saxena-7/Cost-Efficient-Stock-Forecasting-with-Enhanced-LightGBM/blob/master/Cost%20Awareness.md)
6. [Performance and Measurement](https://github.com/aditya-saxena-7/Cost-Efficient-Stock-Forecasting-with-Enhanced-LightGBM/blob/master/Performance%20and%20Measurement.md)
7. [Conclusion](https://github.com/aditya-saxena-7/Cost-Efficient-Stock-Forecasting-with-Enhanced-LightGBM/blob/master/Conclusion.md)
