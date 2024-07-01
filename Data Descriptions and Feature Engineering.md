### Data Descriptions and Feature Engineering 📊💡

This section focuses on how data was selected, processed, and prepared for the stock prediction model. We'll cover data selection, input variables, and labeling methods, with detailed explanations and real-world examples.

#### A. Data Selection and Process

**Data Selection**:
- The study focuses on data from the Shanghai Stock Exchange's main board, selecting 1,560 stocks from 2010 to 2019. 📅🏦
- **Z-score Normalization**: This technique standardizes the data by converting it to a scale where the mean (µ) is zero and the standard deviation (σ) is one. This helps in handling different scales and making the data comparable. 

  Z = (x - mu)/sigma

  - **Real-World Example**: Imagine comparing heights and weights of people. Heights are in centimeters and weights in kilograms. Z-score normalization scales these measurements so they can be compared directly. 📏⚖️

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

     CCI} = (Typical Price - SMATP) / (0.015 * Mean Deviation)

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

