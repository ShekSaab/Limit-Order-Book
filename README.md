This repository contains the numerical simulations for the Agent-based Modelling in Finance.
The project explores market microstructure dynamics by implementing a continuous double-auction Limit Order Book (LOB).

Zero-Intelligence Traders (ZIT)
    
    Mechanism: Agents submit orders based on a simple no-loss constraint.
  
    Observation on Agent Ordering: In this implementation, agents were partitioned with buyers listed first and sellers after. This resulted in an unusual autocorrelation pattern at lags 1 and 5, whereas real financial markets typically show more alternating buyer/seller activity.
  
    Parameters: The model used a minimum price bound (P_min) of 1000 and a price tick size (delta_P) of 1.
  
    Stylized Facts: The ZIT excess kurtosis was 0.0938, which is close to a normal distribution. This indicates that the ZIT model failed to replicate the "Fat Tails" typically seen in real market data.

Behavioural Traders (BT)
  
    Mechanism: Agents (referred to as Normal Agents) form expectations by weighting fundamental, technical (momentum), and noise signals.
  
    Expected Return: The expected return is calculated as a weighted average of:
    
      1. Fundamental Signal: The difference between fundamental price and last mid-price.
    
      2. Technical Signal: Momentum based on a private look-back window.
    
      3. Noise Signal: A random "gut feeling" drawn from a normal distribution.
  
    Stylized Facts: The BT model successfully produced "Fat Tails" with an excess kurtosis of 3.6577. While it showed better volatility clustering than ZIT, it did not fully reproduce the persistence seen in real S&P 500 data. This may be due to simplified agent interactions or parameter sensitivity compared to the models cited in Mizuta & Yagi (2025).

Empirical Benchmark (S&P 500)
  
    Data: Daily closing prices from 2013-01-01 to 2018-01-01.
  
    Analysis: Real-world data showed high excess kurtosis (2.8915) and significant volatility clustering, which the BT model approximates more effectively than the ZIT model.

Learning Process & Reinforcement Learning
  
    Learning Mechanism: Agents dynamically update weights for their strategies. If a strategy's prediction matches the sign of the actual market return, the agent "rewards" it by increasing its weight. If incorrect, the weight is "punished" (decreased).
  
    Reinforcement Learning (RL): This process is a form of RL where the agent (Trader) interacts with an environment (LOB) and adjusts its policy (signal weights) based on feedback (market returns). While it lacks the complexity of Deep RL, it follows the core principle of learning optimal behavior through environmental rewards.

  
