# Importing essential libraries/ packages
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from statsmodels.tsa.stattools import acf
import time


def warm_start_book(params):
    fundamental_price = params['fundamental_price']
    levels = params['levels_per_side']
    step = params['tick_step']
    depth = params['depth_per_level']
    expiry_ticks = params['order_expiry_ticks']
    bids_book = {}
    asks_book = {}

    for i in range(1, levels + 1):
        bid_price = fundamental_price - (i * step)
        ask_price = fundamental_price + (i * step)
        bids_book[bid_price] = []
        asks_book[ask_price] = []
        for _ in range(depth):
            order_data = {'placed_at': 0, 'expires_at': expiry_ticks}
            bids_book[bid_price].append(order_data)
            asks_book[ask_price].append(order_data.copy())
    return bids_book, asks_book


def cancel_expired_orders(current_tick, bids_book, asks_book):
    for book in [bids_book, asks_book]:
        prices_to_delete = []
        for price in list(book.keys()):
            orders_at_price = book[price]
            while orders_at_price and orders_at_price[0]['expires_at'] < current_tick:
                orders_at_price.pop(0)
            if not orders_at_price:
                prices_to_delete.append(price)
        for price in prices_to_delete:
            del book[price]


def process_order(order_type, limit_price, current_tick, expiry_time, bids_book, asks_book):
    order_data = {'placed_at': current_tick, 'expires_at': expiry_time}
    trade_executed = False

    if order_type == 'buy':
        if asks_book:
            best_ask = min(asks_book.keys())
            if limit_price >= best_ask:
                trade_executed = True
                asks_book[best_ask].pop(0)
                if not asks_book[best_ask]:
                    del asks_book[best_ask]
        if not trade_executed:
            if limit_price not in bids_book:
                bids_book[limit_price] = []
            bids_book[limit_price].append(order_data)
    else: 
        if bids_book:
            best_bid = max(bids_book.keys())
            if limit_price <= best_bid:
                trade_executed = True
                bids_book[best_bid].pop(0)
                if not bids_book[best_bid]:
                    del bids_book[best_bid]
        if not trade_executed:
            if limit_price not in asks_book:
                asks_book[limit_price] = []
            asks_book[limit_price].append(order_data)


def get_current_mid_price(bids_book, asks_book, last_mid_price):
    best_bid = max(bids_book.keys()) if bids_book else None
    best_ask = min(asks_book.keys()) if asks_book else None
    if best_bid and best_ask:
        return (best_bid + best_ask) / 2.0
    else:
        return last_mid_price


def calculate_stats(price_log, burn_in, sampling_ticks=100):
    retained_series = price_log[burn_in:]
    if len(retained_series) <= sampling_ticks:
        return np.nan, np.full(6, np.nan)
    sample_indices = np.arange(0, len(retained_series), sampling_ticks)
    sampled_mid_prices = retained_series[sample_indices]
    if len(sampled_mid_prices) < 2:
        return np.nan, np.full(6, np.nan)
    log_returns = np.diff(np.log(sampled_mid_prices))
    if len(log_returns) < 2:
        return np.nan, np.full(6, np.nan)
    excess_kurt = kurtosis(log_returns, fisher=True)
    squared_returns = log_returns**2
    if len(squared_returns) <= 6:
        return excess_kurt, np.full(6, np.nan)
    acf_sq_returns = acf(squared_returns, nlags=6, fft=False)[1:]
    return excess_kurt, acf_sq_returns


def create_bt_agents(params):
   # Each dictionary is one "agent" with its own properties.
    agents = []
    N = params['num_agents']
    for _ in range(N):
        agent = {
            'w1': np.random.uniform(0, params['w1_max']),
            'w2': np.random.uniform(0, params['w2_max']),
            'w3': np.random.uniform(0, params['w3_max']),
            'tau': np.random.randint(1, params['tau_max'] + 1)
        }
        agents.append(agent)
    return agents

def generate_bt_order(agent, current_tick, price_history, params):
    # Get agent's private parameters
    w1, w2, w3, tau = agent['w1'], agent['w2'], agent['w3'], agent['tau']
    
    # Get market parameters
    Pf = params['fundamental_price']
    Pd = params['order_half_width']
    sigma_epsilon = params['noise_std_dev']
    deltaP = params['tick_size']
    
    P_t_minus_1 = price_history[current_tick - 1]
    
    # Fundamental Signal
    # Prevent log(0)
    if P_t_minus_1 == 0:
        return 'buy', 0 # Failsafe, skip turn
        
    fundamental_signal = np.log(Pf / P_t_minus_1)
    
    # Technical Signal
    if current_tick <= tau + 1:
        technical_signal = 0.0
    else:
        P_past = price_history[current_tick - tau - 1]
        technical_signal = 0.0 if P_past == 0 else np.log(P_t_minus_1 / P_past)
    
    # Noise Signal
    noise_signal = np.random.normal(0, sigma_epsilon)
    
    # Form Expected Return
    numerator = (w1 * fundamental_signal) + (w2 * technical_signal) + (w3 * noise_signal)
    denominator = w1 + w2 + w3
    r_e = 0.0 if denominator == 0 else numerator / denominator
        
    # Form Expected Price
    P_e = P_t_minus_1 * np.exp(r_e)
    
    # Generate Order Price
    rho_j_t = np.random.uniform(0, 1)
    P_o = P_e + Pd * (2 * rho_j_t - 1)
    
    # Determine Buy/Sell
    if P_e > P_o:
        order_type = 'buy'
        limit_price = np.floor(P_o / deltaP) * deltaP # Round down
    else:
        order_type = 'sell'
        limit_price = np.ceil(P_o / deltaP) * deltaP # Round up
        
    return order_type, limit_price

# Main Simulation

def run_bt_simulation(market_params, seed):
    # Main runner function for the BT model.
    
    np.random.seed(seed)
    
    # Unpack parameters
    N = market_params['num_agents']
    total_ticks = market_params['total_ticks']
    expiry_ticks = market_params['order_expiry_ticks']
    burn_in_count = market_params['burn_in_count']
    
    # Initialisation
    agents = create_bt_agents(market_params) 
    bids_book, asks_book = warm_start_book(market_params)
    
    agent_idx = 0 
    mid_price_log = np.zeros(total_ticks + 1)
    last_mid_price = get_current_mid_price(bids_book, asks_book, market_params['fundamental_price'])
    mid_price_log[0] = last_mid_price

    # Event flow
    for t in range(1, total_ticks + 1):
        
        # Cancel expired orders
        cancel_expired_orders(t, bids_book, asks_book)
        
        # Generate new order
        current_agent = agents[agent_idx]
        order_type, limit_price = generate_bt_order(current_agent, t, mid_price_log, market_params)
        
        # Process the order
        expiry_time = t + expiry_ticks
        process_order(order_type, limit_price, t, expiry_time, bids_book, asks_book)
        
        # Record mid price
        last_mid_price = get_current_mid_price(bids_book, asks_book, last_mid_price)
        mid_price_log[t] = last_mid_price
        
        # Update agent counter
        agent_idx = (agent_idx + 1) % N

    # Post-Processing
    return calculate_stats(mid_price_log, burn_in_count)

# Main execution

if __name__ == "__main__":
    
    PARAMS_Q2_FULL = {
        'num_agents': 500,
        'w1_max': 1,
        'w2_max': 10,
        'w3_max': 1,
        'fundamental_price': 10000,
        'tau_max': 10000,
        'noise_std_dev': 0.03,
        'order_expiry_ticks': 10000,
        'order_half_width': 1000, # Pd
        'total_ticks': 100000,
        'levels_per_side': 200,
        'tick_step': 1,
        'depth_per_level': 5,
        'burn_in_count': int(0.01 * 100000), # 1000
        'num_runs': 30,
        'tick_size': 1 
    }
    
    params_to_use = PARAMS_Q2_FULL
    R_runs = params_to_use['num_runs']

    print(f"--- Running BT Simulation (Full Parameters, Human Style) ---")
    print(f"R = {R_runs} runs, TE = {params_to_use['total_ticks']} ticks")
    
    all_stats = []
    start_time = time.time()

    for i in range(R_runs):
        print(f"Starting Run {i+1}/{R_runs}...")
        seed = i
        kurt, acf_lags = run_bt_simulation(params_to_use, seed)
        
        if not np.isnan(kurt):
            all_stats.append([kurt] + list(acf_lags))
        else:
            print(f"Run {i+1} resulted in invalid stats (NaN), skipping.")
        
        print(f"Run {i+1} complete. Time elapsed: {time.time() - start_time:.2f}s")

    end_time = time.time()
    print(f"\nSimulation complete. Total time: {end_time - start_time:.2f} seconds")

    if all_stats:
        stats_df = pd.DataFrame(all_stats, columns=['Excess Kurtosis'] + [f'ACF Lag {i}' for i in range(1, 7)])
        mean_stats = stats_df.mean()
        
        print("\nResults for Table 1 (BT Column):")
        print("****************************************")
        print(f"Excess kurtosis of returns: {mean_stats['Excess Kurtosis']:.4f}")
        for i in range(1, 7):
            print(f"ACF of squared returns lag {i}: {mean_stats[f'ACF Lag {i}']:.4f}")
        print("****************************************")
    else:
        print("No valid simulation runs completed.")