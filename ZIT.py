# Importing essential libraries / packages
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from statsmodels.tsa.stattools import acf
import time


def warm_start_book(params):
    # Create the initial order book ladder
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
        
        # Initialize the list for each price
        bids_book[bid_price] = []
        asks_book[ask_price] = []
        
        for _ in range(depth):
            order_data = {'placed_at': 0, 'expires_at': expiry_ticks}
            bids_book[bid_price].append(order_data)
            asks_book[ask_price].append(order_data.copy())
            
    return bids_book, asks_book

def cancel_expired_orders(current_tick, bids_book, asks_book):
    # Clears out-of-date orders from both sides of the book
    
    for book in [bids_book, asks_book]:
        prices_to_delete = []
        for price in list(book.keys()):
            orders_at_price = book[price]
            
            # Keep checking first order (FIFO) until not expired
            while orders_at_price and orders_at_price[0]['expires_at'] < current_tick:
                orders_at_price.pop(0)      # Remove the expired order
            
            if not orders_at_price:
                prices_to_delete.append(price)
        
        # Delete the empty price levels
        for price in prices_to_delete:
            del book[price]

def generate_zit_order(agent_id, num_buyers, params):
    # Creates a new order based on the ZIT agent rules
    Pf = params['fundamental_price']
    S = params['private_value_span']
    Pmin = params['price_min']
    Pmax = params['price_max']
    deltaP = params['tick_size']

    if agent_id < num_buyers:
        # Agent is a Buyer
        vi = np.random.uniform(Pf, Pf + S)
        bi = np.random.uniform(Pmin, vi)
        limit_price = np.floor(bi / deltaP) * deltaP # Round down
        return 'buy', limit_price
    else:
        # Agent is a Seller
        ci = np.random.uniform(Pf - S, Pf)
        ai = np.random.uniform(ci, Pmax)
        limit_price = np.ceil(ai / deltaP) * deltaP # Round up
        return 'sell', limit_price

def process_order(order_type, limit_price, current_tick, expiry_time, bids_book, asks_book):
    # Matches or adds a new order to the book.
    
    order_data = {'placed_at': current_tick, 'expires_at': expiry_time}
    trade_executed = False

    if order_type == 'buy':
        # Check if any asks exist
        if asks_book:
            best_ask = min(asks_book.keys())
            if limit_price >= best_ask:
                
                trade_executed = True
                # Remove the resting order (FIFO)
                asks_book[best_ask].pop(0)
                if not asks_book[best_ask]:
                    del asks_book[best_ask]
        
        if not trade_executed:
            if limit_price not in bids_book:
                bids_book[limit_price] = []
            bids_book[limit_price].append(order_data)
            
    else: 
        # Check if any bids exist
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
    # Calculates the current mid-price, or holds the last one
    best_bid = max(bids_book.keys()) if bids_book else None
    best_ask = min(asks_book.keys()) if asks_book else None
    
    if best_bid and best_ask:
        return (best_bid + best_ask) / 2.0
    else:
        return last_mid_price

def calculate_stats(price_log, burn_in, sampling_ticks=100):
    # Calculate final statistics for the run
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

# Main Simulation

def run_simulation(market_params, seed):
    # This is the main function that runs one full simulation.
    np.random.seed(seed)

    # Unpack parameters
    N = market_params['num_agents']
    total_ticks = market_params['total_ticks']
    expiry_ticks = market_params['order_expiry_ticks']
    burn_in_count = market_params['burn_in_count']
    num_buyers = N // 2
    
    # Initialization
    bids_book, asks_book = warm_start_book(market_params)
    
    agent_idx = 0 
    
    mid_price_log = np.zeros(total_ticks + 1)
    
    # Get initial mid-price
    last_mid_price = get_current_mid_price(bids_book, asks_book, 0)
    mid_price_log[0] = last_mid_price

    for t in range(1, total_ticks + 1):
        
        # Cancel expired orders
        cancel_expired_orders(t, bids_book, asks_book)
        
        # Generate new order
        agent_id = agent_idx
        order_type, limit_price = generate_zit_order(agent_id, num_buyers, market_params)
        
        # Process the order
        expiry_time = t + expiry_ticks
        process_order(order_type, limit_price, t, expiry_time, bids_book, asks_book)
        
        # Record mid-price
        last_mid_price = get_current_mid_price(bids_book, asks_book, last_mid_price)
        mid_price_log[t] = last_mid_price
        
        # Update agent counter
        agent_idx = (agent_idx + 1) % N

    # Post-Processing
    return calculate_stats(mid_price_log, burn_in_count)

# Main execution

if __name__ == "__main__":
    
    PARAMS_FULL = {
        'num_agents': 500,
        'fundamental_price': 10000,
        'private_value_span': 1000,
        'tick_size': 1,
        'order_expiry_ticks': 10000,
        'total_ticks': 100000,
        'price_min': 1000,
        'price_max': 2 * (10000 + 1000), # 2(Pf + S)
        'levels_per_side': 200,
        'tick_step': 1,
        'depth_per_level': 5,
        'burn_in_count': int(0.01 * 100000), # 1000
        'num_runs': 30
    }
    
    params_to_use = PARAMS_FULL
    R_runs = params_to_use['num_runs']

    print(f"--- Running ZIT Simulation (Full Parameters, Human Style) ---")
    print(f"R = {R_runs} runs, TE = {params_to_use['total_ticks']} ticks")
    
    all_stats = []
    start_time = time.time()

    for i in range(R_runs):
        print(f"Starting Run {i+1}/{R_runs}...")
        seed = i
        kurt, acf_lags = run_simulation(params_to_use, seed)
        
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
        
        print("\nResults for Table 1 (ZIT Column):")
        print("*************************************")
        print(f"Excess kurtosis of returns: {mean_stats['Excess Kurtosis']:.4f}")
        for i in range(1, 7):
            print(f"ACF of squared returns lag {i}: {mean_stats[f'ACF Lag {i}']:.4f}")
        print("*************************************")
    else:
        print("No valid simulation runs completed.")