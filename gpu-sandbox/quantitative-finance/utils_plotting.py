import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.patches import Rectangle


def find_closest_time_index(df, date, target_time):
    """Find the index of the closest time to target_time"""
    time_diffs = abs(df['DateTime'].dt.time.apply(lambda x: 
        datetime.datetime.combine(date, x) - 
        datetime.datetime.combine(date, target_time)
    ))
    closest_idx = time_diffs.idxmin()
    result = list(df.index).index(closest_idx)
    print(f'closest time index for {target_time} on {date}: {result}')
    return result


def plot_spread_analysis(df, desired_date):    
    recent_data = df[df['DateTime'].dt.date == desired_date].copy()
    
    # Get first 15 minutes (9:30 to 9:45).
    first_15min = recent_data[
        (recent_data['DateTime'].dt.time >= datetime.time(9, 30)) & 
        (recent_data['DateTime'].dt.time <= datetime.time(9, 45))
    ]
    max_15min = first_15min['High'].max()
    min_15min = first_15min['Low'].min()
    ##print(f'ORB: {first_15min["Time"].iloc[0]} -> {first_15min["Time"].iloc[-1]}')
    ##print(f'First 15min data shape: {first_15min.shape}')
    ##print(f'First 15min time range: {first_15min["DateTime"].min()} to {first_15min["DateTime"].max()}')
    ##print(first_15min[['DateTime', 'High', 'Low']].head(16))

    # Determine line color based on trend vs previous close.
    current_close = first_15min['Close'].iloc[-1]
    ##print(f'15min close: {current_close}')

    prev_dates = df[df['DateTime'].dt.date < desired_date]['DateTime'].dt.date.unique()
    if len(prev_dates) > 0:
        prev_date = max(prev_dates)  # Most recent date before today
        print(f'using {prev_date} as previous date')
        prev_close_data = df[df['DateTime'].dt.date == prev_date]
        prev_close = prev_close_data['Close'].iloc[-1]
        print(f'prev day close: {prev_close}')
        line_color = 'green' if current_close > prev_close else 'red'
    else:
        print('no data for previous day found')
        line_color = 'black'

    day_close = recent_data['Close'].iloc[-1]
    print(f'price at close: {day_close}')

    
    #fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # ===== MAIN CANDLESTICK CHART =====
    for i, (idx, row) in enumerate(recent_data.iterrows()):
        open_price = row['Open']
        close_price = row['Close']
        high_price = row['High']
        low_price = row['Low']
        
        # Determine candle color.
        color = 'green' if close_price > open_price else 'red'
        
        # Draw the high-low line.
        ax1.plot([i, i], [low_price, high_price], color=color, linewidth=1)
        
        # Draw the open-close rectangle.
        height = abs(close_price - open_price)
        bottom = min(open_price, close_price)
        rect = Rectangle((i-0.3, bottom), 0.6, height, facecolor=color, alpha=0.7, edgecolor=color)
        ax1.add_patch(rect)

    ##########################################
    # Add pre-market and post-market shading #
    ##########################################
    market_open_time = datetime.time(9, 30)
    market_close_time = datetime.time(16, 0)

    # Find positions for market open/close.
    market_open_idx = recent_data[recent_data['DateTime'].dt.time >= market_open_time].index
    market_close_idx = recent_data[recent_data['DateTime'].dt.time <= market_close_time].index

    if len(market_open_idx) > 0:
        market_open_pos = list(recent_data.index).index(market_open_idx[0])
    else:
        market_open_pos = 0

    if len(market_close_idx) > 0:
        market_close_pos = list(recent_data.index).index(market_close_idx[-1])
    else:
        market_close_pos = len(recent_data) - 1

    # Shade pre-market (start to market open)
    if market_open_pos > 0:
        ax1.axvspan(0, market_open_pos, alpha=0.5, color='lightgray', label='Pre-market')

    # Shade post-market (market close to end)
    if market_close_pos < len(recent_data) - 1:
        ax1.axvspan(market_close_pos, len(recent_data) - 1, alpha=0.5, color='lightgray', label='Post-market')

       
    ##################################################
    # Plot horizontal lines for first 15min high/low #
    ##################################################
    ax1.axhline(y=max_15min, color=line_color, linestyle='-', linewidth=3, alpha=0.3, label=f'15min High: {max_15min:.2f}')
    ax1.axhline(y=min_15min, color=line_color, linestyle='-', linewidth=3, alpha=0.3, label=f'15min Low: {min_15min:.2f}')
    ax1.axhline(y=prev_close, color='black', linestyle='--', linewidth=2, alpha=0.3, label=f'prev day close: {prev_close:.2f}')


    # Calculate where 10-minute marks would be on your x-axis
    first_time = recent_data['DateTime'].iloc[0]
    last_time = recent_data['DateTime'].iloc[-1]
    print(f'first time: {first_time}, last time: {last_time}')
    # Convert these times to x-axis positions
    tick_times_desired = pd.date_range(
        start=first_time.floor('30min'),  
        end=last_time.floor('30min'),      
        freq='30min'
    )
    
    tick_positions = []
    tick_labels = []
    for tick_time in tick_times_desired:
        # Find the closest actual data point to this time
        time_diffs = abs(recent_data['DateTime'] - tick_time)
        closest_idx = time_diffs.idxmin()
        
        # Convert to position in the plot (0 to len(recent_data)-1)
        position = list(recent_data.index).index(closest_idx)
        
        tick_positions.append(position)
        tick_labels.append(tick_time.strftime('%H:%M'))
    
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=45)

    ######################################################
    # Find the indices for 9:30 AM, 9:45 AM, and 3:00 PM #
    ######################################################
    pos_930 = find_closest_time_index(recent_data, desired_date, datetime.time(9, 30))
    ax1.axvline(x=pos_930, color='black', linestyle='--', linewidth=1, alpha=0.7, label='9:30 AM')

    pos_945 = find_closest_time_index(recent_data, desired_date, datetime.time(9, 45))
    ax1.axvline(x=pos_945, color='black', linestyle='--', linewidth=1, alpha=0.7, label='9:45 AM')

    pos_3pm = find_closest_time_index(recent_data, desired_date, datetime.time(15, 0))
    ax1.axvline(x=pos_3pm, color='black', linestyle='--', linewidth=1, alpha=0.7, label='3:00 PM')
    ##print(f'{pos_3pm=}')

    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()