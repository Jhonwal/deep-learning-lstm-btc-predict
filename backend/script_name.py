# # import time
# # import csv
# # import datetime
# # import requests
# # import pandas as pd
# # from datetime import datetime, timedelta

# # def fetch_btc_historical_data(start_time, end_time, interval='1m', limit=1000):
# #     """
# #     Fetch historical BTC/USDT data from Binance API
    
# #     Parameters:
# #     - start_time: Start time in milliseconds (Unix timestamp)
# #     - end_time: End time in milliseconds (Unix timestamp)
# #     - interval: Kline/candlestick interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
# #     - limit: Maximum number of records to return (max 1000)
    
# #     Returns:
# #     - DataFrame with columns: 'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', etc.
# #     """
# #     url = 'https://api.binance.com/api/v3/klines'
# #     params = {
# #         'symbol': 'BTCUSDT',
# #         'interval': interval,
# #         'startTime': start_time,
# #         'endTime': end_time,
# #         'limit': limit
# #     }
    
# #     response = requests.get(url, params=params)
    
# #     if response.status_code == 200:
# #         data = response.json()
# #         df = pd.DataFrame(data, columns=[
# #             'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
# #             'Close time', 'Quote asset volume', 'Number of trades',
# #             'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
# #         ])
        
# #         # Convert numeric columns
# #         numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
# #         for col in numeric_columns:
# #             df[col] = pd.to_numeric(df[col])
        
# #         # Convert timestamp to datetime
# #         df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        
# #         # Select only required columns
# #         return df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
# #     else:
# #         print(f"Error fetching data: {response.status_code}")
# #         return None

# # def collect_btc_data(total_records=4046400, output_file='btc_price_data.csv'):
# #     """
# #     Collect historical BTC price data and save to CSV
    
# #     Parameters:
# #     - total_records: Total number of 1-minute records to collect
# #     - output_file: Output CSV file name
# #     """
# #     print(f"Starting collection of {total_records} BTC price records...")
    
# #     # Calculate how many API calls we need (max 1000 per call)
# #     batch_size = 1000
# #     num_batches = (total_records + batch_size - 1) // batch_size
    
# #     # Calculate end time (now) and start time based on total_records
# #     end_time = int(datetime.now().timestamp() * 1000)
    
# #     # Create/open the CSV file
# #     with open(output_file, 'w', newline='') as csvfile:
# #         csv_writer = csv.writer(csvfile)
# #         # Write header
# #         csv_writer.writerow(['Open time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
# #         records_collected = 0
        
# #         for batch in range(num_batches):
# #             if records_collected >= total_records:
# #                 break
                
# #             # Calculate batch end_time and start_time (in milliseconds)
# #             batch_end_time = end_time - (batch * batch_size * 60 * 1000)
# #             batch_start_time = batch_end_time - (batch_size * 60 * 1000)
            
# #             print(f"Fetching batch {batch+1}/{num_batches}: {batch_start_time} to {batch_end_time}")
            
# #             # Fetch data
# #             df = fetch_btc_historical_data(
# #                 start_time=batch_start_time,
# #                 end_time=batch_end_time,
# #                 interval='1m',
# #                 limit=batch_size
# #             )
            
# #             if df is not None and not df.empty:
# #                 # Write to CSV
# #                 for _, row in df.iterrows():
# #                     if records_collected >= total_records:
# #                         break
# #                     csv_writer.writerow([
# #                         row['Open time'].strftime('%Y-%m-%d %H:%M:%S'),
# #                         row['Open'],
# #                         row['High'],
# #                         row['Low'],
# #                         row['Close'],
# #                         row['Volume']
# #                     ])
# #                     records_collected += 1
                
# #                 print(f"Collected {records_collected}/{total_records} records so far")
                
# #                 # Sleep to avoid API rate limits
# #                 if batch < num_batches - 1:
# #                     time.sleep(1)
# #             else:
# #                 print("Failed to fetch data for this batch, retrying...")
# #                 time.sleep(5)
# #                 batch -= 1
    
# #     print(f"Data collection complete! {records_collected} records saved to {output_file}")

# # if __name__ == "__main__":
# #     # Collect 100,000 minute-by-minute BTC price records
# #     collect_btc_data(total_records=400000, output_file='btc_minute_data2.csv')

# # # import pandas as pd
# # # import math

# # # # Load your CSV file
# # # df = pd.read_csv('btc_minute_data.csv')

# # # # Calculate size of each part
# # # chunk_size = math.ceil(len(df) / 8)

# # # # Split and save each part
# # # for i in range(8):
# # #     start = i * chunk_size
# # #     end = start + chunk_size
# # #     chunk = df.iloc[start:end]
# # #     chunk.to_csv(f'part_{i+1}.csv', index=False)
# import pandas as pd
# import ccxt
# import time
# from datetime import datetime, timedelta
# import os

# def collect_btc_data(days_back=30):
#     """
#     Collect minute-by-minute BTC/USD price data from Binance
#     Args:
#         days_back: Number of days of historical data to collect
#     Returns:
#         DataFrame with OHLCV data
#     """
#     print(f"Collecting {days_back} days of BTC/USD minute data...")
    
#     # Initialize the Binance exchange API (using the public API, no authentication needed)
#     exchange = ccxt.binance()
    
#     # Calculate the start timestamp (now - days_back)
#     end_time = datetime.now()
#     start_time = end_time - timedelta(days=days_back)
    
#     # Convert to millisecond timestamps
#     start_timestamp = int(start_time.timestamp() * 1000)
#     end_timestamp = int(end_time.timestamp() * 1000)
    
#     all_ohlcv = []
#     current_timestamp = start_timestamp
    
#     # Binance has a limit of 1000 candles per request, so we need to paginate
#     while current_timestamp < end_timestamp:
#         try:
#             # Fetch OHLCV data (Open, High, Low, Close, Volume)
#             ohlcv = exchange.fetch_ohlcv(
#                 symbol='BTC/USDT',
#                 timeframe='1m',
#                 since=current_timestamp,
#                 limit=1000
#             )
            
#             if len(ohlcv) > 0:
#                 all_ohlcv.extend(ohlcv)
#                 # Update timestamp for next iteration (last timestamp + 1 minute)
#                 current_timestamp = ohlcv[-1][0] + 60000  # Add 1 minute in milliseconds
#                 print(f"Collected data up to {datetime.fromtimestamp(current_timestamp/1000)}")
#             else:
#                 break
                
#             # Rate limiting to avoid hitting API limits
#             time.sleep(1)
            
#         except Exception as e:
#             print(f"Error fetching data: {e}")
#             time.sleep(10)  # Wait longer on error
    
#     # Convert to DataFrame
#     df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
#     # Convert timestamp to datetime
#     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
#     # Add additional features that might be useful for prediction
#     df['hour'] = df['timestamp'].dt.hour
#     df['minute'] = df['timestamp'].dt.minute
#     df['day_of_week'] = df['timestamp'].dt.dayofweek
#     df['price_change'] = df['close'].pct_change()
#     df['volatility'] = (df['high'] - df['low']) / df['low']
    
#     # Calculate moving averages
#     df['ma_5'] = df['close'].rolling(window=5).mean()
#     df['ma_10'] = df['close'].rolling(window=10).mean()
#     df['ma_20'] = df['close'].rolling(window=20).mean()
    
#     # Calculate RSI (Relative Strength Index)
#     delta = df['close'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#     rs = gain / loss
#     df['rsi'] = 100 - (100 / (1 + rs))
    
#     # Drop rows with NaN values
#     df.dropna(inplace=True)
    
#     # Save to CSV
#     csv_path = 'btc_minute_data3.csv'
#     df.to_csv(csv_path, index=False)
#     print(f"Data saved to {csv_path}")
    
#     return df

# if __name__ == "__main__":
#     # Collect 30 days of minute-by-minute data by default
#     collect_btc_data(days_back=365)

import pandas as pd
import ccxt
import time
from datetime import datetime, timedelta
import os

def collect_btc_data(days_back=30):
    """
    Collect hour-by-hour BTC/USD price data from Binance
    Args:
        days_back: Number of days of historical data to collect
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Collecting {days_back} days of BTC/USD hourly data...")
    
    # Initialize the Binance exchange API (using the public API, no authentication needed)
    exchange = ccxt.binance()
    
    # Calculate the start timestamp (now - days_back)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)
    
    # Convert to millisecond timestamps
    start_timestamp = int(start_time.timestamp() * 1000)
    end_timestamp = int(end_time.timestamp() * 1000)
    
    all_ohlcv = []
    current_timestamp = start_timestamp
    
    # Binance has a limit of 1000 candles per request, so we need to paginate
    while current_timestamp < end_timestamp:
        try:
            # Fetch OHLCV data (Open, High, Low, Close, Volume) with 1h timeframe
            ohlcv = exchange.fetch_ohlcv(
                symbol='BTC/USDT',
                timeframe='1h',
                since=current_timestamp,
                limit=1000
            )
            
            if len(ohlcv) > 0:
                all_ohlcv.extend(ohlcv)
                # Update timestamp for next iteration (last timestamp + 1 hour)
                current_timestamp = ohlcv[-1][0] + 3600000  # Add 1 hour in milliseconds
                print(f"Collected data up to {datetime.fromtimestamp(current_timestamp/1000)}")
            else:
                break
                
            # Rate limiting to avoid hitting API limits
            time.sleep(1)
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(10)  # Wait longer on error
    
    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Add additional features that might be useful for prediction
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['price_change'] = df['close'].pct_change()
    df['volatility'] = (df['high'] - df['low']) / df['low']
    
    # Calculate moving averages
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    
    # Calculate RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    # Save to CSV
    csv_path = 'btc_hourly_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")
    
    return df

if __name__ == "__main__":
    # Collect 30 days of hour-by-hour data by default
    collect_btc_data(days_back=900)