import time
import csv
import datetime
import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_btc_historical_data(start_time, end_time, interval='1m', limit=1000):
    """
    Fetch historical BTC/USDT data from Binance API
    
    Parameters:
    - start_time: Start time in milliseconds (Unix timestamp)
    - end_time: End time in milliseconds (Unix timestamp)
    - interval: Kline/candlestick interval (1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M)
    - limit: Maximum number of records to return (max 1000)
    
    Returns:
    - DataFrame with columns: 'Open time', 'Open', 'High', 'Low', 'Close', 'Volume', etc.
    """
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': 'BTCUSDT',
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': limit
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=[
            'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
            'Close time', 'Quote asset volume', 'Number of trades',
            'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
        ])
        
        # Convert numeric columns
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # Convert timestamp to datetime
        df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
        
        # Select only required columns
        return df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']]
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

def collect_btc_data(total_records=100000, output_file='btc_price_data.csv'):
    """
    Collect historical BTC price data and save to CSV
    
    Parameters:
    - total_records: Total number of 1-minute records to collect
    - output_file: Output CSV file name
    """
    print(f"Starting collection of {total_records} BTC price records...")
    
    # Calculate how many API calls we need (max 1000 per call)
    batch_size = 1000
    num_batches = (total_records + batch_size - 1) // batch_size
    
    # Calculate end time (now) and start time based on total_records
    end_time = int(datetime.now().timestamp() * 1000)
    
    # Create/open the CSV file
    with open(output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['Open time', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        records_collected = 0
        
        for batch in range(num_batches):
            if records_collected >= total_records:
                break
                
            # Calculate batch end_time and start_time (in milliseconds)
            batch_end_time = end_time - (batch * batch_size * 60 * 1000)
            batch_start_time = batch_end_time - (batch_size * 60 * 1000)
            
            print(f"Fetching batch {batch+1}/{num_batches}: {batch_start_time} to {batch_end_time}")
            
            # Fetch data
            df = fetch_btc_historical_data(
                start_time=batch_start_time,
                end_time=batch_end_time,
                interval='1m',
                limit=batch_size
            )
            
            if df is not None and not df.empty:
                # Write to CSV
                for _, row in df.iterrows():
                    if records_collected >= total_records:
                        break
                    csv_writer.writerow([
                        row['Open time'].strftime('%Y-%m-%d %H:%M:%S'),
                        row['Open'],
                        row['High'],
                        row['Low'],
                        row['Close'],
                        row['Volume']
                    ])
                    records_collected += 1
                
                print(f"Collected {records_collected}/{total_records} records so far")
                
                # Sleep to avoid API rate limits
                if batch < num_batches - 1:
                    time.sleep(1)
            else:
                print("Failed to fetch data for this batch, retrying...")
                time.sleep(5)
                batch -= 1
    
    print(f"Data collection complete! {records_collected} records saved to {output_file}")

if __name__ == "__main__":
    # Collect 100,000 minute-by-minute BTC price records
    collect_btc_data(total_records=100000, output_file='btc_minute_data.csv')