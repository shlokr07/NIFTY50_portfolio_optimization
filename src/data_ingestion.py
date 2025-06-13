import yfinance as yf
import pandas as pd
import requests
import datetime
import time
import random
import logging
import os
from pathlib import Path
from urllib.parse import quote
import json
from playwright.sync_api import sync_playwright

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Constants
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
scrip_list_dir = data_dir / "scrip_list"
scrip_list_dir.mkdir(exist_ok=True)
hist_dir = data_dir / "historical_data"
hist_dir.mkdir(exist_ok=True)
last_updated_file = data_dir / "last_updated.txt"

# Configuration
START_DATE = None  # Set to None for max data, or specify date like "2020-01-01"
PRICE_UPDATE_INTERVAL_DAYS = 1  # Daily updates
SCRIP_UPDATE_INTERVAL_DAYS = 7  # Weekly updates

INDEX_NAMES = [
    "NIFTY AUTO", "NIFTY BANK", "NIFTY FINANCIAL SERVICES", "NIFTY IT", 
    "NIFTY FMCG", "NIFTY MEDIA", "NIFTY METAL", "NIFTY PHARMA", "NIFTY HEALTHCARE INDEX", 
    "NIFTY CONSUMER DURABLES", "NIFTY REALTY", "NIFTY OIL AND GAS"
]

# === Utility Functions ===
def sanitize_filename(name):
    """Convert index name to valid filename."""
    return name.replace(" ", "_").replace("&", "AND").upper()

def get_nse_session():
    """
    Create and return a properly configured session for NSE with required headers
    and cookie initialization.
    """
    session = requests.Session()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/json,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    }
    
    # Initialize session with NSE homepage to get cookies
    try:
        logger.info("Initializing NSE session...")
        session.headers.update(headers)
        response = session.get("https://www.nseindia.com", timeout=10)
        response.raise_for_status()
        
        # Verify essential cookies are present
        cookies = session.cookies.get_dict()
        essential_cookies = ["NSE_APP_ID", "NSIT", "bm_sv"]
        missing_cookies = [cookie for cookie in essential_cookies if cookie not in cookies]
        
        if missing_cookies:
            logger.warning(f"Missing essential cookies: {missing_cookies}")
        else:
            logger.info("All essential cookies acquired")
            
        return session
    except requests.RequestException as e:
        logger.error(f"Failed to initialize NSE session: {e}")
        raise

def get_index_scrips(index_name):
    """
    Fetch scrips for a specific index from NSE API using the simplified session.
    """
    base_url = "https://www.nseindia.com"
    index_encoded = quote(index_name)
    api_url = f"{base_url}/api/equity-stockIndices?index={index_encoded}"

    session = get_nse_session()

    try:
        print(f"Fetching data for {index_name}...")
        response = session.get(api_url, timeout=10)
        response.raise_for_status()

        data = response.json()
        symbols = [item["symbol"] for item in data.get("data", [])]

        print(f"Retrieved {len(symbols)} symbols:")
        print(symbols)
        return symbols

    except requests.RequestException as e:
        print(f"Failed to fetch index data: {e}")
        return []

def fetch_and_save_data(symbol, index_name, min_days_required=252):
    """
    Fetch and save historical data for a symbol in the appropriate index directory.
    """
    try:
        sanitized_index = sanitize_filename(index_name)
        index_hist_dir = hist_dir / sanitized_index
        index_hist_dir.mkdir(exist_ok=True)
        
        file_path = index_hist_dir / f"{symbol}.csv"
        existing_data = pd.DataFrame()
        fetch_start_date = START_DATE

        # Load existing data if available
        if file_path.exists():
            try:
                existing_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if not existing_data.empty:
                    last_date = existing_data.index.max().date()
                    # Check if data is recent enough
                    if (datetime.date.today() - last_date).days <= PRICE_UPDATE_INTERVAL_DAYS:
                        logger.info(f"{symbol} ({index_name}) data is up to date.")
                        return True
                    # Set start date for incremental update
                    fetch_start_date = last_date + datetime.timedelta(days=1)
                    logger.info(f"Fetching incremental data for {symbol} ({index_name}) from {fetch_start_date}")
            except Exception as e:
                logger.warning(f"Error reading existing data for {symbol} ({index_name}): {e}")
                existing_data = pd.DataFrame()

        # Fetch new data
        yahoo_symbol = symbol + ".NS"
        if fetch_start_date:
            logger.info(f"Fetching data for {yahoo_symbol} from {fetch_start_date}")
            new_data = yf.download(yahoo_symbol, start=fetch_start_date, progress=False, auto_adjust=False)
        else:
            logger.info(f"Fetching maximum historical data for {yahoo_symbol}")
            new_data = yf.download(yahoo_symbol, progress=False, auto_adjust=False)

        # Handle empty data
        if new_data.empty and existing_data.empty:
            logger.warning(f"No data available for {yahoo_symbol}")
            return False

        # Combine data
        if not existing_data.empty and not new_data.empty:
            full_data = pd.concat([existing_data, new_data])
            full_data = full_data[~full_data.index.duplicated(keep='last')]
        elif not new_data.empty:
            full_data = new_data
        else:
            full_data = existing_data

        full_data.sort_index(inplace=True)

        # Check minimum data requirement
        if full_data.empty or (pd.Timestamp.today() - full_data.index.min()).days < min_days_required:
            logger.warning(f"{yahoo_symbol} has insufficient historical data ({(pd.Timestamp.today() - full_data.index.min()).days} days). Minimum required: {min_days_required} days.")
            return False

        # Save data
        full_data.to_csv(file_path)
        logger.info(f"Saved {len(full_data)} records for {yahoo_symbol} in {sanitized_index}")
        return True

    except Exception as e:
        logger.error(f"Error fetching data for {symbol} ({index_name}): {e}")
        return False

def main():
    """
    Main data ingestion pipeline for all indices.
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting Multi-Index Data Ingestion Pipeline")
        logger.info("=" * 60)

        today = datetime.date.today()
        
        # Read last updated dates
        hist_last_updated = None
        scrips_last_updated = None
        
        if last_updated_file.exists():
            try:
                with open(last_updated_file, "r") as f:
                    lines = f.read().strip().split('\n')
                if len(lines) >= 2:
                    parts = lines[1].split(',')
                    if len(parts) >= 2:
                        hist_date_str = parts[0].strip()
                        scrips_date_str = parts[1].strip()
                        
                        if hist_date_str and hist_date_str != 'None':
                            hist_last_updated = datetime.datetime.strptime(hist_date_str, "%Y-%m-%d").date()
                        if scrips_date_str and scrips_date_str != 'None':
                            scrips_last_updated = datetime.datetime.strptime(scrips_date_str, "%Y-%m-%d").date()
                        
                        logger.info(f"Last updates - Historical: {hist_last_updated}, Scrips: {scrips_last_updated}")
            except Exception as e:
                logger.warning(f"Could not read last updated dates: {e}")

        # Determine what needs updating
        price_update_needed = (hist_last_updated is None or 
                              (today - hist_last_updated).days >= PRICE_UPDATE_INTERVAL_DAYS)
        
        scrip_update_needed = (scrips_last_updated is None or 
                              (today - scrips_last_updated).days >= SCRIP_UPDATE_INTERVAL_DAYS)

        logger.info(f"Price update needed: {price_update_needed}")
        logger.info(f"Scrip update needed: {scrip_update_needed}")

        if not price_update_needed and not scrip_update_needed:
            logger.info("No updates required today.")
            return

        # Process each index
        all_index_symbols = {}
        scrips_updated = False
        hist_updated = False
        
        for index_name in INDEX_NAMES:
            logger.info(f"\n--- Processing {index_name} ---")
            sanitized_name = sanitize_filename(index_name)
            scrip_file = scrip_list_dir / f"{sanitized_name}.csv"
            
            # Handle scrip list updates for this index
            current_symbols = []
            if scrip_update_needed:
                logger.info(f"Fetching scrips for {index_name}...")
                new_symbols = get_index_scrips(index_name)
                
                if new_symbols:
                    # Check if scrips have changed
                    if scrip_file.exists():
                        try:
                            existing_symbols = pd.read_csv(scrip_file, header=None)[0].tolist()
                            if set(new_symbols) != set(existing_symbols):
                                logger.info(f"Scrip list for {index_name} has changed. Updating file.")
                                pd.Series(new_symbols).to_csv(scrip_file, index=False, header=False)
                                scrips_updated = True
                            else:
                                logger.info(f"Scrip list for {index_name} unchanged.")
                        except Exception as e:
                            logger.warning(f"Error comparing scrips for {index_name}: {e}")
                            pd.Series(new_symbols).to_csv(scrip_file, index=False, header=False)
                            scrips_updated = True
                    else:
                        # First time - create scrips file
                        pd.Series(new_symbols).to_csv(scrip_file, index=False, header=False)
                        scrips_updated = True
                        logger.info(f"Created new scrips file for {index_name}.")
                    
                    current_symbols = new_symbols
                else:
                    logger.warning(f"Could not fetch scrips for {index_name}")
            
            # Load scrips for price updates
            if not current_symbols and scrip_file.exists():
                try:
                    current_symbols = pd.read_csv(scrip_file, header=None)[0].tolist()
                    logger.info(f"Loaded {len(current_symbols)} symbols for {index_name}.")
                except Exception as e:
                    logger.error(f"Failed to load scrips for {index_name}: {e}")
                    continue
            
            if current_symbols:
                all_index_symbols[index_name] = current_symbols
            else:
                logger.warning(f"No symbols available for {index_name}")

        # Handle historical data updates
        if price_update_needed and all_index_symbols:
            logger.info(f"\n--- Updating Historical Data for All Indices ---")
            
            total_successful = 0
            total_symbols = sum(len(symbols) for symbols in all_index_symbols.values())
            current_symbol_count = 0
            
            for index_name, symbols in all_index_symbols.items():
                logger.info(f"\nUpdating historical data for {index_name} ({len(symbols)} symbols)")
                
                successful_updates = 0
                for symbol in symbols:
                    current_symbol_count += 1
                    logger.info(f"Processing {symbol} ({current_symbol_count}/{total_symbols}) - {index_name}")
                    
                    if fetch_and_save_data(symbol, index_name):
                        successful_updates += 1
                        total_successful += 1
                    
                    # Add small delay to avoid overwhelming the API
                    if current_symbol_count < total_symbols:
                        time.sleep(0.1)
                
                logger.info(f"Successfully updated {successful_updates}/{len(symbols)} symbols for {index_name}")
            
            logger.info(f"Overall: Successfully updated {total_successful}/{total_symbols} symbols across all indices.")
            hist_updated = total_successful > 0

        # Mark scrips as updated if we processed them (even if no changes)
        if scrip_update_needed:
            scrips_updated = True

        # Update last_updated.txt
        final_hist_date = today if hist_updated else hist_last_updated
        final_scrips_date = today if scrips_updated else scrips_last_updated
        
        with open(last_updated_file, "w") as f:
            f.write("historical_data,scrip_list\n")
            f.write(f"{final_hist_date},{final_scrips_date}\n")
        
        logger.info(f"Updated last_updated.txt - Historical: {final_hist_date}, Scrips: {final_scrips_date}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Indices processed: {len(INDEX_NAMES)}")
        logger.info(f"Total scrip files: {len(all_index_symbols)}")
        logger.info(f"Scrips updated: {scrips_updated}")
        logger.info(f"Historical data updated: {hist_updated}")
        logger.info("Data ingestion pipeline completed successfully.")

    except Exception as e:
        logger.critical(f"Critical error in data ingestion pipeline: {e}")
        raise

if __name__ == "__main__":
    main()