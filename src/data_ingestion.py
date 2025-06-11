import yfinance as yf
import pandas as pd
import requests
import datetime
import time
import random
import logging
import os
from pathlib import Path

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
data_dir = Path("src/data")
data_dir.mkdir(exist_ok=True)
hist_dir = data_dir / "historical_data"
hist_dir.mkdir(exist_ok=True)
scrips_file = data_dir / "scrips.csv"
last_updated_file = data_dir / "last_updated.txt"
MAX_YEARS = 20

# === Date Tracking ===

def read_last_updated():
    if not last_updated_file.exists():
        return None, None
    try:
        with open(last_updated_file, "r") as f:
            lines = f.read().strip().split('\n')
        if len(lines) >= 2:
            hist_date_str, scrips_date_str = [d.strip() for d in lines[1].split(',')]
            hist_date = datetime.datetime.strptime(hist_date_str, "%Y-%m-%d").date() if hist_date_str else None
            scrips_date = datetime.datetime.strptime(scrips_date_str, "%Y-%m-%d").date() if scrips_date_str else None
            return hist_date, scrips_date
    except Exception as e:
        logger.warning(f"Could not read last updated dates: {e}")
    return None, None

def update_last_updated(hist_date=None, scrips_date=None):
    current_hist_date, current_scrips_date = read_last_updated()
    today = datetime.date.today()
    final_hist_date = hist_date or current_hist_date or today
    final_scrips_date = scrips_date or current_scrips_date or today
    with open(last_updated_file, "w") as f:
        f.write("historical_data, nifty50_scrips\n")
        f.write(f"{final_hist_date}, {final_scrips_date}\n")

def is_historical_data_update_due():
    hist_date, _ = read_last_updated()
    return not hist_date or (datetime.date.today() - hist_date).days >= 14

def is_scrips_update_due():
    _, scrips_date = read_last_updated()
    return not scrips_date or (datetime.date.today() - scrips_date).days >= 30

# === Scrip Fetching ===

def get_nifty50_scrips(max_retries=3, base_sleep=1, max_jitter=2):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive"
    }

    session = requests.Session()
    session.headers.update(headers)

    try:
        session.get("https://www.nseindia.com", timeout=10)
        time.sleep(1 + random.uniform(0, 1))
    except Exception as e:
        logger.warning(f"Failed to initialize NSE session: {e}")

    api_url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050"
    symbols = []

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"NSE API attempt {attempt}")
            response = session.get(api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                raw_symbols = data.get("data", [])
                symbols = [entry["symbol"] for entry in raw_symbols if entry["symbol"] != "NIFTY 50"]
                if len(symbols) >= 45:
                    logger.info(f"Fetched {len(symbols)} symbols from NSE API.")
                    break
                else:
                    logger.warning(f"Incomplete symbol list: {len(symbols)} entries.")
                    symbols = []
            else:
                logger.warning(f"NSE API error: {response.status_code}")
        except Exception as e:
            logger.error(f"NSE fetch error attempt {attempt}: {e}")
        time.sleep(base_sleep + random.uniform(0, max_jitter))

    if not symbols:
        logger.warning("Falling back to local scrips file.")
        try:
            if scrips_file.exists():
                symbols = pd.read_csv(scrips_file, header=None)[0].tolist()
                logger.info(f"Loaded {len(symbols)} symbols from local file.")
            else:
                logger.critical("Local scrips file not found.")
        except Exception as e:
            logger.critical(f"Failed to read local scrips file: {e}")
            symbols = []

    return symbols

# === Historical Data Fetch ===

def fetch_and_save_data(symbol, min_days_required=365):
    try:
        file_path = hist_dir / f"{symbol}.csv"
        max_start_date = datetime.date.today() - datetime.timedelta(days=MAX_YEARS * 365)

        if file_path.exists():
            existing_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            last_date = existing_data.index.max().date()
            if (datetime.date.today() - last_date).days < 14:
                logger.info(f"{symbol} already updated within 2 weeks.")
                return True
            existing_data = existing_data[existing_data.index.date >= max_start_date]
            start_date = last_date + datetime.timedelta(days=1)
        else:
            existing_data = pd.DataFrame()
            start_date = max_start_date

        logger.info(f"Fetching data for {symbol} from {start_date}")
        new_data = yf.download(symbol, start=start_date, progress=False)

        if new_data.empty and existing_data.empty:
            logger.warning(f"No data available for {symbol}")
            return False

        full_data = pd.concat([existing_data, new_data]).sort_index().drop_duplicates()
        full_data = full_data[full_data.index.date >= max_start_date]

        if full_data.empty or (pd.Timestamp.today() - full_data.index.min()).days < min_days_required:
            logger.warning(f"{symbol} has insufficient valid data. Skipping.")
            return False

        full_data.to_csv(file_path)
        logger.info(f"Saved {len(full_data)} records for {symbol}")
        return True

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return False

# === Main Execution ===

def main():
    try:
        logger.info("Starting Data Ingestion Pipeline")

        hist_update_due = is_historical_data_update_due()
        scrips_update_due = is_scrips_update_due()

        if not hist_update_due and not scrips_update_due:
            logger.info("No updates required today.")
            return

        nifty_symbols = []
        scrips_updated = False

        if scrips_update_due:
            logger.info("Fetching NIFTY 50 scrips...")
            nifty_symbols = get_nifty50_scrips()
            if not nifty_symbols:
                logger.error("Could not fetch NIFTY symbols.")
                return

            current_scrips_df = pd.Series(nifty_symbols)
            if not scrips_file.exists() or not current_scrips_df.equals(pd.read_csv(scrips_file, header=None)[0]):
                logger.info("Updating scrips file.")
                current_scrips_df.to_csv(scrips_file, index=False, header=False)
            scrips_updated = True
        else:
            if scrips_file.exists():
                nifty_symbols = pd.read_csv(scrips_file, header=None)[0].tolist()
            else:
                logger.error("No scrips file found and scrips update not due.")
                return

        hist_updated = False
        if hist_update_due and nifty_symbols:
            logger.info("Updating historical data...")
            yahoo_symbols = [sym + ".NS" for sym in nifty_symbols]
            successful_updates = sum(fetch_and_save_data(symbol) for symbol in yahoo_symbols)
            logger.info(f"Updated {successful_updates}/{len(yahoo_symbols)} symbols.")
            hist_updated = successful_updates > 0

        today = datetime.date.today()
        update_last_updated(
            hist_date=today if hist_updated else None,
            scrips_date=today if scrips_updated else None
        )

        logger.info("Data ingestion pipeline completed.")

    except Exception as e:
        logger.critical(f"Unexpected error occurred: {e}")
        raise

if __name__ == "__main__":
    main()