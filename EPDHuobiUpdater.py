#!/usr/bin/env python3
"""
HTX Crypto Data Collector v5.0 - High-Speed Edition
====================================================
High-performance cryptocurrency data collection for USDT pairs
Features:
✅ Parallel processing for multiple symbols
✅ HTX API key support for higher rate limits
✅ Multiple exchange sources with automatic fallback
✅ Concurrent data fetching from multiple sources
✅ Chinese market indicators (成交额, 振幅, 涨跌幅, 涨跌额, 换手率)
✅ Optimized for speed with thread pools
✅ Smart caching to avoid re-downloading

Author: Data Collection Specialist
Version: 5.0 - High-Speed Edition
"""

import pandas as pd
import json
import time
import random
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor, \
    TimeoutError as FuturesTimeoutError
import signal
import sys
import requests
import requests.adapters
import numpy as np
import os
from collections import defaultdict, deque
from enum import Enum
from urllib.parse import urlencode, urlparse
import traceback
import hashlib
import hmac
import base64
from functools import lru_cache
import pickle

# Try importing optional dependencies
try:
    import ccxt

    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False
    print("⚠️ CCXT not installed. Install with: pip install ccxt")

try:
    import aiohttp
    import asyncio

    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    print("⚠️ aiohttp not installed for async support. Install with: pip install aiohttp")

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# CONFIGURATION CONSTANTS
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
BATCH_SIZE = 50  # Increased for parallel processing

# HTX API Configuration
HTX_API_BASE = "https://api.htx.com"

# Data directory
DATA_BASE_DIR = "Chinese_Market/data/huobi/csv"


def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("Chinese_Market/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / 'data_collector.log', encoding='utf-8')
        ]
    )


setup_logging()
logger = logging.getLogger(__name__)


class HTXSigner:
    """HTX API request signer for authentication"""

    def __init__(self, access_key: str, secret_key: str):
        self.access_key = access_key
        self.secret_key = secret_key

    def sign(self, method: str, host: str, path: str, params: dict = None) -> dict:
        """Generate signature for HTX API request"""
        sorted_params = sorted(params.items()) if params else []
        encoded_params = urlencode(sorted_params)

        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        payload = [method.upper(), host, path, encoded_params]
        payload = '\n'.join(payload)
        payload = payload.encode(encoding='UTF-8')

        secret_key = self.secret_key.encode(encoding='UTF-8')
        signature = base64.b64encode(
            hmac.new(secret_key, payload, digestmod=hashlib.sha256).digest()
        ).decode()

        auth_params = {
            'AccessKeyId': self.access_key,
            'SignatureMethod': 'HmacSHA256',
            'SignatureVersion': '2',
            'Timestamp': timestamp,
            'Signature': signature
        }

        if params:
            auth_params.update(params)

        return auth_params


@dataclass
class Config:
    """Configuration for high-speed data collection"""

    # Directory settings
    data_dir: str = DATA_BASE_DIR

    # HTX API credentials (for higher rate limits)
    htx_access_key: str = ""
    htx_secret_key: str = ""

    # Data sources
    enable_htx: bool = True
    enable_ccxt: bool = True
    preferred_exchanges: List[str] = field(default_factory=lambda: [
        'gate', 'kucoin', 'mexc', 'bitget', 'bybit', 'okx', 'binance'
    ])

    # Performance settings - OPTIMIZED FOR SPEED
    max_workers: int = field(default_factory=lambda: min(10, os.cpu_count() * 2))  # Reduced for stability
    concurrent_symbols: int = 5  # Start with 5 symbols at once for stability
    timeout_seconds: int = 15  # Reduced timeout

    # Data settings
    fetch_all_history: bool = True
    quote_currency: str = 'USDT'

    # Rate limiting - OPTIMIZED
    adaptive_rate_limit: bool = True
    min_delay_ms: int = 50  # Reduced delays
    max_delay_ms: int = 200

    # Cache settings
    enable_cache: bool = True
    cache_completed_symbols: bool = True

    def __post_init__(self):
        """Initialize configuration"""
        self._load_credentials()
        self._setup_directories()

    def _load_credentials(self):
        """Load HTX API credentials"""
        # Try environment variables first
        self.htx_access_key = os.environ.get('HTX_ACCESS_KEY', self.htx_access_key)
        self.htx_secret_key = os.environ.get('HTX_SECRET_KEY', self.htx_secret_key)

        # Try config file
        config_file = Path('htx_config.json')
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    self.htx_access_key = config_data.get('htx_access_key', self.htx_access_key)
                    self.htx_secret_key = config_data.get('htx_secret_key', self.htx_secret_key)

                if self.htx_access_key and self.htx_secret_key:
                    logger.info("HTX API credentials loaded - Higher rate limits enabled")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")

    def _setup_directories(self):
        """Setup directory structure"""
        data_path = Path(self.data_dir)
        data_path.mkdir(parents=True, exist_ok=True)

        subdirs = ['spot', 'failed', 'metadata', 'cache']
        for subdir in subdirs:
            (data_path / subdir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Data directory: {self.data_dir}")


class HTXAPIClient:
    """HTX API client with authentication support"""

    def __init__(self, config: Config):
        self.config = config
        self.base_url = HTX_API_BASE

        # Setup session with connection pooling
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=2
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

        self.session.headers.update({
            'User-Agent': 'DataCollector/5.0',
            'Content-Type': 'application/json'
        })

        # Setup authentication if available
        self.signer = None
        if config.htx_access_key and config.htx_secret_key:
            self.signer = HTXSigner(config.htx_access_key, config.htx_secret_key)
            logger.info("HTX API authentication enabled")

        # Rate limiter
        self.last_request_time = 0
        self.request_lock = threading.Lock()

    def _request(self, method: str, path: str, params: dict = None, signed: bool = False) -> Optional[dict]:
        """Make API request with optional signing"""
        with self.request_lock:
            # Rate limiting
            if self.signer:
                min_interval = 0.05  # 20 requests per second with API key
            else:
                min_interval = 0.2  # 5 requests per second without API key

            elapsed = time.time() - self.last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            try:
                url = f"{self.base_url}{path}"

                if signed and self.signer:
                    host = urlparse(self.base_url).netloc
                    params = self.signer.sign(method, host, path, params)

                if method == 'GET':
                    response = self.session.get(url, params=params, timeout=self.config.timeout_seconds)
                else:
                    response = self.session.post(url, json=params, timeout=self.config.timeout_seconds)

                self.last_request_time = time.time()

                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'ok':
                        return data

                return None

            except Exception as e:
                logger.debug(f"Request failed: {e}")
                return None

    def get_usdt_symbols(self) -> List[dict]:
        """Get all USDT trading pairs"""
        symbols = []

        logger.info("Fetching symbol list from HTX...")

        # Try with authentication for higher limits
        data = self._request('GET', '/v2/settings/common/symbols', signed=bool(self.signer))

        if not data or 'data' not in data:
            logger.info("v2 API failed, trying v1...")
            # Fallback to v1
            data = self._request('GET', '/v1/common/symbols')

        if data and 'data' in data:
            logger.info(f"Parsing {len(data['data'])} symbols...")
            for item in data['data']:
                parsed = self._parse_symbol_info(item)
                if parsed and parsed['quote'] == 'USDT':
                    symbols.append(parsed)
            logger.info(f"Found {len(symbols)} USDT pairs")
        else:
            logger.error("Failed to get symbols from HTX")

        return symbols

    def _parse_symbol_info(self, item: dict) -> Optional[dict]:
        """Parse symbol info"""
        try:
            if 'bc' in item and 'qc' in item:
                base = item.get('bcdn', item['bc']).upper()
                quote = item.get('qcdn', item['qc']).upper()
                return {
                    'symbol': f"{base}/{quote}",
                    'base': base,
                    'quote': quote,
                    'state': item.get('state', 'unknown'),
                    'original_code': item.get('sc', '')
                }
            elif 'base-currency' in item and 'quote-currency' in item:
                base = item['base-currency'].upper()
                quote = item['quote-currency'].upper()
                return {
                    'symbol': f"{base}/{quote}",
                    'base': base,
                    'quote': quote,
                    'state': item.get('state', 'unknown'),
                    'original_code': item.get('symbol', '')
                }
        except Exception:
            pass
        return None

    def get_all_klines(self, symbol: str) -> pd.DataFrame:
        """Get ALL available historical daily klines with timeout protection"""
        all_data = []
        symbol_clean = symbol.replace('/', '').lower()

        end_time = None
        consecutive_empty = 0
        max_requests = 10  # Reduced from 100 for speed

        for i in range(max_requests):
            try:
                params = {
                    'symbol': symbol_clean,
                    'period': '1day',
                    'size': 2000
                }

                if end_time:
                    params['to'] = end_time

                # Use signed request if available for higher limits
                data = self._request('GET', '/market/history/kline', params, signed=bool(self.signer))

                if data and 'data' in data and len(data['data']) > 0:
                    all_data.extend(data['data'])
                    earliest = min(item['id'] for item in data['data'])
                    end_time = earliest - 1
                    consecutive_empty = 0

                    # Stop if we have enough data (about 5 years)
                    if len(all_data) > 1800:
                        break
                else:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        break

            except Exception as e:
                logger.debug(f"Error in kline request: {e}")
                break

        if all_data:
            return self._convert_klines(all_data, symbol)

        return pd.DataFrame()

    def _convert_klines(self, data: list, symbol: str) -> pd.DataFrame:
        """Convert klines to DataFrame with Chinese indicators"""
        if not data:
            return pd.DataFrame()

        records = []
        data.sort(key=lambda x: x['id'])
        prev_close = None

        for item in data:
            try:
                timestamp = item['id']
                open_price = float(item['open'])
                high = float(item['high'])
                low = float(item['low'])
                close = float(item['close'])
                volume = float(item.get('amount', 0))
                volume_quote = float(item.get('vol', 0))

                # Chinese indicators
                amplitude = ((high - low) / open_price * 100) if open_price > 0 else 0

                if prev_close:
                    change_rate = ((close - prev_close) / prev_close * 100)
                    change_amount = close - prev_close
                else:
                    change_rate = 0
                    change_amount = 0

                records.append({
                    'timestamp': datetime.fromtimestamp(timestamp),
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                    'volume_quote': volume_quote,
                    'count': int(item.get('count', 0)),
                    'symbol': symbol,
                    'price_change': change_rate,
                    '成交额': volume_quote,
                    '振幅': amplitude,
                    '涨跌幅': change_rate,
                    '涨跌额': change_amount,
                    '换手率': 0
                })

                prev_close = close

            except Exception:
                continue

        if records:
            df = pd.DataFrame(records)
            df = df.set_index('timestamp')
            df = df.sort_index()
            return df

        return pd.DataFrame()


class CCXTDataCollector:
    """Multi-exchange data collector with better initialization"""

    def __init__(self, config: Config):
        self.config = config
        self.exchanges = {}
        self._init_exchanges()

    def _init_exchanges(self):
        """Initialize exchanges with better error handling"""
        if not CCXT_AVAILABLE:
            logger.warning("CCXT not available")
            return

        # Extended list of exchanges to try
        exchange_configs = {
            'gate': {'enableRateLimit': True},
            'kucoin': {'enableRateLimit': True},
            'mexc': {'enableRateLimit': True},
            'bitget': {'enableRateLimit': True},
            'bybit': {'enableRateLimit': True},
            'okx': {'enableRateLimit': True, 'options': {'defaultType': 'spot'}},
            'binance': {'enableRateLimit': True, 'options': {'defaultType': 'spot'}},
            'huobi': {'enableRateLimit': True},
            'htx': {'enableRateLimit': True},
        }

        for exchange_name, config in exchange_configs.items():
            try:
                if hasattr(ccxt, exchange_name):
                    exchange_class = getattr(ccxt, exchange_name)
                    exchange = exchange_class({
                        **config,
                        'timeout': self.config.timeout_seconds * 1000
                    })

                    # Test connection with timeout
                    exchange.load_markets()
                    self.exchanges[exchange_name] = exchange
                    logger.info(f"✅ Initialized {exchange_name} exchange")

            except Exception as e:
                logger.debug(f"Failed to initialize {exchange_name}: {str(e)[:100]}")

    def get_historical_data_fast(self, symbol: str) -> pd.DataFrame:
        """Get historical data from fastest available exchange with timeout"""
        if not self.exchanges:
            return pd.DataFrame()

        # Try exchanges with shorter timeout
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []

            # Only try first 2 exchanges for speed
            for name, exchange in list(self.exchanges.items())[:2]:
                future = executor.submit(self._fetch_from_exchange, exchange, symbol, name)
                futures.append((name, future))

            for name, future in futures:
                try:
                    result = future.result(timeout=5)  # 5 second timeout per exchange
                    if result is not None and not result.empty:
                        return result
                except Exception:
                    continue

        return pd.DataFrame()

    def _fetch_from_exchange(self, exchange, symbol: str, name: str) -> Optional[pd.DataFrame]:
        """Fetch data from specific exchange with optimizations"""
        try:
            if not exchange.has['fetchOHLCV']:
                return None

            # Quick check if symbol exists
            if symbol not in exchange.markets:
                return None

            all_data = []
            since = exchange.parse8601('2020-01-01T00:00:00Z')  # Only get last 5 years for speed

            # Limit iterations to prevent getting stuck
            max_iterations = 3  # Reduced from unlimited

            for i in range(max_iterations):
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=since, limit=1000)

                    if not ohlcv:
                        break

                    all_data.extend(ohlcv)

                    if len(ohlcv) < 1000:
                        break

                    since = ohlcv[-1][0] + 86400000

                except Exception as e:
                    logger.debug(f"Exchange fetch error: {e}")
                    break

            if all_data and len(all_data) > 50:  # Need at least 50 days of data
                return self._convert_ccxt_data(all_data, symbol, name)

        except Exception as e:
            logger.debug(f"Failed to fetch from {name}: {e}")

        return None

    def _convert_ccxt_data(self, data: list, symbol: str, source: str) -> pd.DataFrame:
        """Convert CCXT data to DataFrame"""
        records = []
        prev_close = None

        for item in data:
            try:
                timestamp = item[0] / 1000
                open_price = float(item[1])
                high = float(item[2])
                low = float(item[3])
                close = float(item[4])
                volume = float(item[5])

                volume_quote = volume * close
                amplitude = ((high - low) / open_price * 100) if open_price > 0 else 0

                if prev_close:
                    change_rate = ((close - prev_close) / prev_close * 100)
                    change_amount = close - prev_close
                else:
                    change_rate = 0
                    change_amount = 0

                records.append({
                    'timestamp': datetime.fromtimestamp(timestamp),
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                    'volume_quote': volume_quote,
                    'count': 0,
                    'symbol': symbol,
                    'source': source,
                    'price_change': change_rate,
                    '成交额': volume_quote,
                    '振幅': amplitude,
                    '涨跌幅': change_rate,
                    '涨跌额': change_amount,
                    '换手率': 0
                })

                prev_close = close

            except Exception:
                continue

        if records:
            df = pd.DataFrame(records)
            df = df.set_index('timestamp')
            df = df.sort_index()
            return df

        return pd.DataFrame()


class HighSpeedDataCollector:
    """Main high-speed data collector with parallel processing"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.htx_client = HTXAPIClient(self.config)
        self.ccxt_collector = CCXTDataCollector(self.config)

        # Cache for completed symbols
        self.cache_file = Path(self.config.data_dir) / 'cache' / 'completed_symbols.pkl'
        self.completed_symbols = self._load_cache()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.shutdown_requested = False

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        if not self.shutdown_requested:
            logger.info("Shutdown requested...")
            self.shutdown_requested = True
            self.executor.shutdown(wait=False)

    def _load_cache(self) -> set:
        """Load cached completed symbols"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                pass
        return set()

    def _save_cache(self):
        """Save completed symbols to cache"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.completed_symbols, f)
        except Exception:
            pass

    def collect_all_usdt_pairs_parallel(self) -> dict:
        """Collect all USDT pairs in parallel for maximum speed"""
        results = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'total_records': 0,
            'errors': [],
            'start_time': time.time()
        }

        # Get USDT symbols
        logger.info("Fetching USDT symbols list...")
        symbols_info = self.htx_client.get_usdt_symbols()

        if not symbols_info:
            logger.error("No USDT symbols found")
            return results

        # Filter active symbols
        active_symbols = [
            s for s in symbols_info
            if s.get('state') in ['online', 'enabled', 'trading']
               and s['symbol'] not in self.completed_symbols  # Skip already completed
        ]

        results['total'] = len(active_symbols)
        results['skipped'] = len(symbols_info) - len(active_symbols) + len(self.completed_symbols)

        logger.info(
            f"Processing {len(active_symbols)} active USDT pairs (skipping {len(self.completed_symbols)} cached)")

        # Reduce batch size for better stability
        batch_size = min(5, self.config.concurrent_symbols)  # Start with smaller batches
        batches = [active_symbols[i:i + batch_size] for i in range(0, len(active_symbols), batch_size)]

        for batch_idx, batch in enumerate(batches):
            if self.shutdown_requested:
                break

            batch_start = time.time()
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} symbols)")

            # Show symbols being processed
            batch_symbols = [s.get('symbol', 'Unknown') for s in batch]
            logger.info(f"Symbols: {', '.join(batch_symbols[:3])}...")

            # Submit all symbols in batch for parallel processing
            futures = {}
            for symbol_info in batch:
                if self.shutdown_requested:
                    break

                symbol = symbol_info.get('symbol')
                if symbol:
                    try:
                        future = self.executor.submit(self._collect_symbol_fast, symbol)
                        futures[future] = symbol
                    except Exception as e:
                        logger.error(f"Failed to submit {symbol}: {e}")
                        results['failed'] += 1

            # Collect results with shorter timeout
            completed = 0
            timeout_count = 0

            try:
                for future in as_completed(futures, timeout=60):  # 60 second timeout for whole batch
                    symbol = futures[future]

                    try:
                        success, records = future.result(timeout=15)  # 15 second timeout per symbol

                        if success:
                            results['success'] += 1
                            results['total_records'] += records
                            self.completed_symbols.add(symbol)
                            logger.info(f"✅ {symbol}: {records} records")
                        else:
                            results['failed'] += 1
                            logger.info(f"⚠️ {symbol}: No data available")

                        completed += 1

                        # Show progress within batch
                        if completed % 2 == 0:
                            logger.info(f"  Batch progress: {completed}/{len(batch)}")

                    except FuturesTimeoutError:
                        results['failed'] += 1
                        results['errors'].append(f"{symbol}: Timeout")
                        logger.warning(f"⏱️ {symbol}: Timeout - skipping")
                        future.cancel()  # Cancel the stuck future
                        timeout_count += 1

                    except Exception as e:
                        results['failed'] += 1
                        results['errors'].append(f"{symbol}: {str(e)[:50]}")
                        logger.error(f"❌ {symbol}: {str(e)[:50]}")

            except FuturesTimeoutError:
                logger.warning(f"Batch timeout - processing remaining symbols sequentially")
                # Cancel all pending futures
                for future in futures:
                    future.cancel()

                # Process remaining symbols sequentially
                for symbol_info in batch[completed:]:
                    symbol = symbol_info.get('symbol')
                    if symbol and symbol not in self.completed_symbols:
                        try:
                            success, records = self._collect_symbol_fast(symbol)
                            if success:
                                results['success'] += 1
                                results['total_records'] += records
                                self.completed_symbols.add(symbol)
                                logger.info(f"✅ {symbol}: {records} records (sequential)")
                            else:
                                results['failed'] += 1
                        except Exception as e:
                            results['failed'] += 1
                            logger.error(f"❌ {symbol}: {e}")

            # If too many timeouts, reduce batch size
            if timeout_count > len(batch) * 0.3:
                batch_size = max(2, batch_size - 1)
                logger.info(f"Reducing batch size to {batch_size} due to timeouts")

            # Save cache after each batch
            self._save_cache()

            # Report batch completion
            batch_elapsed = time.time() - batch_start
            logger.info(f"Batch {batch_idx + 1} completed in {batch_elapsed:.1f}s")

            # Adaptive batch size based on performance
            if batch_elapsed < 30 and batch_size < 10:
                batch_size = min(batch_size + 1, 10)
            elif batch_elapsed > 90 and batch_size > 2:
                batch_size = max(batch_size - 1, 2)

            # Brief pause between batches
            if batch_idx < len(batches) - 1:
                time.sleep(0.5)

        # Calculate statistics
        elapsed = time.time() - results['start_time']
        results['elapsed_time'] = elapsed
        results['symbols_per_minute'] = (results['success'] / elapsed * 60) if elapsed > 0 else 0

        logger.info(f"Collection completed in {elapsed:.1f}s - {results['symbols_per_minute']:.1f} symbols/min")
        logger.info(
            f"Success: {results['success']}, Failed: {results['failed']}, Records: {results['total_records']:,}")

        return results

    def _collect_symbol_fast(self, symbol: str) -> Tuple[bool, int]:
        """Collect data for a single symbol - optimized for speed"""
        start_time = time.time()

        try:
            # Check if already exists
            file_path = self._get_file_path(symbol)
            if file_path.exists() and self.config.cache_completed_symbols:
                try:
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                    if len(df) > 100:  # Consider it complete if has good amount of data
                        return True, len(df)
                except Exception:
                    pass  # File might be corrupted, re-download

            # Set timeout for entire operation
            max_time = 20  # 20 seconds max per symbol

            # Try HTX first (usually fastest for HTX pairs)
            data = pd.DataFrame()

            if time.time() - start_time < max_time:
                try:
                    data = self.htx_client.get_all_klines(symbol)
                except Exception as e:
                    logger.debug(f"HTX failed for {symbol}: {e}")

            # If HTX fails or returns little data, try CCXT
            if (data.empty or len(data) < 100) and time.time() - start_time < max_time:
                try:
                    ccxt_data = self.ccxt_collector.get_historical_data_fast(symbol)
                    if not ccxt_data.empty:
                        if data.empty:
                            data = ccxt_data
                        else:
                            # Merge data
                            data = pd.concat([data, ccxt_data])
                            data = data[~data.index.duplicated(keep='first')]
                            data = data.sort_index()
                except Exception as e:
                    logger.debug(f"CCXT failed for {symbol}: {e}")

            # Save any data we got
            if not data.empty:
                self._save_data(data, symbol)
                return True, len(data)

            return False, 0

        except Exception as e:
            logger.debug(f"Error collecting {symbol}: {e}")
            return False, 0

    def _get_file_path(self, symbol: str) -> Path:
        """Get file path for symbol"""
        safe_symbol = symbol.replace('/', '_').replace('-', '_')
        return Path(self.config.data_dir) / 'spot' / f"{safe_symbol}.csv"

    def _save_data(self, data: pd.DataFrame, symbol: str):
        """Save data to CSV"""
        try:
            file_path = self._get_file_path(symbol)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Save to CSV
            data.to_csv(file_path, encoding='utf-8')

            # Save metadata
            metadata = {
                'symbol': symbol,
                'records': len(data),
                'start_date': str(data.index.min()),
                'end_date': str(data.index.max()),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            metadata_file = Path(self.config.data_dir) / 'metadata' / f"{symbol.replace('/', '_')}_meta.json"
            metadata_file.parent.mkdir(parents=True, exist_ok=True)

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to save {symbol}: {e}")

    def get_collection_stats(self) -> dict:
        """Get collection statistics"""
        stats = {
            'total_files': 0,
            'total_size_mb': 0,
            'cached_symbols': len(self.completed_symbols),
            'exchanges_available': list(self.ccxt_collector.exchanges.keys())
        }

        data_path = Path(self.config.data_dir) / 'spot'
        if data_path.exists():
            csv_files = list(data_path.glob("*.csv"))
            stats['total_files'] = len(csv_files)
            stats['total_size_mb'] = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)

        return stats

    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        self._save_cache()


def create_config_file():
    """Create configuration file with API keys"""
    config = {
        "htx_access_key": "your_access_key_here",
        "htx_secret_key": "your_secret_key_here",
        "max_workers": 10,
        "concurrent_symbols": 5,
        "preferred_exchanges": ["gate", "kucoin", "mexc", "bitget", "bybit"]
    }

    with open('htx_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("✅ Configuration file created: htx_config.json")
    print("📌 Add your HTX API credentials for 5-10x faster collection!")


def main():
    """Main function for high-speed data collection"""
    print("=" * 60)
    print("🚀 HTX Crypto Data Collector v5.0 - High-Speed Edition")
    print("⚡ Optimized for parallel processing and speed")
    print("=" * 60)

    # Check for config file
    config_path = Path('htx_config.json')
    if not config_path.exists():
        print("\n⚠️ No configuration file found")
        print("Creating one now...")
        create_config_file()
        print("\n👉 Please edit htx_config.json with your HTX API keys")
        print("   This will increase speed by 5-10x!")

        proceed = input("\nProceed without API keys? (y/n): ").strip().lower()
        if proceed != 'y':
            return

    # Initialize configuration
    config = Config()

    print(f"\n📁 Data directory: {config.data_dir}")
    print(f"⚡ Max workers: {config.max_workers}")
    print(f"🔄 Concurrent symbols: {config.concurrent_symbols}")
    print(f"🔑 HTX API: {'Authenticated ✅' if config.htx_access_key else 'Public (slower)'}")

    # Initialize collector
    collector = HighSpeedDataCollector(config)

    # Quick test to ensure HTX is working
    print("\n🔍 Testing HTX connection...")
    test_symbols = collector.htx_client.get_usdt_symbols()
    if test_symbols:
        print(f"✅ HTX connected - Found {len(test_symbols)} USDT pairs")
    else:
        print("⚠️ HTX connection issue - data collection may be limited")

    # Show available exchanges
    stats = collector.get_collection_stats()
    if stats['exchanges_available']:
        print(f"📊 Available exchanges: {', '.join(stats['exchanges_available'])}")

    while True:
        print("\n" + "=" * 50)
        print("📋 Main Menu")
        print("=" * 50)
        print("1. ⚡ FAST Collection - ALL USDT Pairs")
        print("2. 🎯 Collect Specific Symbols")
        print("3. 📊 Collection Statistics")
        print("4. 🔄 Resume Interrupted Collection")
        print("5. 🗑️ Clear Cache (Recollect All)")
        print("6. 🧪 Test Mode (Collect 5 symbols)")
        print("7. 🚪 Exit")

        choice = input("\nEnter choice (1-7): ").strip()

        if choice == "1":
            print("\n⚡ HIGH-SPEED COLLECTION MODE")
            print(f"Processing up to {config.concurrent_symbols} symbols simultaneously")
            print("This version has improved timeout handling!")

            # Estimate time
            symbols_list = collector.htx_client.get_usdt_symbols()
            if symbols_list:
                total_symbols = len(symbols_list)
                uncached = total_symbols - len(collector.completed_symbols)
                estimated_time = (uncached / 30) * 60  # Assuming 30 symbols per hour

                print(f"\n📊 Total: {total_symbols} symbols")
                print(f"📊 To collect: {uncached} symbols")
                print(f"⏱️ Estimated time: {estimated_time:.0f} minutes")

            confirm = input("\nStart high-speed collection? (y/n): ").strip().lower()
            if confirm == 'y':
                start_time = time.time()
                results = collector.collect_all_usdt_pairs_parallel()

                print(f"\n✅ Collection completed in {results['elapsed_time']:.1f} seconds!")
                print(f"⚡ Speed: {results['symbols_per_minute']:.1f} symbols/minute")
                print(f"📊 Success: {results['success']}")
                print(f"❌ Failed: {results['failed']}")
                print(f"📁 Total records: {results['total_records']:,}")

                if results['errors'][:5]:
                    print("\nFirst 5 errors:")
                    for error in results['errors'][:5]:
                        print(f"  - {error}")

        elif choice == "2":
            symbols_input = input("\nEnter symbols (comma-separated): ").strip()
            if symbols_input:
                symbols = [s.strip().upper() for s in symbols_input.split(',')]

                print(f"\n🎯 Collecting {len(symbols)} symbols...")

                for symbol in symbols:
                    success, records = collector._collect_symbol_fast(symbol)
                    if success:
                        print(f"✅ {symbol}: {records} records")
                    else:
                        print(f"❌ {symbol}: Failed")

        elif choice == "3":
            stats = collector.get_collection_stats()
            print("\n📊 Collection Statistics:")
            print(f"Files collected: {stats['total_files']}")
            print(f"Total size: {stats['total_size_mb']:.2f} MB")
            print(f"Cached symbols: {stats['cached_symbols']}")
            print(f"Available exchanges: {', '.join(stats['exchanges_available'])}")

        elif choice == "4":
            print("\n🔄 Resuming collection...")
            print(f"Skipping {len(collector.completed_symbols)} already completed symbols")

            results = collector.collect_all_usdt_pairs_parallel()
            print(f"\n✅ Resumed collection completed!")
            print(f"New symbols collected: {results['success']}")

        elif choice == "5":
            confirm = input("\n⚠️ Clear cache and recollect all data? (y/n): ").strip().lower()
            if confirm == 'y':
                collector.completed_symbols.clear()
                collector._save_cache()
                print("✅ Cache cleared")

        elif choice == "6":
            print("\n🧪 TEST MODE - Collecting 5 symbols to test speed")
            test_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']

            print(f"Testing with: {', '.join(test_symbols)}")
            start_time = time.time()

            for symbol in test_symbols:
                symbol_start = time.time()
                success, records = collector._collect_symbol_fast(symbol)
                symbol_time = time.time() - symbol_start

                if success:
                    print(f"✅ {symbol}: {records} records in {symbol_time:.1f}s")
                else:
                    print(f"❌ {symbol}: Failed in {symbol_time:.1f}s")

            total_time = time.time() - start_time
            print(f"\n📊 Test completed in {total_time:.1f}s")
            print(f"⚡ Average: {total_time / 5:.1f}s per symbol")

        elif choice == "7":
            print("\n👋 Goodbye!")
            collector.cleanup()
            break

        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.error(f"Fatal error:\n{traceback.format_exc()}")
