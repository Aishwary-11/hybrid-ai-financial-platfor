"""
Real-Time Market Data Engine
Sub-millisecond market data streaming and processing
"""

import asyncio
import websockets
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import numpy as np
from collections import defaultdict, deque
import logging

class ExchangeType(Enum):
    NYSE = "nyse"
    NASDAQ = "nasdaq"
    CME = "cme"
    CBOE = "cboe"
    ICE = "ice"

class DataType(Enum):
    TRADE = "trade"
    QUOTE = "quote"
    OPTIONS = "options"
    FUTURES = "futures"
    LEVEL2 = "level2"

@dataclass
class MarketTick:
    symbol: str
    exchange: ExchangeType
    data_type: DataType
    price: float
    volume: int
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    metadata: Dict[str, Any] = None

@dataclass
class OptionsChain:
    underlying_symbol: str
    expiration_date: datetime
    strike_price: float
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    timestamp: datetime

class RealTimeMarketEngine:
    """High-performance real-time market data engine"""
    
    def __init__(self):
        self.connections = {}
        self.subscribers = defaultdict(list)
        self.data_buffer = defaultdict(lambda: deque(maxlen=10000))
        self.latency_tracker = deque(maxlen=1000)
        self.correlation_engine = CrossAssetCorrelationEngine()
        self.options_engine = OptionsChainEngine()
        self.level2_engine = Level2DataEngine()
        
        # Performance targets
        self.target_latency_ms = 1.0  # 1ms target
        self.max_latency_ms = 5.0     # 5ms max acceptable
        
        self.logger = logging.getLogger(__name__)
        
    async def initialize_connections(self):
        """Initialize WebSocket connections to all exchanges"""
        
        exchange_configs = {
            ExchangeType.NYSE: {
                'url': 'wss://api.nyse.com/v1/market-data',
                'auth': self._get_nyse_auth(),
                'symbols': ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            },
            ExchangeType.NASDAQ: {
                'url': 'wss://api.nasdaq.com/v1/real-time',
                'auth': self._get_nasdaq_auth(),
                'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
            },
            ExchangeType.CME: {
                'url': 'wss://api.cmegroup.com/v1/market-data',
                'auth': self._get_cme_auth(),
                'symbols': ['ES', 'NQ', 'YM', 'RTY', 'GC', 'CL', 'ZN', 'ZB']
            }
        }
        
        for exchange, config in exchange_configs.items():
            try:
                connection = await self._establish_websocket_connection(
                    exchange, config['url'], config['auth'], config['symbols']
                )
                self.connections[exchange] = connection
                self.logger.info(f"Connected to {exchange.value} exchange")
            except Exception as e:
                self.logger.error(f"Failed to connect to {exchange.value}: {e}")
    
    async def _establish_websocket_connection(self, exchange: ExchangeType, 
                                           url: str, auth: Dict, symbols: List[str]):
        """Establish WebSocket connection with sub-millisecond optimization"""
        
        async def connection_handler():
            try:
                async with websockets.connect(
                    url,
                    extra_headers=auth,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as websocket:
                    
                    # Subscribe to symbols
                    subscribe_message = {
                        'action': 'subscribe',
                        'symbols': symbols,
                        'types': ['trade', 'quote', 'level2']
                    }
                    await websocket.send(json.dumps(subscribe_message))
                    
                    # Process incoming messages
                    async for message in websocket:
                        receive_time = time.time_ns()
                        await self._process_market_message(
                            exchange, message, receive_time
                        )
                        
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning(f"Connection to {exchange.value} closed, reconnecting...")
                await asyncio.sleep(1)
                await self._establish_websocket_connection(exchange, url, auth, symbols)
            except Exception as e:
                self.logger.error(f"Error in {exchange.value} connection: {e}")
                await asyncio.sleep(5)
                await self._establish_websocket_connection(exchange, url, auth, symbols)
        
        # Start connection handler as background task
        asyncio.create_task(connection_handler())
        return True
    
    async def _process_market_message(self, exchange: ExchangeType, 
                                    message: str, receive_time: int):
        """Process incoming market data with sub-millisecond latency"""
        
        try:
            data = json.loads(message)
            process_start = time.time_ns()
            
            # Extract timestamp from message
            msg_timestamp = datetime.fromtimestamp(data.get('timestamp', time.time()))
            
            # Calculate latency
            latency_ns = receive_time - (data.get('exchange_timestamp', receive_time))
            latency_ms = latency_ns / 1_000_000
            self.latency_tracker.append(latency_ms)
            
            # Create market tick
            tick = MarketTick(
                symbol=data['symbol'],
                exchange=exchange,
                data_type=DataType(data['type']),
                price=float(data['price']),
                volume=int(data.get('volume', 0)),
                timestamp=msg_timestamp,
                bid=data.get('bid'),
                ask=data.get('ask'),
                bid_size=data.get('bid_size'),
                ask_size=data.get('ask_size'),
                metadata=data.get('metadata', {})
            )
            
            # Store in buffer
            self.data_buffer[tick.symbol].append(tick)
            
            # Notify subscribers
            await self._notify_subscribers(tick)
            
            # Update correlations in real-time
            await self.correlation_engine.update_correlation(tick)
            
            # Process options data if applicable
            if tick.data_type == DataType.OPTIONS:
                await self.options_engine.process_options_tick(tick)
            
            # Process Level 2 data
            if tick.data_type == DataType.LEVEL2:
                await self.level2_engine.process_level2_data(tick)
            
            # Performance monitoring
            process_end = time.time_ns()
            processing_time_ms = (process_end - process_start) / 1_000_000
            
            if processing_time_ms > self.max_latency_ms:
                self.logger.warning(
                    f"High processing latency: {processing_time_ms:.2f}ms for {tick.symbol}"
                )
                
        except Exception as e:
            self.logger.error(f"Error processing market message: {e}")
    
    async def _notify_subscribers(self, tick: MarketTick):
        """Notify all subscribers of new market data"""
        
        # Symbol-specific subscribers
        symbol_subscribers = self.subscribers.get(tick.symbol, [])
        
        # Exchange-specific subscribers
        exchange_subscribers = self.subscribers.get(f"exchange:{tick.exchange.value}", [])
        
        # Data type subscribers
        type_subscribers = self.subscribers.get(f"type:{tick.data_type.value}", [])
        
        all_subscribers = symbol_subscribers + exchange_subscribers + type_subscribers
        
        # Notify all subscribers concurrently
        if all_subscribers:
            await asyncio.gather(
                *[subscriber(tick) for subscriber in all_subscribers],
                return_exceptions=True
            )
    
    def subscribe(self, subscription_key: str, callback: Callable):
        """Subscribe to real-time market data"""
        self.subscribers[subscription_key].append(callback)
        
    def unsubscribe(self, subscription_key: str, callback: Callable):
        """Unsubscribe from market data"""
        if callback in self.subscribers[subscription_key]:
            self.subscribers[subscription_key].remove(callback)
    
    async def get_real_time_quote(self, symbol: str) -> Optional[MarketTick]:
        """Get the latest real-time quote for a symbol"""
        if symbol in self.data_buffer and self.data_buffer[symbol]:
            return self.data_buffer[symbol][-1]
        return None
    
    async def get_historical_ticks(self, symbol: str, 
                                 minutes: int = 5) -> List[MarketTick]:
        """Get historical ticks for the last N minutes"""
        if symbol not in self.data_buffer:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            tick for tick in self.data_buffer[symbol]
            if tick.timestamp >= cutoff_time
        ]
    
    def get_latency_stats(self) -> Dict[str, float]:
        """Get latency statistics"""
        if not self.latency_tracker:
            return {}
        
        latencies = list(self.latency_tracker)
        return {
            'avg_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'max_latency_ms': np.max(latencies),
            'min_latency_ms': np.min(latencies)
        }
    
    def _get_nyse_auth(self) -> Dict[str, str]:
        """Get NYSE authentication headers"""
        return {
            'Authorization': 'Bearer YOUR_NYSE_API_KEY',
            'X-API-Version': 'v1'
        }
    
    def _get_nasdaq_auth(self) -> Dict[str, str]:
        """Get NASDAQ authentication headers"""
        return {
            'Authorization': 'Bearer YOUR_NASDAQ_API_KEY',
            'X-API-Version': 'v1'
        }
    
    def _get_cme_auth(self) -> Dict[str, str]:
        """Get CME authentication headers"""
        return {
            'Authorization': 'Bearer YOUR_CME_API_KEY',
            'X-API-Version': 'v1'
        }

class CrossAssetCorrelationEngine:
    """Real-time cross-asset correlation analysis"""
    
    def __init__(self):
        self.price_history = defaultdict(lambda: deque(maxlen=1000))
        self.correlation_matrix = {}
        self.update_frequency = 100  # Update correlations every 100 ticks
        self.tick_count = 0
        
    async def update_correlation(self, tick: MarketTick):
        """Update correlation matrix with new tick data"""
        
        # Store price data
        self.price_history[tick.symbol].append(tick.price)
        self.tick_count += 1
        
        # Update correlations periodically
        if self.tick_count % self.update_frequency == 0:
            await self._calculate_correlations()
    
    async def _calculate_correlations(self):
        """Calculate real-time correlations between assets"""
        
        symbols = list(self.price_history.keys())
        if len(symbols) < 2:
            return
        
        # Convert to numpy arrays for efficient calculation
        price_data = {}
        min_length = min(len(self.price_history[symbol]) for symbol in symbols)
        
        for symbol in symbols:
            prices = list(self.price_history[symbol])[-min_length:]
            if len(prices) > 10:  # Minimum data points for correlation
                price_data[symbol] = np.array(prices)
        
        # Calculate correlation matrix
        correlation_matrix = {}
        for symbol1 in price_data:
            correlation_matrix[symbol1] = {}
            for symbol2 in price_data:
                if symbol1 == symbol2:
                    correlation_matrix[symbol1][symbol2] = 1.0
                else:
                    corr = np.corrcoef(price_data[symbol1], price_data[symbol2])[0, 1]
                    correlation_matrix[symbol1][symbol2] = float(corr) if not np.isnan(corr) else 0.0
        
        self.correlation_matrix = correlation_matrix
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        if (symbol1 in self.correlation_matrix and 
            symbol2 in self.correlation_matrix[symbol1]):
            return self.correlation_matrix[symbol1][symbol2]
        return 0.0
    
    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get the full correlation matrix"""
        return self.correlation_matrix.copy()

class OptionsChainEngine:
    """Real-time options chain processing"""
    
    def __init__(self):
        self.options_data = defaultdict(dict)
        self.implied_vol_calculator = ImpliedVolatilityCalculator()
        self.greeks_calculator = GreeksCalculator()
    
    async def process_options_tick(self, tick: MarketTick):
        """Process options market data"""
        
        if not tick.metadata or 'option_details' not in tick.metadata:
            return
        
        option_details = tick.metadata['option_details']
        
        # Create options chain entry
        option_chain = OptionsChain(
            underlying_symbol=option_details['underlying'],
            expiration_date=datetime.fromisoformat(option_details['expiration']),
            strike_price=float(option_details['strike']),
            option_type=option_details['type'],
            bid=tick.bid or 0.0,
            ask=tick.ask or 0.0,
            last_price=tick.price,
            volume=tick.volume,
            open_interest=option_details.get('open_interest', 0),
            implied_volatility=0.0,  # Will be calculated
            delta=0.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            timestamp=tick.timestamp
        )
        
        # Calculate implied volatility and Greeks
        option_chain.implied_volatility = await self.implied_vol_calculator.calculate(
            option_chain
        )
        
        greeks = await self.greeks_calculator.calculate(option_chain)
        option_chain.delta = greeks['delta']
        option_chain.gamma = greeks['gamma']
        option_chain.theta = greeks['theta']
        option_chain.vega = greeks['vega']
        
        # Store options data
        key = f"{option_chain.underlying_symbol}_{option_chain.expiration_date.date()}_{option_chain.strike_price}_{option_chain.option_type}"
        self.options_data[option_chain.underlying_symbol][key] = option_chain
    
    def get_options_chain(self, underlying_symbol: str) -> List[OptionsChain]:
        """Get complete options chain for an underlying symbol"""
        if underlying_symbol in self.options_data:
            return list(self.options_data[underlying_symbol].values())
        return []

class Level2DataEngine:
    """Level 2 market data processing (order book)"""
    
    def __init__(self):
        self.order_books = defaultdict(lambda: {'bids': {}, 'asks': {}})
        self.book_depth = 10  # Track top 10 levels
    
    async def process_level2_data(self, tick: MarketTick):
        """Process Level 2 order book data"""
        
        if not tick.metadata or 'level2' not in tick.metadata:
            return
        
        level2_data = tick.metadata['level2']
        symbol = tick.symbol
        
        # Update order book
        if level2_data['side'] == 'bid':
            if level2_data['size'] == 0:
                # Remove level
                self.order_books[symbol]['bids'].pop(level2_data['price'], None)
            else:
                # Add/update level
                self.order_books[symbol]['bids'][level2_data['price']] = level2_data['size']
        
        elif level2_data['side'] == 'ask':
            if level2_data['size'] == 0:
                # Remove level
                self.order_books[symbol]['asks'].pop(level2_data['price'], None)
            else:
                # Add/update level
                self.order_books[symbol]['asks'][level2_data['price']] = level2_data['size']
        
        # Trim to book depth
        self._trim_order_book(symbol)
    
    def _trim_order_book(self, symbol: str):
        """Trim order book to specified depth"""
        
        # Sort and trim bids (highest first)
        bids = dict(sorted(
            self.order_books[symbol]['bids'].items(),
            key=lambda x: x[0],
            reverse=True
        )[:self.book_depth])
        
        # Sort and trim asks (lowest first)
        asks = dict(sorted(
            self.order_books[symbol]['asks'].items(),
            key=lambda x: x[0]
        )[:self.book_depth])
        
        self.order_books[symbol]['bids'] = bids
        self.order_books[symbol]['asks'] = asks
    
    def get_order_book(self, symbol: str) -> Dict[str, Dict[float, int]]:
        """Get current order book for a symbol"""
        return self.order_books[symbol].copy()

class ImpliedVolatilityCalculator:
    """Calculate implied volatility for options"""
    
    async def calculate(self, option: OptionsChain) -> float:
        """Calculate implied volatility using Black-Scholes"""
        # Simplified IV calculation - in production, use more sophisticated methods
        # This would typically use numerical methods like Newton-Raphson
        return 0.25  # Placeholder - implement actual IV calculation

class GreeksCalculator:
    """Calculate option Greeks"""
    
    async def calculate(self, option: OptionsChain) -> Dict[str, float]:
        """Calculate option Greeks"""
        # Simplified Greeks calculation - implement actual Black-Scholes Greeks
        return {
            'delta': 0.5,
            'gamma': 0.1,
            'theta': -0.05,
            'vega': 0.2
        }

# Demo implementation
async def demo_realtime_market_engine():
    """Demonstrate real-time market data engine"""
    
    print("🚀 Real-Time Market Data Engine Demo")
    print("=" * 50)
    
    # Initialize engine
    engine = RealTimeMarketEngine()
    
    # Sample callback for market data
    async def market_data_callback(tick: MarketTick):
        print(f"📊 {tick.symbol}: ${tick.price:.2f} "
              f"Vol: {tick.volume:,} "
              f"Exchange: {tick.exchange.value} "
              f"Time: {tick.timestamp.strftime('%H:%M:%S.%f')[:-3]}")
    
    # Subscribe to market data
    engine.subscribe('AAPL', market_data_callback)
    engine.subscribe('TSLA', market_data_callback)
    engine.subscribe('SPY', market_data_callback)
    
    print("✅ Subscribed to real-time market data")
    print("✅ Simulating market data feed...")
    
    # Simulate market data (in production, this comes from WebSocket feeds)
    symbols = ['AAPL', 'TSLA', 'SPY', 'QQQ', 'MSFT']
    base_prices = {'AAPL': 175.0, 'TSLA': 240.0, 'SPY': 450.0, 'QQQ': 380.0, 'MSFT': 380.0}
    
    for i in range(20):
        for symbol in symbols:
            # Simulate price movement
            price_change = np.random.normal(0, 0.5)
            new_price = base_prices[symbol] + price_change
            base_prices[symbol] = new_price
            
            # Create simulated tick
            tick = MarketTick(
                symbol=symbol,
                exchange=ExchangeType.NYSE,
                data_type=DataType.TRADE,
                price=new_price,
                volume=np.random.randint(100, 10000),
                timestamp=datetime.now(),
                bid=new_price - 0.01,
                ask=new_price + 0.01,
                bid_size=np.random.randint(100, 1000),
                ask_size=np.random.randint(100, 1000)
            )
            
            # Process tick
            await engine._notify_subscribers(tick)
            await engine.correlation_engine.update_correlation(tick)
        
        await asyncio.sleep(0.1)  # 100ms between updates
    
    # Show correlation matrix
    print("\n📈 Real-Time Correlation Matrix:")
    correlation_matrix = engine.correlation_engine.get_correlation_matrix()
    
    if correlation_matrix:
        symbols_list = list(correlation_matrix.keys())
        print(f"{'Symbol':<8}", end="")
        for symbol in symbols_list:
            print(f"{symbol:<8}", end="")
        print()
        
        for symbol1 in symbols_list:
            print(f"{symbol1:<8}", end="")
            for symbol2 in symbols_list:
                corr = correlation_matrix[symbol1].get(symbol2, 0.0)
                print(f"{corr:>7.3f} ", end="")
            print()
    
    # Show latency statistics
    print(f"\n⚡ Performance Statistics:")
    latency_stats = engine.get_latency_stats()
    if latency_stats:
        print(f"   Average Latency: {latency_stats.get('avg_latency_ms', 0):.2f}ms")
        print(f"   95th Percentile: {latency_stats.get('p95_latency_ms', 0):.2f}ms")
        print(f"   99th Percentile: {latency_stats.get('p99_latency_ms', 0):.2f}ms")
    
    print("\n🎉 Real-Time Market Data Engine Demo Complete!")

if __name__ == "__main__":
    asyncio.run(demo_realtime_market_engine())