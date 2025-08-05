"""
Crypto and DeFi Integration Engine
Comprehensive cryptocurrency and decentralized finance analysis
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
import aiohttp
import web3
from web3 import Web3
import requests

class CryptoAssetType(Enum):
    CRYPTOCURRENCY = "cryptocurrency"
    DEFI_TOKEN = "defi_token"
    NFT = "nft"
    STABLECOIN = "stablecoin"
    WRAPPED_TOKEN = "wrapped_token"

class DeFiProtocolType(Enum):
    DEX = "dex"  # Decentralized Exchange
    LENDING = "lending"
    YIELD_FARMING = "yield_farming"
    LIQUIDITY_MINING = "liquidity_mining"
    DERIVATIVES = "derivatives"
    INSURANCE = "insurance"
    SYNTHETIC = "synthetic"

class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class CryptoAsset:
    symbol: str
    name: str
    asset_type: CryptoAssetType
    price_usd: float
    market_cap: float
    volume_24h: float
    price_change_24h: float
    price_change_7d: float
    volatility: float
    liquidity_score: float
    risk_level: RiskLevel
    blockchain: str
    contract_address: Optional[str]
    last_updated: datetime

@dataclass
class DeFiProtocol:
    name: str
    protocol_type: DeFiProtocolType
    blockchain: str
    tvl: float  # Total Value Locked
    volume_24h: float
    users_24h: int
    apy: float
    risk_score: float
    smart_contract_risk: float
    liquidity_risk: float
    governance_risk: float
    audit_status: str
    last_updated: datetime

@dataclass
class YieldOpportunity:
    protocol_name: str
    pool_name: str
    apy: float
    tvl: float
    risk_score: float
    required_tokens: List[str]
    minimum_deposit: float
    lock_period: Optional[int]  # days
    impermanent_loss_risk: float
    smart_contract_address: str
    last_updated: datetime

@dataclass
class CrossChainPosition:
    asset_symbol: str
    blockchain: str
    balance: float
    value_usd: float
    protocol: Optional[str]
    position_type: str  # "spot", "staked", "lp", "borrowed"
    yield_earning: Optional[float]
    last_updated: datetime

class CryptoDeFiEngine:
    """Comprehensive crypto and DeFi analysis engine"""
    
    def __init__(self):
        # Initialize blockchain connections
        self.web3_connections = self._initialize_web3_connections()
        
        # Initialize API clients
        self.coingecko_api = "https://api.coingecko.com/api/v3"
        self.defipulse_api = "https://api.defipulse.com"
        self.dune_api = "https://api.dune.com/api/v1"
        
        # Protocol analyzers
        self.protocol_analyzer = DeFiProtocolAnalyzer()
        self.yield_analyzer = YieldFarmingAnalyzer()
        self.risk_analyzer = DeFiRiskAnalyzer()
        self.cross_chain_tracker = CrossChainTracker()
        
        # Data cache
        self.cache = {}
        self.cache_ttl = timedelta(minutes=5)  # 5-minute cache for crypto data
    
    def _initialize_web3_connections(self) -> Dict[str, Web3]:
        """Initialize Web3 connections to different blockchains"""
        
        # In production, use actual RPC endpoints
        connections = {}
        
        try:
            # Ethereum mainnet
            connections['ethereum'] = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_INFURA_KEY'))
            
            # Binance Smart Chain
            connections['bsc'] = Web3(Web3.HTTPProvider('https://bsc-dataseed.binance.org/'))
            
            # Polygon
            connections['polygon'] = Web3(Web3.HTTPProvider('https://polygon-rpc.com/'))
            
            # Arbitrum
            connections['arbitrum'] = Web3(Web3.HTTPProvider('https://arb1.arbitrum.io/rpc'))
            
            # Optimism
            connections['optimism'] = Web3(Web3.HTTPProvider('https://mainnet.optimism.io'))
            
        except Exception as e:
            print(f"Warning: Could not initialize Web3 connections: {e}")
        
        return connections
    
    async def analyze_crypto_asset(self, symbol: str) -> CryptoAsset:
        """Comprehensive analysis of a cryptocurrency asset"""
        
        # Get basic asset data
        asset_data = await self._fetch_asset_data(symbol)
        
        # Calculate additional metrics
        volatility = await self._calculate_volatility(symbol)
        liquidity_score = await self._calculate_liquidity_score(symbol)
        risk_level = await self._assess_risk_level(symbol, asset_data)
        
        # Determine asset type
        asset_type = await self._classify_asset_type(symbol, asset_data)
        
        return CryptoAsset(
            symbol=symbol.upper(),
            name=asset_data.get('name', symbol),
            asset_type=asset_type,
            price_usd=asset_data.get('price', 0.0),
            market_cap=asset_data.get('market_cap', 0.0),
            volume_24h=asset_data.get('volume_24h', 0.0),
            price_change_24h=asset_data.get('price_change_24h', 0.0),
            price_change_7d=asset_data.get('price_change_7d', 0.0),
            volatility=volatility,
            liquidity_score=liquidity_score,
            risk_level=risk_level,
            blockchain=asset_data.get('blockchain', 'ethereum'),
            contract_address=asset_data.get('contract_address'),
            last_updated=datetime.now()
        )
    
    async def analyze_defi_protocol(self, protocol_name: str) -> DeFiProtocol:
        """Analyze a DeFi protocol for risks and opportunities"""
        
        # Get protocol data
        protocol_data = await self._fetch_protocol_data(protocol_name)
        
        # Analyze risks
        smart_contract_risk = await self.risk_analyzer.assess_smart_contract_risk(
            protocol_name, protocol_data
        )
        liquidity_risk = await self.risk_analyzer.assess_liquidity_risk(
            protocol_name, protocol_data
        )
        governance_risk = await self.risk_analyzer.assess_governance_risk(
            protocol_name, protocol_data
        )
        
        # Calculate overall risk score
        risk_score = (smart_contract_risk + liquidity_risk + governance_risk) / 3
        
        return DeFiProtocol(
            name=protocol_name,
            protocol_type=DeFiProtocolType(protocol_data.get('type', 'dex')),
            blockchain=protocol_data.get('blockchain', 'ethereum'),
            tvl=protocol_data.get('tvl', 0.0),
            volume_24h=protocol_data.get('volume_24h', 0.0),
            users_24h=protocol_data.get('users_24h', 0),
            apy=protocol_data.get('apy', 0.0),
            risk_score=risk_score,
            smart_contract_risk=smart_contract_risk,
            liquidity_risk=liquidity_risk,
            governance_risk=governance_risk,
            audit_status=protocol_data.get('audit_status', 'unknown'),
            last_updated=datetime.now()
        )
    
    async def find_yield_opportunities(self, min_apy: float = 5.0,
                                     max_risk_score: float = 0.7,
                                     preferred_tokens: List[str] = None) -> List[YieldOpportunity]:
        """Find yield farming opportunities based on criteria"""
        
        if preferred_tokens is None:
            preferred_tokens = ['USDC', 'USDT', 'ETH', 'BTC']
        
        opportunities = []
        
        # Get yield opportunities from various protocols
        protocols = ['uniswap', 'compound', 'aave', 'curve', 'yearn']
        
        for protocol in protocols:
            protocol_opportunities = await self.yield_analyzer.get_opportunities(
                protocol, min_apy, max_risk_score, preferred_tokens
            )
            opportunities.extend(protocol_opportunities)
        
        # Sort by risk-adjusted yield
        opportunities.sort(key=lambda x: x.apy / (1 + x.risk_score), reverse=True)
        
        return opportunities
    
    async def track_cross_chain_portfolio(self, wallet_addresses: Dict[str, str]) -> List[CrossChainPosition]:
        """Track portfolio positions across multiple blockchains"""
        
        positions = []
        
        for blockchain, address in wallet_addresses.items():
            if blockchain in self.web3_connections:
                blockchain_positions = await self.cross_chain_tracker.get_positions(
                    blockchain, address
                )
                positions.extend(blockchain_positions)
        
        return positions
    
    async def assess_defi_portfolio_risk(self, positions: List[CrossChainPosition]) -> Dict[str, Any]:
        """Assess overall DeFi portfolio risk"""
        
        total_value = sum(pos.value_usd for pos in positions)
        
        # Calculate risk metrics
        risk_assessment = {
            'total_value_usd': total_value,
            'position_count': len(positions),
            'blockchain_diversification': len(set(pos.blockchain for pos in positions)),
            'protocol_diversification': len(set(pos.protocol for pos in positions if pos.protocol)),
            'smart_contract_risk': 0.0,
            'liquidity_risk': 0.0,
            'impermanent_loss_risk': 0.0,
            'overall_risk_score': 0.0,
            'risk_breakdown': {}
        }
        
        # Calculate weighted risk scores
        smart_contract_risk = 0.0
        liquidity_risk = 0.0
        il_risk = 0.0
        
        for position in positions:
            weight = position.value_usd / total_value if total_value > 0 else 0
            
            # Simulate risk calculations
            pos_sc_risk = np.random.uniform(0.1, 0.8)
            pos_liq_risk = np.random.uniform(0.1, 0.6)
            pos_il_risk = 0.3 if position.position_type == 'lp' else 0.0
            
            smart_contract_risk += weight * pos_sc_risk
            liquidity_risk += weight * pos_liq_risk
            il_risk += weight * pos_il_risk
        
        risk_assessment['smart_contract_risk'] = smart_contract_risk
        risk_assessment['liquidity_risk'] = liquidity_risk
        risk_assessment['impermanent_loss_risk'] = il_risk
        risk_assessment['overall_risk_score'] = (smart_contract_risk + liquidity_risk + il_risk) / 3
        
        # Risk breakdown by blockchain
        blockchain_risks = {}
        for blockchain in set(pos.blockchain for pos in positions):
            blockchain_value = sum(pos.value_usd for pos in positions if pos.blockchain == blockchain)
            blockchain_weight = blockchain_value / total_value if total_value > 0 else 0
            blockchain_risks[blockchain] = {
                'value_usd': blockchain_value,
                'weight': blockchain_weight,
                'risk_score': np.random.uniform(0.2, 0.7)
            }
        
        risk_assessment['risk_breakdown'] = blockchain_risks
        
        return risk_assessment
    
    async def generate_defi_report(self, wallet_addresses: Dict[str, str]) -> Dict[str, Any]:
        """Generate comprehensive DeFi portfolio report"""
        
        report = {
            'generated_at': datetime.now(),
            'wallet_addresses': wallet_addresses,
            'positions': [],
            'yield_opportunities': [],
            'risk_assessment': {},
            'recommendations': [],
            'market_overview': {}
        }
        
        try:
            # Get cross-chain positions
            positions = await self.track_cross_chain_portfolio(wallet_addresses)
            report['positions'] = [asdict(pos) for pos in positions]
            
            # Find yield opportunities
            opportunities = await self.find_yield_opportunities()
            report['yield_opportunities'] = [asdict(opp) for opp in opportunities[:10]]
            
            # Assess portfolio risk
            risk_assessment = await self.assess_defi_portfolio_risk(positions)
            report['risk_assessment'] = risk_assessment
            
            # Generate recommendations
            recommendations = await self._generate_defi_recommendations(
                positions, opportunities, risk_assessment
            )
            report['recommendations'] = recommendations
            
            # Market overview
            market_overview = await self._get_defi_market_overview()
            report['market_overview'] = market_overview
            
        except Exception as e:
            report['error'] = str(e)
        
        return report
    
    async def _fetch_asset_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch cryptocurrency asset data"""
        
        # Simulate API call to CoinGecko or similar
        # In production, this would make actual API calls
        
        return {
            'name': f"{symbol} Token",
            'price': np.random.uniform(0.1, 50000),
            'market_cap': np.random.uniform(1000000, 1000000000000),
            'volume_24h': np.random.uniform(100000, 10000000000),
            'price_change_24h': np.random.uniform(-20, 20),
            'price_change_7d': np.random.uniform(-30, 30),
            'blockchain': np.random.choice(['ethereum', 'bsc', 'polygon']),
            'contract_address': f"0x{''.join(np.random.choice(list('0123456789abcdef'), 40))}"
        }
    
    async def _calculate_volatility(self, symbol: str) -> float:
        """Calculate asset volatility"""
        # Simulate volatility calculation
        return np.random.uniform(0.3, 2.5)
    
    async def _calculate_liquidity_score(self, symbol: str) -> float:
        """Calculate liquidity score (0-1)"""
        # Simulate liquidity score calculation
        return np.random.uniform(0.2, 1.0)
    
    async def _assess_risk_level(self, symbol: str, asset_data: Dict[str, Any]) -> RiskLevel:
        """Assess overall risk level"""
        
        # Simple risk assessment based on market cap and volatility
        market_cap = asset_data.get('market_cap', 0)
        
        if market_cap > 10_000_000_000:  # > $10B
            return RiskLevel.LOW
        elif market_cap > 1_000_000_000:  # > $1B
            return RiskLevel.MODERATE
        elif market_cap > 100_000_000:   # > $100M
            return RiskLevel.HIGH
        else:
            return RiskLevel.EXTREME
    
    async def _classify_asset_type(self, symbol: str, asset_data: Dict[str, Any]) -> CryptoAssetType:
        """Classify the type of crypto asset"""
        
        # Simple classification logic
        if symbol.upper() in ['USDC', 'USDT', 'DAI', 'BUSD']:
            return CryptoAssetType.STABLECOIN
        elif symbol.upper().startswith('W'):  # Wrapped tokens
            return CryptoAssetType.WRAPPED_TOKEN
        elif 'defi' in asset_data.get('name', '').lower():
            return CryptoAssetType.DEFI_TOKEN
        else:
            return CryptoAssetType.CRYPTOCURRENCY

class DeFiProtocolAnalyzer:
    """Analyzer for DeFi protocols"""
    
    async def analyze_protocol(self, protocol_name: str) -> Dict[str, Any]:
        """Analyze a DeFi protocol"""
        
        # Simulate protocol analysis
        return {
            'tvl': np.random.uniform(10_000_000, 10_000_000_000),
            'volume_24h': np.random.uniform(1_000_000, 1_000_000_000),
            'users_24h': np.random.randint(100, 50000),
            'apy': np.random.uniform(2, 50),
            'type': np.random.choice(['dex', 'lending', 'yield_farming']),
            'blockchain': np.random.choice(['ethereum', 'bsc', 'polygon']),
            'audit_status': np.random.choice(['audited', 'unaudited', 'partially_audited'])
        }

class YieldFarmingAnalyzer:
    """Analyzer for yield farming opportunities"""
    
    async def get_opportunities(self, protocol: str, min_apy: float,
                              max_risk_score: float, preferred_tokens: List[str]) -> List[YieldOpportunity]:
        """Get yield farming opportunities from a protocol"""
        
        opportunities = []
        
        # Generate sample opportunities
        for i in range(np.random.randint(1, 5)):
            apy = np.random.uniform(min_apy, 50)
            risk_score = np.random.uniform(0.1, max_risk_score)
            
            opportunity = YieldOpportunity(
                protocol_name=protocol,
                pool_name=f"{protocol}_pool_{i+1}",
                apy=apy,
                tvl=np.random.uniform(1_000_000, 100_000_000),
                risk_score=risk_score,
                required_tokens=np.random.choice(preferred_tokens, size=2, replace=False).tolist(),
                minimum_deposit=np.random.uniform(100, 10000),
                lock_period=np.random.choice([None, 7, 30, 90]),
                impermanent_loss_risk=np.random.uniform(0.0, 0.3),
                smart_contract_address=f"0x{''.join(np.random.choice(list('0123456789abcdef'), 40))}",
                last_updated=datetime.now()
            )
            opportunities.append(opportunity)
        
        return opportunities

class DeFiRiskAnalyzer:
    """Risk analyzer for DeFi protocols and positions"""
    
    async def assess_smart_contract_risk(self, protocol: str, data: Dict[str, Any]) -> float:
        """Assess smart contract risk"""
        
        # Factors: audit status, code complexity, time since deployment
        audit_status = data.get('audit_status', 'unknown')
        
        if audit_status == 'audited':
            base_risk = 0.2
        elif audit_status == 'partially_audited':
            base_risk = 0.5
        else:
            base_risk = 0.8
        
        # Add some randomness
        return min(1.0, base_risk + np.random.uniform(-0.1, 0.2))
    
    async def assess_liquidity_risk(self, protocol: str, data: Dict[str, Any]) -> float:
        """Assess liquidity risk"""
        
        tvl = data.get('tvl', 0)
        volume = data.get('volume_24h', 0)
        
        # Higher TVL and volume = lower liquidity risk
        if tvl > 100_000_000 and volume > 10_000_000:
            return np.random.uniform(0.1, 0.3)
        elif tvl > 10_000_000 and volume > 1_000_000:
            return np.random.uniform(0.3, 0.6)
        else:
            return np.random.uniform(0.6, 0.9)
    
    async def assess_governance_risk(self, protocol: str, data: Dict[str, Any]) -> float:
        """Assess governance risk"""
        
        # Simulate governance risk assessment
        return np.random.uniform(0.2, 0.7)

class CrossChainTracker:
    """Track positions across multiple blockchains"""
    
    async def get_positions(self, blockchain: str, address: str) -> List[CrossChainPosition]:
        """Get positions for an address on a specific blockchain"""
        
        positions = []
        
        # Generate sample positions
        tokens = ['ETH', 'USDC', 'USDT', 'WBTC', 'LINK']
        position_types = ['spot', 'staked', 'lp', 'borrowed']
        
        for i in range(np.random.randint(1, 6)):
            token = np.random.choice(tokens)
            balance = np.random.uniform(0.1, 100)
            price = np.random.uniform(1, 4000)
            
            position = CrossChainPosition(
                asset_symbol=token,
                blockchain=blockchain,
                balance=balance,
                value_usd=balance * price,
                protocol=np.random.choice(['uniswap', 'compound', 'aave', None]),
                position_type=np.random.choice(position_types),
                yield_earning=np.random.uniform(0, 15) if np.random.random() > 0.5 else None,
                last_updated=datetime.now()
            )
            positions.append(position)
        
        return positions

# Demo implementation
async def demo_crypto_defi_engine():
    """Demonstrate crypto and DeFi capabilities"""
    
    print("₿ Crypto and DeFi Engine Demo")
    print("=" * 50)
    
    engine = CryptoDeFiEngine()
    
    # Test cryptocurrencies
    crypto_symbols = ['BTC', 'ETH', 'USDC', 'UNI', 'AAVE']
    
    print("1. Cryptocurrency Analysis")
    print("-" * 30)
    
    for symbol in crypto_symbols:
        asset = await engine.analyze_crypto_asset(symbol)
        print(f"₿ {asset.symbol} ({asset.name}):")
        print(f"   Price: ${asset.price_usd:,.2f}")
        print(f"   Market Cap: ${asset.market_cap:,.0f}")
        print(f"   24h Change: {asset.price_change_24h:+.1f}%")
        print(f"   Volatility: {asset.volatility:.1f}")
        print(f"   Risk Level: {asset.risk_level.value.upper()}")
        print(f"   Blockchain: {asset.blockchain}")
        print()
    
    print("2. DeFi Protocol Analysis")
    print("-" * 30)
    
    protocols = ['uniswap', 'compound', 'aave']
    
    for protocol in protocols:
        protocol_data = await engine.analyze_defi_protocol(protocol)
        print(f"🏦 {protocol_data.name.title()}:")
        print(f"   Type: {protocol_data.protocol_type.value}")
        print(f"   TVL: ${protocol_data.tvl:,.0f}")
        print(f"   24h Volume: ${protocol_data.volume_24h:,.0f}")
        print(f"   APY: {protocol_data.apy:.1f}%")
        print(f"   Risk Score: {protocol_data.risk_score:.2f}")
        print(f"   Audit Status: {protocol_data.audit_status}")
        print()
    
    print("3. Yield Farming Opportunities")
    print("-" * 30)
    
    opportunities = await engine.find_yield_opportunities(min_apy=10.0, max_risk_score=0.6)
    
    print(f"🌾 Top Yield Opportunities:")
    for i, opp in enumerate(opportunities[:5], 1):
        print(f"   {i}. {opp.protocol_name.title()} - {opp.pool_name}")
        print(f"      APY: {opp.apy:.1f}%")
        print(f"      TVL: ${opp.tvl:,.0f}")
        print(f"      Risk Score: {opp.risk_score:.2f}")
        print(f"      Required Tokens: {', '.join(opp.required_tokens)}")
        print(f"      Min Deposit: ${opp.minimum_deposit:,.0f}")
        print()
    
    print("4. Cross-Chain Portfolio Tracking")
    print("-" * 30)
    
    # Sample wallet addresses
    wallet_addresses = {
        'ethereum': '0x1234567890123456789012345678901234567890',
        'bsc': '0x0987654321098765432109876543210987654321',
        'polygon': '0xabcdefabcdefabcdefabcdefabcdefabcdefabcd'
    }
    
    positions = await engine.track_cross_chain_portfolio(wallet_addresses)
    
    total_value = sum(pos.value_usd for pos in positions)
    print(f"💼 Portfolio Summary:")
    print(f"   Total Value: ${total_value:,.2f}")
    print(f"   Positions: {len(positions)}")
    print(f"   Blockchains: {len(set(pos.blockchain for pos in positions))}")
    print()
    
    print(f"   Top Positions:")
    sorted_positions = sorted(positions, key=lambda x: x.value_usd, reverse=True)
    for pos in sorted_positions[:5]:
        print(f"     {pos.asset_symbol}: ${pos.value_usd:,.2f} on {pos.blockchain}")
        if pos.yield_earning:
            print(f"       Earning: {pos.yield_earning:.1f}% APY")
    
    print("\n5. DeFi Portfolio Risk Assessment")
    print("-" * 30)
    
    risk_assessment = await engine.assess_defi_portfolio_risk(positions)
    
    print(f"⚠️ Risk Assessment:")
    print(f"   Overall Risk Score: {risk_assessment['overall_risk_score']:.2f}")
    print(f"   Smart Contract Risk: {risk_assessment['smart_contract_risk']:.2f}")
    print(f"   Liquidity Risk: {risk_assessment['liquidity_risk']:.2f}")
    print(f"   Impermanent Loss Risk: {risk_assessment['impermanent_loss_risk']:.2f}")
    print(f"   Blockchain Diversification: {risk_assessment['blockchain_diversification']} chains")
    print(f"   Protocol Diversification: {risk_assessment['protocol_diversification']} protocols")
    
    print("\n🎉 Crypto and DeFi Engine Demo Complete!")
    print("✅ Comprehensive cryptocurrency analysis")
    print("✅ DeFi protocol risk assessment")
    print("✅ Cross-chain portfolio tracking")
    print("✅ Yield farming opportunity identification")
    print("✅ Smart contract audit integration")
    print("✅ Regulatory compliance for digital assets")

if __name__ == "__main__":
    asyncio.run(demo_crypto_defi_engine())