"""
Enhanced Stage Analyzer
Implements 10 professional analysis stages with advanced metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from scipy.stats import linregress

from .base_stage_analyzer import BaseStageAnalyzer, StageResult


class EnhancedStageAnalyzer(BaseStageAnalyzer):
    """
    Enhanced stage analyzer with 10 professional analysis stages
    Implements comprehensive multi-stage analysis pipeline
    """
    
    def __init__(self, blacklist_manager=None):
        super().__init__(blacklist_manager)
        self.version = "6.1"
        self.stages = [
            "Smart Money Detection",
            "Accumulation Analysis",
            "Technical Confluence",
            "Volume Profiling",
            "Momentum Analysis",
            "Pattern Recognition",
            "Risk Assessment",
            "Entry Optimization",
            "Breakout Probability",
            "Professional Grade"
        ]
    
    def get_supported_stages(self) -> List[str]:
        """Return all supported stages"""
        return self.stages.copy()
    
    def run_all_stages(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> List[StageResult]:
        """
        Run all 10 enhanced stages
        
        Args:
            df: OHLCV data
            symbol: Symbol being analyzed
            is_crypto: Whether this is cryptocurrency data
            
        Returns:
            List of StageResult objects
        """
        if not self.validate_data(df):
            return [StageResult(stage, False, 0, {"error": "Invalid data"}, "❌", [], 0) 
                   for stage in self.stages]
        
        results = []
        
        # Stage 1: Smart Money Detection
        results.append(self.stage_1_smart_money_detection(df, symbol, is_crypto))
        
        # Stage 2: Accumulation Analysis
        results.append(self.stage_2_accumulation_analysis(df, symbol, is_crypto))
        
        # Stage 3: Technical Confluence
        results.append(self.stage_3_technical_confluence(df, symbol, is_crypto))
        
        # Stage 4: Volume Profiling
        results.append(self.stage_4_volume_profiling(df, symbol, is_crypto))
        
        # Stage 5: Momentum Analysis
        results.append(self.stage_5_momentum_analysis(df, symbol, is_crypto))
        
        # Stage 6: Advanced Pattern Recognition
        results.append(self.stage_6_pattern_recognition(df, symbol, is_crypto))
        
        # Stage 7: Risk Assessment
        results.append(self.stage_7_risk_assessment(df, symbol, is_crypto))
        
        # Stage 8: Entry Optimization
        results.append(self.stage_8_entry_optimization(df, symbol, is_crypto))
        
        # Stage 9: Breakout Probability
        results.append(self.stage_9_breakout_probability(df, symbol, is_crypto))
        
        # Stage 10: Professional Grade
        results.append(self.stage_10_professional_grade(df, symbol, is_crypto))
        
        return results

    # ================== STAGE IMPLEMENTATIONS ==================
    
    def stage_1_smart_money_detection(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Detect smart money activity"""
        try:
            # Smart money indicators
            smart_money_factors = []
            
            # Factor 1: Volume analysis on down days
            if 'Volume' in df.columns:
                volume_analysis = self._analyze_smart_money_volume(df.tail(20))
                smart_money_factors.append(volume_analysis)
            else:
                smart_money_factors.append(0.3)
            
            # Factor 2: Price stability with volume increase
            price_stability = self._calculate_price_stability_with_volume(df.tail(15))
            smart_money_factors.append(price_stability)
            
            # Factor 3: Support level accumulation
            support_accumulation = self._analyze_support_accumulation(df.tail(30))
            smart_money_factors.append(support_accumulation)
            
            # Factor 4: Hidden divergences
            hidden_divergence = self._detect_hidden_divergences(df.tail(40))
            smart_money_factors.append(hidden_divergence)
            
            # Factor 5: Institutional patterns
            institutional_pattern = self._detect_institutional_patterns(df.tail(25))
            smart_money_factors.append(institutional_pattern)
            
            # Calculate combined score
            total_score = np.mean(smart_money_factors) * 100
            passed = total_score >= (55 if is_crypto else 60)
            
            confidence = min(100, total_score * 1.2)
            indicator = "💰" if passed else "❌"
            
            if passed and total_score > 70 and symbol:
                self.stage_top_stocks["Smart Money Detection"].append((symbol, total_score))
            
            details = {
                'volume_analysis': smart_money_factors[0],
                'price_stability': smart_money_factors[1],
                'support_accumulation': smart_money_factors[2],
                'hidden_divergence': smart_money_factors[3],
                'institutional_pattern': smart_money_factors[4],
                'total_score': total_score,
                'smart_money_probability': confidence
            }
            
            return StageResult(
                "Smart Money Detection",
                passed,
                total_score,
                details,
                indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Smart Money Detection", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_2_accumulation_analysis(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Advanced accumulation zone analysis"""
        try:
            accumulation_factors = []
            
            # Factor 1: Price consolidation
            price_consolidation = self._analyze_price_consolidation(df.tail(30))
            accumulation_factors.append(price_consolidation)
            
            # Factor 2: Volume accumulation patterns
            if 'Volume' in df.columns:
                volume_accumulation = self._analyze_volume_accumulation_patterns(df.tail(25))
                accumulation_factors.append(volume_accumulation)
            else:
                accumulation_factors.append(0.4)
            
            # Factor 3: Support level strength
            support_strength = self._calculate_support_level_strength(df.tail(40))
            accumulation_factors.append(support_strength)
            
            # Factor 4: Accumulation/Distribution line
            if 'Volume' in df.columns:
                ad_line_strength = self._calculate_ad_line_strength(df.tail(30))
                accumulation_factors.append(ad_line_strength)
            else:
                accumulation_factors.append(0.4)
            
            # Factor 5: Time-based accumulation
            time_accumulation = self._analyze_time_based_accumulation(df.tail(50))
            accumulation_factors.append(time_accumulation)
            
            total_score = np.mean(accumulation_factors) * 100
            passed = total_score >= (60 if is_crypto else 65)
            
            confidence = min(100, total_score * 1.1)
            indicator = "🏗️" if passed else "❌"
            
            if passed and total_score > 75 and symbol:
                self.stage_top_stocks["Accumulation Analysis"].append((symbol, total_score))
            
            details = {
                'price_consolidation': accumulation_factors[0],
                'volume_accumulation': accumulation_factors[1],
                'support_strength': accumulation_factors[2],
                'ad_line_strength': accumulation_factors[3],
                'time_accumulation': accumulation_factors[4],
                'accumulation_probability': confidence
            }
            
            return StageResult(
                "Accumulation Analysis",
                passed,
                total_score,
                details,
                indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Accumulation Analysis", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_3_technical_confluence(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Advanced technical confluence analysis"""
        try:
            confluence_signals = []
            
            # Multiple timeframe analysis
            mtf_alignment = self._calculate_mtf_alignment(df)
            confluence_signals.append(('mtf_alignment', mtf_alignment))
            
            # Support/Resistance confluence
            sr_confluence = self._calculate_sr_confluence(df)
            confluence_signals.append(('sr_confluence', sr_confluence))
            
            # Moving average confluence
            ma_confluence = self._calculate_ma_confluence(df)
            confluence_signals.append(('ma_confluence', ma_confluence))
            
            # Fibonacci confluence
            fib_confluence = self._calculate_fibonacci_confluence(df)
            confluence_signals.append(('fib_confluence', fib_confluence))
            
            # Technical indicator confluence
            indicator_confluence = self._calculate_indicator_confluence(df)
            confluence_signals.append(('indicator_confluence', indicator_confluence))
            
            # Calculate confluence strength
            valid_signals = [score for name, score in confluence_signals if score > 0.3]
            confluence_strength = np.mean(valid_signals) if valid_signals else 0
            
            # Bonus for multiple confirming signals
            strong_signals = len([score for name, score in confluence_signals if score > 0.6])
            confluence_bonus = min(0.3, strong_signals * 0.1)
            
            total_score = (confluence_strength + confluence_bonus) * 100
            passed = confluence_strength >= 0.6 and strong_signals >= 2
            
            confidence = min(100, total_score * 1.15)
            indicator = "🎯" if passed else "❌"
            
            if passed and total_score > 70 and symbol:
                self.stage_top_stocks["Technical Confluence"].append((symbol, total_score))
            
            details = {
                'mtf_alignment': mtf_alignment,
                'sr_confluence': sr_confluence,
                'ma_confluence': ma_confluence,
                'fib_confluence': fib_confluence,
                'indicator_confluence': indicator_confluence,
                'confluence_strength': confluence_strength,
                'strong_signals_count': strong_signals,
                'confluence_probability': confidence
            }
            
            return StageResult(
                "Technical Confluence",
                passed,
                total_score,
                details,
                indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Technical Confluence", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_4_volume_profiling(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Advanced volume profile analysis"""
        try:
            if 'Volume' not in df.columns:
                return StageResult("Volume Profiling", False, 30, {"error": "No volume data"}, "❌", [], 0)
            
            volume_factors = []
            
            # Volume profile distribution
            volume_profile = self._calculate_volume_profile_distribution(df.tail(40))
            volume_factors.append(volume_profile['strength'])
            
            # Volume trend analysis
            volume_trend = self._analyze_volume_trend_patterns(df.tail(30))
            volume_factors.append(volume_trend)
            
            # Large volume day analysis
            large_volume_analysis = self._analyze_large_volume_days(df.tail(25))
            volume_factors.append(large_volume_analysis)
            
            # Volume-price relationship
            volume_price_relationship = self._analyze_volume_price_relationship(df.tail(35))
            volume_factors.append(volume_price_relationship)
            
            # Institutional volume patterns
            institutional_volume = self._detect_institutional_volume_patterns(df.tail(20))
            volume_factors.append(institutional_volume)
            
            total_score = np.mean(volume_factors) * 100
            
            # Enhanced criteria
            threshold = 60 if is_crypto else 65
            passed = total_score >= threshold and volume_profile['concentration'] > 0.4
            
            confidence = min(100, total_score * 1.2)
            indicator = "📊" if passed else "❌"
            
            if passed and total_score > 70 and symbol:
                self.stage_top_stocks["Volume Profiling"].append((symbol, total_score))
            
            details = {
                'volume_profile': volume_profile,
                'volume_trend': volume_trend,
                'large_volume_analysis': large_volume_analysis,
                'volume_price_relationship': volume_price_relationship,
                'institutional_volume': institutional_volume,
                'volume_strength': total_score / 100,
                'volume_confidence': confidence
            }
            
            return StageResult(
                "Volume Profiling",
                passed,
                total_score,
                details,
                indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Volume Profiling", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_5_momentum_analysis(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Advanced momentum analysis with multiple oscillators"""
        try:
            momentum_indicators = {}
            
            # RSI analysis
            rsi_analysis = self._analyze_rsi_momentum(df)
            momentum_indicators['rsi'] = rsi_analysis
            
            # MACD analysis
            macd_analysis = self._analyze_macd_momentum(df)
            momentum_indicators['macd'] = macd_analysis
            
            # Stochastic analysis
            stochastic_analysis = self._analyze_stochastic_momentum(df)
            momentum_indicators['stochastic'] = stochastic_analysis
            
            # Williams %R analysis
            williams_analysis = self._analyze_williams_momentum(df)
            momentum_indicators['williams'] = williams_analysis
            
            # Custom momentum composite
            momentum_composite = self._calculate_momentum_composite_score(df)
            momentum_indicators['composite'] = momentum_composite
            
            # Calculate overall momentum score
            momentum_scores = []
            for indicator, data in momentum_indicators.items():
                if isinstance(data, dict) and 'score' in data:
                    momentum_scores.append(data['score'])
                elif isinstance(data, (int, float)):
                    momentum_scores.append(data)
            
            if momentum_scores:
                total_score = np.mean(momentum_scores)
            else:
                total_score = 50
            
            # Enhanced momentum criteria
            bullish_count = sum(1 for indicator, data in momentum_indicators.items()
                               if isinstance(data, dict) and data.get('bullish', False))
            
            passed = total_score >= 60 and bullish_count >= 3
            
            confidence = min(100, total_score * 1.1 + bullish_count * 5)
            indicator = "🚀" if passed else "❌"
            
            if passed and total_score > 70 and symbol:
                self.stage_top_stocks["Momentum Analysis"].append((symbol, total_score))
            
            details = momentum_indicators.copy()
            details.update({
                'overall_momentum_score': total_score,
                'bullish_indicators_count': bullish_count,
                'momentum_confidence': confidence
            })
            
            return StageResult(
                "Momentum Analysis",
                passed,
                total_score,
                details,
                indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Momentum Analysis", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_6_pattern_recognition(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Advanced pattern recognition"""
        try:
            detected_patterns = []
            pattern_scores = []
            
            # Traditional chart patterns
            traditional_patterns = self._detect_traditional_chart_patterns(df)
            if traditional_patterns['detected']:
                detected_patterns.extend(traditional_patterns['patterns'])
                pattern_scores.append(traditional_patterns['score'])
            
            # Candlestick patterns
            candlestick_patterns = self._detect_candlestick_patterns(df)
            if candlestick_patterns['detected']:
                detected_patterns.extend(candlestick_patterns['patterns'])
                pattern_scores.append(candlestick_patterns['score'])
            
            # Volume patterns
            if 'Volume' in df.columns:
                volume_patterns = self._detect_volume_patterns(df)
                if volume_patterns['detected']:
                    detected_patterns.extend(volume_patterns['patterns'])
                    pattern_scores.append(volume_patterns['score'])
            
            # Momentum patterns
            momentum_patterns = self._detect_momentum_patterns(df)
            if momentum_patterns['detected']:
                detected_patterns.extend(momentum_patterns['patterns'])
                pattern_scores.append(momentum_patterns['score'])
            
            # Support/Resistance patterns
            sr_patterns = self._detect_support_resistance_patterns(df)
            if sr_patterns['detected']:
                detected_patterns.extend(sr_patterns['patterns'])
                pattern_scores.append(sr_patterns['score'])
            
            # Calculate pattern recognition score
            if pattern_scores:
                total_score = np.mean(pattern_scores)
                pattern_strength = len(detected_patterns) * 5
                total_score = min(100, total_score + pattern_strength)
            else:
                total_score = 0
            
            passed = len(detected_patterns) >= 2 and total_score >= 60
            
            confidence = min(100, total_score * 1.1 + len(detected_patterns) * 3)
            indicator = "🔍" if passed else "❌"
            
            if passed and total_score > 70 and symbol:
                self.stage_top_stocks["Pattern Recognition"].append((symbol, total_score))
            
            details = {
                'detected_patterns': detected_patterns,
                'traditional_patterns': traditional_patterns,
                'candlestick_patterns': candlestick_patterns,
                'volume_patterns': volume_patterns if 'Volume' in df.columns else {'detected': False},
                'momentum_patterns': momentum_patterns,
                'sr_patterns': sr_patterns,
                'pattern_count': len(detected_patterns),
                'pattern_strength': total_score,
                'recognition_confidence': confidence
            }
            
            return StageResult(
                "Pattern Recognition",
                passed,
                total_score,
                details,
                indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Pattern Recognition", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_7_risk_assessment(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Comprehensive risk assessment"""
        try:
            risk_factors = {}
            
            # Volatility risk - IMPROVED
            volatility_risk = self._calculate_volatility_risk_improved(df, is_crypto)
            risk_factors['volatility'] = volatility_risk
            
            # Liquidity risk
            liquidity_risk = self._calculate_liquidity_risk(df)
            risk_factors['liquidity'] = liquidity_risk
            
            # Technical risk
            technical_risk = self._calculate_technical_risk(df)
            risk_factors['technical'] = technical_risk
            
            # Market structure risk
            market_structure_risk = self._calculate_market_structure_risk(df, is_crypto)
            risk_factors['market_structure'] = market_structure_risk
            
            # Drawdown risk
            drawdown_risk = self._calculate_drawdown_risk(df)
            risk_factors['drawdown'] = drawdown_risk
            
            # Calculate overall risk score
            risk_scores = []
            for factor in risk_factors.values():
                if isinstance(factor, dict) and 'score' in factor:
                    risk_scores.append(factor['score'])
            
            overall_risk = np.mean(risk_scores) if risk_scores else 0.5
            
            # Risk-adjusted score (higher is better)
            risk_adjusted_score = max(0, 100 - overall_risk * 100)
            
            # Risk level classification - IMPROVED
            if overall_risk < 0.35:  # More lenient
                risk_level = "LOW"
                risk_indicator = "🟢"
            elif overall_risk < 0.65:  # More lenient
                risk_level = "MEDIUM"
                risk_indicator = "🟡"
            else:
                risk_level = "HIGH"
                risk_indicator = "🔴"
            
            # Pass if risk is acceptable - MORE LENIENT
            passed = overall_risk < 0.75 and risk_adjusted_score >= 35
            
            confidence = min(100, risk_adjusted_score * 1.2)
            
            if passed and risk_adjusted_score > 60 and symbol:
                self.stage_top_stocks["Risk Assessment"].append((symbol, risk_adjusted_score))
            
            details = risk_factors.copy()
            details.update({
                'overall_risk': overall_risk,
                'risk_level': risk_level,
                'risk_adjusted_score': risk_adjusted_score,
                'risk_confidence': confidence
            })
            
            return StageResult(
                "Risk Assessment",
                passed,
                risk_adjusted_score,
                details,
                risk_indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Risk Assessment", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_8_entry_optimization(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Advanced entry point optimization"""
        try:
            entry_analysis = {}
            
            # Optimal entry timing
            entry_timing = self._calculate_optimal_entry_timing(df, is_crypto)
            entry_analysis['timing'] = entry_timing
            
            # Entry zone calculation
            entry_zone = self._calculate_entry_zone_advanced(df, is_crypto)
            entry_analysis['zone'] = entry_zone
            
            # Risk-reward optimization
            risk_reward = self._optimize_risk_reward_advanced(df, is_crypto)
            entry_analysis['risk_reward'] = risk_reward
            
            # Position sizing optimization
            position_sizing = self._calculate_optimal_position_sizing(df, is_crypto)
            entry_analysis['position_sizing'] = position_sizing
            
            # Entry execution probability
            execution_prob = self._calculate_entry_execution_probability(df, is_crypto)
            entry_analysis['execution'] = execution_prob
            
            # Calculate entry optimization score
            entry_scores = []
            for analysis in entry_analysis.values():
                if isinstance(analysis, dict) and 'score' in analysis:
                    entry_scores.append(analysis['score'])
            
            if entry_scores:
                total_score = np.mean(entry_scores) * 100
            else:
                total_score = 50
            
            # Enhanced criteria - MORE LENIENT
            passed = (total_score >= 60 and  # Reduced from 65
                     risk_reward.get('ratio', 0) >= 1.8 and  # Reduced from 2.0
                     entry_timing.get('optimal', False))
            
            confidence = min(100, total_score * 1.15)
            indicator = "🎯" if passed else "❌"
            
            if passed and total_score > 70 and symbol:
                self.stage_top_stocks["Entry Optimization"].append((symbol, total_score))
            
            details = entry_analysis.copy()
            details.update({
                'entry_score': total_score,
                'entry_confidence': confidence,
                'recommendation': 'BUY' if passed else 'WAIT'
            })
            
            return StageResult(
                "Entry Optimization",
                passed,
                total_score,
                details,
                indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Entry Optimization", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_9_breakout_probability(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Calculate breakout probability using advanced methods"""
        try:
            breakout_factors = {}
            
            # Historical breakout analysis
            historical_breakouts = self._analyze_historical_breakout_patterns(df)
            breakout_factors['historical'] = historical_breakouts
            
            # Volume breakout indicators
            volume_breakout = self._analyze_volume_breakout_setup(df)
            breakout_factors['volume'] = volume_breakout
            
            # Price action breakout signals
            price_action = self._analyze_price_action_breakout_setup(df)
            breakout_factors['price_action'] = price_action
            
            # Technical breakout probability
            technical_breakout = self._calculate_technical_breakout_setup(df)
            breakout_factors['technical'] = technical_breakout
            
            # Time-based breakout analysis
            time_analysis = self._analyze_breakout_timing_factors(df)
            breakout_factors['time'] = time_analysis
            
            # Volatility breakout analysis
            volatility_breakout = self._analyze_volatility_breakout_setup(df)
            breakout_factors['volatility'] = volatility_breakout
            
            # Calculate overall breakout probability
            breakout_scores = []
            for factor in breakout_factors.values():
                if isinstance(factor, dict) and 'probability' in factor:
                    breakout_scores.append(factor['probability'])
            
            if breakout_scores:
                breakout_probability = np.mean(breakout_scores)
                total_score = breakout_probability * 100
            else:
                breakout_probability = 0.5
                total_score = 50
            
            # Enhanced breakout criteria
            high_probability_factors = sum(1 for factor in breakout_factors.values()
                                          if isinstance(factor, dict) and factor.get('probability', 0) > 0.7)
            
            passed = breakout_probability >= 0.65 and high_probability_factors >= 3
            
            confidence = min(100, total_score * 1.2 + high_probability_factors * 5)
            indicator = "💥" if passed else "❌"
            
            if passed and total_score > 75 and symbol:
                self.stage_top_stocks["Breakout Probability"].append((symbol, total_score))
            
            details = breakout_factors.copy()
            details.update({
                'breakout_probability': breakout_probability,
                'breakout_score': total_score,
                'high_probability_factors': high_probability_factors,
                'breakout_confidence': confidence,
                'expected_breakout_timeframe': self._estimate_breakout_timeframe(breakout_factors),
                'breakout_target': self._calculate_breakout_price_target(df, breakout_probability)
            })
            
            return StageResult(
                "Breakout Probability",
                passed,
                total_score,
                details,
                indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Breakout Probability", False, 0, {"error": str(e)}, "❌", [], 0)

    def stage_10_professional_grade(self, df: pd.DataFrame, symbol: str = "", is_crypto: bool = False) -> StageResult:
        """Final professional-grade evaluation"""
        try:
            professional_criteria = {}
            
            # Institutional quality assessment
            institutional_quality = self._assess_institutional_investment_quality(df, is_crypto)
            professional_criteria['institutional_quality'] = institutional_quality
            
            # Professional trader grade
            trader_grade = self._calculate_professional_trader_grade(df, is_crypto)
            professional_criteria['trader_grade'] = trader_grade
            
            # Risk management excellence
            risk_management = self._evaluate_risk_management_quality(df)
            professional_criteria['risk_management'] = risk_management
            
            # Execution quality assessment
            execution_quality = self._assess_execution_quality(df, is_crypto)
            professional_criteria['execution_quality'] = execution_quality
            
            # Alpha generation potential
            alpha_potential = self._calculate_alpha_generation_potential(df, is_crypto)
            professional_criteria['alpha_potential'] = alpha_potential
            
            # Market edge assessment
            market_edge = self._assess_market_edge(df, is_crypto)
            professional_criteria['market_edge'] = market_edge
            
            # Calculate professional grade
            professional_scores = []
            for criteria in professional_criteria.values():
                if isinstance(criteria, dict) and 'grade' in criteria:
                    professional_scores.append(criteria['grade'])
                elif isinstance(criteria, (int, float)):
                    professional_scores.append(criteria)
            
            if professional_scores:
                professional_grade = np.mean(professional_scores)
                total_score = professional_grade
            else:
                professional_grade = 50
                total_score = 50
            
            # Professional grade classification
            if professional_grade >= 90:
                grade_level = "INSTITUTIONAL"
                grade_indicator = "👑"
            elif professional_grade >= 80:
                grade_level = "PROFESSIONAL"
                grade_indicator = "🏆"
            elif professional_grade >= 70:
                grade_level = "ADVANCED"
                grade_indicator = "⭐"
            elif professional_grade >= 60:
                grade_level = "INTERMEDIATE"
                grade_indicator = "📈"
            else:
                grade_level = "BASIC"
                grade_indicator = "📊"
            
            # Pass if meets professional standards - MORE LENIENT
            passed = professional_grade >= 70 and alpha_potential.get('score', 0) >= 65  # Reduced thresholds
            
            confidence = min(100, professional_grade * 1.1)
            
            if passed and professional_grade > 75 and symbol:  # Reduced from 80
                self.stage_top_stocks["Professional Grade"].append((symbol, professional_grade))
            
            details = professional_criteria.copy()
            details.update({
                'professional_grade': professional_grade,
                'grade_level': grade_level,
                'professional_confidence': confidence,
                'recommendation': 'STRONG BUY' if professional_grade >= 85 else 'BUY' if passed else 'HOLD'
            })
            
            return StageResult(
                "Professional Grade",
                passed,
                total_score,
                details,
                grade_indicator,
                [],
                confidence
            )
        except Exception as e:
            return StageResult("Professional Grade", False, 0, {"error": str(e)}, "❌", [], 0)

    # ================== HELPER METHODS ==================
    
    def _analyze_smart_money_volume(self, df: pd.DataFrame) -> float:
        """Analyze volume patterns for smart money activity"""
        try:
            if 'Volume' not in df.columns:
                return 0.3
            
            # Higher volume on down days suggests accumulation
            down_days = df[df['Close'] < df['Open']]
            up_days = df[df['Close'] > df['Open']]
            
            if len(down_days) == 0 or len(up_days) == 0:
                return 0.3
            
            avg_down_volume = down_days['Volume'].mean()
            avg_up_volume = up_days['Volume'].mean()
            
            smart_money_ratio = avg_down_volume / (avg_down_volume + avg_up_volume)
            return min(1.0, smart_money_ratio * 2)  # Amplify signal
        except:
            return 0.3

    def _calculate_price_stability_with_volume(self, df: pd.DataFrame) -> float:
        """Calculate price stability combined with volume analysis"""
        try:
            price_volatility = df['Close'].pct_change().std()
            
            if 'Volume' in df.columns:
                volume_trend = df['Volume'].tail(5).mean() / df['Volume'].mean()
                # Low volatility + increasing volume = accumulation
                stability_score = (1 - min(1.0, price_volatility * 50)) * 0.7 + min(1.0, volume_trend) * 0.3
            else:
                stability_score = 1 - min(1.0, price_volatility * 50)
            
            return max(0, stability_score)
        except:
            return 0.3

    def _analyze_support_accumulation(self, df: pd.DataFrame) -> float:
        """Analyze accumulation at support levels"""
        try:
            support_level = df['Low'].tail(20).min()
            tolerance = support_level * 0.02
            
            # Count touches and volume at support
            support_touches = 0
            total_volume_at_support = 0
            total_volume = 0
            
            for _, row in df.iterrows():
                if 'Volume' in df.columns:
                    total_volume += row['Volume']
                
                if support_level - tolerance <= row['Low'] <= support_level + tolerance:
                    support_touches += 1
                    if 'Volume' in df.columns:
                        total_volume_at_support += row['Volume']
            
            touch_score = min(1.0, support_touches / 5)
            
            if 'Volume' in df.columns and total_volume > 0:
                volume_score = total_volume_at_support / total_volume
                return (touch_score * 0.6 + volume_score * 0.4)
            else:
                return touch_score * 0.8
        except:
            return 0.3

    def _detect_hidden_divergences(self, df: pd.DataFrame) -> float:
        """Detect hidden bullish divergences"""
        try:
            if len(df) < 20:
                return 0.3
            
            # Simple RSI divergence
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            if len(rsi) < 10:
                return 0.3
            
            # Check for bullish hidden divergence
            price_higher_low = df['Close'].iloc[-1] > df['Close'].iloc[-10]
            rsi_lower_low = rsi.iloc[-1] < rsi.iloc[-10]
            
            if price_higher_low and rsi_lower_low:
                return 0.8
            elif price_higher_low or rsi.iloc[-1] > rsi.iloc[-5]:
                return 0.6
            else:
                return 0.3
        except:
            return 0.3

    def _detect_institutional_patterns(self, df: pd.DataFrame) -> float:
        """Detect institutional trading patterns"""
        try:
            if 'Volume' not in df.columns:
                return 0.3
            
            # Large volume with minimal price impact
            avg_volume = df['Volume'].mean()
            price_impacts = []
            
            for i in range(1, len(df)):
                volume_ratio = df['Volume'].iloc[i] / avg_volume
                price_impact = abs(df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]
                
                if volume_ratio > 1.5:  # Large volume day
                    price_impacts.append(price_impact)
            
            if price_impacts:
                avg_impact = np.mean(price_impacts)
                # Lower impact with high volume suggests institutional activity
                institutional_score = max(0, 1 - avg_impact * 100)
                return min(1.0, institutional_score)
            else:
                return 0.3
        except:
            return 0.3

    def _analyze_price_consolidation(self, df: pd.DataFrame) -> float:
        """Analyze price consolidation patterns"""
        try:
            high = df['High'].max()
            low = df['Low'].min()
            avg_price = df['Close'].mean()
            
            consolidation_range = (high - low) / avg_price
            
            # Tighter consolidation = higher score
            if consolidation_range < 0.05:
                return 0.9
            elif consolidation_range < 0.1:
                return 0.7
            elif consolidation_range < 0.15:
                return 0.5
            else:
                return 0.3
        except:
            return 0.3

    def _analyze_volume_accumulation_patterns(self, df: pd.DataFrame) -> float:
        """Analyze volume accumulation patterns"""
        try:
            if 'Volume' not in df.columns:
                return 0.3
            
            volume_trend = df['Volume'].rolling(5).mean()
            
            # Increasing volume trend
            if len(volume_trend) >= 10:
                recent_avg = volume_trend.tail(5).mean()
                earlier_avg = volume_trend.iloc[-10:-5].mean()
                
                if recent_avg > earlier_avg * 1.1:
                    return 0.8
                elif recent_avg > earlier_avg:
                    return 0.6
                else:
                    return 0.4
            else:
                return 0.4
        except:
            return 0.3

    def _calculate_support_level_strength(self, df: pd.DataFrame) -> float:
        """Calculate support level strength"""
        try:
            lows = df['Low'].values
            
            # Find multiple support levels
            support_levels = [
                np.percentile(lows, 5),
                np.percentile(lows, 10),
                np.percentile(lows, 15)
            ]
            
            max_strength = 0
            for level in support_levels:
                tolerance = level * 0.02
                touches = sum(1 for low in lows if level - tolerance <= low <= level + tolerance)
                strength = min(1.0, touches / 5)
                max_strength = max(max_strength, strength)
            
            return max_strength
        except:
            return 0.3

    def _calculate_ad_line_strength(self, df: pd.DataFrame) -> float:
        """Calculate Accumulation/Distribution line strength"""
        try:
            if 'Volume' not in df.columns:
                return 0.4
            
            ad_values = []
            ad_total = 0
            
            for _, row in df.iterrows():
                if row['High'] != row['Low']:
                    clv = ((row['Close'] - row['Low']) - (row['High'] - row['Close'])) / (row['High'] - row['Low'])
                    ad_total += clv * row['Volume']
                ad_values.append(ad_total)
            
            if len(ad_values) >= 10:
                # Positive slope indicates accumulation
                recent_trend = ad_values[-1] - ad_values[-10]
                if recent_trend > 0:
                    return 0.8
                else:
                    return 0.3
            else:
                return 0.4
        except:
            return 0.4

    def _analyze_time_based_accumulation(self, df: pd.DataFrame) -> float:
        """Analyze time-based accumulation patterns"""
        try:
            # Longer consolidation periods get higher scores
            consolidation_days = len(df)
            
            if consolidation_days >= 40:
                return 0.8
            elif consolidation_days >= 25:
                return 0.6
            elif consolidation_days >= 15:
                return 0.5
            else:
                return 0.3
        except:
            return 0.3

    def _calculate_mtf_alignment(self, df: pd.DataFrame) -> float:
        """Calculate multiple timeframe alignment"""
        try:
            if len(df) < 50:
                return 0.3
            
            # Different period moving averages
            ma_5 = df['Close'].rolling(5).mean()
            ma_20 = df['Close'].rolling(20).mean()
            ma_50 = df['Close'].rolling(50).mean()
            
            current_price = df['Close'].iloc[-1]
            
            # Check alignment
            alignment_score = 0
            if current_price > ma_5.iloc[-1]:
                alignment_score += 0.33
            if ma_5.iloc[-1] > ma_20.iloc[-1]:
                alignment_score += 0.33
            if ma_20.iloc[-1] > ma_50.iloc[-1]:
                alignment_score += 0.34
            
            return alignment_score
        except:
            return 0.3

    def _calculate_sr_confluence(self, df: pd.DataFrame) -> float:
        """Calculate support/resistance confluence"""
        try:
            if len(df) < 30:
                return 0.3
            
            current_price = df['Close'].iloc[-1]
            
            # Calculate various S/R levels
            pivot_high = df['High'].tail(20).max()
            pivot_low = df['Low'].tail(20).min()
            
            # Check proximity to key levels
            confluence_score = 0
            tolerance = current_price * 0.02
            
            levels = [pivot_high, pivot_low]
            
            if len(df) >= 50:
                levels.append(df['High'].tail(50).max())
                levels.append(df['Low'].tail(50).min())
            
            for level in levels:
                if abs(current_price - level) <= tolerance:
                    confluence_score += 0.25
            
            return min(1.0, confluence_score)
        except:
            return 0.3

    def _calculate_ma_confluence(self, df: pd.DataFrame) -> float:
        """Calculate moving average confluence"""
        try:
            if len(df) < 50:
                return 0.3
            
            current_price = df['Close'].iloc[-1]
            tolerance = current_price * 0.01
            
            # Key moving averages
            ma_20 = df['Close'].rolling(20).mean().iloc[-1]
            ma_50 = df['Close'].rolling(50).mean().iloc[-1]
            
            confluence_score = 0
            
            # Check if price is near key MAs
            if abs(current_price - ma_20) <= tolerance:
                confluence_score += 0.5
            if abs(current_price - ma_50) <= tolerance:
                confluence_score += 0.5
            
            return min(1.0, confluence_score)
        except:
            return 0.3

    def _calculate_fibonacci_confluence(self, df: pd.DataFrame) -> float:
        """Calculate Fibonacci level confluence"""
        try:
            if len(df) < 40:
                return 0.3
            
            # Find swing high and low
            high = df['High'].tail(40).max()
            low = df['Low'].tail(40).min()
            
            if high == low:
                return 0.3
            
            # Calculate key Fib levels
            diff = high - low
            fib_levels = [
                high - diff * 0.382,
                high - diff * 0.5,
                high - diff * 0.618
            ]
            
            current_price = df['Close'].iloc[-1]
            tolerance = current_price * 0.01
            
            confluence_score = 0
            for level in fib_levels:
                if abs(current_price - level) <= tolerance:
                    confluence_score += 0.33
            
            return min(1.0, confluence_score)
        except:
            return 0.3

    def _calculate_indicator_confluence(self, df: pd.DataFrame) -> float:
        """Calculate technical indicator confluence"""
        try:
            if len(df) < 30:
                return 0.3
            
            bullish_signals = 0
            total_signals = 0
            
            # RSI
            if len(df) >= 14:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                if rsi.iloc[-1] > 50:
                    bullish_signals += 1
                total_signals += 1
            
            # MACD
            if len(df) >= 26:
                ema_12 = df['Close'].ewm(span=12).mean()
                ema_26 = df['Close'].ewm(span=26).mean()
                macd = ema_12 - ema_26
                
                if macd.iloc[-1] > 0:
                    bullish_signals += 1
                total_signals += 1
            
            # Moving average
            if len(df) >= 20:
                ma_20 = df['Close'].rolling(20).mean()
                if df['Close'].iloc[-1] > ma_20.iloc[-1]:
                    bullish_signals += 1
                total_signals += 1
            
            if total_signals > 0:
                return bullish_signals / total_signals
            else:
                return 0.3
        except:
            return 0.3

    # Continue with more helper methods for remaining stages...
    # Due to space limitations, I'll implement the most critical ones

    def _calculate_volume_profile_distribution(self, df: pd.DataFrame) -> Dict:
        """Calculate volume profile distribution"""
        try:
            if 'Volume' not in df.columns:
                return {'strength': 0.3, 'concentration': 0.0}
            
            # Simple volume profile
            price_levels = np.linspace(df['Low'].min(), df['High'].max(), 10)
            volume_at_levels = []
            
            for i in range(len(price_levels) - 1):
                level_volume = 0
                for _, row in df.iterrows():
                    if price_levels[i] <= row['Close'] <= price_levels[i + 1]:
                        level_volume += row['Volume']
                volume_at_levels.append(level_volume)
            
            total_volume = sum(volume_at_levels)
            if total_volume == 0:
                return {'strength': 0.3, 'concentration': 0.0}
            
            max_volume = max(volume_at_levels)
            concentration = max_volume / total_volume
            strength = min(1.0, concentration * 2)
            
            return {'strength': strength, 'concentration': concentration}
        except:
            return {'strength': 0.3, 'concentration': 0.0}

    def _analyze_volume_trend_patterns(self, df: pd.DataFrame) -> float:
        """Analyze volume trend patterns"""
        try:
            if 'Volume' not in df.columns:
                return 0.3
            
            volume = df['Volume']
            volume_ma = volume.rolling(10).mean()
            
            # Check for increasing volume trend
            if len(volume_ma) >= 5:
                recent_trend = volume_ma.iloc[-1] / volume_ma.iloc[-5]
                if recent_trend > 1.1:
                    return 0.8
                elif recent_trend > 1.0:
                    return 0.6
                else:
                    return 0.4
            else:
                return 0.4
        except:
            return 0.3

    def _analyze_large_volume_days(self, df: pd.DataFrame) -> float:
        """Analyze large volume days"""
        try:
            if 'Volume' not in df.columns:
                return 0.3
            
            avg_volume = df['Volume'].mean()
            large_volume_threshold = avg_volume * 1.5
            
            large_volume_days = (df['Volume'] > large_volume_threshold).sum()
            large_volume_ratio = large_volume_days / len(df)
            
            return min(1.0, large_volume_ratio * 3)
        except:
            return 0.3

    def _analyze_volume_price_relationship(self, df: pd.DataFrame) -> float:
        """Analyze volume-price relationship"""
        try:
            if 'Volume' not in df.columns:
                return 0.3
            
            # Calculate correlation between volume and price changes
            price_changes = df['Close'].pct_change().abs()
            volume_changes = df['Volume'].pct_change().abs()
            
            correlation = price_changes.corr(volume_changes)
            
            # Higher correlation indicates healthy volume-price relationship
            if pd.isna(correlation):
                return 0.3
            
            return min(1.0, max(0, correlation))
        except:
            return 0.3

    def _detect_institutional_volume_patterns(self, df: pd.DataFrame) -> float:
        """Detect institutional volume patterns"""
        try:
            if 'Volume' not in df.columns:
                return 0.3
            
            # Look for block trading patterns
            avg_volume = df['Volume'].mean()
            std_volume = df['Volume'].std()
            
            institutional_threshold = avg_volume + (2 * std_volume)
            institutional_days = (df['Volume'] > institutional_threshold).sum()
            
            institutional_ratio = institutional_days / len(df)
            return min(1.0, institutional_ratio * 5)
        except:
            return 0.3

    # RSI, MACD, and other momentum analysis methods
    def _analyze_rsi_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze RSI momentum"""
        try:
            if len(df) < 14:
                return {'score': 50, 'bullish': False}
            
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1]
            
            # RSI scoring
            if 50 <= current_rsi <= 70:
                score = 80
                bullish = True
            elif 30 <= current_rsi < 50:
                score = 60
                bullish = False
            elif current_rsi > 70:
                score = 40  # Overbought
                bullish = False
            else:
                score = 70  # Oversold, potential reversal
                bullish = True
            
            return {
                'score': score,
                'value': current_rsi,
                'bullish': bullish,
                'overbought': current_rsi > 70,
                'oversold': current_rsi < 30
            }
        except:
            return {'score': 50, 'bullish': False}

    def _analyze_macd_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze MACD momentum"""
        try:
            if len(df) < 26:
                return {'score': 50, 'bullish': False}
            
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            histogram = current_macd - current_signal
            
            score = 50
            bullish = False
            
            if current_macd > current_signal and current_macd > 0:
                score = 85
                bullish = True
            elif current_macd > current_signal:
                score = 70
                bullish = True
            elif current_macd > 0:
                score = 60
                bullish = True
            
            return {
                'score': score,
                'macd': current_macd,
                'signal': current_signal,
                'histogram': histogram,
                'bullish': bullish
            }
        except:
            return {'score': 50, 'bullish': False}

    def _analyze_stochastic_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze Stochastic momentum"""
        try:
            if len(df) < 14:
                return {'score': 50, 'bullish': False}
            
            low_14 = df['Low'].rolling(14).min()
            high_14 = df['High'].rolling(14).max()
            k_percent = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
            d_percent = k_percent.rolling(3).mean()
            
            current_k = k_percent.iloc[-1]
            current_d = d_percent.iloc[-1]
            
            score = 50
            bullish = False
            
            if current_k > current_d and current_k > 20:
                score = 75
                bullish = True
            elif current_k < 20:
                score = 80  # Oversold
                bullish = True
            elif current_k > 80:
                score = 30  # Overbought
                bullish = False
            
            return {
                'score': score,
                'k_value': current_k,
                'd_value': current_d,
                'bullish': bullish,
                'oversold': current_k < 20,
                'overbought': current_k > 80
            }
        except:
            return {'score': 50, 'bullish': False}

    def _analyze_williams_momentum(self, df: pd.DataFrame) -> Dict:
        """Analyze Williams %R momentum"""
        try:
            if len(df) < 14:
                return {'score': 50, 'bullish': False}
            
            high_14 = df['High'].rolling(14).max()
            low_14 = df['Low'].rolling(14).min()
            williams_r = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
            
            current_wr = williams_r.iloc[-1]
            
            score = 50
            bullish = False
            
            if current_wr > -50:
                score = 70
                bullish = True
            elif current_wr < -80:
                score = 75  # Oversold
                bullish = True
            
            return {
                'score': score,
                'value': current_wr,
                'bullish': bullish,
                'oversold': current_wr < -80,
                'overbought': current_wr > -20
            }
        except:
            return {'score': 50, 'bullish': False}

    def _calculate_momentum_composite_score(self, df: pd.DataFrame) -> Dict:
        """Calculate composite momentum score"""
        try:
            momentum_scores = []
            
            # Price momentum
            if len(df) >= 10:
                price_momentum = (df['Close'].iloc[-1] / df['Close'].iloc[-10] - 1) * 100
                momentum_scores.append(min(100, max(0, 50 + price_momentum * 2)))
            
            # Volume momentum
            if 'Volume' in df.columns and len(df) >= 10:
                volume_momentum = (df['Volume'].tail(5).mean() / df['Volume'].iloc[-15:-5].mean() - 1) * 100
                momentum_scores.append(min(100, max(0, 50 + volume_momentum)))
            
            if momentum_scores:
                composite_score = np.mean(momentum_scores)
                return {
                    'score': composite_score,
                    'bullish': composite_score > 60
                }
            else:
                return {'score': 50, 'bullish': False}
        except:
            return {'score': 50, 'bullish': False}

    # Add remaining helper methods for pattern recognition, risk assessment, etc.
    # Due to space constraints, I'll implement key ones

    def _detect_traditional_chart_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect traditional chart patterns"""
        try:
            patterns = []
            score = 0
            
            # Simple triangle detection
            if len(df) >= 20:
                highs = df['High'].tail(20).values
                lows = df['Low'].tail(20).values
                
                # Check for converging pattern
                high_slope, _, high_r, _, _ = linregress(range(len(highs)), highs)
                low_slope, _, low_r, _, _ = linregress(range(len(lows)), lows)
                
                if abs(high_slope) < 0.01 and low_slope > 0.01:
                    patterns.append('Ascending Triangle')
                    score += 30
                elif high_slope < -0.01 and abs(low_slope) < 0.01:
                    patterns.append('Descending Triangle')
                    score += 25
                elif high_slope < -0.01 and low_slope > 0.01:
                    patterns.append('Symmetrical Triangle')
                    score += 35
            
            return {
                'detected': len(patterns) > 0,
                'patterns': patterns,
                'score': score
            }
        except:
            return {'detected': False, 'patterns': [], 'score': 0}

    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect candlestick patterns"""
        try:
            patterns = []
            score = 0
            
            if len(df) < 3:
                return {'detected': False, 'patterns': [], 'score': 0}
            
            # Last 3 candles
            recent = df.tail(3)
            
            # Hammer pattern
            for i, (_, candle) in enumerate(recent.iterrows()):
                body = abs(candle['Close'] - candle['Open'])
                lower_shadow = candle['Open'] - candle['Low'] if candle['Close'] > candle['Open'] else candle['Close'] - candle['Low']
                upper_shadow = candle['High'] - candle['Close'] if candle['Close'] > candle['Open'] else candle['High'] - candle['Open']
                
                if lower_shadow > body * 2 and upper_shadow < body * 0.5:
                    patterns.append('Hammer')
                    score += 20
                    break
            
            # Doji pattern
            last_candle = df.iloc[-1]
            body_size = abs(last_candle['Close'] - last_candle['Open'])
            candle_range = last_candle['High'] - last_candle['Low']
            
            if body_size < candle_range * 0.1:
                patterns.append('Doji')
                score += 15
            
            return {
                'detected': len(patterns) > 0,
                'patterns': patterns,
                'score': score
            }
        except:
            return {'detected': False, 'patterns': [], 'score': 0}

    def _detect_volume_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect volume patterns"""
        try:
            if 'Volume' not in df.columns:
                return {'detected': False, 'patterns': [], 'score': 0}
            
            patterns = []
            score = 0
            
            # Volume spike pattern
            avg_volume = df['Volume'].mean()
            if df['Volume'].iloc[-1] > avg_volume * 2:
                patterns.append('Volume Spike')
                score += 25
            
            # Accumulation pattern
            recent_volume = df['Volume'].tail(5).mean()
            if recent_volume > avg_volume * 1.2:
                patterns.append('Volume Accumulation')
                score += 20
            
            return {
                'detected': len(patterns) > 0,
                'patterns': patterns,
                'score': score
            }
        except:
            return {'detected': False, 'patterns': [], 'score': 0}

    def _detect_momentum_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect momentum patterns"""
        try:
            patterns = []
            score = 0
            
            if len(df) >= 10:
                # Momentum acceleration
                momentum_5 = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
                momentum_10 = (df['Close'].iloc[-1] / df['Close'].iloc[-10] - 1) * 100
                
                if momentum_5 > momentum_10 and momentum_5 > 2:
                    patterns.append('Momentum Acceleration')
                    score += 30
                
                # Price breakout
                resistance = df['High'].tail(20).max()
                if df['Close'].iloc[-1] > resistance * 1.01:
                    patterns.append('Resistance Breakout')
                    score += 25
            
            return {
                'detected': len(patterns) > 0,
                'patterns': patterns,
                'score': score
            }
        except:
            return {'detected': False, 'patterns': [], 'score': 0}

    def _detect_support_resistance_patterns(self, df: pd.DataFrame) -> Dict:
        """Detect support/resistance patterns"""
        try:
            patterns = []
            score = 0
            
            if len(df) >= 20:
                support = df['Low'].tail(20).min()
                resistance = df['High'].tail(20).max()
                current_price = df['Close'].iloc[-1]
                
                # Near support
                if abs(current_price - support) / current_price < 0.02:
                    patterns.append('Near Support')
                    score += 20
                
                # Near resistance
                if abs(current_price - resistance) / current_price < 0.02:
                    patterns.append('Near Resistance')
                    score += 15
            
            return {
                'detected': len(patterns) > 0,
                'patterns': patterns,
                'score': score
            }
        except:
            return {'detected': False, 'patterns': [], 'score': 0}

    # Risk assessment helper methods with improved thresholds
    def _calculate_volatility_risk_improved(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Calculate volatility risk with improved thresholds"""
        try:
            if len(df) < 20:
                return {'score': 0.3, 'level': 'MEDIUM'}
            
            returns = df['Close'].pct_change().dropna()
            hist_vol = returns.std() * np.sqrt(252)
            
            # IMPROVED: More lenient thresholds
            if is_crypto:
                high_vol_threshold = 1.5  # Increased
                medium_vol_threshold = 0.8  # Increased
            else:
                high_vol_threshold = 0.6  # Increased
                medium_vol_threshold = 0.35  # Increased
            
            if hist_vol > high_vol_threshold:
                vol_risk = 0.7  # Reduced
                level = 'HIGH'
            elif hist_vol > medium_vol_threshold:
                vol_risk = 0.4  # Reduced
                level = 'MEDIUM'
            else:
                vol_risk = 0.2
                level = 'LOW'
            
            return {
                'score': vol_risk,
                'level': level,
                'historical_volatility': hist_vol
            }
        except:
            return {'score': 0.3, 'level': 'MEDIUM'}

    def _calculate_liquidity_risk(self, df: pd.DataFrame) -> Dict:
        """Calculate liquidity risk"""
        try:
            if 'Volume' not in df.columns:
                return {'score': 0.4, 'level': 'MEDIUM'}
            
            avg_volume = df['Volume'].mean()
            volume_consistency = 1 - (df['Volume'].std() / avg_volume) if avg_volume > 0 else 0
            
            if volume_consistency > 0.6:
                return {'score': 0.2, 'level': 'LOW'}
            elif volume_consistency > 0.3:
                return {'score': 0.4, 'level': 'MEDIUM'}
            else:
                return {'score': 0.7, 'level': 'HIGH'}
        except:
            return {'score': 0.4, 'level': 'MEDIUM'}

    def _calculate_technical_risk(self, df: pd.DataFrame) -> Dict:
        """Calculate technical breakdown risk"""
        try:
            support_level = df['Low'].tail(20).min()
            current_price = df['Close'].iloc[-1]
            distance_to_support = (current_price - support_level) / current_price
            
            if distance_to_support > 0.08:  # More lenient
                return {'score': 0.2, 'level': 'LOW'}
            elif distance_to_support > 0.04:
                return {'score': 0.4, 'level': 'MEDIUM'}
            else:
                return {'score': 0.7, 'level': 'HIGH'}
        except:
            return {'score': 0.4, 'level': 'MEDIUM'}

    def _calculate_market_structure_risk(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Calculate market structure risk"""
        try:
            # Simplified market structure analysis
            if len(df) >= 30:
                uptrend_intact = df['Close'].iloc[-1] > df['Close'].iloc[-30]
                if uptrend_intact:
                    return {'score': 0.3, 'level': 'MEDIUM'}
                else:
                    return {'score': 0.6, 'level': 'HIGH'}
            else:
                return {'score': 0.4, 'level': 'MEDIUM'}
        except:
            return {'score': 0.4, 'level': 'MEDIUM'}

    def _calculate_drawdown_risk(self, df: pd.DataFrame) -> Dict:
        """Calculate drawdown risk"""
        try:
            rolling_max = df['Close'].expanding().max()
            drawdown = (df['Close'] - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
            
            if max_drawdown < 0.15:  # More lenient
                return {'score': 0.2, 'level': 'LOW'}
            elif max_drawdown < 0.25:
                return {'score': 0.4, 'level': 'MEDIUM'}
            else:
                return {'score': 0.7, 'level': 'HIGH'}
        except:
            return {'score': 0.4, 'level': 'MEDIUM'}

    # Entry optimization and other remaining methods...
    def _calculate_optimal_entry_timing(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Calculate optimal entry timing"""
        try:
            timing_factors = []
            
            # RSI timing
            if len(df) >= 14:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                
                if 30 <= current_rsi <= 50:
                    timing_factors.append(0.8)
                elif 50 < current_rsi <= 65:  # More lenient
                    timing_factors.append(0.7)  # Increased
                else:
                    timing_factors.append(0.4)  # Increased
            else:
                timing_factors.append(0.5)
            
            # Support proximity
            support_level = df['Low'].tail(20).min()
            current_price = df['Close'].iloc[-1]
            support_distance = (current_price - support_level) / current_price
            
            if support_distance < 0.04:  # More lenient
                timing_factors.append(0.9)
            elif support_distance < 0.07:  # More lenient
                timing_factors.append(0.8)  # Increased
            else:
                timing_factors.append(0.5)  # Increased
            
            timing_score = np.mean(timing_factors)
            optimal = timing_score >= 0.65  # More lenient
            
            return {
                'score': timing_score,
                'optimal': optimal,
                'rsi_timing': timing_factors[0],
                'support_timing': timing_factors[1] if len(timing_factors) > 1 else 0.5
            }
        except:
            return {'score': 0.5, 'optimal': False}

    def _calculate_entry_zone_advanced(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Calculate advanced entry zone"""
        try:
            current_price = df['Close'].iloc[-1]
            support_1 = df['Low'].tail(20).min()
            
            entry_zone_low = support_1 * 1.01
            entry_zone_high = current_price * 0.99
            
            zone_width = (entry_zone_high - entry_zone_low) / current_price
            
            if zone_width > 0.04:  # More lenient
                zone_quality = 0.5  # Increased
            elif zone_width > 0.02:
                zone_quality = 0.8  # Increased
            else:
                zone_quality = 0.9
            
            return {
                'score': zone_quality,
                'entry_low': entry_zone_low,
                'entry_high': entry_zone_high,
                'zone_width': zone_width
            }
        except:
            return {'score': 0.5}

    def _optimize_risk_reward_advanced(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Optimize risk-reward ratio with advanced calculations"""
        try:
            current_price = df['Close'].iloc[-1]
            support_level = df['Low'].tail(20).min()
            resistance_level = df['High'].tail(20).max()
            
            stop_loss = support_level * 0.98
            target_1 = resistance_level * 1.02
            
            risk = current_price - stop_loss
            reward = target_1 - current_price
            
            rr_ratio = reward / risk if risk > 0 else 0
            
            # More lenient R/R scoring
            if rr_ratio >= 2.5:
                rr_score = 1.0
            elif rr_ratio >= 2.0:
                rr_score = 0.9  # Increased
            elif rr_ratio >= 1.5:
                rr_score = 0.8  # Increased
            elif rr_ratio >= 1.2:  # More lenient threshold
                rr_score = 0.7  # New tier
            else:
                rr_score = 0.4  # Increased
            
            return {
                'score': rr_score,
                'ratio': rr_ratio,
                'risk_amount': risk,
                'reward_amount': reward
            }
        except:
            return {'score': 0.5, 'ratio': 1.5}

    def _calculate_optimal_position_sizing(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Calculate optimal position sizing"""
        try:
            # Simplified Kelly Criterion
            volatility = df['Close'].pct_change().std()
            
            if volatility < 0.02:
                position_size = 0.1  # 10%
            elif volatility < 0.05:
                position_size = 0.05  # 5%
            else:
                position_size = 0.02  # 2%
            
            return {
                'score': 0.8,
                'position_size_pct': position_size * 100,
                'volatility': volatility
            }
        except:
            return {'score': 0.5, 'position_size_pct': 5.0}

    def _calculate_entry_execution_probability(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Calculate entry execution probability"""
        try:
            execution_factors = []
            
            # Volume adequacy
            if 'Volume' in df.columns:
                recent_volume = df['Volume'].tail(5).mean()
                avg_volume = df['Volume'].mean()
                volume_factor = min(1.0, recent_volume / avg_volume) if avg_volume > 0 else 0.5
                execution_factors.append(volume_factor)
            else:
                execution_factors.append(0.6)  # Better default
            
            # Price stability
            price_volatility = df['Close'].tail(10).std() / df['Close'].tail(10).mean()
            if price_volatility < 0.03:  # More lenient
                execution_factors.append(0.9)
            elif price_volatility < 0.06:  # More lenient
                execution_factors.append(0.8)  # Increased
            else:
                execution_factors.append(0.6)  # Increased
            
            execution_probability = np.mean(execution_factors)
            
            return {
                'score': execution_probability,
                'probability': execution_probability
            }
        except:
            return {'score': 0.6, 'probability': 0.6}

    # Continue with breakout probability and professional grade methods...
    # Due to space constraints, I'll implement the key remaining methods

    def _analyze_historical_breakout_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze historical breakout patterns"""
        try:
            if len(df) < 50:
                return {'probability': 0.5, 'success_rate': 0.5}
            
            breakout_successes = 0
            total_breakouts = 0
            
            # Look for historical resistance breakouts
            for i in range(20, len(df) - 10):
                resistance = df['High'].iloc[i-20:i].max()
                
                if df['Close'].iloc[i] > resistance * 1.02:
                    total_breakouts += 1
                    
                    # Check if breakout was successful
                    future_lows = df['Low'].iloc[i+1:i+6]
                    if len(future_lows) > 0 and future_lows.min() > resistance:
                        breakout_successes += 1
            
            success_rate = breakout_successes / max(1, total_breakouts)
            
            return {
                'probability': success_rate,
                'success_rate': success_rate,
                'total_breakouts': total_breakouts
            }
        except:
            return {'probability': 0.5, 'success_rate': 0.5}

    def _analyze_volume_breakout_setup(self, df: pd.DataFrame) -> Dict:
        """Analyze volume breakout setup"""
        try:
            if 'Volume' not in df.columns:
                return {'probability': 0.5}
            
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            recent_volume = df['Volume'].tail(3).mean()
            
            volume_surge = recent_volume > avg_volume * 1.3  # More lenient
            volume_building = df['Volume'].tail(5).mean() > df['Volume'].tail(10).mean()
            
            probability = 0.5
            if volume_surge:
                probability += 0.3
            if volume_building:
                probability += 0.2
            
            return {
                'probability': min(1.0, probability),
                'volume_surge': volume_surge,
                'volume_building': volume_building
            }
        except:
            return {'probability': 0.5}

    def _analyze_price_action_breakout_setup(self, df: pd.DataFrame) -> Dict:
        """Analyze price action breakout setup"""
        try:
            current_price = df['Close'].iloc[-1]
            resistance_level = df['High'].tail(20).max()
            
            distance_to_resistance = (resistance_level - current_price) / current_price
            momentum = (current_price - df['Close'].iloc[-5]) / df['Close'].iloc[-5]
            
            probability = 0.5
            
            if distance_to_resistance < 0.03:  # More lenient
                probability += 0.3
            if momentum > 0.01:  # More lenient
                probability += 0.2
            
            return {
                'probability': min(1.0, probability),
                'distance_to_resistance': distance_to_resistance,
                'momentum': momentum
            }
        except:
            return {'probability': 0.5}

    def _calculate_technical_breakout_setup(self, df: pd.DataFrame) -> Dict:
        """Calculate technical breakout setup"""
        try:
            indicators = {}
            
            # RSI momentum
            if len(df) >= 14:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = -delta.where(delta < 0, 0).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                indicators['rsi_bullish'] = rsi.iloc[-1] > 50
            
            # MACD momentum
            if len(df) >= 26:
                ema_12 = df['Close'].ewm(span=12).mean()
                ema_26 = df['Close'].ewm(span=26).mean()
                macd = ema_12 - ema_26
                indicators['macd_positive'] = macd.iloc[-1] > 0
            
            positive_indicators = sum(indicators.values())
            total_indicators = len(indicators)
            
            probability = positive_indicators / total_indicators if total_indicators > 0 else 0.5
            
            return {
                'probability': probability,
                'indicators': indicators
            }
        except:
            return {'probability': 0.5}

    def _analyze_breakout_timing_factors(self, df: pd.DataFrame) -> Dict:
        """Analyze breakout timing factors"""
        try:
            # Simple consolidation analysis
            price_range = (df['High'].tail(20).max() - df['Low'].tail(20).min()) / df['Close'].tail(20).mean()
            
            if price_range < 0.1:  # Tight consolidation
                timing_score = 0.8
            elif price_range < 0.15:
                timing_score = 0.6
            else:
                timing_score = 0.4
            
            return {
                'probability': timing_score,
                'consolidation_range': price_range
            }
        except:
            return {'probability': 0.5}

    def _analyze_volatility_breakout_setup(self, df: pd.DataFrame) -> Dict:
        """Analyze volatility breakout setup"""
        try:
            recent_vol = df['Close'].tail(10).pct_change().std()
            historical_vol = df['Close'].pct_change().std()
            
            vol_compression = recent_vol < historical_vol * 0.8
            
            return {
                'probability': 0.7 if vol_compression else 0.4,
                'vol_compression': vol_compression,
                'vol_ratio': recent_vol / historical_vol if historical_vol > 0 else 1.0
            }
        except:
            return {'probability': 0.5}

    def _estimate_breakout_timeframe(self, breakout_factors: Dict) -> str:
        """Estimate breakout timeframe"""
        try:
            high_prob_count = sum(1 for factor in breakout_factors.values()
                                 if isinstance(factor, dict) and factor.get('probability', 0) > 0.7)
            
            if high_prob_count >= 4:
                return "1-5 days"
            elif high_prob_count >= 2:
                return "1-2 weeks"
            else:
                return "2-4 weeks"
        except:
            return "Unknown"

    def _calculate_breakout_price_target(self, df: pd.DataFrame, breakout_probability: float) -> float:
        """Calculate breakout price target"""
        try:
            current_price = df['Close'].iloc[-1]
            resistance_level = df['High'].tail(20).max()
            
            # Measured move
            consolidation_range = resistance_level - df['Low'].tail(20).min()
            base_target = resistance_level + consolidation_range
            
            # Adjust based on probability
            probability_multiplier = 0.5 + (breakout_probability * 0.5)
            adjusted_target = current_price + ((base_target - current_price) * probability_multiplier)
            
            return adjusted_target
        except:
            return df['Close'].iloc[-1] * 1.1

    # Professional grade helper methods
    def _assess_institutional_investment_quality(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Assess institutional investment quality"""
        try:
            quality_factors = []
            
            # Liquidity
            if 'Volume' in df.columns:
                volume_consistency = 1 - (df['Volume'].std() / df['Volume'].mean())
                quality_factors.append(max(0, volume_consistency * 100))
            else:
                quality_factors.append(60)  # Better default
            
            # Price stability
            price_stability = 1 - (df['Close'].tail(30).std() / df['Close'].tail(30).mean())
            quality_factors.append(max(0, price_stability * 100))
            
            # Trend consistency
            if len(df) >= 20:
                ma_20 = df['Close'].rolling(20).mean()
                trend_consistency = (df['Close'] > ma_20).tail(20).mean()
                quality_factors.append(trend_consistency * 100)
            else:
                quality_factors.append(60)
            
            institutional_grade = np.mean(quality_factors)
            
            return {
                'grade': institutional_grade,
                'quality': 'INSTITUTIONAL' if institutional_grade > 80 else 'PROFESSIONAL' if institutional_grade > 60 else 'RETAIL'
            }
        except:
            return {'grade': 60, 'quality': 'RETAIL'}

    def _calculate_professional_trader_grade(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Calculate professional trader grade with improved scoring"""
        try:
            trader_criteria = []
            
            # Risk-reward assessment
            support_level = df['Low'].tail(20).min()
            resistance_level = df['High'].tail(20).max()
            current_price = df['Close'].iloc[-1]
            
            risk = current_price - support_level
            reward = resistance_level - current_price
            rr_ratio = reward / risk if risk > 0 else 2.0
            
            # IMPROVED: More generous R/R scoring
            if rr_ratio >= 2.0:
                trader_criteria.append(85)  # Increased
            elif rr_ratio >= 1.5:
                trader_criteria.append(80)  # Increased
            elif rr_ratio >= 1.2:
                trader_criteria.append(75)  # New tier
            elif rr_ratio >= 1.0:
                trader_criteria.append(65)  # Increased
            else:
                trader_criteria.append(55)  # Increased
            
            # Technical setup quality
            if len(df) >= 50:
                ma_20 = df['Close'].rolling(20).mean().iloc[-1]
                ma_50 = df['Close'].rolling(50).mean().iloc[-1]
                
                alignment_score = 70  # Better base
                if current_price > ma_20 > ma_50:
                    alignment_score = 90
                elif current_price > ma_20:
                    alignment_score = 85  # Increased
                elif current_price > ma_50:
                    alignment_score = 80  # Increased
                
                trader_criteria.append(alignment_score)
            else:
                trader_criteria.append(70)  # Better default
            
            # Entry precision
            distance_to_support = (current_price - support_level) / current_price
            if distance_to_support < 0.03:  # More lenient
                precision_score = 95
            elif distance_to_support < 0.06:  # More lenient
                precision_score = 85
            elif distance_to_support < 0.10:  # More lenient
                precision_score = 75
            else:
                precision_score = 65  # Increased
            
            trader_criteria.append(precision_score)
            
            professional_grade = np.mean(trader_criteria)
            
            # IMPROVED: Better level classification
            if professional_grade >= 85:
                level = 'EXPERT'
            elif professional_grade >= 75:
                level = 'PROFESSIONAL'
            elif professional_grade >= 65:
                level = 'ADVANCED'
            elif professional_grade >= 55:
                level = 'INTERMEDIATE'
            else:
                level = 'BASIC'
            
            return {
                'grade': professional_grade,
                'level': level,
                'risk_reward_grade': trader_criteria[0],
                'technical_grade': trader_criteria[1],
                'precision_grade': trader_criteria[2]
            }
        except:
            return {'grade': 65, 'level': 'INTERMEDIATE'}

    def _evaluate_risk_management_quality(self, df: pd.DataFrame) -> Dict:
        """Evaluate risk management quality"""
        try:
            # Simplified risk management assessment
            volatility = df['Close'].pct_change().std()
            max_drawdown = abs((df['Close'] / df['Close'].expanding().max() - 1).min())
            
            risk_score = 70  # Better baseline
            
            if volatility < 0.03 and max_drawdown < 0.15:  # More lenient
                risk_score = 90
            elif volatility < 0.05 and max_drawdown < 0.25:  # More lenient
                risk_score = 80
            elif volatility < 0.08:  # More lenient
                risk_score = 70
            
            return {
                'grade': risk_score,
                'volatility': volatility,
                'max_drawdown': max_drawdown
            }
        except:
            return {'grade': 65}

    def _assess_execution_quality(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Assess execution quality"""
        try:
            # Simplified execution quality assessment
            if 'Volume' in df.columns:
                volume_consistency = 1 - (df['Volume'].std() / df['Volume'].mean())
                execution_score = max(50, min(100, volume_consistency * 120))  # Better scaling
            else:
                execution_score = 70  # Better default
            
            return {
                'grade': execution_score,
                'quality': 'EXCELLENT' if execution_score > 85 else 'GOOD' if execution_score > 70 else 'FAIR'
            }
        except:
            return {'grade': 65, 'quality': 'FAIR'}

    def _calculate_alpha_generation_potential(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Calculate alpha generation potential"""
        try:
            alpha_factors = []
            
            # Performance
            if len(df) >= 60:
                performance_60d = (df['Close'].iloc[-1] - df['Close'].iloc[-60]) / df['Close'].iloc[-60]
                if performance_60d > 0.15:  # More lenient
                    alpha_factors.append(85)  # Increased
                elif performance_60d > 0.08:  # More lenient
                    alpha_factors.append(75)  # Increased
                elif performance_60d > 0:
                    alpha_factors.append(65)  # Increased
                else:
                    alpha_factors.append(45)  # Increased
            else:
                alpha_factors.append(60)  # Better default
            
            # Momentum
            if len(df) >= 20:
                momentum_20d = (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]
                momentum_score = min(100, max(30, 60 + momentum_20d * 400))  # Better scaling
                alpha_factors.append(momentum_score)
            else:
                alpha_factors.append(60)
            
            alpha_score = np.mean(alpha_factors)
            
            return {
                'score': alpha_score,
                'potential': 'HIGH' if alpha_score > 75 else 'MEDIUM' if alpha_score > 60 else 'LOW',  # More lenient
                'expected_alpha': max(0, (alpha_score - 50) / 50 * 0.15)
            }
        except:
            return {'score': 60, 'potential': 'MEDIUM'}

    def _assess_market_edge(self, df: pd.DataFrame, is_crypto: bool = False) -> Dict:
        """Assess market edge"""
        try:
            edge_factors = []
            
            # Trend strength
            if len(df) >= 30:
                trend_strength = (df['Close'].iloc[-1] - df['Close'].iloc[-30]) / df['Close'].iloc[-30]
                if trend_strength > 0.1:  # More lenient
                    edge_factors.append(80)  # Increased
                elif trend_strength > 0.05:  # More lenient
                    edge_factors.append(70)  # Increased
                else:
                    edge_factors.append(55)  # Increased
            else:
                edge_factors.append(60)
            
            # Volume edge
            if 'Volume' in df.columns:
                volume_edge = df['Volume'].tail(10).mean() / df['Volume'].mean()
                if volume_edge > 1.2:
                    edge_factors.append(75)
                elif volume_edge > 1.0:
                    edge_factors.append(65)
                else:
                    edge_factors.append(55)
            else:
                edge_factors.append(60)
            
            market_edge = np.mean(edge_factors)
            
            return {
                'grade': market_edge,
                'edge': 'STRONG' if market_edge > 75 else 'MODERATE' if market_edge > 60 else 'WEAK'
            }
        except:
            return {'grade': 60, 'edge': 'MODERATE'}
