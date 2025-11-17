import pandas as pd
import numpy as np
import sqlite3
import requests
from datetime import datetime, timedelta
import threading
import time
from typing import List, Dict, Optional, Tuple
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import mplfinance as mpf
from flask import Flask, render_template, request, jsonify
import os
import sys

# Mevcut dizindeki modülleri import et
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from harmonic_patterns import find_xabcd, ALL_PATTERNS, plot_pattern, XABCDFound
    from directional_change import get_extremes, directional_change
    HAS_HARMONIC_MODULES = True
    print("Harmonik pattern modülleri başarıyla yüklendi")
except ImportError as e:
    print(f"Harmonik modüller yüklenemedi: {e}")
    HAS_HARMONIC_MODULES = False
    
    # Fallback sınıf tanımları
    @dataclass
    class XABCDFound:
        X: int
        A: int
        B: int
        C: int
        D: int
        error: float
        name: str
        bull: bool
    
    # Fallback fonksiyonlar
    def find_xabcd(ohlc, extremes, err_thresh):
        return {}
    
    def get_extremes(ohlc, sigma):
        return pd.DataFrame()
    
    def directional_change(close, high, low, sigma):
        return [], []
    
    ALL_PATTERNS = []

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternType(Enum):
    GARTLEY = "Gartley"
    BUTTERFLY = "Butterfly"
    BAT = "Bat"
    CRAB = "Crab"
    SHARK = "Shark"
    CYPHER = "Cypher"
    DEEP_CRAB = "Deep Crab"
    AB = "AB=CD"
    THREE_DRIVES = "Three Drives"
    WOLFEWAVE = "Wolfe Wave"
    NEN_STAR = "Nen Star"

@dataclass
class AdvancedPatternResult:
    name: str
    pattern_type: str
    direction: str
    points: Dict
    ratios: Dict
    strength: float
    status: str
    trading_levels: Dict
    error_score: float
    description: str

class IntegratedHarmonicScanner:
    def __init__(self, error_threshold: float = 0.2, sigma_range: List[float] = None,
                 max_distance_pct: float = 5.0):
        self.error_threshold = error_threshold
        self.sigma_range = sigma_range or [0.01, 0.015, 0.02]
        self.max_distance_pct = max_distance_pct
        self.pattern_mapping = self.create_pattern_mapping()
    
    def create_pattern_mapping(self) -> Dict:
        """İki sistemdeki pattern isimlerini eşleştir"""
        return {
            'Gartley': PatternType.GARTLEY,
            'Bat': PatternType.BAT,
            'Butterfly': PatternType.BUTTERFLY,
            'Crab': PatternType.CRAB,
            'Deep Crab': PatternType.DEEP_CRAB,
            'Cypher': PatternType.CYPHER,
            'Shark': PatternType.SHARK
        }
    
    def scan_with_original_method(self, df: pd.DataFrame, current_price: float) -> List[Dict]:
        """Orijinal harmonic_patterns.py yöntemi ile tarama"""
        if not HAS_HARMONIC_MODULES:
            print("Harmonik modüller yüklenmedi, orijinal method kullanılamıyor")
            return []
            
        try:
            all_patterns = []
            
            for sigma in self.sigma_range:
                extremes = get_extremes(df, sigma)
                if extremes.empty:
                    continue
                    
                output = find_xabcd(df, extremes, self.error_threshold)
                
                for pattern in ALL_PATTERNS:
                    pattern_name = pattern.name
                    if pattern_name not in output:
                        continue
                        
                    # Bull pattern'leri işle
                    for bull_pattern in output[pattern_name]['bull_patterns']:
                        pattern_data = self.convert_xabcdfound_to_advanced(
                            bull_pattern, df, current_price, "BULLISH"
                        )
                        if pattern_data and self.is_pattern_active(pattern_data, current_price):
                            all_patterns.append(pattern_data)
                    
                    # Bear pattern'leri işle
                    for bear_pattern in output[pattern_name]['bear_patterns']:
                        pattern_data = self.convert_xabcdfound_to_advanced(
                            bear_pattern, df, current_price, "BEARISH"
                        )
                        if pattern_data and self.is_pattern_active(pattern_data, current_price):
                            all_patterns.append(pattern_data)
            
            return all_patterns
            
        except Exception as e:
            logger.error(f"Original method scan error: {e}")
            return []
    
    def convert_xabcdfound_to_advanced(self, pattern: XABCDFound, df: pd.DataFrame, 
                                     current_price: float, direction: str) -> Optional[Dict]:
        """XABCDFound'ı advanced formata dönüştür"""
        try:
            # DataFrame indekslerini kontrol et
            if pattern.X >= len(df) or pattern.A >= len(df) or pattern.B >= len(df) or pattern.C >= len(df) or pattern.D >= len(df):
                return None
            
            # Zaman bilgilerini al
            timestamps = df.index
            
            points = {
                'X': {
                    'index': pattern.X, 
                    'value': self.get_price_at_index(df, pattern.X, direction),
                    'time': int(timestamps[pattern.X].timestamp()) if pattern.X < len(timestamps) else 0
                },
                'A': {
                    'index': pattern.A, 
                    'value': self.get_price_at_index(df, pattern.A, not direction),
                    'time': int(timestamps[pattern.A].timestamp()) if pattern.A < len(timestamps) else 0
                },
                'B': {
                    'index': pattern.B, 
                    'value': self.get_price_at_index(df, pattern.B, direction),
                    'time': int(timestamps[pattern.B].timestamp()) if pattern.B < len(timestamps) else 0
                },
                'C': {
                    'index': pattern.C, 
                    'value': self.get_price_at_index(df, pattern.C, not direction),
                    'time': int(timestamps[pattern.C].timestamp()) if pattern.C < len(timestamps) else 0
                },
                'D': {
                    'index': pattern.D, 
                    'value': self.get_price_at_index(df, pattern.D, direction),
                    'time': int(timestamps[pattern.D].timestamp()) if pattern.D < len(timestamps) else 0
                }
            }
            
            # Oranları hesapla
            ratios = self.calculate_ratios_from_points(points)
            
            # Trading seviyeleri
            trading_levels = self.calculate_trading_levels(points, direction, current_price)
            
            # Pattern gücü (error'a göre ters orantılı)
            strength = max(0, 100 - (pattern.error * 100))
            
            return {
                'name': pattern.name,
                'pattern_type': self.pattern_mapping.get(pattern.name, pattern.name),
                'type': direction,
                'points': points,
                'ratios': ratios,
                'trading_levels': trading_levels,
                'strength': strength,
                'error_score': pattern.error,
                'status': self.get_pattern_status(trading_levels, current_price, direction),
                'description': f"{direction} {pattern.name} pattern found with error {pattern.error:.3f}",
                'method': 'original'
            }
            
        except Exception as e:
            logger.error(f"Conversion error: {e}")
            return None
    
    def get_price_at_index(self, df: pd.DataFrame, index: int, is_low: bool) -> float:
        """İndeksteki fiyatı getir"""
        try:
            if is_low:
                return df.iloc[index]['low']
            else:
                return df.iloc[index]['high']
        except:
            return 0.0
    
    def calculate_ratios_from_points(self, points: Dict) -> Dict:
        """Noktalardan oranları hesapla"""
        try:
            X = points['X']['value']
            A = points['A']['value']
            B = points['B']['value']
            C = points['C']['value']
            D = points['D']['value']
            
            xa_move = abs(A - X)
            ab_move = abs(B - A)
            bc_move = abs(C - B)
            cd_move = abs(D - C)
            xd_move = abs(D - X)
            
            return {
                'AB': ab_move / xa_move if xa_move > 0 else 0,
                'BC': bc_move / ab_move if ab_move > 0 else 0,
                'CD': cd_move / bc_move if bc_move > 0 else 0,
                'XD': xd_move / xa_move if xa_move > 0 else 0
            }
        except:
            return {'AB': 0, 'BC': 0, 'CD': 0, 'XD': 0}
    
    def calculate_trading_levels(self, points: Dict, direction: str, current_price: float) -> Dict:
        """Trading seviyelerini hesapla"""
        try:
            X = points['X']['value']
            A = points['A']['value']
            D = points['D']['value']
            C = points['C']['value']
            
            if direction == "BULLISH":
                entry = D
                stop_loss = min(X, C) * 0.995
                # TP'leri entry'den (D) yukarı doğru hesapla - A noktasına doğru
                # BULLISH pattern'de tüm TP'ler entry'nin ÜSTÜNDE olmalı
                # BULLISH pattern'de A > D olmalı (yukarı hareket)
                if A > D:
                    move_from_entry = A - D  # Entry'den A'ya olan pozitif hareket
                    tp_calc1 = D + move_from_entry * 0.382  # İlk hedef (entry'nin üstünde)
                    tp_calc2 = D + move_from_entry * 0.618  # İkinci hedef (entry'nin üstünde)
                    tp_calc3 = max(A, X)  # En yüksek hedef (A veya X'ten büyük olan)
                else:
                    # Anormal durum: A <= D, bu durumda A'yı entry'nin üstünde bir nokta olarak kabul et
                    move_from_entry = abs(A - D) if A != D else entry * 0.01  # Minimum hareket
                    tp_calc1 = entry + move_from_entry * 0.382
                    tp_calc2 = entry + move_from_entry * 0.618
                    tp_calc3 = max(A, X, entry * 1.02)  # En azından entry'nin %2 üstü
                
                # TP'leri entry'den uzaklığa göre sırala (en yakından en uzağa)
                # Sadece entry'nin üstündeki TP'leri al
                tps = [tp for tp in [tp_calc1, tp_calc2, tp_calc3] if tp > entry]
                if len(tps) < 3:
                    # Eksik TP'leri entry'nin üstünde oluştur
                    if not tps:
                        tps = [entry * 1.01, entry * 1.02, entry * 1.03]
                    else:
                        max_tp = max(tps)
                        while len(tps) < 3:
                            tps.append(max_tp + (max_tp - entry) * 0.5)
                
                tps_sorted = sorted(tps)  # Küçükten büyüğe
                tp1 = tps_sorted[0]  # En yakın (entry'ye en yakın)
                tp2 = tps_sorted[1] if len(tps_sorted) > 1 else tps_sorted[0] * 1.1  # Orta
                tp3 = tps_sorted[2] if len(tps_sorted) > 2 else tps_sorted[-1] * 1.2  # En uzak
                
                distance_to_entry = ((current_price - entry) / entry * 100) if entry > 0 else 0
            else:
                entry = D
                stop_loss = max(X, C) * 1.005
                # TP'leri entry'den (D) aşağı doğru hesapla - A noktasına doğru
                # BEARISH pattern'de tüm TP'ler entry'nin ALTINDA olmalı
                # BEARISH pattern'de A < D olmalı (aşağı hareket)
                if A < D:
                    move_from_entry = D - A  # Entry'den A'ya olan pozitif hareket
                    tp_calc1 = D - move_from_entry * 0.382  # İlk hedef (entry'nin altında)
                    tp_calc2 = D - move_from_entry * 0.618  # İkinci hedef (entry'nin altında)
                    tp_calc3 = min(A, X)  # En düşük hedef (A veya X'ten küçük olan)
                else:
                    # Anormal durum: A >= D, bu durumda A'yı entry'nin altında bir nokta olarak kabul et
                    move_from_entry = abs(D - A) if A != D else entry * 0.01  # Minimum hareket
                    tp_calc1 = entry - move_from_entry * 0.382
                    tp_calc2 = entry - move_from_entry * 0.618
                    tp_calc3 = min(A, X, entry * 0.98)  # En azından entry'nin %2 altı
                
                # TP'leri entry'den uzaklığa göre sırala (en yakından en uzağa)
                # Sadece entry'nin altındaki TP'leri al
                tps = [tp for tp in [tp_calc1, tp_calc2, tp_calc3] if tp < entry]
                if len(tps) < 3:
                    # Eksik TP'leri entry'nin altında oluştur
                    if not tps:
                        tps = [entry * 0.99, entry * 0.98, entry * 0.97]
                    else:
                        min_tp = min(tps)
                        while len(tps) < 3:
                            tps.append(min_tp - (entry - min_tp) * 0.5)
                
                tps_sorted = sorted(tps, reverse=True)  # Büyükten küçüğe
                tp1 = tps_sorted[0]  # En yakın (entry'ye en yakın)
                tp2 = tps_sorted[1] if len(tps_sorted) > 1 else tps_sorted[0] * 0.9  # Orta
                tp3 = tps_sorted[2] if len(tps_sorted) > 2 else tps_sorted[-1] * 0.8  # En uzak
                
                distance_to_entry = ((entry - current_price) / entry * 100) if entry > 0 else 0
            
            risk = abs(entry - stop_loss)
            reward1 = abs(tp1 - entry)
            reward2 = abs(tp2 - entry)
            reward3 = abs(tp3 - entry)
            
            return {
                'entry': round(entry, 4),
                'stop_loss': round(stop_loss, 4),
                'tp1': round(tp1, 4),
                'tp2': round(tp2, 4),
                'tp3': round(tp3, 4),
                'risk_reward_1': round(reward1 / risk, 2) if risk > 0 else 0,
                'risk_reward_2': round(reward2 / risk, 2) if risk > 0 else 0,
                'risk_reward_3': round(reward3 / risk, 2) if risk > 0 else 0,
                'distance_to_entry_pct': round(abs(distance_to_entry), 2)
            }
        except Exception as e:
            logger.error(f"Trading levels calculation error: {e}")
            return {}
    
    def get_pattern_status(self, trading_levels: Dict, current_price: float, direction: str) -> str:
        """Pattern durumunu belirle"""
        try:
            entry = trading_levels.get('entry', 0)
            stop_loss = trading_levels.get('stop_loss', 0)
            
            if entry <= 0:
                return "UNKNOWN"
            
            # Stop kontrolü - daha sıkı
            if direction == "BULLISH":
                # BULLISH: Stop entry'nin altında olmalı
                if stop_loss > 0:
                    # Normal durum: stop < entry, fiyat stop'un altına düşerse stop
                    if stop_loss < entry and current_price < stop_loss:
                        return "STOPPED_OUT"
                    # Anormal durum: stop >= entry, fiyat entry'nin altına düşerse de stop sayılabilir
                    elif stop_loss >= entry and current_price < entry:
                        return "STOPPED_OUT"
                return "IN_TRADE" if current_price <= entry else "WAITING_ENTRY"
            else:
                # BEARISH: Stop entry'nin üstünde olmalı
                if stop_loss > 0:
                    # Normal durum: stop > entry, fiyat stop'un üstüne çıkarsa stop
                    if stop_loss > entry and current_price > stop_loss:
                        return "STOPPED_OUT"
                    # Anormal durum: stop <= entry, fiyat entry'nin üstüne çıkarsa da stop sayılabilir
                    elif stop_loss <= entry and current_price > entry:
                        return "STOPPED_OUT"
                return "IN_TRADE" if current_price >= entry else "WAITING_ENTRY"
        except Exception as e:
            logger.error(f"Error in get_pattern_status: {e}")
            return "UNKNOWN"
    
    def is_pattern_active(self, pattern_data: Dict, current_price: float) -> bool:
        """Pattern'in aktif olup olmadığını kontrol et"""
        try:
            trading_levels = pattern_data['trading_levels']
            entry = trading_levels['entry']
            stop_loss = trading_levels.get('stop_loss', 0)
            tp1 = trading_levels.get('tp1', 0)
            tp2 = trading_levels.get('tp2', 0)
            tp3 = trading_levels.get('tp3', 0)
            direction = pattern_data['type']
            distance_pct = trading_levels.get('distance_to_entry_pct', 100)
            
            # Stop olmuş pattern'leri gösterme - daha sıkı kontrol
            if direction == "BULLISH":
                # BULLISH: Stop entry'nin altında olmalı, fiyat stop'un altına düşerse stop olmuş
                if stop_loss > 0:
                    # Stop mantıklı mı kontrol et (entry'den düşük olmalı)
                    if stop_loss < entry and current_price < stop_loss:
                        return False
                    # Eğer stop entry'den yüksekse (yanlış hesaplama), entry'nin altına düşerse stop
                    elif stop_loss >= entry and current_price < entry:
                        return False
                # TP'lere ulaşmış mı kontrol et (tüm TP'ler geçilmişse)
                if tp3 > 0 and current_price >= tp3:
                    return False
            else:
                # BEARISH: Stop entry'nin üstünde olmalı, fiyat stop'un üstüne çıkarsa stop olmuş
                if stop_loss > 0:
                    # Stop mantıklı mı kontrol et (entry'den yüksek olmalı)
                    if stop_loss > entry and current_price > stop_loss:
                        return False
                    # Eğer stop entry'den düşükse (yanlış hesaplama), entry'nin üstüne çıkarsa stop
                    elif stop_loss <= entry and current_price > entry:
                        return False
                # TP'lere ulaşmış mı kontrol et (tüm TP'ler geçilmişse)
                if tp3 > 0 and current_price <= tp3:
                    return False
            
            # Çok uzak pattern'leri gösterme
            if distance_pct > self.max_distance_pct:
                return False
            
            if direction == "BULLISH":
                return current_price <= entry or abs(current_price - entry) / entry * 100 <= self.max_distance_pct
            else:
                return current_price >= entry or abs(current_price - entry) / entry * 100 <= self.max_distance_pct
        except:
            return False
    
    def scan_with_basic_method(self, df: pd.DataFrame, current_price: float) -> List[Dict]:
        """Temel pattern tarama yöntemi (harmonik modüller yoksa)"""
        patterns = []
        
        try:
            # Basit bir zigzag implementasyonu
            highs = df['high'].tolist()
            lows = df['low'].tolist()
            
            # Basit extremum tespiti
            extremas = []
            for i in range(1, len(highs)-1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    extremas.append({'index': i, 'value': highs[i], 'type': 'peak'})
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    extremas.append({'index': i, 'value': lows[i], 'type': 'valley'})
            
            # Son 5 extremum ile basit pattern kontrolü
            if len(extremas) >= 5:
                recent_extremas = extremas[-5:]
                
                # Basit Gartley pattern kontrolü
                pattern_data = self.check_basic_gartley(recent_extremas, df, current_price)
                if pattern_data:
                    patterns.append(pattern_data)
            
        except Exception as e:
            logger.error(f"Basic method scan error: {e}")
        
        return patterns
    
    def check_basic_gartley(self, extremas: List[Dict], df: pd.DataFrame, current_price: float) -> Optional[Dict]:
        """Basit Gartley pattern kontrolü"""
        try:
            if len(extremas) != 5:
                return None
                
            # Pattern noktaları
            X, A, B, C, D = extremas
            
            # Zaman bilgilerini al
            timestamps = df.index
            
            points = {
                'X': {
                    'index': X['index'],
                    'value': X['value'],
                    'time': int(timestamps[X['index']].timestamp()) if X['index'] < len(timestamps) else 0
                },
                'A': {
                    'index': A['index'],
                    'value': A['value'], 
                    'time': int(timestamps[A['index']].timestamp()) if A['index'] < len(timestamps) else 0
                },
                'B': {
                    'index': B['index'],
                    'value': B['value'],
                    'time': int(timestamps[B['index']].timestamp()) if B['index'] < len(timestamps) else 0
                },
                'C': {
                    'index': C['index'],
                    'value': C['value'],
                    'time': int(timestamps[C['index']].timestamp()) if C['index'] < len(timestamps) else 0
                },
                'D': {
                    'index': D['index'],
                    'value': D['value'],
                    'time': int(timestamps[D['index']].timestamp()) if D['index'] < len(timestamps) else 0
                }
            }
            
            ratios = self.calculate_ratios_from_points(points)
            
            # Basit Gartley oran kontrolü
            if (0.5 <= ratios.get('AB', 0) <= 0.8 and
                0.3 <= ratios.get('BC', 0) <= 0.9 and
                1.0 <= ratios.get('CD', 0) <= 1.5):
                
                direction = "BULLISH" if X['value'] < A['value'] else "BEARISH"
                trading_levels = self.calculate_trading_levels(points, direction, current_price)
                
                return {
                    'name': "Gartley",
                    'pattern_type': "Gartley", 
                    'type': direction,
                    'points': points,
                    'ratios': ratios,
                    'trading_levels': trading_levels,
                    'strength': 70.0,
                    'error_score': 0.1,
                    'status': self.get_pattern_status(trading_levels, current_price, direction),
                    'description': f"{direction} Gartley pattern (basic method)",
                    'method': 'basic'
                }
                
        except Exception as e:
            logger.error(f"Basic Gartley check error: {e}")
            
        return None
    
    def check_historical_stop(self, df: pd.DataFrame, pattern: Dict) -> bool:
        """Pattern entry'ye ulaştıktan sonra geçmişte stop'a düşmüş mü kontrol et"""
        try:
            points = pattern.get('points', {})
            trading_levels = pattern.get('trading_levels', {})
            direction = pattern.get('type', 'BULLISH')
            
            entry = trading_levels.get('entry', 0)
            stop_loss = trading_levels.get('stop_loss', 0)
            D_index = points.get('D', {}).get('index', -1)
            
            if D_index < 0 or entry <= 0 or stop_loss <= 0:
                return False
            
            # D noktasından (entry) sonraki tüm fiyatları kontrol et
            if D_index >= len(df):
                return False
            
            # Entry'ye ulaştıktan sonraki fiyatları kontrol et
            for i in range(D_index, len(df)):
                row = df.iloc[i]
                price_low = row['low']
                price_high = row['high']
                
                if direction == "BULLISH":
                    # BULLISH: Eğer low stop'un altına düşmüşse stop olmuş
                    if stop_loss < entry and price_low < stop_loss:
                        return True  # Stop'a düşmüş
                else:
                    # BEARISH: Eğer high stop'un üstüne çıkmışsa stop olmuş
                    if stop_loss > entry and price_high > stop_loss:
                        return True  # Stop'a çıkmış
            
            return False  # Stop'a düşmemiş
        except Exception as e:
            logger.error(f"Error checking historical stop: {e}")
            return False
    
    def scan_combined_patterns(self, df: pd.DataFrame, current_price: float) -> List[Dict]:
        """Tüm tarama yöntemlerini birleştir"""
        all_patterns = []
        
        # Orijinal yöntemle tarama (eğer modüller yüklüyse)
        if HAS_HARMONIC_MODULES:
            original_patterns = self.scan_with_original_method(df, current_price)
            all_patterns.extend(original_patterns)
        
        # Temel yöntemle tarama (her zaman kullanılabilir)
        basic_patterns = self.scan_with_basic_method(df, current_price)
        all_patterns.extend(basic_patterns)
        
        # Pattern'leri güce göre sırala ve duplicate'leri temizle
        unique_patterns = self.remove_duplicate_patterns(all_patterns)
        
        # Son filtreleme: Stop olmuş ve tamamlanmış pattern'leri çıkar
        active_patterns = []
        for pattern in unique_patterns:
            # Status'u güncelle
            trading_levels = pattern.get('trading_levels', {})
            direction = pattern.get('type', 'BULLISH')
            pattern['status'] = self.get_pattern_status(trading_levels, current_price, direction)
            
            # STOPPED_OUT durumundaki pattern'leri direkt filtrele
            if pattern.get('status') == 'STOPPED_OUT':
                continue
            
            # Geçmişte stop'a düşmüş mü kontrol et
            if self.check_historical_stop(df, pattern):
                continue  # Geçmişte stop'a düşmüş, filtrele
            
            # Aktif mi kontrol et (stop ve TP kontrolü dahil)
            if self.is_pattern_active(pattern, current_price):
                active_patterns.append(pattern)
        
        active_patterns.sort(key=lambda x: x.get('strength', 0), reverse=True)
        
        return active_patterns[:15]  # En iyi 15 pattern
    
    def remove_duplicate_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Benzer pattern'leri temizle"""
        unique_patterns = []
        seen_signatures = set()
        
        for pattern in patterns:
            try:
                # Pattern imzası oluştur (X ve D noktalarına göre)
                X_index = pattern['points']['X']['index']
                D_index = pattern['points']['D']['index']
                pattern_type = pattern['pattern_type']
                direction = pattern['type']
                
                signature = f"{X_index}_{D_index}_{pattern_type}_{direction}"
                
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    unique_patterns.append(pattern)
            except:
                continue
        
        return unique_patterns

# DataManager sınıfı
class DataManager:
    def __init__(self, db_path: str = "crypto_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL,
                UNIQUE(symbol, timeframe, timestamp)
            )
        ''')
        conn.commit()
        conn.close()
    
    def fetch_binance_data(self, symbol: str, timeframe: str, limit: int = 500, try_alternatives: bool = True) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Binance'den veri çek. Hata durumunda alternatif pair'leri dene.
        Hata durumunda (DataFrame, error_message) tuple döndür.
        """
        url = "https://api.binance.com/api/v3/klines"
        tf_mapping = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
        }
        
        original_symbol = symbol.strip().upper()
        
        # Symbol'ü temizle ve USDT ekle
        if original_symbol.endswith('USDT') or original_symbol.endswith('BTC') or original_symbol.endswith('ETH') or original_symbol.endswith('BNB'):
            symbol_upper = original_symbol
        else:
            symbol_upper = original_symbol + 'USDT'
        
        # Önce ana symbol'ü dene
        df, error_msg = self._fetch_single_symbol(url, symbol_upper, tf_mapping[timeframe], limit)
        if not df.empty:
            return df, None
        
        # Alternatif pair'leri dene (sadece USDT eklenmişse)
        if try_alternatives and not original_symbol.endswith(('USDT', 'BTC', 'ETH', 'BNB')):
            alternatives = ['BTC', 'ETH', 'BNB']
            for alt in alternatives:
                alt_symbol = original_symbol + alt
                df, _ = self._fetch_single_symbol(url, alt_symbol, tf_mapping[timeframe], limit)
                if not df.empty:
                    logger.info(f"Found data for {alt_symbol} instead of {symbol_upper}")
                    return df, None
        
        # Hiçbiri çalışmadıysa hata mesajını döndür
        return pd.DataFrame(), error_msg or f"Symbol {original_symbol} not found on Binance"
    
    def _fetch_single_symbol(self, url: str, symbol: str, interval: str, limit: int) -> Tuple[pd.DataFrame, Optional[str]]:
        """Tek bir symbol için veri çek"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            # HTTP hata kontrolü
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}"
                try:
                    error_data = response.json()
                    if isinstance(error_data, dict) and 'msg' in error_data:
                        error_msg = error_data['msg']
                except:
                    pass
                return pd.DataFrame(), error_msg
            
            data = response.json()
            
            # Binance API hata kontrolü (JSON içinde code ve msg varsa)
            if isinstance(data, dict) and 'code' in data:
                error_msg = data.get('msg', f"Binance API error: {data.get('code')}")
                return pd.DataFrame(), error_msg
            
            # Veri boş mu kontrol et
            if not data or len(data) == 0:
                error_msg = f"No data available for {symbol}"
                return pd.DataFrame(), error_msg
            
            # Veri listesi değilse hata
            if not isinstance(data, list):
                error_msg = f"Unexpected response format for {symbol}"
                return pd.DataFrame(), error_msg
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']], None
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error: {str(e)}"
            return pd.DataFrame(), error_msg
        except Exception as e:
            error_msg = f"Error processing data: {str(e)}"
            return pd.DataFrame(), error_msg
    
    def save_to_database(self, symbol: str, timeframe: str, df: pd.DataFrame):
        if df.empty:
            return
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            for _, row in df.iterrows():
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO ohlcv 
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, timeframe, 
                    int(row['timestamp'].timestamp()),
                    row['open'], row['high'], row['low'], 
                    row['close'], row['volume']
                ))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
        finally:
            conn.close()
    
    def get_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT timestamp, open, high, low, close, volume 
            FROM ohlcv 
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp
        '''
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
        conn.close()
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
        
        return df

# Flask uygulaması
app = Flask(__name__)
data_manager = DataManager()
pattern_scanner = IntegratedHarmonicScanner(error_threshold=0.3)

def determine_chart_start_index(df_length: int, patterns: List[Dict], 
                                default_limit: int = 250, buffer: int = 5) -> int:
    """
    Grafikte gösterilecek mumların başlangıç indeksini belirle.
    Varsayılan olarak son `default_limit` mumu gösteririz ancak
    mevcut pattern noktalarının grafikte görünmesini garanti etmek için
    en erken pattern noktasını da dikkate alırız.
    """
    chart_start = max(0, df_length - default_limit)
    
    if not patterns:
        return chart_start
    
    pattern_indices = []
    for pattern in patterns:
        points = pattern.get('points') or {}
        for point in points.values():
            idx = point.get('index')
            if isinstance(idx, (int, float)):
                pattern_indices.append(int(idx))
    
    if not pattern_indices:
        return chart_start
    
    earliest_index = max(0, min(pattern_indices) - buffer)
    
    # Pattern mumları grafikte yer alsın diye gerekirse daha erken başla
    return min(chart_start, earliest_index)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        symbol = data.get('symbol', 'BTCUSDT')
        timeframe = data.get('timeframe', '1h')
        
        # Veriyi çek ve kaydet
        df, error_msg = data_manager.fetch_binance_data(symbol, timeframe, 500)
        if df.empty:
            if error_msg:
                return jsonify({"error": error_msg})
            return jsonify({"error": f"Veri çekilemedi: {symbol.upper()}"})
        
        data_manager.save_to_database(symbol, timeframe, df)
        
        # Veritabanından oku
        df = data_manager.get_data(symbol, timeframe)
        if df.empty:
            return jsonify({"error": "Veritabanında veri yok"})
        
        current_price = df['close'].iloc[-1] if len(df) > 0 else 0
        
        # Pattern'leri tara
        patterns = pattern_scanner.scan_combined_patterns(df, current_price)
        
        # Sonuçları hazırla
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'current_price': current_price,
            'patterns_count': len(patterns),
            'patterns': patterns,
            'chart_data': [],
            'module_status': 'loaded' if HAS_HARMONIC_MODULES else 'fallback'
        }
        
        # Pattern noktalarının grafikte görünmesini garanti eden mum aralığını hesapla
        chart_start_index = determine_chart_start_index(len(df), patterns, default_limit=300, buffer=10)
        chart_df = df.iloc[chart_start_index:]
        
        for timestamp, row in chart_df.iterrows():
            try:
                if hasattr(timestamp, 'timestamp'):
                    timestamp_int = int(timestamp.timestamp())
                else:
                    timestamp_int = int(pd.to_datetime(timestamp).timestamp())
                
                result['chart_data'].append({
                    'time': timestamp_int,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close'])
                })
            except Exception as e:
                logger.error(f"Chart data processing error at timestamp {timestamp}: {e}")
                continue
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return jsonify({"error": str(e)})

@app.route('/backtest', methods=['POST'])
def backtest():
    """Basit backtest endpoint'i"""
    try:
        if not HAS_HARMONIC_MODULES:
            return jsonify({"error": "Backtest için harmonik modüller gerekli"})
            
        data = request.json
        symbol = data.get('symbol', 'BTCUSDT')
        timeframe = data.get('timeframe', '1h')
        
        df = data_manager.get_data(symbol, timeframe)
        if df.empty:
            return jsonify({"error": "Veri bulunamadı"})
        
        # Basit backtest
        df['r'] = np.log(df['close']).diff().shift(-1)
        all_combined = np.zeros(len(df))
        sigmas = [0.01, 0.015, 0.02]
        
        for sigma in sigmas:
            extremes = get_extremes(df, sigma)
            if not extremes.empty:
                output = find_xabcd(df, extremes, 0.5)
                sig = np.zeros(len(df))
                for pat in ALL_PATTERNS:
                    pat_name = pat.name
                    if pat_name in output:
                        sig += output[pat_name]['bear_signal'] + output[pat_name]['bull_signal']
                all_combined += sig
        
        if len(sigmas) > 0:
            all_combined /= len(sigmas)
        
        df['combined_signal'] = all_combined
        df['combined_returns'] = df['r'] * df['combined_signal']
        
        win_returns = df[df['combined_returns'] > 0]['combined_returns'].sum() 
        lose_returns = df[df['combined_returns'] < 0]['combined_returns'].abs().sum() 
        combined_pf = win_returns / lose_returns if lose_returns > 0 else 0
        
        return jsonify({
            'profit_factor': round(combined_pf, 3),
            'total_trades': len(df[df['combined_signal'] != 0]),
            'winning_trades': len(df[df['combined_returns'] > 0]),
            'losing_trades': len(df[df['combined_returns'] < 0])
        })
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return jsonify({"error": str(e)})

@app.route('/symbols')
def get_symbols():
    return jsonify(['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'SOLUSDT', 'DOTUSDT', 'DOGEUSDT'])

@app.route('/timeframes')  
def get_timeframes():
    return jsonify(['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'])

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'harmonic_modules_loaded': HAS_HARMONIC_MODULES,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print(f"Harmonik modül durumu: {'YÜKLENDİ' if HAS_HARMONIC_MODULES else 'FALLBACK MOD'}")
    app.run(debug=True, host='0.0.0.0', port=5000)