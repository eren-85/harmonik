import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import requests
import warnings
from datetime import datetime, timedelta
import os
import json

warnings.filterwarnings('ignore')

# =============================================================================
# BINANCE API - GELİŞMİŞ DATA MANAGER
# =============================================================================

class BinanceDataManager:
    """Binance veri yöneticisi - Cache ve file storage ile"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.data_dir = "binance_data"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def get_klines(self, symbol: str, interval: str = "1d", limit: int = 1000) -> pd.DataFrame:
        """Binance'ten OHLCV verisi al - Cache ile"""
        try:
            symbol = symbol.strip().upper()
            if not symbol.endswith('USDT'):
                symbol += 'USDT'
            
            print(f"Veri çekiliyor: {symbol}, {interval}, {limit} bar")
            
            # Cache dosyasını kontrol et
            cache_file = os.path.join(self.data_dir, f"{symbol}_{interval}.csv")
            
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file, index_col='OpenTime', parse_dates=True)
                cached_count = len(df)
                print(f"Cache'ten {cached_count} bar yüklendi")
                
                # Eğer cached veri yeterliyse direkt dön
                if cached_count >= limit:
                    return df.tail(limit)
            
            # API'den yeni veri çek
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if not data:
                raise ValueError(f"{symbol} için veri bulunamadı")
            
            # DataFrame oluştur
            df = pd.DataFrame(data, columns=[
                'OpenTime', 'Open', 'High', 'Low', 'Close', 'Volume',
                'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades',
                'TakerBuyBaseVolume', 'TakerBuyQuoteVolume', 'Ignore'
            ])
            
            # Tarih ve sayısal dönüşüm
            df['OpenTime'] = pd.to_datetime(df['OpenTime'], unit='ms')
            df.set_index('OpenTime', inplace=True)
            
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # NaN değerleri temizle
            df = df.dropna()
            
            if df.empty:
                raise ValueError(f"{symbol} için geçerli veri bulunamadı")
            
            # Cache'e kaydet
            df.to_csv(cache_file)
            print(f"Başarıyla çekildi ve cache'lendi: {len(df)} bar")
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            raise Exception(f"Veri çekme hatası: {e}")
    
    def get_all_timeframes(self):
        """Tüm timeframe'leri getir"""
        return ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]

# =============================================================================
# DATA STRUCTURES - DÜZELTİLMİŞ
# =============================================================================

@dataclass
class PatternDescriptor:
    """Pattern tanımı - DÜZELTİLMİŞ"""
    ab2xa_min: float = 0
    ab2xa_max: float = 0
    bc2ab_min: float = 0
    bc2ab_max: float = 0
    cd2bc_min: float = 0
    cd2bc_max: float = 0
    ad2xa_min: float = 0
    ad2xa_max: float = 0

@dataclass 
class PatternMatch:
    """Pattern eşleşme verisi"""
    pattern_name: str = ""
    XIndex: int = 0
    AIndex: int = 0
    BIndex: int = 0 
    CIndex: int = 0
    DIndex: int = 0
    bullish: bool = True
    quality: float = 0.0

# =============================================================================
# ZIGZAG INDICATOR - GELİŞTİRİLMİŞ
# =============================================================================

class ImprovedZigZag:
    """Geliştirilmiş ZigZag indikatörü"""
    
    def __init__(self, depth=12, deviation=5, backstep=3):
        self.depth = depth
        self.deviation = deviation
        self.backstep = backstep
        
    def calculate(self, high, low):
        """Geliştirilmiş ZigZag hesaplama"""
        n = len(high)
        if n < self.depth:
            return [0]*n, [0]*n
            
        peaks = [0.0] * n
        troughs = [0.0] * n
        
        last_high = high[0]
        last_low = low[0]
        last_high_idx = 0
        last_low_idx = 0
        
        for i in range(1, n):
            # Yüksek nokta kontrolü
            if high[i] > last_high:
                last_high = high[i]
                last_high_idx = i
            elif (last_high - high[i]) / last_high * 100 >= self.deviation:
                if last_high_idx > 0:
                    peaks[last_high_idx] = last_high
                last_high = high[i]
                last_high_idx = i
                
            # Düşük nokta kontrolü
            if low[i] < last_low:
                last_low = low[i]
                last_low_idx = i
            elif (low[i] - last_low) / last_low * 100 >= self.deviation:
                if last_low_idx > 0:
                    troughs[last_low_idx] = last_low
                last_low = low[i]
                last_low_idx = i
        
        return peaks, troughs

# =============================================================================
# PATTERN MATCHER - HATA DÜZELTMELİ
# =============================================================================

class ImprovedPatternMatcher:
    """Geliştirilmiş pattern eşleştirici - Hata düzeltmeli"""
    
    def __init__(self, slack=0.1):
        self.slack = slack
        
    def find_patterns(self, patterns_dict: Dict[str, PatternDescriptor], high: list, low: list) -> List[PatternMatch]:
        """Patternleri bul - DÜZELTİLMİŞ"""
        matches = []
        
        # ZigZag noktalarını bul
        zigzag = ImprovedZigZag()
        peaks, troughs = zigzag.calculate(high, low)
        
        swing_points = self._get_swing_points(peaks, troughs)
        
        if len(swing_points) < 5:
            return matches
        
        # Her pattern için kontrol et
        for pattern_name, pattern in patterns_dict.items():
            pattern_matches = self._check_pattern(pattern, pattern_name, swing_points, high, low)
            matches.extend(pattern_matches)
            
        return matches
    
    def _get_swing_points(self, peaks, troughs):
        """Swing noktalarını topla"""
        points = []
        
        for i, val in enumerate(peaks):
            if val > 0:
                points.append((i, val, 'peak'))
                
        for i, val in enumerate(troughs):
            if val > 0:
                points.append((i, val, 'trough'))
                
        points.sort(key=lambda x: x[0])
        return points
    
    def _check_pattern(self, pattern: PatternDescriptor, pattern_name: str, swing_points: list, high: list, low: list) -> List[PatternMatch]:
        """Pattern kontrolü - DÜZELTİLMİŞ"""
        matches = []
        
        for i in range(len(swing_points) - 4):
            points = swing_points[i:i+5]
            
            # Zaman sırası kontrolü
            indices = [p[0] for p in points]
            if not all(x < y for x, y in zip(indices, indices[1:])):
                continue
                
            # Bullish pattern kontrolü (X-A-B-C-D: Low-High-Low-High-Low)
            if (points[0][2] == 'trough' and points[1][2] == 'peak' and 
                points[2][2] == 'trough' and points[3][2] == 'peak' and 
                points[4][2] == 'trough'):
                
                quality = self._calculate_quality(pattern, points, high, low, True)
                if quality > 0.6:
                    match = PatternMatch(
                        pattern_name=pattern_name,
                        XIndex=points[0][0], AIndex=points[1][0], 
                        BIndex=points[2][0], CIndex=points[3][0], DIndex=points[4][0],
                        bullish=True,
                        quality=quality
                    )
                    matches.append(match)
            
            # Bearish pattern kontrolü (X-A-B-C-D: High-Low-High-Low-High)
            elif (points[0][2] == 'peak' and points[1][2] == 'trough' and 
                  points[2][2] == 'peak' and points[3][2] == 'trough' and 
                  points[4][2] == 'peak'):
                
                quality = self._calculate_quality(pattern, points, high, low, False)
                if quality > 0.6:
                    match = PatternMatch(
                        pattern_name=pattern_name,
                        XIndex=points[0][0], AIndex=points[1][0], 
                        BIndex=points[2][0], CIndex=points[3][0], DIndex=points[4][0],
                        bullish=False,
                        quality=quality
                    )
                    matches.append(match)
                    
        return matches
    
    def _calculate_quality(self, pattern: PatternDescriptor, points: list, high: list, low: list, bullish: bool) -> float:
        """Kalite hesapla - DÜZELTİLMİŞ"""
        try:
            X_idx, A_idx, B_idx, C_idx, D_idx = [p[0] for p in points]
            
            if bullish:
                X = low[X_idx]; A = high[A_idx]; B = low[B_idx]; C = high[C_idx]; D = low[D_idx]
            else:
                X = high[X_idx]; A = low[A_idx]; B = high[B_idx]; C = low[C_idx]; D = high[D_idx]
            
            # Oranları hesapla
            XA = abs(A - X)
            AB = abs(B - A) 
            BC = abs(C - B)
            CD = abs(D - C)
            
            if XA == 0:
                return 0.0
                
            ratios = []
            targets = []
            
            # AB/XA
            ab_xa = AB / XA
            if pattern.ab2xa_min > 0:
                ratios.append(ab_xa)
                targets.append((pattern.ab2xa_min + pattern.ab2xa_max) / 2)
            
            # BC/AB
            if AB > 0:
                bc_ab = BC / AB
                if pattern.bc2ab_min > 0:
                    ratios.append(bc_ab)
                    targets.append((pattern.bc2ab_min + pattern.bc2ab_max) / 2)
            
            # CD/BC  
            if BC > 0:
                cd_bc = CD / BC
                if pattern.cd2bc_min > 0:
                    ratios.append(cd_bc)
                    targets.append((pattern.cd2bc_min + pattern.cd2bc_max) / 2)
            
            # Kalite hesapla
            quality = 1.0
            for ratio, target in zip(ratios, targets):
                if target > 0:
                    deviation = abs(ratio - target) / target
                    quality *= max(0.1, 1 - min(deviation, 1.0))
                
            return quality
            
        except Exception as e:
            print(f"Kalite hesaplama hatası: {e}")
            return 0.0

# =============================================================================
# ANA INDICATOR - TAMİR EDİLMİŞ
# =============================================================================

class HarmonicPatternFinder:
    """Harmonik pattern bulucu - TAMİR EDİLMİŞ"""
    
    def __init__(self):
        self.matcher = ImprovedPatternMatcher()
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> Dict[str, PatternDescriptor]:
        """Pattern tanımları - DÜZELTİLMİŞ"""
        return {
            'Gartley': PatternDescriptor(
                ab2xa_min=0.618, ab2xa_max=0.618,
                bc2ab_min=0.382, bc2ab_max=0.886,
                cd2bc_min=1.272, cd2bc_max=1.618
            ),
            'Bat': PatternDescriptor(
                ab2xa_min=0.382, ab2xa_max=0.5,
                bc2ab_min=0.382, bc2ab_max=0.886,
                cd2bc_min=1.618, cd2bc_max=2.618
            ),
            'Butterfly': PatternDescriptor(
                ab2xa_min=0.786, ab2xa_max=0.786,
                bc2ab_min=0.382, bc2ab_max=0.886,
                cd2bc_min=1.618, cd2bc_max=2.618
            ),
            'Crab': PatternDescriptor(
                ab2xa_min=0.382, ab2xa_max=0.618,
                bc2ab_min=0.382, bc2ab_max=0.886,
                cd2bc_min=2.24, cd2bc_max=3.618
            ),
            'Cypher': PatternDescriptor(
                ab2xa_min=0.382, ab2xa_max=0.618,
                bc2ab_min=1.13, bc2ab_max=1.414,
                cd2bc_min=1.272, cd2bc_max=2.0
            ),
            'Shark': PatternDescriptor(
                ab2xa_min=0.886, ab2xa_max=1.13,
                bc2ab_min=1.618, bc2ab_max=2.24,
                cd2bc_min=0.886, cd2bc_max=1.13
            )
        }
    
    def analyze(self, df: pd.DataFrame, pattern_names: List[str] = None) -> List[PatternMatch]:
        """Analiz yap - DÜZELTİLMİŞ"""
        if pattern_names is None:
            pattern_names = list(self.patterns.keys())
            
        selected_patterns = {name: self.patterns[name] for name in pattern_names 
                           if name in self.patterns}
        
        high = df['High'].values
        low = df['Low'].values
        
        return self.matcher.find_patterns(selected_patterns, high, low)

# =============================================================================
# CANDLESTICK CHART - MUM GRAFİĞİ DESTEĞİ
# =============================================================================

def plot_candlestick(ax, df, width=0.6, width2=0.1):
    """Mum grafiği çiz"""
    # Renkleri belirle
    up = df['Close'] >= df['Open']
    down = df['Open'] > df['Close']
    
    # Renkleri ayarla
    up_color = 'lime'
    down_color = 'red'
    
    # Fitiller
    ax.vlines(df.index[up], df['Low'][up], df['High'][up], color='black', linewidth=0.5)
    ax.vlines(df.index[down], df['Low'][down], df['High'][down], color='black', linewidth=0.5)
    
    # Gövdeler - yeşil mumlar
    if up.any():
        ax.bar(df.index[up], df['Close'][up] - df['Open'][up], width, 
               bottom=df['Open'][up], color=up_color, edgecolor='black', linewidth=0.5)
    
    # Gövdeler - kırmızı mumlar  
    if down.any():
        ax.bar(df.index[down], df['Open'][down] - df['Close'][down], width, 
               bottom=df['Close'][down], color=down_color, edgecolor='black', linewidth=0.5)

# =============================================================================
# USER INTERFACE - MUM GRAFİĞİ İLE
# =============================================================================

class CryptoHarmonicScanner:
    """Crypto Harmonic Pattern Scanner - MUM GRAFİĞİ ile"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Crypto Harmonic Scanner v4.0 - Candlestick")
        self.root.geometry("1600x1000")
        
        # Değişkenler
        self.finder = HarmonicPatternFinder()
        self.data_manager = BinanceDataManager()
        self.current_data = None
        self.current_symbol = "BTCUSDT"
        self.current_interval = "4h"
        
        # UI'yı kur
        self._setup_ui()
        self._initialize_app()
    
    def _setup_ui(self):
        """UI'yi oluştur"""
        # Ana frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Üst kontrol paneli
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Sembol girişi
        ttk.Label(control_frame, text="Coin Pair:", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=(0, 5))
        self.symbol_var = tk.StringVar(value=self.current_symbol)
        self.symbol_entry = ttk.Entry(control_frame, textvariable=self.symbol_var, width=15)
        self.symbol_entry.grid(row=0, column=1, padx=(0, 15))
        self.symbol_entry.bind('<Return>', lambda e: self._load_data())
        
        # Zaman aralığı - TÜM TIMEFRAME'ler
        ttk.Label(control_frame, text="Timeframe:", font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=(0, 5))
        self.interval_var = tk.StringVar(value=self.current_interval)
        self.interval_combo = ttk.Combobox(control_frame, textvariable=self.interval_var, 
                                          values=self.data_manager.get_all_timeframes(),
                                          width=8, state="readonly")
        self.interval_combo.grid(row=0, column=3, padx=(0, 15))
        self.interval_combo.bind('<<ComboboxSelected>>', lambda e: self._load_data())
        
        # Butonlar
        ttk.Button(control_frame, text="Load Data (1000 bars)", 
                  command=self._load_data, style='Accent.TButton').grid(row=0, column=4, padx=(0, 10))
        ttk.Button(control_frame, text="Scan Patterns", 
                  command=self._analyze, style='Accent.TButton').grid(row=0, column=5, padx=(0, 10))
        
        # Progress
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.grid(row=0, column=6, padx=(10, 0), sticky=tk.EW)
        
        control_frame.columnconfigure(6, weight=1)
        
        # Ana içerik alanı
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Sol panel - Ayarlar
        left_panel = ttk.LabelFrame(content_frame, text="Pattern Settings", padding="10", width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Pattern seçimleri
        ttk.Label(left_panel, text="Select Patterns:", font=('Arial', 11, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        self.pattern_vars = {}
        for pattern in self.finder.patterns.keys():
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(left_panel, text=pattern, variable=var)
            cb.pack(anchor=tk.W, pady=2)
            self.pattern_vars[pattern] = var
        
        # Kontrol butonları
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=(15, 0))
        
        ttk.Button(btn_frame, text="Select All", 
                  command=self._select_all).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(btn_frame, text="Deselect All", 
                  command=self._deselect_all).pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # Kalite filtresi
        ttk.Label(left_panel, text="Minimum Quality:", font=('Arial', 11, 'bold')).pack(anchor=tk.W, pady=(20, 5))
        self.quality_var = tk.DoubleVar(value=0.7)
        quality_scale = ttk.Scale(left_panel, from_=0.1, to=1.0, 
                                 variable=self.quality_var, orient=tk.HORIZONTAL)
        quality_scale.pack(fill=tk.X)
        
        ttk.Label(left_panel, text=f"Current: {self.quality_var.get():.1f}").pack(anchor=tk.CENTER)
        self.quality_var.trace('w', lambda *args: self._update_quality_label())
        
        # İstatistikler
        stats_frame = ttk.LabelFrame(left_panel, text="Scan Results", padding="5")
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        self.stats_text = tk.Text(stats_frame, height=20, width=30, font=('Consolas', 9))
        scrollbar = ttk.Scrollbar(stats_frame, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=scrollbar.set)
        
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Sağ panel - Grafik
        right_panel = ttk.LabelFrame(content_frame, text="Candlestick Chart", padding="10")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Matplotlib figure - MUM GRAFİĞİ için
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                                     gridspec_kw={'height_ratios': [3, 1]},
                                                     facecolor='#1e1e1e')
        self.fig.patch.set_facecolor('#1e1e1e')
        
        self.canvas = FigureCanvasTkAgg(self.fig, right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Stil ayarları
        self._configure_styles()
    
    def _configure_styles(self):
        """UI stillerini ayarla"""
        style = ttk.Style()
        style.configure('Accent.TButton', background='#007acc', foreground='white')
    
    def _update_quality_label(self):
        """Kalite label'ını güncelle"""
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Label) and child.cget('text').startswith('Current:'):
                        child.config(text=f"Current: {self.quality_var.get():.1f}")
                        break
    
    def _select_all(self):
        """Tüm patternleri seç"""
        for var in self.pattern_vars.values():
            var.set(True)
    
    def _deselect_all(self):
        """Tüm patternleri kaldır"""
        for var in self.pattern_vars.values():
            var.set(False)
    
    def _initialize_app(self):
        """Uygulamayı başlat"""
        self._load_data()
    
    def _load_data(self):
        """Veri yükle - 1000 BAR"""
        self.progress.start()
        
        try:
            symbol = self.symbol_var.get().strip().upper()
            interval = self.interval_var.get()
            
            if not symbol:
                messagebox.showerror("Error", "Lütfen bir coin sembolü girin!")
                return
            
            print(f"1000 bar veri yükleniyor: {symbol}, {interval}")
            self.current_data = self.data_manager.get_klines(symbol, interval, 1000)
            
            if self.current_data is not None and not self.current_data.empty:
                self.current_symbol = symbol
                self._update_chart()
                self._update_status(f"✓ Veri yüklendi: {symbol} {interval}\n✓ Bar sayısı: {len(self.current_data)}\n✓ Son fiyat: {self.current_data['Close'].iloc[-1]:.4f}")
            else:
                messagebox.showerror("Error", f"{symbol} için veri bulunamadı!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Veri yükleme hatası: {str(e)}")
        finally:
            self.progress.stop()
    
    def _analyze(self):
        """Pattern analizi yap"""
        if self.current_data is None or self.current_data.empty:
            messagebox.showerror("Error", "Önce veri yükleyin!")
            return
        
        self.progress.start()
        
        try:
            # Seçili patternler
            selected_patterns = [p for p, var in self.pattern_vars.items() if var.get()]
            if not selected_patterns:
                messagebox.showerror("Error", "En az bir pattern seçin!")
                return
            
            # Analiz yap
            matches = self.finder.analyze(self.current_data, selected_patterns)
            
            # Kalite filtresi
            min_quality = self.quality_var.get()
            filtered_matches = [m for m in matches if m.quality >= min_quality]
            
            # Sonuçları göster
            self._update_chart(filtered_matches)
            self._update_stats(filtered_matches)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analiz hatası: {str(e)}")
        finally:
            self.progress.stop()
    
    def _update_chart(self, matches=None):
        """Grafiği güncelle - MUM GRAFİĞİ ile"""
        if self.current_data is None:
            return
            
        self.ax1.clear()
        self.ax2.clear()
        
        df = self.current_data
        
        # Mum grafiği çiz
        plot_candlestick(self.ax1, df)
        
        # Patternleri çiz
        if matches:
            for match in matches:
                self._draw_pattern(match, df)
        
        # Volume
        colors = ['lime' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' 
                 for i in range(len(df))]
        self.ax2.bar(df.index, df['Volume'], color=colors, alpha=0.6, width=0.8)
        
        # Styling
        self.ax1.set_facecolor('#1e1e1e')
        self.ax2.set_facecolor('#1e1e1e')
        
        for ax in [self.ax1, self.ax2]:
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.2, color='white')
            ax.set_xlim(df.index[0], df.index[-1])
        
        self.ax1.set_title(f"{self.current_symbol} {self.current_interval} - Harmonic Patterns", 
                          color='white', fontsize=14, fontweight='bold')
        self.ax1.set_ylabel("Price", color='white')
        self.ax2.set_ylabel("Volume", color='white')
        
        # Tarih formatı
        self.fig.autofmt_xdate()
        self.canvas.draw()
    
    def _draw_pattern(self, match: PatternMatch, df: pd.DataFrame):
        """Pattern çiz - MUM GRAFİĞİ üzerine"""
        try:
            indices = [match.XIndex, match.AIndex, match.BIndex, match.CIndex, match.DIndex]
            dates = [df.index[i] for i in indices]
            
            if match.bullish:
                prices = [
                    df['Low'].iloc[match.XIndex],
                    df['High'].iloc[match.AIndex], 
                    df['Low'].iloc[match.BIndex],
                    df['High'].iloc[match.CIndex],
                    df['Low'].iloc[match.DIndex]
                ]
                color = 'lime'
            else:
                prices = [
                    df['High'].iloc[match.XIndex],
                    df['Low'].iloc[match.AIndex],
                    df['High'].iloc[match.BIndex],
                    df['Low'].iloc[match.CIndex],
                    df['High'].iloc[match.DIndex]
                ]
                color = 'red'
            
            # Pattern çizgileri
            line_style = '-' if match.quality > 0.8 else '--'
            line_width = 2.5 if match.quality > 0.8 else 1.5
            
            self.ax1.plot(dates, prices, color=color, linestyle=line_style, 
                         linewidth=line_width, marker='o', 
                         markersize=6, markerfacecolor=color, 
                         markeredgecolor='white', markeredgewidth=1)
            
            # Pattern ismi
            direction = "BULL" if match.bullish else "BEAR"
            self.ax1.annotate(f"{match.pattern_name} {direction}\nQ:{match.quality:.2f}", 
                             xy=(dates[-1], prices[-1]),
                             xytext=(10, -20 if match.bullish else 20), 
                             textcoords='offset points',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9),
                             fontsize=9, color='white', fontweight='bold',
                             arrowprops=dict(arrowstyle='->', color='white', alpha=0.7))
            
        except Exception as e:
            print(f"Pattern çizim hatası: {e}")
    
    def _update_stats(self, matches):
        """İstatistikleri güncelle"""
        self.stats_text.delete(1.0, tk.END)
        
        if not matches:
            self.stats_text.insert(tk.END, "🚫 No patterns found\n")
            self.stats_text.insert(tk.END, f"Minimum quality: {self.quality_var.get():.1f}\n")
            return
        
        # İstatistikleri hesapla
        pattern_counts = {}
        bullish_count = sum(1 for m in matches if m.bullish)
        bearish_count = len(matches) - bullish_count
        
        for match in matches:
            pattern_name = match.pattern_name
            pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1
        
        # Sonuçları göster
        self.stats_text.insert(tk.END, "✅ SCAN RESULTS\n\n")
        self.stats_text.insert(tk.END, f"Symbol: {self.current_symbol}\n")
        self.stats_text.insert(tk.END, f"Timeframe: {self.current_interval}\n")
        self.stats_text.insert(tk.END, f"Total Patterns: {len(matches)}\n")
        self.stats_text.insert(tk.END, f"Bullish: {bullish_count}\n")
        self.stats_text.insert(tk.END, f"Bearish: {bearish_count}\n")
        self.stats_text.insert(tk.END, f"Min Quality: {self.quality_var.get():.1f}\n\n")
        
        # Kalite istatistikleri
        if matches:
            qualities = [m.quality for m in matches]
            avg_quality = np.mean(qualities)
            max_quality = max(qualities)
            self.stats_text.insert(tk.END, f"Avg Quality: {avg_quality:.2f}\n")
            self.stats_text.insert(tk.END, f"Max Quality: {max_quality:.2f}\n\n")
        
        self.stats_text.insert(tk.END, "📊 Pattern Distribution:\n")
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(matches)) * 100
            self.stats_text.insert(tk.END, f"• {pattern}: {count} ({percentage:.1f}%)\n")
        
        # High quality patterns
        high_quality = [m for m in matches if m.quality > 0.8]
        if high_quality:
            self.stats_text.insert(tk.END, f"\n🎯 High Quality (Q>0.8): {len(high_quality)}\n")
            
        # Best pattern
        if matches:
            best_match = max(matches, key=lambda x: x.quality)
            self.stats_text.insert(tk.END, f"\n🏆 Best Pattern:\n")
            self.stats_text.insert(tk.END, f"• {best_match.pattern_name}\n")
            self.stats_text.insert(tk.END, f"• {('BULLISH' if best_match.bullish else 'BEARISH')}\n")
            self.stats_text.insert(tk.END, f"• Quality: {best_match.quality:.2f}\n")
    
    def _update_status(self, message):
        """Durum güncelle"""
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, message)

# =============================================================================
# UYGULAMAYI BAŞLAT
# =============================================================================

def main():
    """Ana uygulama"""
    try:
        root = tk.Tk()
        app = CryptoHarmonicScanner(root)
        root.mainloop()
    except Exception as e:
        print(f"Uygulama hatası: {e}")
        messagebox.showerror("Error", f"Uygulama başlatılamadı: {e}")

if __name__ == "__main__":
    main()