import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from directional_change import get_extremes
from harmonic_patterns import find_xabcd, ALL_PATTERNS, plot_pattern

class HarmonicPatternAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.results = {}
        
    def analyze_multiple_sigmas(self, sigmas=None, err_threshold=0.2):
        """Birden fazla sigma değeri için analiz yapar"""
        if sigmas is None:
            sigmas = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
            
        all_combined = np.zeros(len(self.data))
        
        for sigma in sigmas:
            print(f"Processing sigma: {sigma}")
            extremes = get_extremes(self.data, sigma)
            output = find_xabcd(self.data, extremes, err_threshold)
            
            sig = np.zeros(len(self.data))
            for pat in ALL_PATTERNS:
                sig += output[pat.name]['bear_signal'] + output[pat.name]['bull_signal']
            
            all_combined += sig
            
        all_combined /= len(sigmas)
        return all_combined
    
    def calculate_performance(self, signals):
        """Strateji performansını hesaplar"""
        self.data['r'] = np.log(self.data['close']).diff().shift(-1)
        self.data['strategy_returns'] = self.data['r'] * signals
        
        win_returns = self.data[self.data['strategy_returns'] > 0]['strategy_returns'].sum()
        lose_returns = self.data[self.data['strategy_returns'] < 0]['strategy_returns'].abs().sum()
        
        pf = win_returns / lose_returns if lose_returns != 0 else 0
        total_return = self.data['strategy_returns'].sum()
        win_rate = len(self.data[self.data['strategy_returns'] > 0]) / len(self.data[self.data['strategy_returns'] != 0])
        
        return {
            'profit_factor': pf,
            'total_return': total_return,
            'win_rate': win_rate,
            'num_trades': len(self.data[self.data['strategy_returns'] != 0])
        }
    
    def plot_cumulative_returns(self, signals, title="Cumulative Returns"):
        """Kümülatif getirileri çizer"""
        self.data['r'] = np.log(self.data['close']).diff().shift(-1)
        self.data['strategy_returns'] = self.data['r'] * signals
        
        plt.figure(figsize=(15, 8))
        plt.style.use('dark_background')
        
        cumulative_returns = self.data['strategy_returns'].cumsum()
        cumulative_returns.plot(color='cyan', linewidth=2)
        
        plt.title(title, fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Log Returns')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def pattern_performance_analysis(self, sigma=0.02, err_threshold=0.2):
        """Pattern bazlı performans analizi"""
        extremes = get_extremes(self.data, sigma)
        output = find_xabcd(self.data, extremes, err_threshold)
        
        results = []
        for pat in ALL_PATTERNS:
            sig = output[pat.name]['bear_signal'] + output[pat.name]['bull_signal']
            count = len(output[pat.name]['bear_patterns']) + len(output[pat.name]['bull_patterns'])
            
            if count > 0:
                rets = self.data['r'] * sig
                pf = rets[rets > 0].sum() / rets[rets < 0].abs().sum() if rets[rets < 0].abs().sum() > 0 else 0
                pf = 1.0 if np.isnan(pf) else min(pf, 4.0)  # Cap at 4.0 for visualization
            else:
                pf = 1.0
                
            results.append({
                'Pattern': pat.name,
                'Profit_Factor': pf,
                'Count': count,
                'Sigma': sigma
            })
        
        return pd.DataFrame(results)

if __name__ == '__main__':
    # Örnek kullanım
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    
    analyzer = HarmonicPatternAnalyzer(data)
    
    # Kombine sinyal analizi
    combined_signal = analyzer.analyze_multiple_sigmas()
    performance = analyzer.calculate_performance(combined_signal)
    
    print("Performance Metrics:")
    for key, value in performance.items():
        print(f"{key}: {value:.4f}")
    
    # Grafik çiz
    analyzer.plot_cumulative_returns(combined_signal, "Combined Harmonic Patterns Strategy")