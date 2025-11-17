import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from directional_change import get_extremes, plot_directional_change
from harmonic_patterns import find_xabcd, ALL_PATTERNS, plot_pattern
from pattern_analyzer import HarmonicPatternAnalyzer

def main():
    # Veriyi yükle
    print("Loading data...")
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    
    # Analyzer oluştur
    analyzer = HarmonicPatternAnalyzer(data)
    
    # 1. Directional Change Görselleştirme
    print("1. Directional Change Analysis")
    sigma = 0.02
    extremes = get_extremes(data, sigma)
    print(f"Found {len(extremes)} extremes with sigma={sigma}")
    
    # 2. Harmonik Pattern Analizi
    print("\n2. Harmonic Pattern Analysis")
    output = find_xabcd(data, extremes, err_threshold=0.2)
    
    # Pattern sayılarını yazdır
    for pat in ALL_PATTERNS:
        bull_count = len(output[pat.name]['bull_patterns'])
        bear_count = len(output[pat.name]['bear_patterns'])
        print(f"{pat.name}: {bull_count} bull, {bear_count} bear patterns")
    
    # 3. Kombine Strateji Testi
    print("\n3. Combined Strategy Performance")
    combined_signal = analyzer.analyze_multiple_sigmas()
    performance = analyzer.calculate_performance(combined_signal)
    
    print("Performance Summary:")
    for key, value in performance.items():
        print(f"  {key}: {value:.4f}")
    
    # 4. Grafikler
    print("\n4. Generating plots...")
    analyzer.plot_cumulative_returns(combined_signal)
    
    # 5. Pattern bazlı performans
    print("\n5. Pattern-wise Performance")
    pattern_df = analyzer.pattern_performance_analysis()
    print(pattern_df)
    
    # 6. İlk bulunan patterni görselleştir
    for pat in ALL_PATTERNS:
        if output[pat.name]['bull_patterns']:
            print(f"\nPlotting first {pat.name} pattern...")
            plot_pattern(data, output[pat.name]['bull_patterns'][0])
            break

if __name__ == '__main__':
    main()