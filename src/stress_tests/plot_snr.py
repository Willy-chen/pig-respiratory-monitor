"""
plot_snr.py — SNR robustness curve plotter.
Reads snr_results_aggregated.csv and produces:
  - One line plot per noise type (Farm, White, Pink)
  - X-axis: SNR level (Clean → +10 → 0 → -5 → -10 dB)
  - Y-axis: Macro F1-score
  - Each model is a separate line
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SNR_ORDER = [None, 10, 0, -5, -10]
SNR_LABELS = ['Clean', '+10 dB', '0 dB', '−5 dB', '−10 dB']

FRIENDLY = {
    'ultimate_ast_xgb': '20260302_ultimate (AST+XGBoost)',
    'yin_2021':          'Yin 2021 (AlexNet)',
    'dorr_2026':         'Dörr 2026 (BEATs)',
    'sheikh_2024':       'Sheikh 2024 (Whisper)',
    'wu_2022':           'Wu 2022 (Bi-LSTM)',
    'shen_2022':         'Shen 2022 (LeNet+SVM)',
    'hou_2024':          'Hou 2024 (BP-MLP)',
    'wang_2026':         'Wang 2026 (CoughRNet)',
    'mdpi_2026':         'MDPI 2026 (Dual-CNN)',
    'nithin_2026':       'Nithin 2026 (LSTM-KAN)',
}

# Highlight the ultimate model with a thick red line
HIGHLIGHT = 'ultimate_ast_xgb'

def plot_for_noise_type(df_noise, noise_type, out_path):
    models = df_noise['Model'].unique()
    x = list(range(len(SNR_ORDER)))

    fig, ax = plt.subplots(figsize=(11, 7))

    for model in models:
        sub = df_noise[df_noise['Model'] == model]
        # Align to SNR order
        y_vals = []
        for snr in SNR_ORDER:
            row = sub[sub['SNR_dB'] == snr] if snr is not None else sub[sub['SNR_dB'].isna()]
            y_vals.append(row['F1'].values[0] if len(row) else np.nan)

        label = FRIENDLY.get(model, model)
        if model == HIGHLIGHT:
            ax.plot(x, y_vals, label=label, linewidth=3.0, marker='*',
                    markersize=12, color='crimson', zorder=5)
        else:
            ax.plot(x, y_vals, label=label, linewidth=1.5, marker='o', markersize=6, alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(SNR_LABELS, fontsize=11)
    ax.set_xlabel('Noise Level (SNR)', fontsize=13)
    ax.set_ylabel('Macro F1-Score', fontsize=13)
    ax.set_title(f'SNR Robustness — {noise_type} Noise', fontsize=15)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")


def main():
    os.makedirs("plots", exist_ok=True)

    if not os.path.exists("snr_results_aggregated.csv"):
        print("snr_results_aggregated.csv not found. Run run_snr.py first.")
        return

    df = pd.read_csv("snr_results_aggregated.csv")
    # nan SNR is "Clean"
    df['SNR_dB'] = df['SNR_dB'].where(df['SNR_dB'].notna(), None)

    for noise_type in df['Noise'].unique():
        sub = df[df['Noise'] == noise_type]
        out = f"plots/snr_{noise_type.lower()}.png"
        plot_for_noise_type(sub, noise_type, out)

    # Combined summary: mean across noise types per model/SNR
    summary = df.groupby(['Model', 'SNR_dB'])['F1'].mean().reset_index()
    summary['Noise'] = 'All (Mean)'
    plot_for_noise_type(summary, 'All Noise Types (Mean)', "plots/snr_combined.png")

if __name__ == "__main__":
    main()
