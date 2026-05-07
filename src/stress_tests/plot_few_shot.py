import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    df = pd.read_csv("few_shot_results.csv")
    
    plt.figure(figsize=(10, 6))
    
    # Map friendly names
    name_map = {
        'ultimate_ast_xgb': '20260302_ultimate (AST+XGBoost)',
        'yin_2021': 'Yin et al. 2021 (AlexNet)',
        'dorr_2026': 'Dörr et al. 2026 (BEATs)'
    }
    df['Friendly_Model'] = df['Model'].map(lambda x: name_map.get(x, x))
    
    # Convert Ratio to Percentage
    df['Train_Percentage'] = df['Train_Ratio'] * 100
    
    sns.lineplot(data=df, x='Train_Percentage', y='F1', hue='Friendly_Model', marker='o', linewidth=2.5, markersize=8)
    
    plt.title("Few-Shot Learning Curve: Data Scaling Robustness", fontsize=14)
    plt.xlabel("Percentage of Training Data Used (%)", fontsize=12)
    plt.ylabel("Macro F1-Score", fontsize=12)
    
    # Set y limit carefully
    plt.ylim(max(0, df['F1'].min() - 0.1), df['F1'].max() + 0.05)
    
    plt.xticks([25, 50, 75, 100])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Model Architecture')
    
    plt.tight_layout()
    plt.savefig("plots/few_shot_curve.png", dpi=300)
    print("Saved plots/few_shot_curve.png")

if __name__ == "__main__":
    import os
    os.makedirs("plots", exist_ok=True)
    main()
