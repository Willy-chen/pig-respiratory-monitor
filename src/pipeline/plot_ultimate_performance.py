import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

LOG_PATH = "results/joint_optimization_log.csv"
OUTPUT_DIR = "results/plots"

def main():
    if not os.path.exists(LOG_PATH):
        print(f"Log file not found at {LOG_PATH}. Run run_ultimate_search.py first.")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(pd.read_csv(LOG_PATH))
    
    # 1. Best Overall Highlight
    best = df.loc[df['F1'].idxmax()]
    print(f"Best Configuration Found:")
    print(best)
    
    # 2. Multiplier Heatmap (Mean F1 over thresholds)
    # We aggregate the best F1 achieved for each (N_Mult, A_Mult) pair
    pivot_df = df.pivot_table(index='N_Mult', columns='A_Mult', values='F1', aggfunc='max')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="YlGnBu")
    plt.title("Best Macro F1 per Multiplier Combination")
    plt.xlabel("Abnormal Multiplier (A_Mult)")
    plt.ylabel("Normal Multiplier (N_Mult)")
    
    # Highlight best
    plt.scatter(best['A_Mult'] - 1 + 0.5, best['N_Mult'] - 1 + 0.5, color='red', s=200, marker='*', label='Best')
    plt.savefig(f"{OUTPUT_DIR}/multiplier_heatmap.png")
    print(f"Saved multiplier heatmap to {OUTPUT_DIR}/multiplier_heatmap.png")
    
    # 3. Threshold Landscape for the BEST multiplier combo
    best_nm = best['N_Mult']
    best_am = best['A_Mult']
    subset = df[(df['N_Mult'] == best_nm) & (df['A_Mult'] == best_am)]
    
    thresh_pivot = subset.pivot_table(index='T_N', columns='T_A', values='F1')
    
    # Format axis labels to 2 decimal places
    thresh_pivot.index = [f"{x:.2f}" for x in thresh_pivot.index]
    thresh_pivot.columns = [f"{x:.2f}" for x in thresh_pivot.columns]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(thresh_pivot, annot=False, cmap="viridis")
    plt.title(f"Threshold Landscape for N={best_nm}, A={best_am}")
    plt.xlabel("Abnormal Threshold (T_A)")
    plt.ylabel("Normal Threshold (T_N)")
    
    # Highlight best threshold
    # Need to map threshold values to plot indices. 
    # Thresh candidates are 0.1, 0.15 ... 0.9.
    # index = (value - 0.1) / 0.05
    tn_idx = (best['T_N'] - 0.1) / 0.05
    ta_idx = (best['T_A'] - 0.1) / 0.05
    plt.scatter(ta_idx + 0.5, tn_idx + 0.5, color='red', s=200, marker='*', label='Best')
    
    plt.savefig(f"{OUTPUT_DIR}/threshold_landscape.png")
    print(f"Saved threshold landscape to {OUTPUT_DIR}/threshold_landscape.png")
    
    # 4. Summary Table of Top 10
    top10 = df.sort_values(by='F1', ascending=False).head(10)
    top10.to_csv(f"{OUTPUT_DIR}/top10_configs.csv", index=False)
    print(f"Saved Top 10 configs to {OUTPUT_DIR}/top10_configs.csv")

if __name__ == "__main__":
    main()
