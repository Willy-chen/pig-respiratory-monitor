import os
import glob
import json
import pandas as pd

RESULTS_DIR = "classifiers"

def main():
    print("--- 20260330 Classifier Benchmark Results ---")
    
    results_files = glob.glob(os.path.join(RESULTS_DIR, "*", "results.json"))
    
    if not results_files:
        print("No results.json found. Please run the baselines first using ./run_all.sh")
        return
        
    data = []
    for fpath in results_files:
        model_name = os.path.basename(os.path.dirname(fpath))
        with open(fpath, 'r') as f:
            res = json.load(f)
            
        data.append({
            'Model': model_name,
            'Macro F1': res.get('macro_f1', 0.0),
            'Accuracy': res.get('accuracy', 0.0),
            'TrainTime(s)': res.get('total_train_time_s', 0.0),
            'Latency(ms/seg)': res.get('inference_latency_ms', 0.0)
        })
        
    df = pd.DataFrame(data)
    df = df.sort_values(by="Macro F1", ascending=False).reset_index(drop=True)
    
    print("\n" + df.to_string(index=False, float_format="%.4f"))
    df.to_csv("benchmark_summary.csv", index=False)
    print("\nSaved full summary to benchmark_summary.csv")

if __name__ == "__main__":
    main()
