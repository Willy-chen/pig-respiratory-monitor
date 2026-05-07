import time
import torch
import numpy as np
import os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../20260322')))
from baselines.yin_2021.model import SpectrogramAlexNet
from baselines.dorr_2026.model import DorrBEATsOfficial

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_dl_model(model_name, model_class, input_shape, device='cuda'):
    print(f"\n--- Benchmarking {model_name} on {device.upper()} ---")
    model = model_class().to(device)
    model.eval()
    
    print(f"Trainable Parameters: {count_parameters(model):,}")
    
    # Dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
            
    # Measure
    if device == 'cuda': torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    N_RUNS = 100
    for _ in range(N_RUNS):
        with torch.no_grad():
            _ = model(dummy_input)
            
    if device == 'cuda': torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    latency_ms = ((end_time - start_time) / N_RUNS) * 1000
    print(f"Latency per segment: {latency_ms:.2f} ms")
    return latency_ms

def benchmark_ultimate(device='cuda'):
    print(f"\n--- Benchmarking Ultimate (AST + XGBoost) on {device.upper()} ---")
    # Part 1. AST Encoding
    from transformers import ASTModel, ASTFeatureExtractor
    ast_model = ASTModel.from_pretrained("../20260209_n/best_ast_model", output_hidden_states=True).to(device)
    ast_model.eval()
    print(f"Trainable Parameters (AST): {count_parameters(ast_model):,}")
    
    processor = ASTFeatureExtractor.from_pretrained("../20260209_n/best_ast_model")
    
    dummy_wav = np.random.randn(16000 * 10).astype(np.float32)
    inputs = processor(dummy_wav, sampling_rate=16000, return_tensors="pt").input_values.to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            out = ast_model(inputs)
            
    if device == 'cuda': torch.cuda.synchronize()
    start_ast = time.perf_counter()
    
    N_RUNS = 100
    ast_features = []
    for _ in range(N_RUNS):
        with torch.no_grad():
            out = ast_model(inputs)
            hs = torch.stack(out.hidden_states[-3:])
            avg = torch.mean(hs, dim=0)
            gp = torch.mean(avg, dim=1).cpu().numpy()
            ast_features.append(gp)
            
    if device == 'cuda': torch.cuda.synchronize()
    ast_latency_ms = ((time.perf_counter() - start_ast) / N_RUNS) * 1000
    print(f"AST Encoding Latency: {ast_latency_ms:.2f} ms")
    
    # Part 2. XGBoost Inference
    import xgboost as xgb
    # Create dummy booster
    # We can measure it precisely using xgb
    test_feature = np.random.randn(1, 768)
    dmat = xgb.DMatrix(test_feature)
    
    # Mock model
    import tempfile
    X_train = np.random.randn(10, 768)
    y_train = np.random.randint(0, 3, 10)
    bst = xgb.train({'objective': 'multi:softprob', 'num_class':3}, xgb.DMatrix(X_train, label=y_train), num_boost_round=100)
    
    # Warmup
    for _ in range(10): bst.predict(dmat)
    
    start_xgb = time.perf_counter()
    for _ in range(N_RUNS): bst.predict(dmat)
    xgb_latency_ms = ((time.perf_counter() - start_xgb) / N_RUNS) * 1000
    print(f"XGBoost Latency: {xgb_latency_ms:.4f} ms")
    
    total_latency = ast_latency_ms + xgb_latency_ms
    print(f"Total Latency per segment: {total_latency:.2f} ms")
    return total_latency

def main():
    has_cuda = torch.cuda.is_available()
    devices = ['cuda', 'cpu'] if has_cuda else ['cpu']
    
    for d in devices:
        benchmark_ultimate(d)
        
        benchmark_dl_model(
            "Yin 2021 (AlexNet)",
            lambda: SpectrogramAlexNet(num_classes=3, pretrained=False),
            (1, 160000), # Model creates spectrogram internally
            d
        )
        
        benchmark_dl_model(
            "Dorr 2026 (BEATs)",
            lambda: DorrBEATsOfficial(ckpt_path="../20260322/pretrained_models/BEATs_iter3_plus_AS2M.pt", num_classes=3),
            (1, 160000),
            d
        )

if __name__ == "__main__":
    main()
