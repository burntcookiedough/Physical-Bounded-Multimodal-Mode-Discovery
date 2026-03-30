import time
import sys
import numpy as np

def benchmark_pipeline(processing_function, df_stream, batch_sizes=[10, 50, 100, 500, 1000]):
    """
    Evaluates system performance for real-time and batch operational mode discovery.
    
    processing_function: A callable that takes a chunk of dataframe and returns modes.
    df_stream: The synthetic or real historical data to process.
    batch_sizes: List of window sizes to evaluate scalability.
    """
    metrics = {}
    
    # 1. Inference Time (Latency) & Throughput on fixed window
    default_window = 100 # Default real-time buffer
    if len(df_stream) >= default_window:
        test_chunk = df_stream.iloc[:default_window]
        
        # Warmup
        _ = processing_function(test_chunk)
        
        # Benchmark
        latencies = []
        for _ in range(50): # 50 iterations
            start = time.perf_counter()
            _ = processing_function(test_chunk)
            latencies.append(time.perf_counter() - start)
            
        mean_latency = np.mean(latencies)
        metrics['Mean_Latency_Per_Window_Seconds'] = mean_latency
        metrics['P99_Latency_Seconds'] = np.percentile(latencies, 99)
        
        # Throughput: windows processed per second
        metrics['Throughput_Windows_Per_Second'] = 1.0 / mean_latency if mean_latency > 0 else float('inf')
        
    # 2. Memory Usage (Proxy)
    # Shallow estimate of data footprint
    metrics['Memory_Footprint_MB'] = sys.getsizeof(df_stream) / (1024 * 1024)
    
    # 3. Scalability (Big O empirical estimation)
    scalability_results = {}
    for size in batch_sizes:
        if len(df_stream) >= size:
            chunk = df_stream.iloc[:size]
            start = time.perf_counter()
            _ = processing_function(chunk)
            duration = time.perf_counter() - start
            scalability_results[size] = duration
            
    metrics['Scalability_Profile'] = scalability_results
    
    return metrics

if __name__ == "__main__":
    from dataset_simulator import IndustrialMultimodalSimulator
    
    sim = IndustrialMultimodalSimulator(n_samples=2000)
    df = sim.generate_domain()
    
    # Mock processing function (simulate HDBSCAN + Consensus + Physics overhead)
    def dummy_pipeline(data_chunk):
        time.sleep(0.005) # Assume 5ms computation time
        # Random labels
        return np.random.randint(0, 4, size=len(data_chunk))
        
    metrics = benchmark_pipeline(dummy_pipeline, df)
    
    for k, v in metrics.items():
        if isinstance(v, dict):
            print(f"\n{k}:")
            for sub_k, sub_v in v.items():
                print(f"  Size {sub_k}: {sub_v:.4f}s")
        else:
            print(f"{k}: {v:.4f}")
