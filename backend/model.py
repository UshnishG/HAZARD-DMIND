import time
import os
import psutil
import cv2
try:
    import torch
except ImportError:
    torch = None

# This will trigger app.py to load the models globally
print("Loading models and warming up system... Please wait.")
from app import annotate_frame, process_video

def get_memory_usage():
    """Gets current RAM and GPU VRAM usage."""
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    
    vram_mb = 0
    if torch and torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        
    return ram_mb, vram_mb

def benchmark_pipeline(color_image_path, grayscale_image_path, video_path):
    print("\n" + "="*50)
    print("📊 SYSTEM PERFORMANCE BENCHMARK REPORT")
    print("="*50)

    # --- 1. MEMORY USAGE ---
    ram, vram = get_memory_usage()
    print(f"\n[1] RESOURCE EFFICIENCY")
    print(f"    System RAM Usage:  {ram:.2f} MB")
    if vram > 0:
        print(f"    GPU VRAM Usage:    {vram:.2f} MB")
    else:
        print(f"    GPU VRAM Usage:    Not detected (Running on CPU)")

    # --- 2. PIPELINE LATENCY ---
    print(f"\n[2] PIPELINE LATENCY (End-to-End Frame Processing)")
    
    # Worst-case scenario: Color image (triggers Human + Fire + Landslide API)
    if os.path.exists(color_image_path):
        img_color = cv2.imread(color_image_path)
        annotate_frame(img_color, color_image_path) # Warmup
        
        start = time.perf_counter()
        annotate_frame(img_color, color_image_path)
        color_time = time.perf_counter() - start
        print(f"    Color Frame (Human + Fire + Landslide): {color_time:.4f} sec/frame")
    else:
        print("    [!] Color test image not found.")

    # Best-case scenario / Conditional routing: Grayscale (triggers Human + Landmine)
    if os.path.exists(grayscale_image_path):
        img_gray = cv2.imread(grayscale_image_path)
        annotate_frame(img_gray, grayscale_image_path) # Warmup
        
        start = time.perf_counter()
        annotate_frame(img_gray, grayscale_image_path)
        gray_time = time.perf_counter() - start
        print(f"    Grayscale Frame (Human + Landmine):     {gray_time:.4f} sec/frame")
    else:
        print("    [!] Grayscale test image not found.")

    # --- 3. VIDEO THROUGHPUT (FPS) ---
    print(f"\n[3] VIDEO THROUGHPUT (Application FPS)")
    if os.path.exists(video_path):
        print("    Processing test video (this will log to your mission.db)...")
        start = time.perf_counter()
        
        # Using the actual function from app.py to measure real-world performance
        _, _, total_frames, _ = process_video(video_path, "benchmark_test_video.mp4")
        
        total_time = time.perf_counter() - start
        if total_time > 0:
            fps = total_frames / total_time
            print(f"    Processed {total_frames} frames in {total_time:.2f} seconds.")
            print(f"    Effective Application Speed: {fps:.2f} FPS")
    else:
        print(f"    [!] Video file '{video_path}' not found.")
        
    print("\n" + "="*50)

if __name__ == "__main__":
    # Update these paths to files that exist on your machine
    TEST_COLOR_IMG = "test_fire.jpg"       
    TEST_GRAY_IMG  = "test_landmine.jpg"   
    TEST_VIDEO     = "test_drone_feed.mp4" 

    benchmark_pipeline(TEST_COLOR_IMG, TEST_GRAY_IMG, TEST_VIDEO)