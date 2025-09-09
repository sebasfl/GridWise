# reset_data.py
import shutil
from pathlib import Path
import argparse

def reset_directories(keep_external=True):
    """Reset data and model directories, optionally keeping external data."""
    
    base_path = Path(".")
    
    # Directories to clean
    directories_to_clean = [
        "data/processed",
        "data/analysis", 
        "data/forecasts",
        "models"
    ]
    
    if not keep_external:
        directories_to_clean.append("data/external")
    
    print("🧹 Resetting directories...")
    
    for dir_path in directories_to_clean:
        full_path = base_path / dir_path
        
        if full_path.exists():
            print(f"  Removing: {dir_path}")
            shutil.rmtree(full_path)
        
        print(f"  Creating: {dir_path}")
        full_path.mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep to preserve directory in git
        gitkeep = full_path / ".gitkeep"
        gitkeep.touch()
    
    print("✅ Reset completed!")
    print("\nDirectories ready for fresh data:")
    for dir_path in directories_to_clean:
        print(f"  📁 {dir_path}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset data and model directories")
    parser.add_argument("--include-external", action="store_true", 
                       help="Also reset external data (will need to re-download BDG2)")
    
    args = parser.parse_args()
    
    if args.include_external:
        print("⚠️  WARNING: This will also delete external data (BDG2 download)")
        confirm = input("Continue? (y/N): ").lower().strip()
        if confirm != 'y':
            print("❌ Cancelled")
            exit(1)
    
    reset_directories(keep_external=not args.include_external)