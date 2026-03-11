"""
Dashboard Launcher

Simple script to launch the Streamlit dashboard.
"""

import subprocess
import sys

def main():
    """Launch the Streamlit dashboard."""
    print("="*60)
    print("LAUNCHING SUPPORT TICKET AUTO-TAGGING DASHBOARD")
    print("="*60)
    print("\nStarting Streamlit server...")
    print("Dashboard will open in your browser automatically.")
    print("\nPress Ctrl+C to stop the server.")
    print("="*60)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.headless=false"
        ])
    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTry running manually:")
        print("  streamlit run app.py")

if __name__ == "__main__":
    main()
