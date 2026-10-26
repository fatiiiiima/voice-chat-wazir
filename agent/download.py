"""Download and warm up the speech models: uv run python -m agent.download"""

from agent.models import preload

if __name__ == "__main__":
    preload()
    print("Speech models ready.")
