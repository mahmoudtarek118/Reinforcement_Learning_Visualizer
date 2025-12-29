"""
Main entry point for the RL Learning Tool backend.
"""

import uvicorn
from backend.api.server import app


def main():
    """Run the FastAPI server."""
    uvicorn.run(
        "backend.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


if __name__ == "__main__":
    main()
