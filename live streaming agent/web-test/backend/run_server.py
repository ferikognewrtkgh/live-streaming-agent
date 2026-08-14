import argparse
from pathlib import Path

import uvicorn

from app.logging_setup import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LiveStreamingAgent backend.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--reload",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend_dir = Path(__file__).resolve().parent
    configure_logging()
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(backend_dir)] if args.reload else None,
        log_config=None,
        access_log=True,
    )


if __name__ == "__main__":
    main()
