"""Short video and social post extraction MCP server."""

__version__ = "1.3.0"
__author__ = "yzfly"
__email__ = "yz.liu.me@gmail.com"

def main():
    from .server import main as server_main

    return server_main()

__all__ = ["main"]
