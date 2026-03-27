"""Social Post Extractor MCP server."""

__version__ = "0.1.0"
__author__ = "JNHFlow21"
__email__ = "JNHFlow21@users.noreply.github.com"

def main():
    from .server import main as server_main

    return server_main()

__all__ = ["main"]
