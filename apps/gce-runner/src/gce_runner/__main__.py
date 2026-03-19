"""Entrypoint for the gce-runner MCP server."""

from gce_runner.server import mcp


def main() -> None:
    """Run the MCP server in stdio transport mode."""
    mcp.run()


if __name__ == "__main__":
    main()
