"""
Mock MCP execution layer.

Returns canned responses for tool calls without hitting real APIs.
Used for end-to-end evaluation where the model needs to process tool results.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class MockMCPExecutor:
    """Executes tool calls by returning mock/canned responses.

    Lookup order:
    1. Exact match: {server}_{tool}_{arg_hash}.json
    2. Tool default: {server}_{tool}.json
    3. Generic fallback response
    """

    def __init__(self, mock_responses_dir: str | Path):
        self.mock_dir = Path(mock_responses_dir)
        self._cache: dict[str, dict] = {}

    def _arg_hash(self, arguments: dict) -> str:
        """Generate a short hash of tool arguments for cache key."""
        arg_str = json.dumps(arguments, sort_keys=True, default=str)
        return hashlib.md5(arg_str.encode()).hexdigest()[:8]

    def _load_mock(self, filename: str) -> dict | None:
        """Load a mock response file from the mock directory."""
        if filename in self._cache:
            return self._cache[filename]

        filepath = self.mock_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                data = json.load(f)
                self._cache[filename] = data
                return data
        return None

    def execute(
        self,
        server: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """Execute a mock tool call and return a JSON string response.

        Args:
            server: Server name (e.g., 'sec_edgar')
            tool_name: Tool name (e.g., 'sec_get_income_statement')
            arguments: Tool call arguments from the model

        Returns:
            JSON string of the mock response
        """
        # Try exact match with argument hash
        arg_hash = self._arg_hash(arguments)
        exact_filename = f"{server}_{tool_name}_{arg_hash}.json"
        mock = self._load_mock(exact_filename)
        if mock:
            response = mock.get("response", mock.get("default_response", mock))
            return json.dumps(response, indent=2)

        # Try tool default
        default_filename = f"{server}_{tool_name}.json"
        mock = self._load_mock(default_filename)
        if mock:
            response = mock.get("default_response", mock)
            # Substitute common argument values into the response
            response = self._substitute_args(response, arguments)
            return json.dumps(response, indent=2)

        # Generic fallback
        return json.dumps(self._generic_response(server, tool_name, arguments), indent=2)

    def _substitute_args(self, response: Any, arguments: dict) -> Any:
        """Substitute argument values into mock response where field names match."""
        if isinstance(response, dict):
            result = {}
            for key, value in response.items():
                if key in arguments and isinstance(value, (str, int, float)):
                    result[key] = arguments[key]
                else:
                    result[key] = self._substitute_args(value, arguments)
            return result
        elif isinstance(response, list):
            return [self._substitute_args(item, arguments) for item in response]
        return response

    def _generic_response(
        self, server: str, tool_name: str, arguments: dict
    ) -> dict:
        """Generate a generic mock response for tools without specific mock data."""
        return {
            "status": "success",
            "server": server,
            "tool": tool_name,
            "arguments_received": arguments,
            "data": {
                "message": f"Mock response from {server}.{tool_name}",
                "note": "This is a generic mock response. Add a specific mock file for realistic data.",
                "mock_results": [
                    {"id": "mock-001", "title": f"Result 1 for {tool_name}", "value": 42},
                    {"id": "mock-002", "title": f"Result 2 for {tool_name}", "value": 84},
                ],
            },
        }

    def has_mock(self, server: str, tool_name: str) -> bool:
        """Check if a specific mock response exists for a tool."""
        filename = f"{server}_{tool_name}.json"
        return (self.mock_dir / filename).exists()

    def list_available_mocks(self) -> list[str]:
        """List all available mock response files."""
        if not self.mock_dir.exists():
            return []
        return [f.stem for f in sorted(self.mock_dir.glob("*.json"))]


def resolve_server_for_tool(
    tool_name: str, schemas: list[dict]
) -> str | None:
    """Given a tool name, find which server it belongs to.

    Args:
        tool_name: The tool name to look up
        schemas: List of server schemas

    Returns:
        Server name or None if not found
    """
    for schema in schemas:
        for tool in schema.get("tools", []):
            if tool["name"] == tool_name:
                return schema["server"]
    return None
