"""
Schema generator for MCP tool eval framework.

Supports two generation modes:
1. LLM-assisted generation from YAML templates (for long-tail scaling)
2. Conversion from raw API docs text to MCP tool schemas

Usage:
    # Generate schemas from template
    python -m registry.generator --generate --template registry/templates/office_template.yaml --count 10

    # Convert API docs to schema
    python -m registry.generator --from-docs path/to/api_docs.md --server-name my_server --domain office

    # List existing schemas
    python -m registry.generator --list
"""

import argparse
import json
import os
import random
import string
import sys
from pathlib import Path
from typing import Any

import yaml

# Attempt to import openai for LLM-based generation
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


REGISTRY_DIR = Path(__file__).parent / "schemas"

# ============================================================================
# Schema loading utilities
# ============================================================================


def load_all_schemas(registry_dir: Path | None = None) -> list[dict]:
    """Load all tool schemas from the registry directory."""
    registry_dir = registry_dir or REGISTRY_DIR
    schemas = []
    for domain_dir in sorted(registry_dir.iterdir()):
        if domain_dir.is_dir():
            for schema_file in sorted(domain_dir.glob("*.json")):
                with open(schema_file) as f:
                    schemas.append(json.load(f))
    return schemas


def load_schemas_by_servers(server_names: list[str], registry_dir: Path | None = None) -> list[dict]:
    """Load schemas for specific servers by name."""
    all_schemas = load_all_schemas(registry_dir)
    return [s for s in all_schemas if s["server"] in server_names]


def get_all_tools_flat(schemas: list[dict]) -> list[dict]:
    """Flatten all schemas into a list of (server, tool) pairs for model consumption."""
    tools = []
    for schema in schemas:
        server = schema["server"]
        for tool in schema["tools"]:
            tools.append({
                "server": server,
                "domain": schema.get("domain", "unknown"),
                **tool,
            })
    return tools


def schemas_to_openai_tools(schemas: list[dict]) -> list[dict]:
    """Convert MCP tool schemas to OpenAI function calling format."""
    openai_tools = []
    for schema in schemas:
        for tool in schema["tools"]:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["inputSchema"],
                },
            })
    return openai_tools


def schemas_to_anthropic_tools(schemas: list[dict]) -> list[dict]:
    """Convert MCP tool schemas to Anthropic tool use format."""
    anthropic_tools = []
    for schema in schemas:
        for tool in schema["tools"]:
            anthropic_tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["inputSchema"],
            })
    return anthropic_tools


# ============================================================================
# Template-based generation (LLM-assisted)
# ============================================================================

GENERATION_SYSTEM_PROMPT = """You are a tool schema generator for MCP (Model Context Protocol) servers.
Given a product name, domain archetype, and tool patterns, generate a realistic JSON tool schema.

Rules:
- Tool names should be snake_case prefixed with a short product abbreviation
- Descriptions should be clear, specific, and realistic
- Parameter types should use JSON Schema (string, integer, number, boolean, array, object)
- Include realistic enum values where appropriate
- Include helpful description for every parameter
- Every tool must have an inputSchema with type "object" and properties
- Return valid JSON only, no markdown or explanation"""

GENERATION_USER_PROMPT = """Generate a realistic MCP tool schema for the following product:

Product name: {product_name}
Domain: {domain}
Server description pattern: {description_template}

Generate tools based on these patterns:
{tool_patterns}

Additional variation instructions:
- Add 1-2 extra tools beyond the patterns that would be realistic for this product
- Vary parameter names slightly from the patterns to be product-specific
- Add product-specific enum values where appropriate

Return a single JSON object with this structure:
{{
  "server": "<lowercase_product_id>",
  "domain": "{domain}",
  "description": "<filled description>",
  "tools": [...]
}}"""


def load_template(template_path: str) -> dict:
    """Load a YAML template file."""
    with open(template_path) as f:
        return yaml.safe_load(f)


def generate_schema_with_llm(
    product_name: str,
    archetype: dict,
    domain: str,
    client: Any,
    model: str = "gpt-4o",
) -> dict:
    """Generate a tool schema using an LLM based on template archetype."""
    tool_patterns_str = yaml.dump(archetype["tool_patterns"], default_flow_style=False)
    description_template = archetype["description_template"]

    user_prompt = GENERATION_USER_PROMPT.format(
        product_name=product_name,
        domain=domain,
        description_template=description_template,
        tool_patterns=tool_patterns_str,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )

    return json.loads(response.choices[0].message.content)


def generate_from_template(
    template_path: str,
    count: int = 10,
    model: str = "gpt-4o",
    output_dir: Path | None = None,
) -> list[dict]:
    """Generate multiple schemas from a template using LLM."""
    if not HAS_OPENAI:
        print("Error: openai package required for LLM generation. pip install openai")
        sys.exit(1)

    template = load_template(template_path)
    domain = template["domain"]
    output_dir = output_dir or REGISTRY_DIR / domain

    client = openai.OpenAI()
    generated = []

    # Collect all example products across archetypes
    product_pool = []
    for archetype in template["archetypes"]:
        for product in archetype["example_products"]:
            product_pool.append((product, archetype))

    # Shuffle and pick
    random.shuffle(product_pool)
    selections = product_pool[:count]

    for product_name, archetype in selections:
        print(f"  Generating schema for: {product_name} ({archetype['name']})")
        try:
            schema = generate_schema_with_llm(
                product_name=product_name,
                archetype=archetype,
                domain=domain,
                client=client,
                model=model,
            )

            # Ensure required fields
            schema.setdefault("domain", domain)
            if "server" not in schema:
                schema["server"] = product_name.lower().replace(" ", "_").replace(".", "")

            # Save
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = schema["server"] + ".json"
            filepath = output_dir / filename
            with open(filepath, "w") as f:
                json.dump(schema, f, indent=2)

            generated.append(schema)
            print(f"    -> Saved to {filepath}")

        except Exception as e:
            print(f"    -> Error generating {product_name}: {e}")

    return generated


# ============================================================================
# API docs conversion
# ============================================================================

DOCS_CONVERSION_PROMPT = """Convert the following API documentation into an MCP tool schema JSON.

API Documentation:
{docs_content}

Server name: {server_name}
Domain: {domain}

Rules:
- Extract the most important API endpoints and convert them to MCP tools
- Each tool should have a clear name (snake_case, prefixed with short server name), description, and inputSchema
- Include all relevant parameters with proper types and descriptions
- Limit to the 5-8 most important/commonly used endpoints
- Return valid JSON only

Return format:
{{
  "server": "{server_name}",
  "domain": "{domain}",
  "description": "<one-line description of the server>",
  "tools": [
    {{
      "name": "...",
      "description": "...",
      "inputSchema": {{ "type": "object", "properties": {{...}}, "required": [...] }}
    }}
  ]
}}"""


def generate_from_docs(
    docs_path: str,
    server_name: str,
    domain: str,
    model: str = "gpt-4o",
    output_dir: Path | None = None,
) -> dict:
    """Convert API documentation to MCP tool schema using LLM."""
    if not HAS_OPENAI:
        print("Error: openai package required. pip install openai")
        sys.exit(1)

    with open(docs_path) as f:
        docs_content = f.read()

    # Truncate if too long
    if len(docs_content) > 30000:
        docs_content = docs_content[:30000] + "\n... (truncated)"

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": DOCS_CONVERSION_PROMPT.format(
                    docs_content=docs_content,
                    server_name=server_name,
                    domain=domain,
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    schema = json.loads(response.choices[0].message.content)
    schema.setdefault("server", server_name)
    schema.setdefault("domain", domain)

    output_dir = output_dir or REGISTRY_DIR / domain
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{server_name}.json"
    with open(filepath, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"Saved schema to {filepath}")
    return schema


# ============================================================================
# Deterministic generation (no LLM needed — for testing / offline use)
# ============================================================================


def generate_deterministic_from_template(
    template_path: str,
    count: int = 10,
    output_dir: Path | None = None,
) -> list[dict]:
    """Generate schemas deterministically from templates without LLM.

    Uses the template patterns directly to create tool schemas.
    Useful for offline testing or when LLM API is not available.
    """
    template = load_template(template_path)
    domain = template["domain"]
    output_dir = output_dir or REGISTRY_DIR / domain
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    product_pool = []
    for archetype in template["archetypes"]:
        for product in archetype["example_products"]:
            product_pool.append((product, archetype))

    random.seed(42)  # Deterministic
    random.shuffle(product_pool)
    selections = product_pool[:count]

    for product_name, archetype in selections:
        server_id = product_name.lower().replace(" ", "_").replace(".", "")
        prefix = "".join(w[0] for w in product_name.split()).lower()
        if len(prefix) < 2:
            prefix = server_id[:4]

        tools = []
        for pattern in archetype["tool_patterns"]:
            tool_name = f"{prefix}_{pattern['action']}"
            properties = {}
            for param in pattern["typical_params"]:
                # Infer type from param name
                if param.endswith("_id") or param in ("query", "text", "name", "title",
                    "description", "subject", "body", "email", "folder", "channel",
                    "status", "assignee", "owner", "format", "type", "role",
                    "permission", "template", "agenda", "notes", "location",
                    "domain", "company", "platform", "indicator", "protocol",
                    "indicator_id", "coin_id", "app_id", "ticker", "ric", "isin",
                    "deal_type", "category", "country", "chain", "exchange",
                    "vs_currency", "interval", "period", "frequency",
                    "sort_by", "stage", "version"):
                    ptype = "string"
                elif param in ("limit", "count", "num_periods", "max_results",
                    "duration", "lookback_days", "horizon", "page_size",
                    "years", "peer_count"):
                    ptype = "integer"
                elif param in ("min_value", "value"):
                    ptype = "number"
                elif param in ("is_private", "adjusted", "include_content"):
                    ptype = "boolean"
                elif param in ("to", "cc", "bcc", "members", "participants",
                    "tickers", "countries", "metrics", "fields", "ratios",
                    "criteria", "labels", "attachments", "sorts", "periods"):
                    ptype = "array"
                elif param in ("date_range", "filter", "properties"):
                    ptype = "object"
                elif param.endswith("_date") or param in ("after", "before", "date"):
                    ptype = "string"
                else:
                    ptype = "string"

                prop: dict[str, Any] = {"type": ptype, "description": f"{param.replace('_', ' ').title()}"}
                if ptype == "array":
                    prop["items"] = {"type": "string"}
                if ptype == "integer" and "limit" in param:
                    prop["default"] = 25
                properties[param] = prop

            tools.append({
                "name": tool_name,
                "description": f"{pattern['description']} in {product_name}",
                "inputSchema": {
                    "type": "object",
                    "properties": properties,
                    "required": pattern["required_params"],
                },
            })

        schema = {
            "server": server_id,
            "domain": domain,
            "description": archetype["description_template"].format(product_name=product_name),
            "tools": tools,
        }

        filepath = output_dir / f"{server_id}.json"
        with open(filepath, "w") as f:
            json.dump(schema, f, indent=2)

        generated.append(schema)
        print(f"  Generated: {filepath} ({len(tools)} tools)")

    return generated


# ============================================================================
# List / stats
# ============================================================================


def list_schemas(registry_dir: Path | None = None) -> None:
    """Print a summary of all schemas in the registry."""
    schemas = load_all_schemas(registry_dir)
    total_tools = 0
    print(f"\n{'Server':<25} {'Domain':<10} {'Tools':<6} Description")
    print("-" * 90)
    for s in schemas:
        n_tools = len(s.get("tools", []))
        total_tools += n_tools
        print(f"{s['server']:<25} {s.get('domain', '?'):<10} {n_tools:<6} {s.get('description', '')[:50]}")
    print("-" * 90)
    print(f"Total: {len(schemas)} servers, {total_tools} tools\n")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="MCP Tool Schema Generator")
    parser.add_argument("--list", action="store_true", help="List all existing schemas")
    parser.add_argument("--generate", action="store_true", help="Generate schemas from template (LLM)")
    parser.add_argument("--generate-deterministic", action="store_true",
                        help="Generate schemas from template (no LLM)")
    parser.add_argument("--from-docs", type=str, help="Path to API docs file to convert")
    parser.add_argument("--template", type=str, help="Path to YAML template file")
    parser.add_argument("--count", type=int, default=10, help="Number of schemas to generate")
    parser.add_argument("--server-name", type=str, help="Server name (for --from-docs)")
    parser.add_argument("--domain", type=str, default="office", help="Domain (office or data)")
    parser.add_argument("--model", type=str, default="gpt-4o", help="LLM model for generation")
    parser.add_argument("--output-dir", type=str, help="Override output directory")

    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.list:
        list_schemas()
    elif args.generate:
        if not args.template:
            parser.error("--template required for --generate")
        schemas = generate_from_template(args.template, args.count, args.model, output_dir)
        print(f"\nGenerated {len(schemas)} schemas")
    elif args.generate_deterministic:
        if not args.template:
            parser.error("--template required for --generate-deterministic")
        schemas = generate_deterministic_from_template(args.template, args.count, output_dir)
        print(f"\nGenerated {len(schemas)} schemas (deterministic)")
    elif args.from_docs:
        if not args.server_name:
            parser.error("--server-name required for --from-docs")
        generate_from_docs(args.from_docs, args.server_name, args.domain, args.model, output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
