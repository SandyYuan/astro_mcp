#!/usr/bin/env python3

"""
Claude MCP Client for DESI Data Access

This script demonstrates how to use Claude with the DESI MCP server to answer
natural language queries about astronomical data.

Example usage:
    python claude_mcp_client.py

Example queries:
    - "find the nearest galaxy to ra=10.68, dec=41.27, and return its spectrum and redshift"
    - "search for high redshift quasars between z=2 and z=3"
    - "what galaxies are in the region from RA 150 to 151 degrees and Dec 2 to 3 degrees?"
"""

import asyncio
import json
import os
from typing import Any, Dict, List
import anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ClaudeMCPClient:
    def __init__(self, anthropic_api_key: str):
        """Initialize the Claude MCP client."""
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.server_params = StdioServerParameters(
            command="python",
            args=["server.py"],
            env=None
        )
        self.tools_for_claude = []
        self.session = None
        self.system_prompt = """You are an expert astronomical data analyst with access to DESI (Dark Energy Spectroscopic Instrument) survey data through specialized tools. 

Your role is to help users query and analyze DESI spectroscopic data by:
1. Understanding natural language queries about astronomical objects
2. Using the appropriate DESI tools to search for and retrieve data
3. Interpreting the results and providing clear, informative responses

Available DESI tools:
- find_spectra_by_coordinates: Search for spectra near specific sky coordinates
- get_spectrum_by_id: Retrieve detailed information about a specific spectrum
- search_by_object_type: Find objects by type (GALAXY, QSO, STAR) with optional filters
- search_in_region: Search for objects in rectangular sky regions

When users ask about astronomical objects, coordinates, spectra, or redshifts, use these tools to provide accurate, data-driven answers. Always explain what you're doing and interpret the results in an accessible way."""
        
    async def __aenter__(self):
        """Async context manager entry."""
        # Start the MCP server and create session
        self.stdio_context = stdio_client(self.server_params)
        self.read, self.write = await self.stdio_context.__aenter__()
        
        self.session = ClientSession(self.read, self.write)
        await self.session.__aenter__()
        await self.session.initialize()
        
        # Get available tools and convert them for Claude
        await self._setup_tools()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if hasattr(self, 'stdio_context'):
            await self.stdio_context.__aexit__(exc_type, exc_val, exc_tb)
    
    async def _setup_tools(self):
        """Setup tools for Claude by converting MCP tool definitions."""
        mcp_tools = await self.session.list_tools()
        
        for tool in mcp_tools.tools:
            claude_tool = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            self.tools_for_claude.append(claude_tool)
        
        print(f"✅ Connected to DESI MCP Server with {len(self.tools_for_claude)} tools")
        for tool in self.tools_for_claude:
            print(f"   - {tool['name']}")
    
    async def query(self, user_query: str) -> str:
        """
        Process a natural language query using Claude and MCP tools.
        
        Args:
            user_query: Natural language query about DESI data
            
        Returns:
            Claude's response after potentially using MCP tools
        """
        print(f"\n🔍 Processing query: {user_query}")
        
        # Format initial message for Claude
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_query
                    }
                ]
            }
        ]
        
        # Send query to Claude with available tools
        response = await self._call_claude(messages)
        
        # Handle tool usage if Claude wants to use tools
        if response.stop_reason == "tool_use":
            # Add Claude's response (including tool calls) to conversation
            messages.append({
                "role": "assistant", 
                "content": response.content
            })
            
            # Process all tool calls
            for content_block in response.content:
                if content_block.type == "tool_use":
                    tool_result = await self._execute_tool(content_block)
                    
                    # Add tool result to conversation
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": tool_result
                            }
                        ]
                    })
            
            # Get Claude's final response with tool results
            final_response = await self._call_claude(messages)
            return self._extract_text_content(final_response.content)
        else:
            return self._extract_text_content(response.content)
    
    async def _call_claude(self, messages: List[Dict]) -> Any:
        """Call Claude API with messages and tools using correct syntax."""
        return self.anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            temperature=0.1,
            system=self.system_prompt,
            messages=messages,
            tools=self.tools_for_claude
        )
    
    async def _execute_tool(self, tool_call) -> str:
        """Execute an MCP tool call."""
        tool_name = tool_call.name
        tool_input = tool_call.input
        
        print(f"🔧 Executing tool: {tool_name} with args: {tool_input}")
        
        try:
            result = await self.session.call_tool(tool_name, arguments=tool_input)
            
            # Extract text content from MCP result
            if result.content and len(result.content) > 0:
                content = result.content[0]
                if hasattr(content, 'text'):
                    return content.text
                else:
                    return str(content)
            else:
                return "Tool executed successfully but returned no content."
                
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def _extract_text_content(self, content) -> str:
        """Extract text from Claude's response content."""
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if hasattr(item, 'text'):
                    text_parts.append(item.text)
                elif hasattr(item, 'type') and item.type == 'text':
                    text_parts.append(item.text)
            return ''.join(text_parts)
        elif hasattr(content, 'text'):
            return content.text
        else:
            return str(content)

async def interactive_mode(client: ClaudeMCPClient):
    """Run the client in interactive mode."""
    print("\n" + "="*60)
    print("🌟 DESI MCP + Claude Interactive Client")
    print("="*60)
    print("Ask questions about DESI astronomical data in natural language!")
    print("Examples:")
    print("  • 'find the nearest galaxy to ra=10.68, dec=41.27'")
    print("  • 'search for quasars with redshift between 2 and 3'")
    print("  • 'what objects are in the region RA 150-151, Dec 2-3?'")
    print("\nType 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("🔭 Your query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Process the query
            response = await client.query(user_input)
            print(f"\n🤖 Claude's Response:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

async def run_example_queries(client: ClaudeMCPClient):
    """Run some example queries to demonstrate capabilities."""
    print("\n" + "="*60)
    print("🧪 Running Example Queries")
    print("="*60)
    
    example_queries = [
        "find the nearest galaxy to ra=10.68, dec=41.27, and return its spectrum and redshift",
        "search for high redshift quasars between z=2.0 and z=3.0, limit to 5 results",
        "what objects are in the sky region from RA 150 to 151 degrees and Dec 2 to 3 degrees?",
        "find stars near coordinates ra=45.2, dec=-12.8 within 30 arcseconds"
    ]
    
    for i, query in enumerate(example_queries, 1):
        print(f"\n📝 Example {i}: {query}")
        print("-" * 50)
        
        try:
            response = await client.query(query)
            print(f"🤖 Response: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50)

async def main():
    """Main function."""
    # Get Anthropic API key from environment (loaded from .env)
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Please set ANTHROPIC_API_KEY in your .env file")
        print("   Create a .env file in this directory with:")
        print("   ANTHROPIC_API_KEY='your-key-here'")
        print("   You can get an API key from: https://console.anthropic.com/")
        return
    
    print("🚀 Starting DESI MCP + Claude Client...")
    
    try:
        async with ClaudeMCPClient(api_key) as client:
            # Check if user wants to run examples or interactive mode
            print("\nChoose mode:")
            print("1. Run example queries")
            print("2. Interactive mode")
            
            choice = input("Enter choice (1 or 2): ").strip()
            
            if choice == "1":
                await run_example_queries(client)
            else:
                await interactive_mode(client)
                
    except Exception as e:
        print(f"❌ Failed to start client: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the mcp conda environment")
        print("2. Verify ANTHROPIC_API_KEY is set correctly in .env file")
        print("3. Ensure server.py is in the current directory")
        print("4. Install missing dependencies: pip install python-dotenv")

if __name__ == "__main__":
    asyncio.run(main()) 