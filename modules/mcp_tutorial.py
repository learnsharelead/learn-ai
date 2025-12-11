import streamlit as st

def show():
    st.title("🔌 MCP: Model Context Protocol")
    
    st.markdown("""
    ### The Universal Language for AI Tools
    
    **MCP (Model Context Protocol)** is an open standard that lets AI models connect to
    external tools and data sources in a standardized way. Think of it as "USB for AI".
    """)
    
    tabs = st.tabs([
        "🎯 What is MCP?",
        "🏗️ Architecture",
        "🔧 Building Servers",
        "📱 Building Clients",
        "🌐 MCP Ecosystem"
    ])
    
    # TAB 1: What is MCP
    with tabs[0]:
        st.header("🎯 What is MCP?")
        
        st.markdown("""
        ### The Problem MCP Solves
        
        Every AI tool integration requires custom code:
        - OpenAI function calling format
        - Anthropic tool use format
        - Google Gemini format
        - Each app needs custom adapters
        
        ### MCP: One Standard to Rule Them All
        
        MCP provides a **universal protocol** so tools work everywhere.
        """)
        
        st.graphviz_chart("""
        digraph MCP {
            rankdir=LR;
            node [shape=box, style=filled];
            
            subgraph cluster_before {
                label="Before MCP";
                style=dashed;
                App1 [label="Cursor", fillcolor=lightblue];
                App2 [label="Claude Desktop", fillcolor=lightgreen];
                App3 [label="Your App", fillcolor=lightyellow];
                Tool1 [label="GitHub API", fillcolor=orange];
                Tool2 [label="Database", fillcolor=orange];
                
                App1 -> Tool1 [label="Custom"];
                App1 -> Tool2 [label="Custom"];
                App2 -> Tool1 [label="Different"];
                App2 -> Tool2 [label="Different"];
                App3 -> Tool1 [label="Another"];
            }
        }
        """)
        
        st.markdown("**After MCP:**")
        
        st.graphviz_chart("""
        digraph MCP {
            rankdir=LR;
            node [shape=box, style=filled];
            
            subgraph cluster_after {
                label="With MCP";
                style=solid;
                color=green;
                
                App1 [label="Any App", fillcolor=lightblue];
                MCP [label="MCP Protocol", fillcolor=lightgreen, shape=diamond];
                Server1 [label="MCP Server\\n(GitHub)", fillcolor=orange];
                Server2 [label="MCP Server\\n(Database)", fillcolor=orange];
                
                App1 -> MCP [label="Standard"];
                MCP -> Server1;
                MCP -> Server2;
            }
        }
        """)
        
        st.success("""
        **Key Benefits:**
        - ✅ Build once, use everywhere
        - ✅ Security by design (explicit permissions)
        - ✅ Growing ecosystem of servers
        - ✅ Works with any AI model
        """)
    
    # TAB 2: Architecture
    with tabs[1]:
        st.header("🏗️ MCP Architecture")
        
        st.markdown("""
        ### Three Core Concepts
        """)
        
        concepts = [
            {
                "name": "📦 **Resources**",
                "desc": "Data that the AI can read (files, database records, API responses)",
                "example": "file:///documents/policy.pdf"
            },
            {
                "name": "🔧 **Tools**",
                "desc": "Functions the AI can execute (create file, send email, query API)",
                "example": "create_github_issue(title, body)"
            },
            {
                "name": "💬 **Prompts**",
                "desc": "Reusable prompt templates with variables",
                "example": "summarize_document(document_uri)"
            }
        ]
        
        for c in concepts:
            with st.expander(c["name"]):
                st.markdown(f"**What:** {c['desc']}")
                st.code(c["example"])
        
        st.markdown("---")
        
        st.subheader("Client-Server Model")
        
        st.markdown("""
        | Component | Role | Examples |
        |-----------|------|----------|
        | **Host** | The AI application | Claude Desktop, Cursor, VS Code |
        | **Client** | Connects to servers | Built into the host |
        | **Server** | Provides tools/resources | Custom servers, community servers |
        """)
        
        st.info("""
        **Communication:** MCP uses JSON-RPC 2.0 over:
        - **stdio**: For local processes
        - **HTTP/SSE**: For remote servers
        """)
    
    # TAB 3: Building Servers
    with tabs[2]:
        st.header("🔧 Building MCP Servers")
        
        st.markdown("""
        ### Create Your Own Tools
        
        MCP servers expose tools that any MCP client can use.
        """)
        
        st.info("**Install:** `pip install mcp`")
        
        st.subheader("Simple MCP Server")
        
        st.code('''
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Create server
server = Server("my-tools")

# Define a tool
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a city",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        )
    ]

# Implement the tool
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        city = arguments["city"]
        # Your logic here
        weather = fetch_weather(city)
        return [TextContent(type="text", text=f"Weather in {city}: {weather}")]
    
    raise ValueError(f"Unknown tool: {name}")

# Run the server
async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
        ''', language="python")
        
        st.markdown("---")
        
        st.subheader("Adding Resources")
        
        st.code('''
@server.list_resources()
async def list_resources():
    return [
        Resource(
            uri="file:///config/settings.json",
            name="App Settings",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def read_resource(uri: str):
    if uri == "file:///config/settings.json":
        content = read_file("settings.json")
        return content
        ''', language="python")
    
    # TAB 4: Building Clients
    with tabs[3]:
        st.header("📱 Building MCP Clients")
        
        st.markdown("""
        ### Connect to MCP Servers
        """)
        
        st.subheader("Python Client")
        
        st.code('''
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Connect to server
    server_params = StdioServerParameters(
        command="python",
        args=["my_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            
            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools.tools]}")
            
            # Call a tool
            result = await session.call_tool(
                "get_weather",
                arguments={"city": "Tokyo"}
            )
            print(result)
        ''', language="python")
        
        st.markdown("---")
        
        st.subheader("Using with Claude Desktop")
        
        st.markdown("""
        Claude Desktop supports MCP natively. Add servers to your config:
        """)
        
        st.code('''
// ~/Library/Application Support/Claude/claude_desktop_config.json
// (macOS) or %APPDATA%/Claude/claude_desktop_config.json (Windows)

{
  "mcpServers": {
    "my-tools": {
      "command": "python",
      "args": ["/path/to/my_server.py"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_..."
      }
    }
  }
}
        ''', language="json")
    
    # TAB 5: Ecosystem
    with tabs[4]:
        st.header("🌐 MCP Ecosystem")
        
        st.markdown("""
        ### Growing Library of Servers
        """)
        
        servers = [
            ("📂 **Filesystem**", "@modelcontextprotocol/server-filesystem", "Read/write local files"),
            ("🐙 **GitHub**", "@modelcontextprotocol/server-github", "Issues, PRs, repos"),
            ("🔗 **Slack**", "@modelcontextprotocol/server-slack", "Send messages, read channels"),
            ("🗄️ **PostgreSQL**", "@modelcontextprotocol/server-postgres", "Query databases"),
            ("🌐 **Brave Search**", "@modelcontextprotocol/server-brave-search", "Web search"),
            ("📊 **Google Drive**", "@modelcontextprotocol/server-gdrive", "Read/write docs"),
            ("🐍 **Python REPL**", "Custom", "Execute Python code"),
        ]
        
        for name, package, desc in servers:
            st.markdown(f"- {name}: `{package}` - {desc}")
        
        st.info("""
        **Find More:** [github.com/modelcontextprotocol](https://github.com/modelcontextprotocol)
        """)
        
        st.markdown("---")
        
        st.subheader("MCP vs Function Calling")
        
        st.markdown("""
        | Aspect | MCP | Function Calling |
        |--------|-----|------------------|
        | Standard | Open, universal | Provider-specific |
        | Discovery | Servers list their tools | Defined at call time |
        | Resources | Supports data access | Tools only |
        | Clients | Any MCP client | Specific SDK |
        | Security | Built-in permissions | Custom implementation |
        """)
        
        st.success("""
        **When to Use MCP:**
        - Building tools for multiple AI apps
        - Need standardized security
        - Want to share tools with the community
        
        **When to Use Function Calling:**
        - Quick prototypes
        - Single-app integrations
        - Provider-specific features
        """)
