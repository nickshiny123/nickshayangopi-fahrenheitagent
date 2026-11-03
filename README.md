Nick Clarisse
A LangChain-based roleplay chatbot inspired by Fahrenheit 451.

Features
Clarisse McClellan mode
The Family (Parlor Wall) mode
Tavily web search
Ollama local model support
Gemini fallback support
Installation
Clone
git clone https://github.com/nickshiny123/nick-clarisse.git
cd nick-clarisse
Create virtual environment
uv venv
Install dependencies
uv sync
Environment Variables
Create a .env file:

GOOGLE_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
Run
uv run python src/nick_clarisse/__init__.py
Commands
/clarisse
/parlor
/quit

