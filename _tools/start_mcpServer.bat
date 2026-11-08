@echo off
REM Starts the Rigby2D MCP server on stdio (JSON-RPC 2.0, one message per line).
REM Logs go to stderr. Keep stdin/stdout for the MCP client: never run the
REM server through npm, because npm writes to stdout.
REM
REM The workspace root is the repository root (the folder above _tools).

setlocal
set "HERE=%~dp0"
set "ROOT=%HERE%.."
set "SERVER=%HERE%mcp-server\src\mcp\server.js"

if not exist "%SERVER%" (
  echo MCP server not found: %SERVER% 1>&2
  exit /b 1
)

node "%SERVER%" --root "%ROOT%"

