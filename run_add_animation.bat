@echo off
SET NODE_EXE=node
rem "C:\Program Files\nodejs\node.exe"
SET RIGBY_MCP_SERVER=_tools/mcp-server/src/mcp/server.js
SET RIGBY_WORKSPACE=../../Projects
python rigby_mcp_arms.py
