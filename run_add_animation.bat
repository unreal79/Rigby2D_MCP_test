@echo off
SET NODE_EXE=node
rem "C:\Program Files\nodejs\node.exe"
SET RIGBY_MCP_SERVER=_tools/mcp-server/src/mcp/server.js
SET RIGBY_WORKSPACE=.
rem ../../Projects/rigby_mcp_test
python add_arms_up.py input/NILA.rigby output/NILA_arms_up.rigby
rem python add_arms_up.py input/rigbytron.rigby output/rigbytron_arms_up.rigby
