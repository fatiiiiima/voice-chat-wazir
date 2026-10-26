# Starts the Wazir voice agent. Start the LLM first (scripts\start_llm.ps1).
#   powershell -ExecutionPolicy Bypass -File scripts\start_agent.ps1
Set-Location (Split-Path $PSScriptRoot -Parent)
Write-Host "Open http://127.0.0.1:7860 in Chrome or Edge once you see 'Ready'." -ForegroundColor Green
python -m uv run python -m agent.server
