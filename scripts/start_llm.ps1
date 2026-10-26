# Starts the local LLM (Qwen3-4B-Instruct, 4-bit) on the Intel Arc GPU via llama.cpp + Vulkan.
# The first start downloads the model (~2.5 GB) from Hugging Face.
#   powershell -ExecutionPolicy Bypass -File scripts\start_llm.ps1
Set-Location (Split-Path $PSScriptRoot -Parent)
# Refuse to start if something else already owns the port (otherwise the agent
# silently talks to the wrong program and gets 404s).
$busy = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($busy) {
    $proc = Get-Process -Id $busy.OwningProcess -ErrorAction SilentlyContinue
    Write-Host "Port 8080 is already used by $($proc.ProcessName) (PID $($busy.OwningProcess))." -ForegroundColor Red
    Write-Host "Stop it first:  taskkill /PID $($busy.OwningProcess) /F"
    exit 1
}

$model = if ($env:LLM_GGUF) { $env:LLM_GGUF } else { "unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M" }

# -ngl 99       : put every layer on the GPU
# --jinja       : use the model's chat template (needed for tool calling)
# -c / --parallel: 16k context shared by 4 slots = 4 simultaneous conversations of up to 4k tokens
& tools\llama.cpp\llama-server.exe `
    -hf $model `
    --jinja `
    -ngl 99 `
    -c 16384 --parallel 4 `
    --host 127.0.0.1 --port 8080 `
    --alias qwen3-4b-instruct
