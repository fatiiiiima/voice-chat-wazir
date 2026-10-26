# One-time setup on Windows 11. Run from the repo folder in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "== 1/4 Checking Python ==" -ForegroundColor Cyan
# Any installed Python can run uv. uv then downloads its own Python 3.12 for the
# project (pinned in .python-version), so the system version doesn't matter.
python --version
if ($LASTEXITCODE -ne 0) { throw "Python not found. Install it from python.org." }

# uv = fast Python package manager. Use it as "python -m uv" so PATH doesn't matter.
$hasUv = $true
try { python -m uv --version *> $null; if ($LASTEXITCODE -ne 0) { $hasUv = $false } } catch { $hasUv = $false }
if (-not $hasUv) {
    Write-Host "Installing uv..."
    python -m pip install --user --upgrade uv
}

Write-Host "== 2/4 Installing Python packages ==" -ForegroundColor Cyan
python -m uv sync --python 3.12

Write-Host "== 3/4 Downloading llama.cpp (Vulkan build, runs the LLM on the Intel Arc GPU) ==" -ForegroundColor Cyan
$llamaDir = "tools\llama.cpp"
if (-not (Test-Path "$llamaDir\llama-server.exe")) {
    $release = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    $asset = $release.assets | Where-Object { $_.name -match "bin-win-vulkan-x64\.zip$" } | Select-Object -First 1
    if (-not $asset) { throw "Could not find a Windows Vulkan build in llama.cpp release $($release.tag_name)" }
    New-Item -ItemType Directory -Force -Path $llamaDir | Out-Null
    $zip = "$llamaDir\llama.zip"
    Write-Host "Downloading $($asset.name)..."
    Invoke-WebRequest $asset.browser_download_url -OutFile $zip
    Expand-Archive $zip -DestinationPath $llamaDir -Force
    Remove-Item $zip
    # Some releases nest files in a subfolder
    $exe = Get-ChildItem $llamaDir -Recurse -Filter llama-server.exe | Select-Object -First 1
    if ($exe.DirectoryName -ne (Resolve-Path $llamaDir).Path) {
        Get-ChildItem $exe.DirectoryName | Move-Item -Destination $llamaDir -Force
    }
}
Write-Host "llama.cpp ready"

Write-Host "== 4/4 Downloading speech models (Parakeet STT, Kokoro TTS) ==" -ForegroundColor Cyan
python -m uv run python -m agent.download

if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
Write-Host ""
Write-Host "Setup done. Next:" -ForegroundColor Green
Write-Host "  Terminal 1:  powershell -ExecutionPolicy Bypass -File scripts\start_llm.ps1"
Write-Host "  Terminal 2:  powershell -ExecutionPolicy Bypass -File scripts\start_agent.ps1"
