param([int]$Port = 8000)
$projectPython = Join-Path $PSScriptRoot '.venv-ser\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw '缺少 .venv-ser，请按 README.md 重建 Python 3.11 运行环境并安装 requirements.txt。'
}
& $projectPython (Join-Path $PSScriptRoot 'run.py') --port $Port
