$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
        uv venv .venv --python 3.14
        if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
    }
    uv pip sync --python .venv/Scripts/python.exe requirements-lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency setup failed' }
    Invoke-RestMethod http://127.0.0.1:11434/api/version -TimeoutSec 5 | Out-Null
    & .venv/Scripts/python.exe -m unittest -v
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
    & .venv/Scripts/python.exe experiment.py prepare
    if ($LASTEXITCODE -ne 0) { throw 'Dataset preparation failed' }
    $runs = @(
        @{ Partition = 'development'; Count = 24; Name = 'pilot-8b-v1' },
        @{ Partition = 'controller_train'; Count = 96; Name = 'train-8b-v1' },
        @{ Partition = 'validation'; Count = 32; Name = 'validation-8b-v1' },
        @{ Partition = 'audit'; Count = 32; Name = 'audit-8b-v1' }
    )
    foreach ($run in $runs) {
        & .venv/Scripts/python.exe experiment.py run --model qwen3:8b --partition $run.Partition --count $run.Count --name $run.Name
        if ($LASTEXITCODE -ne 0) { throw "Execution failed: $($run.Name)" }
    }
    & .venv/Scripts/python.exe controller.py --train train-8b-v1 --validation validation-8b-v1 --audit audit-8b-v1
    if ($LASTEXITCODE -ne 0) { throw 'Controller fitting failed' }
}
finally {
    Pop-Location
}
