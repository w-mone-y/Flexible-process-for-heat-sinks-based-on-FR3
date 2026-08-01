[CmdletBinding()]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]] $ArgumentList
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

function Resolve-CondaExe {
    $command = Get-Command conda.exe -CommandType Application -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    $candidates = @()
    if ($env:CONDA_EXE) {
        $candidates += $env:CONDA_EXE
    }
    if ($env:SCOOP) {
        $candidates += Join-Path $env:SCOOP "apps\miniconda3\current\Scripts\conda.exe"
    }
    $candidates += @(
        (Join-Path ([Environment]::GetFolderPath("UserProfile")) "miniconda3\Scripts\conda.exe"),
        (Join-Path ([Environment]::GetFolderPath("UserProfile")) "anaconda3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:LOCALAPPDATA "anaconda3\Scripts\conda.exe"),
        (Join-Path $env:ProgramData "miniconda3\Scripts\conda.exe"),
        (Join-Path $env:ProgramData "anaconda3\Scripts\conda.exe")
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return $candidate
        }
    }
    throw "找不到 conda.exe。请先安装 Conda，或把 conda 加入当前 PowerShell 的 PATH；此脚本不会安装依赖。"
}

$conda = Resolve-CondaExe
$probe = "import importlib.util, sys; required = ('mujoco', 'numpy', 'yaml', 'PySide6'); missing = [name for name in required if importlib.util.find_spec(name) is None]; sys.exit('conda env wy 缺少: ' + ', '.join(missing) + '. 请在 wy 环境中补齐依赖，不要使用全局 Python。') if missing else print('Using conda env wy')"

& $conda run --no-capture-output -n wy python -c $probe
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$entrypoint = Join-Path $repoRoot "brazing_line_v2.py"
& $conda run --no-capture-output -n wy python $entrypoint @ArgumentList
exit $LASTEXITCODE
