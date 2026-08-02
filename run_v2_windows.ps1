#Requires -Version 7.0

[CmdletBinding()]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)]
    [string[]] $ArgumentList
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$envFile = Join-Path $repoRoot ".env"

if (-not (Test-Path -LiteralPath $envFile -PathType Leaf)) {
    throw "找不到 .env。请复制 .env.example 为 .env，并设置 CONDA_ENV_PATH；此脚本不会安装依赖。"
}

$settings = @{}
foreach ($line in Get-Content -LiteralPath $envFile) {
    $text = $line.Trim()
    if ([string]::IsNullOrWhiteSpace($text) -or $text.StartsWith("#")) {
        continue
    }
    $separator = $text.IndexOf("=")
    if ($separator -lt 1) {
        throw "无法解析 .env 行：$text"
    }
    $key = $text.Substring(0, $separator).Trim()
    $value = $text.Substring($separator + 1).Trim()
    if ($value.Length -ge 2 -and (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'")))) {
        $value = $value.Substring(1, $value.Length - 2)
    }
    $settings[$key] = $value
}

$envPathValue = $settings["CONDA_ENV_PATH"]
if ([string]::IsNullOrWhiteSpace($envPathValue)) {
    throw ".env 缺少 CONDA_ENV_PATH。请填写 Conda 环境目录的绝对路径；此脚本不会安装依赖。"
}

$envPath = [Environment]::ExpandEnvironmentVariables($envPathValue)
if (-not [IO.Path]::IsPathRooted($envPath)) {
    $envPath = Join-Path $repoRoot $envPath
}
$envPath = [IO.Path]::GetFullPath($envPath)
$python = Join-Path $envPath "python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "CONDA_ENV_PATH 未找到 python.exe：$envPath"
}

$probe = "import importlib.util, sys; required = ('mujoco', 'numpy', 'yaml', 'PySide6'); missing = [name for name in required if importlib.util.find_spec(name) is None]; sys.exit('配置的 Conda 环境缺少: ' + ', '.join(missing) + '。请在该环境中补齐依赖，不要使用全局 Python。') if missing else print('Using Conda environment: ' + sys.prefix)"

& $python -c $probe
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$entrypoint = Join-Path $repoRoot "brazing_line_v2.py"
& $python $entrypoint @ArgumentList
exit $LASTEXITCODE
