<#
Run the agent evaluation on AWS through cdeploy, then merge the results.

  # 1. free pipeline check (simulator, nothing is created on AWS)
  .\eval\run_cloud_eval.ps1 -CdeployHome C:\path\to\Cloud_deployer

  # 2. the real run (cdeploy shows the plan and asks before spending money)
  .\eval\run_cloud_eval.ps1 -CdeployHome C:\path\to\Cloud_deployer -Real -Matches 6 -Shards 3

Run it from the repo root (the folder with a3.py).
AWS keys: set $env:AWS_ACCESS_KEY_ID / $env:AWS_SECRET_ACCESS_KEY for this
shell, or keep them in %USERPROFILE%\.aws\credentials. This script never
writes or prints them.
#>
param(
    [string]$CdeployHome = $env:CDEPLOY_HOME,
    [switch]$Real,
    [int]$Matches = 6,
    [int]$Shards = 2,
    [string]$Opponents = "full,baseline",
    [string]$Instance = "c6i.xlarge",
    [string]$Region = "us-east-1",
    [double]$Budget = 2.0,
    [ValidateSet("skypilot", "libcloud")][string]$Backend = "skypilot",
    [string]$ImageId = "",
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
if (-not (Test-Path "$Repo\a3.py")) { throw "run this from the repo (a3.py not found in $Repo)" }

# cdeploy: either pip-installed, or importable from its folder
if ($CdeployHome) {
    if (-not (Test-Path (Join-Path $CdeployHome "cdeploy"))) {
        throw "no cdeploy package in $CdeployHome (expected $CdeployHome\cdeploy\)"
    }
    $env:PYTHONPATH = "$CdeployHome;$env:PYTHONPATH"
    $env:CDEPLOY_HOME = $CdeployHome
}
& $Python -c "import cdeploy" 2>$null
if ($LASTEXITCODE -ne 0) { throw "cannot import cdeploy: pass -CdeployHome <Cloud_deployer folder> or pip install -e it" }

$Mode = "dry-run"
if ($Real) {
    $Mode = "real"
    $haveEnv = $env:AWS_ACCESS_KEY_ID -and $env:AWS_SECRET_ACCESS_KEY
    $haveFile = Test-Path (Join-Path $HOME ".aws\credentials")
    if (-not ($haveEnv -or $haveFile)) {
        throw "no AWS credentials: set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or run 'aws configure'"
    }
    if ($Backend -eq "libcloud" -and -not $haveEnv) {
        throw "the libcloud backend reads the keys from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY"
    }
}

$Spec = "eval\cdeploy_eval.json"
$genArgs = @("eval\gen_spec.py", "--mode", $Mode, "--matches", $Matches,
             "--shards", $Shards, "--opponents", $Opponents,
             "--instance", $Instance, "--region", $Region,
             "--budget", $Budget, "--backend", $Backend, "--out", $Spec)
if ($ImageId) { $genArgs += @("--image-id", $ImageId) }
& $Python @genArgs
if ($LASTEXITCODE -ne 0) { throw "gen_spec failed" }

Write-Host "== plan"
& $Python -m cdeploy plan $Spec
if ($LASTEXITCODE -ne 0) { throw "cdeploy plan failed" }

Write-Host "== up ($Mode)"
if ($Real) { & $Python -m cdeploy up $Spec --real } else { & $Python -m cdeploy up $Spec }
$upCode = $LASTEXITCODE

Write-Host "== results"
& $Python eval\aggregate.py
if ($upCode -ne 0) {
    Write-Host "cdeploy exited with $upCode. Check for leftovers with: $Python -m cdeploy status  /  $Python -m cdeploy down <run-id>"
    exit $upCode
}
