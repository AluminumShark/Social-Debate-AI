# Deploy Social Debate AI to the test machine and build the GPU training image.
#
# Reads connection settings from .env (TESTBOX_HOST / TESTBOX_USER /
# TESTBOX_PASSWORD / TESTBOX_HOSTKEY). Requires PuTTY (plink/pscp).
#
# Usage:  powershell -ExecutionPolicy Bypass -File scripts/deploy_testbox.ps1
#
# Steps: package repo -> copy -> extract -> docker compose build.
# Training itself is launched separately (long-running); see README / PLAN.

$ErrorActionPreference = 'Stop'

# --- Load .env ---
$envFile = Join-Path $PSScriptRoot '..\.env'
$cfg = @{}
Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*([A-Z_]+)\s*=\s*(.*)$') { $cfg[$matches[1]] = $matches[2].Trim() }
}
$h = $cfg['TESTBOX_HOST']; $u = $cfg['TESTBOX_USER']
$pw = $cfg['TESTBOX_PASSWORD']; $hk = $cfg['TESTBOX_HOSTKEY']
$plink = 'C:\Program Files\PuTTY\plink.exe'
$pscp = 'C:\Program Files\PuTTY\pscp.exe'
$remoteDir = 'social-debate'

Write-Host "Packaging repo..."
$root = Join-Path $PSScriptRoot '..'
$tgz = Join-Path $root 'social-debate.tgz'
tar -czf $tgz --exclude=.venv --exclude=.git --exclude=__pycache__ `
    --exclude=data --exclude='*.tgz' --exclude=node_modules -C $root .

Write-Host "Copying to ${u}@${h}..."
& $pscp -pw $pw -hostkey $hk $tgz "${u}@${h}:~/social-debate.tgz"

Write-Host "Extracting + building image on remote (this can take a while)..."
$remote = @"
set -e
mkdir -p ~/$remoteDir
tar -xzf ~/social-debate.tgz -C ~/$remoteDir
cd ~/$remoteDir
docker compose -f docker/docker-compose.train.yml build
echo BUILD_DONE
"@
& $plink -ssh "${u}@${h}" -pw $pw -batch -hostkey $hk $remote

Write-Host "Deploy complete. To train:"
Write-Host "  ssh ${u}@${h} 'cd ~/$remoteDir && docker compose -f docker/docker-compose.train.yml run --rm train'"
