# Standalone data provisioning uses the same preflight and rollback owner as installation.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'desktop-installation.ps1')
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$plan = New-DesktopInstallPlan -Project $project -LocalData ([Environment]::GetFolderPath('LocalApplicationData')) -Programs ([Environment]::GetFolderPath('Programs')) -DataOnly
Invoke-DesktopInstallation -Plan $plan -StopApps {
    param($plan)
    $exe = Join-Path $plan.Install 'TajsAnagrams.exe'
    Get-Process -Name TajsAnagrams -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $exe } | ForEach-Object {
        if (-not $_.CloseMainWindow() -or -not $_.WaitForExit(15000)) { throw 'Close TajsAnagrams before provisioning data.' }
    }
}
