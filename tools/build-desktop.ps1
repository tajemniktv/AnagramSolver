[CmdletBinding()]
param([switch]$TestOnly, [switch]$KeepBuildCache)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$desktop = Join-Path $project 'apps/desktop'
$build = Join-Path $project '.codex/temp/desktop-build'
$install = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Programs/TajemnikTV/TajsAnagrams'
$exe = Join-Path $install 'TajsAnagrams.exe'
$backup = Join-Path $install 'backup/TajsAnagrams.exe'
$oldTarget = $env:CARGO_TARGET_DIR
Push-Location $desktop
try {
    & pnpm install --frozen-lockfile
    if ($LASTEXITCODE -ne 0) { throw 'Desktop dependency installation failed' }
    & pnpm check
    if ($LASTEXITCODE -ne 0) { throw 'Desktop type check failed' }
    $env:CARGO_TARGET_DIR = $build
    & node (Join-Path $desktop 'node_modules/@tauri-apps/cli/tauri.js') build --no-bundle -- --locked
    if ($LASTEXITCODE -ne 0) { throw 'Desktop build failed; installed app unchanged' }
    $candidate = Join-Path $build 'release/TajsAnagrams.exe'
    if (-not (Test-Path -LiteralPath $candidate)) { throw 'Desktop executable missing' }
    if ($TestOnly -or $env:CI) { Write-Output "Test/CI executable (not installed or restarted): $candidate"; return }

    # Stop only this installation, gracefully, so the native owner cancels/joins work.
    Get-Process -Name TajsAnagrams -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq $exe } | ForEach-Object {
        if (-not $_.CloseMainWindow() -or -not $_.WaitForExit(15000)) { throw 'Close TajsAnagrams before installing; no process was forcibly terminated.' }
    }
    # Preserve the complete previous installation and its user-owned data on rename.
    $legacy = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Programs/TajemnikTV/AnagramSolver'
    if (Test-Path -LiteralPath $legacy) {
        if (Test-Path -LiteralPath $install) { throw 'Both app installations exist; inspect before merging user data.' }
        Get-Process -Name AnagramSolver -ErrorAction SilentlyContinue | Where-Object { $_.Path -eq (Join-Path $legacy 'AnagramSolver.exe') } | ForEach-Object {
            if (-not $_.CloseMainWindow() -or -not $_.WaitForExit(15000)) { throw 'Close the previous app before migrating its data.' }
        }
        $parent = [IO.Path]::GetFullPath((Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'Programs/TajemnikTV'))
        foreach ($path in @($legacy, $install)) {
            if ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($path)) -ne $parent) { throw 'Unexpected migration target' }
        }
        if ((Get-Item -LiteralPath $legacy -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refusing redirected legacy installation' }
        Move-Item -LiteralPath $legacy -Destination $install
        $legacyExe = Join-Path $install 'AnagramSolver.exe'
        if (Test-Path -LiteralPath $legacyExe) { Move-Item -LiteralPath $legacyExe -Destination $exe }
        $legacyShortcut = Join-Path ([Environment]::GetFolderPath('Programs')) 'AnagramSolver.lnk'
        if (Test-Path -LiteralPath $legacyShortcut) { Move-Item -LiteralPath $legacyShortcut -Destination (Join-Path $install 'legacy-shortcut.lnk') }
    }
    & (Join-Path $PSScriptRoot 'provision-desktop-data.ps1')
    New-Item -ItemType Directory -Path $install,(Join-Path $install 'backup') -Force | Out-Null
    foreach ($path in @($install,(Join-Path $install 'backup'))) {
        if ((Get-Item -LiteralPath $path -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing redirected install path: $path" }
    }
    $staged = Join-Path $install 'TajsAnagrams.next.exe'
    Copy-Item -LiteralPath $candidate -Destination $staged -Force
    $hash = (Get-FileHash -LiteralPath $candidate).Hash
    if ((Get-FileHash -LiteralPath $staged).Hash -ne $hash) { throw 'Staged executable hash mismatch' }
    $rollback = $null
    if (Test-Path -LiteralPath $exe) {
        # Preserve this attempt's predecessor independently of the rotating backup.
        # An identical rebuild must not roll back to an older distinct release.
        $rollback = Join-Path $project ('.codex/temp/install-rollback-' + [guid]::NewGuid().ToString('N') + '.exe')
        $rollbackHash = (Get-FileHash -LiteralPath $exe).Hash
        Copy-Item -LiteralPath $exe -Destination $rollback
        if ((Get-FileHash -LiteralPath $rollback).Hash -ne $rollbackHash) { throw 'Pre-install snapshot verification failed' }
        if ((Get-FileHash -LiteralPath $exe).Hash -ne $hash -or -not (Test-Path -LiteralPath $backup)) {
            Copy-Item -LiteralPath $exe -Destination $backup -Force
            if ((Get-FileHash -LiteralPath $exe).Hash -ne (Get-FileHash -LiteralPath $backup).Hash) { throw 'Backup verification failed' }
        }
    }
    # The rename must not leave two rotating executable backups behind.
    $legacyBackup = Join-Path $install 'backup/AnagramSolver.exe'
    if ((Test-Path -LiteralPath $legacyBackup -PathType Leaf) -and (Test-Path -LiteralPath $backup -PathType Leaf)) {
        if ((Get-Item -LiteralPath $legacyBackup -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refusing redirected legacy backup' }
        Remove-Item -LiteralPath $legacyBackup
        Write-Output 'Removed obsolete pre-rename executable backup; the previous release is retained as backup/TajsAnagrams.exe.'
    }
    Move-Item -LiteralPath $staged -Destination $exe -Force
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Programs')) 'TajsAnagrams.lnk'))
    $shortcut.TargetPath = $exe
    $shortcut.WorkingDirectory = $install
    $shortcut.Description = 'Solve anagrams locally'
    $shortcut.Save()
    # This is the requested interactive app, not a background helper.
    $process = Start-Process -FilePath $exe -WorkingDirectory $install -WindowStyle Normal -PassThru
    Start-Sleep -Seconds 3
    if ($process.HasExited) {
        if ($rollback) {
            Copy-Item -LiteralPath $rollback -Destination $staged
            if ((Get-FileHash -LiteralPath $staged).Hash -ne $rollbackHash) { throw 'Startup failed and rollback staging verification failed; backup retained.' }
            Move-Item -LiteralPath $staged -Destination $exe -Force
            if ((Get-FileHash -LiteralPath $exe).Hash -ne $rollbackHash) { throw 'Startup failed and rollback verification failed; backup retained.' }
            Start-Process -FilePath $exe -WorkingDirectory $install -WindowStyle Normal
            Remove-Item -LiteralPath $rollback
            throw 'New app exited during startup; restored the verified previous release and requested its restart.'
        }
        throw 'Installed app exited during startup; this first install has no previous release to restore.'
    }
    Write-Output "Installed and started: $exe (PID $($process.Id), SHA256 $hash)"
    if ($rollback) { Remove-Item -LiteralPath $rollback }
    if ($KeepBuildCache) {
        Write-Output 'Build cache retained explicitly; installed release rotation is unchanged.'
        return
    }
    # Installed current/backup are the retained artifacts, not Cargo's build cache.
    $resolvedBuild = [IO.Path]::GetFullPath($build)
    if ($resolvedBuild -ne [IO.Path]::GetFullPath((Join-Path $project '.codex/temp/desktop-build'))) { throw 'Unexpected build cleanup path' }
    $ancestor = $resolvedBuild
    while ($ancestor -ne $project) {
        if ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing redirected cleanup path: $ancestor" }
        $ancestor = Split-Path -Parent $ancestor
    }
    if (Get-ChildItem -LiteralPath $resolvedBuild -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint } | Select-Object -First 1) { throw 'Refusing redirected build tree' }
    Remove-Item -LiteralPath $resolvedBuild -Recurse -Force
    Write-Output 'Current installed release and one rotating backup retained; Cargo build scratch removed.'
} finally {
    $env:CARGO_TARGET_DIR = $oldTarget
    Pop-Location
}
