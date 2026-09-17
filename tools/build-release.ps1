# Build and retain only the current native CLI release and one rotating backup.
# This is a development CLI build, not a desktop-app dogfood/CI entrypoint.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$target = Join-Path $project 'target'
$tempRoot = Join-Path $project '.codex/temp'
$scratch = Join-Path $tempRoot ('release-' + [guid]::NewGuid().ToString('N'))
$current = Join-Path $target 'release/anagram-cli.exe'
$backup = Join-Path $target 'backup/anagram-cli.exe'

function Assert-ProjectTree([string]$Path) {
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($project + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing filesystem operation outside project: $full"
    }
    # Reject junctions/symlinks in the target or its ancestors before recursion.
    $ancestor = $full
    while ($ancestor -and $ancestor -ne $project) {
        if (Test-Path -LiteralPath $ancestor) {
            if ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Refusing reparse-point path: $ancestor"
            }
        }
        $ancestor = Split-Path -Parent $ancestor
    }
    if (Test-Path -LiteralPath $full) {
        if (Get-ChildItem -LiteralPath $full -Force -Recurse | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint } | Select-Object -First 1) {
            throw "Refusing recursive cleanup of a tree containing reparse points: $full"
        }
    }
}

if (Get-Process -Name cargo,rustc,anagram-cli -ErrorAction SilentlyContinue) {
    throw 'Close running Rust builds and the native CLI before rotating releases.'
}
Assert-ProjectTree $target
Assert-ProjectTree $scratch
$before = (Get-ChildItem -LiteralPath $target -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
New-Item -ItemType Directory -Path $scratch -Force | Out-Null
Push-Location $project
try {
    $build = Join-Path $scratch 'build'
    & cargo build --release --locked -p anagram-cli --target-dir $build
    if ($LASTEXITCODE -ne 0) { throw 'Release build failed; existing releases are untouched.' }
    $candidate = Join-Path $build 'release/anagram-cli.exe'
    $fixtures = Get-Content -LiteralPath (Join-Path $project 'contracts/fixtures.json') -Raw | ConvertFrom-Json
    $request = ($fixtures | Where-Object { $_.schema -eq 'GenerateRequest' -and $_.valid } | Select-Object -First 1).value
    $output = ($request | ConvertTo-Json -Depth 20 -Compress) | & $candidate generate (Join-Path $project 'tests/parity/dictionary.txt')
    if ($LASTEXITCODE -ne 0 -or ($output | ConvertFrom-Json).kind -ne 'generation_only') {
        throw 'Release smoke check failed; existing releases are untouched.'
    }
    $stagedCurrent = Join-Path $scratch 'current.exe'
    $stagedBackup = Join-Path $scratch 'backup.exe'
    Copy-Item -LiteralPath $candidate -Destination $stagedCurrent
    $previous = if (Test-Path -LiteralPath $current) { $current } elseif (Test-Path -LiteralPath $backup) { $backup } else { $null }
    if ($previous) {
        # Rebuilding an identical executable must not displace a distinct backup.
        if ((Get-FileHash -LiteralPath $previous).Hash -eq (Get-FileHash -LiteralPath $candidate).Hash -and (Test-Path -LiteralPath $backup)) { $previous = $backup }
        Copy-Item -LiteralPath $previous -Destination $stagedBackup
        if ((Get-FileHash -LiteralPath $previous).Hash -ne (Get-FileHash -LiteralPath $stagedBackup).Hash) { throw 'Backup verification failed.' }
    }
    if ((Get-FileHash -LiteralPath $candidate).Hash -ne (Get-FileHash -LiteralPath $stagedCurrent).Hash) { throw 'Release verification failed.' }
    Assert-ProjectTree $target
    # All recoverable executables are staged outside target before build-cache removal.
    if (Test-Path -LiteralPath $target) {
        Get-ChildItem -LiteralPath $target -Force | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
    }
    New-Item -ItemType Directory -Path (Join-Path $target 'release'),(Join-Path $target 'backup') -Force | Out-Null
    Copy-Item -LiteralPath $stagedCurrent -Destination $current
    if (Test-Path -LiteralPath $stagedBackup) { Copy-Item -LiteralPath $stagedBackup -Destination $backup }
    foreach ($path in @($current, $backup)) {
        if (Test-Path -LiteralPath $path) {
            $expected = if ($path -eq $current) { $stagedCurrent } else { $stagedBackup }
            if ((Get-FileHash -LiteralPath $path).Hash -ne (Get-FileHash -LiteralPath $expected).Hash) { throw 'Published executable hash mismatch.' }
        }
    }
    Assert-ProjectTree $scratch
    Remove-Item -LiteralPath $scratch -Recurse -Force
    $after = (Get-ChildItem -LiteralPath $target -Recurse -File | Measure-Object Length -Sum).Sum
    Write-Output ('Release verified; target reduced from {0:N2} GiB to {1:N2} MiB.' -f ($before / 1GB), ($after / 1MB))
    Write-Output "Current: $current"
    if (Test-Path -LiteralPath $backup) { Write-Output "Backup:  $backup" }
} catch {
    Write-Warning "Recovery/build files retained in $scratch"
    throw
} finally {
    Pop-Location
}
