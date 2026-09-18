# One owner for preflight, migration, provisioning and executable publication.
# Rollback covers synchronous failures; a process/power failure retains recovery
# snapshots and a journal in the project's .codex/temp for manual recovery.
function Assert-DesktopPath([string]$Path, [switch]$Tree) {
    $ancestor = [IO.Path]::GetFullPath($Path)
    while ($ancestor) {
        if (Test-Path -LiteralPath $ancestor) {
            if ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Refusing redirected path: $ancestor" }
        }
        $ancestor = Split-Path -Parent $ancestor
    }
    if ($Tree -and (Test-Path -LiteralPath $Path -PathType Container)) {
        if (Get-ChildItem -LiteralPath $Path -Force -Recurse | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint } | Select-Object -First 1) { throw "Refusing redirected tree: $Path" }
    }
}

function New-DesktopInstallPlan {
    param([string]$Project, [string]$LocalData, [string]$Programs, [switch]$DataOnly)
    $projectPath = [IO.Path]::GetFullPath($Project)
    $parent = [IO.Path]::GetFullPath((Join-Path $LocalData 'Programs/TajemnikTV'))
    $install = Join-Path $parent 'TajsAnagrams'
    $legacy = Join-Path $parent 'AnagramSolver'
    $oldProfile = [IO.Path]::GetFullPath((Join-Path $LocalData 'tv.tajemnik.anagramsolver'))
    $legacyExists = Test-Path -LiteralPath $legacy
    if ($legacyExists -and (Test-Path -LiteralPath $install)) { throw 'Both app installations exist; inspect before merging user data.' }
    if ($DataOnly -and $legacyExists) { throw 'Use build-desktop.ps1 to migrate the legacy installation and its shortcut together.' }
    $effective = if ($legacyExists) { $legacy } else { $install }
    foreach ($path in @($effective, $oldProfile)) { Assert-DesktopPath $path -Tree }
    foreach ($path in @($install, $Programs, (Join-Path $projectPath '.codex/temp'))) { Assert-DesktopPath $path }
    foreach ($path in @($effective, $oldProfile)) {
        if ((Test-Path -LiteralPath $path) -and -not (Test-Path -LiteralPath $path -PathType Container)) { throw "Expected a directory: $path" }
        if (Test-Path -LiteralPath $path) {
            if (Get-ChildItem -LiteralPath $path -Recurse -Force -File | Where-Object { $_.Name -like '*.installing' -or $_.Name -eq 'TajsAnagrams.next.exe' } | Select-Object -First 1) { throw "Incomplete staged artifact requires inspection: $path" }
        }
    }
    if ($legacyExists -and (Test-Path -LiteralPath (Join-Path $legacy 'AnagramSolver.exe')) -and (Test-Path -LiteralPath (Join-Path $legacy 'TajsAnagrams.exe'))) { throw 'Both legacy and current executable names exist.' }
    $oldCorpora = Join-Path $oldProfile 'corpora'
    if ((Test-Path -LiteralPath $oldCorpora) -and (Test-Path -LiteralPath (Join-Path $effective 'corpora'))) { throw 'Both legacy and current corpus folders exist; neither has been changed.' }
    $newProfile = Join-Path $effective 'user-data'
    # Ignore only an old diagnostics-only profile recreated by WebView after migration.
    $moveProfile = (Test-Path -LiteralPath $oldProfile) -and (
        -not (Test-Path -LiteralPath $newProfile) -or
        (Test-Path -LiteralPath (Join-Path $oldProfile 'settings.json')) -or
        (Test-Path -LiteralPath (Join-Path $oldProfile 'results.sqlite')) -or
        (Test-Path -LiteralPath $oldCorpora))
    if ($moveProfile -and (Test-Path -LiteralPath $newProfile)) { throw 'Both old and new profiles exist; preserve and inspect them before migration.' }
    $legacyShortcut = Join-Path $Programs 'AnagramSolver.lnk'
    $shortcut = Join-Path $Programs 'TajsAnagrams.lnk'
    foreach ($path in @($legacyShortcut, $shortcut)) { Assert-DesktopPath $path }
    if ($legacyExists -and (Test-Path -LiteralPath $legacyShortcut) -and (Test-Path -LiteralPath (Join-Path $effective 'legacy-shortcut.lnk'))) { throw 'A previous legacy shortcut backup already exists.' }
    foreach ($relative in @('corpora', 'user-data', 'backup')) {
        $path = Join-Path $effective $relative
        if ((Test-Path -LiteralPath $path) -and -not (Test-Path -LiteralPath $path -PathType Container)) { throw "Expected a directory: $path" }
    }
    $files = @('dictionary/normal_user_v2.txt', 'ngrams/count_1w.txt', 'ngrams/count_2w.txt')
    $files += @('index.noun','index.verb','index.adj','index.adv','data.verb','noun.exc','verb.exc') | ForEach-Object { "wordnet31/dict/$_" }
    $copies = @()
    foreach ($relative in $files) {
        $existing = if (Test-Path -LiteralPath $oldCorpora) { Join-Path $oldCorpora $relative } else { Join-Path $effective "corpora/$relative" }
        Assert-DesktopPath $existing
        if (Test-Path -LiteralPath $existing) {
            if (-not (Test-Path -LiteralPath $existing -PathType Leaf)) { throw "Expected corpus file: $existing" }
            continue
        }
        $source = Join-Path $projectPath ".anagram_data/$relative"
        Assert-DesktopPath $source
        if (Test-Path -LiteralPath $source -PathType Leaf) {
            $copies += [pscustomobject]@{ Source=$source; Destination=(Join-Path $install "corpora/$relative"); Hash=(Get-FileHash -LiteralPath $source).Hash }
        }
    }
    [pscustomobject]@{ Project=$projectPath; LocalData=$LocalData; Programs=$Programs; DataOnly=[bool]$DataOnly; Install=$install; Legacy=$legacy; MoveInstall=$legacyExists; OldProfile=$oldProfile; MoveProfile=$moveProfile; OldCorpora=$oldCorpora; MoveCorpora=(Test-Path -LiteralPath $oldCorpora); LegacyShortcut=$legacyShortcut; Shortcut=$shortcut; Copies=$copies }
}

function Add-DesktopJournal($Tx, $Entry) {
    $Tx.Journal.Add($Entry)
    ConvertTo-Json -InputObject @($Tx.Journal.ToArray()) -Depth 5 | Set-Content -LiteralPath (Join-Path $Tx.Recovery 'journal.json') -Encoding utf8
}
function Complete-DesktopStep($Tx, [string]$Name) { & $Tx.AfterStep $Name }
function New-DesktopDirectory($Tx, [string]$Path) {
    Assert-DesktopPath $Path
    if (Test-Path -LiteralPath $Path -PathType Container) { return }
    New-DesktopDirectory $Tx (Split-Path -Parent $Path)
    [void][IO.Directory]::CreateDirectory($Path)
    Add-DesktopJournal $Tx ([pscustomobject]@{ Kind='directory'; Destination=$Path })
}
function Move-DesktopItem($Tx, [string]$Source, [string]$Destination) {
    Assert-DesktopPath $Source -Tree; Assert-DesktopPath $Destination
    if (Test-Path -LiteralPath $Destination) { throw "Refusing to replace migration destination: $Destination" }
    New-DesktopDirectory $Tx (Split-Path -Parent $Destination)
    if (Test-Path -LiteralPath $Source -PathType Container) { [IO.Directory]::Move($Source, $Destination) } else { [IO.File]::Move($Source, $Destination) }
    Add-DesktopJournal $Tx ([pscustomobject]@{ Kind='move'; Source=$Source; Destination=$Destination })
    Complete-DesktopStep $Tx "move:$Destination"
}
function Save-DesktopFile($Tx, [string]$Destination) {
    Assert-DesktopPath $Destination
    $original = $null
    if (Test-Path -LiteralPath $Destination) {
        $original = Join-Path $Tx.Recovery ([guid]::NewGuid().ToString('N') + '.original')
        [IO.File]::Copy($Destination, $original, $false)
        if ((Get-FileHash -LiteralPath $Destination).Hash -ne (Get-FileHash -LiteralPath $original).Hash) { throw "Snapshot verification failed: $Destination" }
    }
    Add-DesktopJournal $Tx ([pscustomobject]@{ Kind='file'; Source=$original; Destination=$Destination })
    return $original
}
function Set-DesktopFile($Tx, [string]$Source, [string]$Destination, [string]$ExpectedHash = '') {
    Assert-DesktopPath $Source; Assert-DesktopPath $Destination
    New-DesktopDirectory $Tx (Split-Path -Parent $Destination)
    if (-not $ExpectedHash) { $ExpectedHash = (Get-FileHash -LiteralPath $Source).Hash }
    $original = Save-DesktopFile $Tx $Destination
    [IO.File]::Copy($Source, $Destination, [bool]$original)
    if ((Get-FileHash -LiteralPath $Destination).Hash -ne $ExpectedHash) { throw "Copy verification failed: $Destination" }
    Complete-DesktopStep $Tx "file:$Destination"
}
function Undo-DesktopInstallation($Tx) {
    $failures = @()
    for ($i = $Tx.Journal.Count - 1; $i -ge 0; --$i) {
        $entry = $Tx.Journal[$i]
        try {
            Assert-DesktopPath $entry.Destination -Tree
            switch ($entry.Kind) {
                'file' {
                    if ($entry.Source) {
                        [IO.File]::Copy($entry.Source, $entry.Destination, $true)
                        if ((Get-FileHash -LiteralPath $entry.Source).Hash -ne (Get-FileHash -LiteralPath $entry.Destination).Hash) { throw 'Rollback hash mismatch' }
                    } elseif (Test-Path -LiteralPath $entry.Destination -PathType Leaf) { [IO.File]::Delete($entry.Destination) }
                }
                'move' {
                    Assert-DesktopPath $entry.Source
                    if (Test-Path -LiteralPath $entry.Source) { throw "Rollback source unexpectedly exists: $($entry.Source)" }
                    if (Test-Path -LiteralPath $entry.Destination -PathType Container) { [IO.Directory]::Move($entry.Destination, $entry.Source) } else { [IO.File]::Move($entry.Destination, $entry.Source) }
                }
                'directory' { [IO.Directory]::Delete($entry.Destination, $false) } # Empty only; never erase unowned files.
            }
        } catch { $failures += $_.Exception.Message; break } # Preserve ancestor paths if an inner undo failed.
    }
    if ($failures.Count) { throw "Rollback incomplete; recovery retained at $($Tx.Recovery): $($failures -join '; ')" }
}

function Invoke-DesktopInstallation {
    param($Plan, [string]$Candidate = '', [scriptblock]$StopApps, [scriptblock]$Activate,
          [scriptblock]$CreateShortcut, [scriptblock]$AfterStep = {})
    # Repeat preflight immediately before mutation; no process or disk changes on rejection.
    $plan = New-DesktopInstallPlan -Project $Plan.Project -LocalData $Plan.LocalData -Programs $Plan.Programs -DataOnly:$Plan.DataOnly
    if ($Candidate) { Assert-DesktopPath $Candidate; $candidateHash = (Get-FileHash -LiteralPath $Candidate).Hash }
    if ($StopApps) { & $StopApps $plan }
    $temp = Join-Path $plan.Project '.codex/temp'
    [void][IO.Directory]::CreateDirectory($temp)
    $recovery = Join-Path $temp ('desktop-install-' + [guid]::NewGuid().ToString('N'))
    [void][IO.Directory]::CreateDirectory($recovery)
    $tx = @{ Recovery=$recovery; Journal=[Collections.Generic.List[object]]::new(); AfterStep=$AfterStep }
    try {
        if ($plan.MoveInstall) {
            Move-DesktopItem $tx $plan.Legacy $plan.Install
            $oldExe = Join-Path $plan.Install 'AnagramSolver.exe'
            if (Test-Path -LiteralPath $oldExe) { Move-DesktopItem $tx $oldExe (Join-Path $plan.Install 'TajsAnagrams.exe') }
            if (Test-Path -LiteralPath $plan.LegacyShortcut) { Move-DesktopItem $tx $plan.LegacyShortcut (Join-Path $plan.Install 'legacy-shortcut.lnk') }
        }
        if ($plan.MoveCorpora) { Move-DesktopItem $tx $plan.OldCorpora (Join-Path $plan.Install 'corpora') }
        if ($plan.MoveProfile) { Move-DesktopItem $tx $plan.OldProfile (Join-Path $plan.Install 'user-data') }
        foreach ($copy in $plan.Copies) { Set-DesktopFile $tx $copy.Source $copy.Destination $copy.Hash }
        if ($Candidate) {
            $exe = Join-Path $plan.Install 'TajsAnagrams.exe'
            $backup = Join-Path $plan.Install 'backup/TajsAnagrams.exe'
            if ((Test-Path -LiteralPath $exe) -and ((Get-FileHash -LiteralPath $exe).Hash -ne $candidateHash -or -not (Test-Path -LiteralPath $backup))) { Set-DesktopFile $tx $exe $backup }
            $oldBackup = Join-Path $plan.Install 'backup/AnagramSolver.exe'
            if ((Test-Path -LiteralPath $oldBackup) -and (Test-Path -LiteralPath $backup)) { Move-DesktopItem $tx $oldBackup (Join-Path $recovery 'legacy-backup.exe') }
            Set-DesktopFile $tx $Candidate $exe $candidateHash
            if ($CreateShortcut) {
                $link = Join-Path $recovery 'TajsAnagrams.lnk'
                & $CreateShortcut $exe $plan.Install $link
                Set-DesktopFile $tx $link $plan.Shortcut
            }
            if ($Activate) {
                # Startup persists relocated paths before WebView construction can fail.
                # Restore that content before reversing the profile/install moves.
                $null = Save-DesktopFile $tx (Join-Path $plan.Install 'user-data/settings.json')
                & $Activate $exe $plan.Install
            }
        }
    } catch {
        $failure = $_
        Undo-DesktopInstallation $tx
        throw "Installation failed and filesystem changes were rolled back. Recovery: $recovery. Cause: $($failure.Exception.Message)"
    }
    # Commit: only our generated recovery directory is removed, never app/user data.
    if ([IO.Path]::GetDirectoryName([IO.Path]::GetFullPath($recovery)) -ne [IO.Path]::GetFullPath($temp)) { throw 'Unexpected recovery cleanup path' }
    Assert-DesktopPath $recovery -Tree
    Remove-Item -LiteralPath $recovery -Recurse -Force
    Write-Output "Installation/provisioning committed: $($plan.Install). Missing corpora can be prepared in Settings > Data."
}
