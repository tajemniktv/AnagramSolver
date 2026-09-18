# Filesystem integration tests: never touches the real installation or starts an app.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'desktop-installation.ps1')
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$root = Join-Path $project ('.codex/temp/installation-tests-' + [guid]::NewGuid().ToString('N'))
function Write-Fixture([string]$Path, [string]$Content) {
    [void][IO.Directory]::CreateDirectory((Split-Path -Parent $Path))
    [IO.File]::WriteAllText($Path, $Content)
}
function Assert-True($Value, [string]$Message) { if (-not $Value) { throw $Message } }
function New-Fixture([string]$Name, [switch]$Current) {
    $base = Join-Path $root $Name
    $f = @{ Project=(Join-Path $base 'project'); LocalData=(Join-Path $base 'local'); Programs=(Join-Path $base 'shortcuts'); Candidate=(Join-Path $base 'project/candidate.exe') }
    $installName = if ($Current) { 'TajsAnagrams' } else { 'AnagramSolver' }
    $install = Join-Path $f.LocalData "Programs/TajemnikTV/$installName"
    Write-Fixture (Join-Path $install "$installName.exe") 'current executable'
    Write-Fixture (Join-Path $install 'backup/TajsAnagrams.exe') 'older distinct backup'
    $profile = if ($Current) { Join-Path $install 'user-data' } else { Join-Path $f.LocalData 'tv.tajemnik.anagramsolver' }
    Write-Fixture (Join-Path $profile 'settings.json') '{"custom":"keep me"}'
    $corpus = if ($Current) { Join-Path $install 'corpora' } else { Join-Path $profile 'corpora' }
    Write-Fixture (Join-Path $corpus 'dictionary/normal_user_v2.txt') 'user corpus'
    Write-Fixture (Join-Path $f.Programs "$installName.lnk") 'original shortcut'
    Write-Fixture (Join-Path $f.Project '.anagram_data/ngrams/count_1w.txt') 'a 100'
    Write-Fixture (Join-Path $f.Project '.anagram_data/ngrams/count_2w.txt') 'a word 100'
    Write-Fixture $f.Candidate 'new executable'
    return $f
}
function Get-UserSnapshot($Fixture) {
    $rows = foreach ($dir in @($Fixture.LocalData, $Fixture.Programs)) {
        Get-ChildItem -LiteralPath $dir -Recurse -Force | Sort-Object FullName | ForEach-Object {
            $hash = if ($_.PSIsContainer) { 'directory' } else { (Get-FileHash -LiteralPath $_.FullName).Hash }
            "$($_.FullName):$hash"
        }
    }
    return $rows -join "`n"
}
$link = { param($exe, $install, $path) [IO.File]::WriteAllText($path, $exe) }
function Run-Fixture($f, [scriptblock]$After = {}, [scriptblock]$Activate = {}) {
    $plan = New-DesktopInstallPlan -Project $f.Project -LocalData $f.LocalData -Programs $f.Programs
    Invoke-DesktopInstallation -Plan $plan -Candidate $f.Candidate -CreateShortcut $link -Activate $Activate -AfterStep $After | Out-Null
}
$steps = [Collections.Generic.List[string]]::new()
$f = New-Fixture 'success'
Run-Fixture $f { param($step) $steps.Add($step) }
$installed = Join-Path $f.LocalData 'Programs/TajemnikTV/TajsAnagrams'
Assert-True ([IO.File]::ReadAllText((Join-Path $installed 'TajsAnagrams.exe')) -eq 'new executable') 'New executable missing'
Assert-True ([IO.File]::ReadAllText((Join-Path $installed 'backup/TajsAnagrams.exe')) -eq 'current executable') 'Previous release not retained'
Assert-True ([IO.File]::ReadAllText((Join-Path $installed 'user-data/settings.json')) -eq '{"custom":"keep me"}') 'Profile changed'
Assert-True ([IO.File]::ReadAllText((Join-Path $installed 'corpora/dictionary/normal_user_v2.txt')) -eq 'user corpus') 'Corpus overwritten'
Assert-True (-not (Test-Path -LiteralPath (Join-Path $f.LocalData 'Programs/TajemnikTV/AnagramSolver'))) 'Legacy install left behind'
for ($failAt = 1; $failAt -le $steps.Count; ++$failAt) {
    $f = New-Fixture "rollback-$failAt"
    $before = Get-UserSnapshot $f
    $state = @{ Count=0; Fail=$failAt }
    $hook = { param($step) $state.Count++; if ($state.Count -eq $state.Fail) { throw "Injected failure: $step" } }.GetNewClosure()
    try { Run-Fixture $f $hook; throw 'Expected injected failure' } catch { Assert-True ($_.Exception.Message -like '*Injected failure*') $_.Exception.Message }
    Assert-True ((Get-UserSnapshot $f) -eq $before) "Rollback changed original files/directories at step $failAt"
}
foreach ($current in @($false, $true)) {
    $f = New-Fixture "activation-$current" -Current:$current
    # Same-hash update must restore this attempt's current executable, not its older backup.
    Write-Fixture $f.Candidate 'current executable'
    $before = Get-UserSnapshot $f
    try {
        Run-Fixture $f -Activate {
            param($exe, $install)
            # Desktop::new rewrites relocated settings before constructing the WebView.
            [IO.File]::WriteAllText((Join-Path $install 'user-data/settings.json'), '{"corpora":"new installation path"}')
            throw 'Injected startup failure'
        }
        throw 'Expected startup failure'
    } catch { Assert-True ($_.Exception.Message -like '*Injected startup failure*') $_.Exception.Message }
    Assert-True ((Get-UserSnapshot $f) -eq $before) 'Startup rollback did not restore exact installation/profile/shortcut/backup'
}
foreach ($case in @('installs','profiles','corpora','staging','redirect')) {
    $f = New-Fixture "preflight-$case"
    $legacy = Join-Path $f.LocalData 'Programs/TajemnikTV/AnagramSolver'
    switch ($case) {
        'installs' { Write-Fixture (Join-Path $f.LocalData 'Programs/TajemnikTV/TajsAnagrams/other') 'conflict' }
        'profiles' { Write-Fixture (Join-Path $legacy 'user-data/settings.json') 'conflict' }
        'corpora' { Write-Fixture (Join-Path $legacy 'corpora/other') 'conflict' }
        'staging' { Write-Fixture (Join-Path $legacy 'leftover.installing') 'incomplete' }
        'redirect' { New-Item -ItemType Junction -Path (Join-Path $legacy 'redirect') -Target $f.Project | Out-Null }
    }
    $before = Get-UserSnapshot $f
    $state = @{ Mutated=$false }
    try { Run-Fixture $f { $state.Mutated=$true }; throw 'Expected preflight failure' } catch { Assert-True ($_.Exception.Message -notlike '*Expected preflight failure*') $_.Exception.Message }
    Assert-True (-not $state.Mutated) 'Preflight performed mutation'
    Assert-True ((Get-UserSnapshot $f) -eq $before) "Preflight changed user state: $case"
}
$f = New-Fixture 'copy-hash-failure'
$before = Get-UserSnapshot $f
$source = Join-Path $f.Project '.anagram_data/ngrams/count_1w.txt'
$state = @{ Once=$false }
$hook = { param($step) if (-not $state.Once) { $state.Once=$true; [IO.File]::WriteAllText($source, 'changed during installation') } }.GetNewClosure()
try { Run-Fixture $f $hook; throw 'Expected hash failure' } catch { Assert-True ($_.Exception.Message -like '*Copy verification failed*') $_.Exception.Message }
Assert-True ((Get-UserSnapshot $f) -eq $before) 'Failed copy was not rolled back'
Write-Output "Installation integration checks passed: success, $($steps.Count) injected mutation failures, startup rollback (legacy/current), five preflight rejections, and copy hash failure. Fixtures: $root"
