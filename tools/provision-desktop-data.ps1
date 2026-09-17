# Provision runtime corpora once, independently of executable release rotation.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$source = Join-Path $project '.anagram_data'
$localData = [Environment]::GetFolderPath('LocalApplicationData')
$destination = [IO.Path]::GetFullPath((Join-Path $localData 'Programs/TajemnikTV/TajsAnagrams/corpora'))
$previous = [IO.Path]::GetFullPath((Join-Path $localData 'tv.tajemnik.anagramsolver/corpora'))
if ((Test-Path -LiteralPath $previous) -and (Test-Path -LiteralPath $destination)) {
    throw "Both legacy and current corpus folders exist. Reconcile them before migration; neither has been changed: $previous; $destination"
}
# Move only this app's previously provisioned data, never settings/cache or a broad root.
if ((Test-Path -LiteralPath $previous -PathType Container) -and -not (Test-Path -LiteralPath $destination)) {
    foreach ($path in @($previous, $destination)) {
        if (-not $path.StartsWith([IO.Path]::GetFullPath($localData) + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected corpus migration path' }
        $ancestor = $path
        while ($ancestor -and $ancestor -ne [IO.Path]::GetPathRoot($ancestor)) {
            if ((Test-Path -LiteralPath $ancestor) -and ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "Refusing redirected migration path: $ancestor" }
            $ancestor = Split-Path -Parent $ancestor
        }
    }
    if (Get-ChildItem -LiteralPath $previous -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint } | Select-Object -First 1) { throw 'Refusing redirected corpus tree' }
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Move-Item -LiteralPath $previous -Destination $destination
}
$files = @('dictionary/normal_user_v2.txt', 'ngrams/count_1w.txt', 'ngrams/count_2w.txt')
# Only the WordNet files the native core reads; license notices remain in the files.
$files += @('index.noun','index.verb','index.adj','index.adv','data.verb','noun.exc','verb.exc') | ForEach-Object { "wordnet31/dict/$_" }
foreach ($relative in $files) {
    $target = [IO.Path]::GetFullPath((Join-Path $destination $relative))
    $ancestor = $target
    while ($ancestor -and $ancestor -ne [IO.Path]::GetPathRoot($ancestor)) {
        if ((Test-Path -LiteralPath $ancestor) -and ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Refusing redirected corpus destination: $ancestor"
        }
        $ancestor = Split-Path -Parent $ancestor
    }
    # Existing provisioned corpora are user data, not overwritten on app upgrades.
    if (Test-Path -LiteralPath $target -PathType Leaf) { continue }
    $inputFile = Join-Path $source $relative
    if (-not (Test-Path -LiteralPath $inputFile -PathType Leaf)) {
        Write-Warning "Corpus not bundled: $relative. After launch, use Settings > Data to download/prepare a corpus set, select it, and save settings."
        continue
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
    $staged = "$target.installing"
    if (Test-Path -LiteralPath $staged) { throw "Previous incomplete corpus copy requires inspection: $staged" }
    Copy-Item -LiteralPath $inputFile -Destination $staged
    if ((Get-FileHash -LiteralPath $inputFile).Hash -ne (Get-FileHash -LiteralPath $staged).Hash) { throw "Corpus copy verification failed: $relative" }
    # No -Force: never replace data created by another installer in the meantime.
    Move-Item -LiteralPath $staged -Destination $target
}
Write-Output "Available runtime corpora copied to: $destination. Missing corpora can be prepared in Settings > Data."
$oldProfile = [IO.Path]::GetFullPath((Join-Path $localData 'tv.tajemnik.anagramsolver'))
$newProfile = [IO.Path]::GetFullPath((Join-Path $localData 'Programs/TajemnikTV/TajsAnagrams/user-data'))
# The build script closes the app before provisioning. Move the entire profile,
# including WebView state, without merging or overwriting existing user files.
# An older WebView crash reporter may recreate its old diagnostics directory
# after migration. It must not block later builds or replace the active profile.
if ((Test-Path -LiteralPath $oldProfile) -and (
    -not (Test-Path -LiteralPath $newProfile) -or
    (Test-Path -LiteralPath (Join-Path $oldProfile 'settings.json')) -or
    (Test-Path -LiteralPath (Join-Path $oldProfile 'results.sqlite')))) {
    if (Test-Path -LiteralPath $newProfile) { throw "Both old and new profiles exist; preserve and inspect them before migration: $oldProfile; $newProfile" }
    foreach ($path in @($oldProfile, $newProfile)) {
        if (-not $path.StartsWith([IO.Path]::GetFullPath($localData) + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected profile migration path' }
        $ancestor = $path
        while ($ancestor -and $ancestor -ne [IO.Path]::GetPathRoot($ancestor)) {
            if ((Test-Path -LiteralPath $ancestor) -and ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw "Refusing redirected profile path: $ancestor" }
            $ancestor = Split-Path -Parent $ancestor
        }
    }
    if (Get-ChildItem -LiteralPath $oldProfile -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint } | Select-Object -First 1) { throw 'Refusing redirected profile tree' }
    Move-Item -LiteralPath $oldProfile -Destination $newProfile
    Write-Output "Settings, solver cache and WebView profile moved to: $newProfile"
}
