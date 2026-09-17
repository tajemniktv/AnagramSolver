# Explicit offline-data preparation; never runs during builds or ordinary solves.
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$OutputDirectory,
    [Parameter(Mandatory)][string]$NativeCli,
    [string]$SourceDirectory,
    [switch]$IncludeWiktionary,
    [switch]$IncludeWikipedia
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$output = [IO.Path]::GetFullPath($OutputDirectory)
$cli = (Resolve-Path -LiteralPath $NativeCli).Path
if (Test-Path -LiteralPath $output) { throw 'Choose a new output directory; existing corpora are never overwritten.' }
$scratch = Join-Path $project ('.codex/temp/native-data-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $scratch, $output | Out-Null
function Fetch([string]$Url, [string]$Path) {
    if ($SourceDirectory) {
        Copy-Item -LiteralPath (Join-Path $SourceDirectory (Split-Path -Leaf $Path)) -Destination $Path
        return
    }
    Invoke-WebRequest -Uri $Url -OutFile "$Path.partial" -UserAgent 'TajsAnagrams-native-data/1.0 (https://github.com/tajemniktv/AnagramSolver)' -TimeoutSec 120
    Move-Item -LiteralPath "$Path.partial" -Destination $Path
}
foreach ($directory in @('dictionary', 'ngrams', 'wordnet31/dict')) {
    New-Item -ItemType Directory -Path (Join-Path $output $directory) | Out-Null
}
Fetch 'https://phillipmfeldman.org/English/large.txt' (Join-Path $scratch 'base.txt')
Fetch 'https://norvig.com/ngrams/count_1w.txt' (Join-Path $output 'ngrams/count_1w.txt')
Fetch 'https://norvig.com/ngrams/count_2w.txt' (Join-Path $output 'ngrams/count_2w.txt')
$policy = & $cli prepare-dictionary (Join-Path $scratch 'base.txt') (Join-Path $output 'ngrams/count_1w.txt') (Join-Path $output 'dictionary/normal_user_v2.txt')
if ($LASTEXITCODE -ne 0) { throw "Dictionary preparation failed: $policy" }
$policy | Set-Content -LiteralPath (Join-Path $output 'dictionary/policy.json') -Encoding utf8
$archive = Join-Path $scratch 'wordnet.tar.gz'
Fetch 'https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz' $archive
# Extract only seven exact members, not the archive's arbitrary paths or links.
$required = @('index.noun','index.verb','index.adj','index.adv','data.verb','noun.exc','verb.exc')
$members = & tar -tzf $archive
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect WordNet archive' }
foreach ($name in $required) {
    $matches = @($members | Where-Object { $_ -match ('^(\./)?dict/' + [regex]::Escape($name) + '$') })
    if ($matches.Count -ne 1) { throw "Expected exactly one WordNet dict/$name member" }
    $member = $matches[0]
    & tar -xzf $archive -C $scratch -- $member
    if ($LASTEXITCODE -ne 0) { throw "WordNet extraction failed: $member" }
    $extracted = Get-Item -LiteralPath (Join-Path $scratch $member)
    if ($extracted.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Refusing linked WordNet member' }
    Copy-Item -LiteralPath $extracted.FullName -Destination (Join-Path $output "wordnet31/dict/$name")
}
$titles = @()
foreach ($source in @('enwiktionary','enwiki')) {
    if (($source -eq 'enwiktionary' -and $IncludeWiktionary) -or ($source -eq 'enwiki' -and $IncludeWikipedia)) {
        $path = Join-Path $scratch "$source.gz"
        Fetch "https://dumps.wikimedia.org/$source/latest/$source-latest-all-titles-in-ns0.gz" $path
        $titles += $path
    }
}
if ($titles.Count) {
    & $cli build-phrases (Join-Path $output 'phrases.db') @titles
    if ($LASTEXITCODE -ne 0) { throw 'Phrase build failed; source downloads retained for retry' }
}
Get-ChildItem -LiteralPath $output -File -Recurse | Get-FileHash | Select-Object Path,Hash |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $output 'manifest.json') -Encoding utf8
Write-Output "Prepared: $output. Select these paths in Settings. Downloads retained at $scratch. See dictionary/policy.json for corpus-derived extra short words."
