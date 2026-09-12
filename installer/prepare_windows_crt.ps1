# App-local Microsoft runtime, from the runner's official Visual Studio redist.
# Never install/update the user's system-wide VC++ runtime.
param([Parameter(Mandatory=$true)][string]$Runtime)
$ErrorActionPreference = 'Stop'
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vs) { throw 'Visual Studio C++ redistributable source not found' }
$versions = Get-ChildItem "$vs\VC\Redist\MSVC" -Directory | Where-Object { $_.Name -match '^\d+\.\d+\.\d+$' } | Sort-Object { [version]$_.Name } -Descending
$crt = $null
foreach ($version in $versions) {
    $candidate = Join-Path $version.FullName 'x64\Microsoft.VC143.CRT'
    if (Test-Path $candidate) { $crt = $candidate; break }
}
if (!$crt) { throw 'Microsoft.VC143.CRT x64 redistributable directory not found' }
$records = @()
foreach ($dll in (Get-ChildItem "$crt\*.dll")) {
    $signature = Get-AuthenticodeSignature $dll.FullName
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
        throw "Unverified Microsoft redistributable: $($dll.Name)"
    }
    Copy-Item -LiteralPath $dll.FullName -Destination (Join-Path $Runtime $dll.Name) -Force
    $records += [ordered]@{ name=$dll.Name; version=$dll.VersionInfo.FileVersion; sha256=(Get-FileHash $dll.FullName -Algorithm SHA256).Hash.ToLower(); signer=$signature.SignerCertificate.Subject }
}
foreach ($required in @('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll')) {
    if (!(Test-Path (Join-Path $Runtime $required))) { throw "Missing private runtime library: $required" }
}
$records | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 (Join-Path $Runtime 'sigma-desktop\microsoft-crt.json')
Write-Host "Packaged $($records.Count) signed Microsoft CRT libraries app-locally"
