# CI-only display driver, never embedded in or run by the SIGMA installer.
# GitHub's Windows VM does not provide a working hardware OpenGL context.
$ErrorActionPreference = 'Stop'
if ($env:GITHUB_ACTIONS -ne 'true' -or !$env:RUNNER_TEMP) {
    throw 'This script is restricted to disposable GitHub Actions runners.'
}
$archive = Join-Path $env:RUNNER_TEMP 'mesa3d-24.3.0-release-msvc.7z'
$destination = Join-Path $env:RUNNER_TEMP 'sigma-ci-mesa'
Invoke-WebRequest -Uri 'https://github.com/pal1000/mesa-dist-win/releases/download/24.3.0/mesa3d-24.3.0-release-msvc.7z' -OutFile $archive
$expected = '824d74f847dc25df8b3ec6b38096cdc79aaadf93c446e3973b6db4f307c38db9'
if ((Get-FileHash -Algorithm SHA256 $archive).Hash.ToLowerInvariant() -ne $expected) {
    throw 'Mesa archive SHA-256 mismatch'
}
& 7z x $archive "-o$destination" -y
if ($LASTEXITCODE -ne 0) { throw 'Mesa extraction failed' }
Push-Location $destination
try {
    & .\systemwidedeploy.cmd 1
    if ($LASTEXITCODE -ne 0) { throw 'Mesa OpenGL deployment failed' }
    & .\systemwidedeploy.cmd 7
    if ($LASTEXITCODE -ne 0) { throw 'Mesa deployment refresh failed' }
} finally { Pop-Location }
'GALLIUM_DRIVER=llvmpipe' >> $env:GITHUB_ENV
'LP_NUM_THREADS=0' >> $env:GITHUB_ENV
