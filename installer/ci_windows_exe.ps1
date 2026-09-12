param([Parameter(Mandatory=$true)][ValidateSet('install','uninstall')][string]$Mode)
$ErrorActionPreference = 'Stop'
$output = (Resolve-Path dist-desktop).Path
$checks = Join-Path $output 'verification'
New-Item -ItemType Directory -Force $checks | Out-Null
if ($Mode -eq 'install') {
    $exe = (Get-ChildItem "$output\*.exe").FullName
    if (@($exe).Count -ne 1) { throw 'Expected exactly one complete EXE' }
    # Installation gets ONLY this one file, with an empty working directory.
    $download = "$env:RUNNER_TEMP\Single EXE Download"
    New-Item -ItemType Directory $download | Out-Null
    $setup = Join-Path $download (Split-Path $exe -Leaf)
    Copy-Item $exe $setup
    $prefix = "$env:LOCALAPPDATA\SIGMA Desktop Test\Private Runtime"
    $arguments = "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /DIR=`"$prefix`" /LOG=`"$checks\installation.log`""
    $proc = Start-Process $setup -ArgumentList $arguments -WorkingDirectory $download -Wait -PassThru
    if ($proc.ExitCode -ne 0) { throw "Installation failed: $($proc.ExitCode)" }
    # A second install must safely refuse a nonempty target.
    $protected = Join-Path $prefix 'user-image-sentinel.txt'
    Set-Content -LiteralPath $protected -Value 'must remain unchanged'
    $proc = Start-Process $setup -ArgumentList $arguments.Replace('installation.log','reinstall-refused.log') -WorkingDirectory $download -Wait -PassThru
    if ($proc.ExitCode -eq 0 -or (Get-Content $protected) -ne 'must remain unchanged') { throw 'Nonempty-folder protection failed' }
    Remove-Item -LiteralPath $protected
    Remove-Item -LiteralPath $download -Recurse -Force
    # No original wheel cache or build prefix may help the installed program.
    foreach ($directory in @("$env:RUNNER_TEMP\sigma-build", "$env:RUNNER_TEMP\sigma-dependency-cache")) {
        if (Test-Path $directory) { Remove-Item -LiteralPath $directory -Recurse -Force }
    }
    "SIGMA_TEST_PREFIX=$prefix" >> $env:GITHUB_ENV
    $record = Get-Content "$prefix\sigma-desktop\shortcuts.json" | ConvertFrom-Json
    $shell = New-Object -ComObject WScript.Shell
    foreach ($path in $record.paths) {
        $link = $shell.CreateShortcut($path)
        if ($link.TargetPath -ne "$prefix\pythonw.exe") { throw "Wrong shortcut target: $path" }
        if ($link.Arguments -ne "-I -B `"$prefix\sigma-desktop\launch.py`"") { throw "Non-isolated shortcut: $path" }
        if ($link.IconLocation -ne "$prefix\sigma-desktop\sigma.ico,0") { throw "Missing shortcut artwork: $path" }
    }
    $key = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\org.fenghuibao.sigma.desktop_is1'
    $entry = Get-ItemProperty $key
    if ($entry.DisplayName -ne 'SIGMA' -or $entry.DisplayIcon -ne "$prefix\sigma-desktop\sigma.ico") { throw 'Uninstall name/icon is incorrect' }
    python installer/verify_windows_icons.py --installer $exe --prefix $prefix --output "$checks\icons.json"
    if ($LASTEXITCODE -ne 0) { throw 'Installer PE icon verification failed' }
    python installer/verify_windows_payload.py --prefix $prefix --manifest "$output\payload-sha256.json" --output "$checks\payload.json"
    if ($LASTEXITCODE -ne 0) { throw 'Installed payload differs from build payload' }
} else {
    $prefix = $env:SIGMA_TEST_PREFIX
    $record = Get-Content "$checks\verification.json" | ConvertFrom-Json
    # Settings and arbitrary user-created files are intentionally preserved.
    $sentinel = Join-Path $prefix 'user-data-sentinel.txt'
    Set-Content -LiteralPath $sentinel -Value 'do not erase user data'
    $proc = Start-Process "$prefix\unins000.exe" -ArgumentList "/VERYSILENT /SUPPRESSMSGBOXES /NORESTART /LOG=`"$checks\uninstallation.log`"" -Wait -PassThru
    if ($proc.ExitCode -ne 0) { throw 'Uninstall failed' }
    foreach ($shortcut in $record.shortcuts.paths) {
        if (Test-Path $shortcut) { throw "Shortcut remains: $shortcut" }
    }
    foreach ($relative in @('python.exe','pythonw.exe','sigma-desktop\launch.py','Lib\site-packages\torch\__init__.py','unins000.exe')) {
        if (Test-Path (Join-Path $prefix $relative)) { throw "Application file remains: $relative" }
    }
    if (Test-Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\org.fenghuibao.sigma.desktop_is1') { throw 'Uninstall registry entry remains' }
    if ((Get-Content $sentinel) -ne 'do not erase user data') { throw 'User file was modified/deleted' }
    @{status='passed'; preserved_user_file=$true; removed_shortcuts=$true; removed_runtime=$true} | ConvertTo-Json | Set-Content "$checks\uninstall.json"
}
