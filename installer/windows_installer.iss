; Compile with /DRuntimeDir=... /DOutputDir=... /DAppVersion=...
; Everything is inside the EXE. No download, pip, PATH change or driver install.
[Setup]
AppId=org.fenghuibao.sigma.desktop
AppName=SIGMA
AppVersion={#AppVersion}
AppVerName=SIGMA
AppPublisher=SIGMA
AppPublisherURL=https://github.com/fenghuibao/napari-sigma
DefaultDirName={localappdata}\Programs\SIGMA
DefaultGroupName=SIGMA
DisableProgramGroupPage=yes
DisableDirPage=no
UsePreviousAppDir=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible and not arm64
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputDir}
OutputBaseFilename=SIGMA-{#AppVersion}-windows-x86_64-cu130
Compression=lzma2/ultra64
SolidCompression=yes
DiskSpanning=no
SetupIconFile={#RuntimeDir}\sigma-desktop\sigma.ico
UninstallDisplayIcon={app}\sigma-desktop\sigma.ico
UninstallDisplayName=SIGMA
WizardStyle=modern
LicenseFile={#RuntimeDir}\sigma-desktop\NOTICE.txt
InfoAfterFile={#RuntimeDir}\sigma-desktop\QUICKSTART.txt
CloseApplications=yes
RestartApplications=no
ChangesEnvironment=no
UninstallLogging=yes

[Files]
Source: "{#RuntimeDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\SIGMA"; Filename: "{app}\pythonw.exe"; Parameters: "-I -B ""{app}\sigma-desktop\launch.py"""; WorkingDir: "{app}"; IconFilename: "{app}\sigma-desktop\sigma.ico"; AppUserModelID: "SIGMA.Desktop"
Name: "{userdesktop}\SIGMA"; Filename: "{app}\pythonw.exe"; Parameters: "-I -B ""{app}\sigma-desktop\launch.py"""; WorkingDir: "{app}"; IconFilename: "{app}\sigma-desktop\sigma.ico"; AppUserModelID: "SIGMA.Desktop"

[Run]
Filename: "{app}\pythonw.exe"; Parameters: "-I -B ""{app}\sigma-desktop\launch.py"""; Description: "Open SIGMA"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Only our generated metadata, never user data/preferences or arbitrary files.
Type: files; Name: "{app}\sigma-desktop\shortcuts.json"

[Code]
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Entry: TFindRec;
begin
  Result := '';
  if FindFirst(ExpandConstant('{app}\*'), Entry) then
  begin
    try
      repeat
        if (Entry.Name <> '.') and (Entry.Name <> '..') then
        begin
          Result := 'The selected folder is not empty. Uninstall the previous SIGMA desktop build first, or choose a new empty folder. Your image data and settings will not be removed.';
          Exit;
        end;
      until not FindNext(Entry);
    finally
      FindClose(Entry);
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ExitCode: Integer;
  Parameters: String;
begin
  if CurStep = ssPostInstall then
  begin
    Parameters := ExpandConstant('-I -B "{app}\sigma-desktop\record_windows_install.py" "{group}\SIGMA.lnk" "{userdesktop}\SIGMA.lnk"');
    if not Exec(ExpandConstant('{app}\python.exe'), Parameters, ExpandConstant('{app}'), SW_HIDE, ewWaitUntilTerminated, ExitCode) then
      RaiseException('SIGMA installation could not be verified.');
    if ExitCode <> 0 then
      RaiseException('SIGMA shortcut verification failed. Please uninstall and retry.');
  end;
end;
