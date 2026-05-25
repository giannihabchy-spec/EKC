Set fso = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
htaPath = fso.BuildPath(scriptDir, "launcher.hta")

WshShell.Run """" & htaPath & """", 1, False
