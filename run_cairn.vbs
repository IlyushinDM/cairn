Dim oShell
Set oShell = CreateObject("WScript.Shell")

Dim scriptDir
scriptDir = Left(WScript.ScriptFullName, _
    InStrRev(WScript.ScriptFullName, "\"))

oShell.Run """" & scriptDir & "setup_and_run.bat""", 0, False

Set oShell = Nothing