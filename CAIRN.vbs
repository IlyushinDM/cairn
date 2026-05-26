Dim shell
Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = Left(WScript.ScriptFullName, InStrRev(WScript.ScriptFullName, "\") - 1)

Dim venv
venv = shell.CurrentDirectory & "\.venv\Scripts\pythonw.exe"

If CreateObject("Scripting.FileSystemObject").FileExists(venv) Then
    shell.Run """" & venv & """ -m cairn", 0, False
Else
    shell.Run "pythonw -m cairn", 0, False
End If
