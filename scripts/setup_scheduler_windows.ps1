<#
.SYNOPSIS
    Mendaftarkan jadwal Windows Task Scheduler agar model SiPeka otomatis
    mengecek data baru, melatih ulang (Champion vs Challenger), dan memperbarui
    hasil prediksi/forecast secara berkala.

.EXAMPLE
    # Jalankan setiap hari Senin jam 02:00 WIB:
    powershell -ExecutionPolicy Bypass -File scripts/setup_scheduler_windows.ps1

    # Hapus jadwal:
    Unregister-ScheduledTask -TaskName "SiPeka-AutoRetrain" -Confirm:$false
#>

param(
    [string]$TaskName = "SiPeka-AutoRetrain",
    [string]$DayOfWeek = "Monday",
    [string]$StartTime = "02:00",
    [ValidateSet("fast", "full")]
    [string]$Mode = "fast"
)

$RepoRoot = (Resolve-Path "$PSScriptRoot\..").Path
$PythonExe = (Get-Command python).Source
$ScriptPath = Join-Path $RepoRoot "scripts\auto_retrain.py"

$Action = New-ScheduledTaskAction `
    -Execute $PythonExe `
    -Argument "`"$ScriptPath`" --mode $Mode" `
    -WorkingDirectory $RepoRoot

$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $DayOfWeek -At $StartTime
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Auto-retraining dan batch scoring berkala untuk model prediksi sampah SiPeka." `
    -Force

Write-Host "Berhasil mendaftarkan jadwal '$TaskName' setiap $DayOfWeek pukul $StartTime (mode=$Mode)."
Write-Host "Uji manual kapan saja dengan: Start-ScheduledTask -TaskName '$TaskName'"
