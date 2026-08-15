# Rebuild asr_live.exe + ocr.exe (WinRT DNN recognizer / offline OCR). Pure ASCII, run: powershell -File build.ps1
$tools = $PSScriptRoot
$fx = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319'
$fa = 'C:\Program Files (x86)\Reference Assemblies\Microsoft\Framework\.NETFramework\v4.8\Facades'
$wm = 'C:\Windows\System32\WinMetadata'
$baseRefs = @(
  "/r:$fx\System.Runtime.WindowsRuntime.dll",
  "/r:$fx\System.Runtime.InteropServices.WindowsRuntime.dll",
  "/r:$fa\System.Runtime.dll",
  "/r:$fa\System.ObjectModel.dll"
)

$asr = Join-Path $tools 'asr_live.cs'
$asrDst = Join-Path $tools 'asr_live.exe'
& "$fx\csc.exe" /nologo /target:exe /out:$asrDst "/r:$wm\Windows.Foundation.winmd" "/r:$wm\Windows.Media.winmd" "/r:$wm\Windows.Globalization.winmd" @baseRefs $asr
Write-Output ("ASR_CSC_EXIT=" + $LASTEXITCODE)

$ocr = Join-Path $tools 'ocr.cs'
$ocrDst = Join-Path $tools 'ocr.exe'
$sdkVer = Get-ChildItem 'C:\Program Files (x86)\Windows Kits\10\UnionMetadata' -Directory -ErrorAction SilentlyContinue |
          Where-Object { $_.Name -match '^\d+\.\d+\.\d+\.\d+$' } |
          Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
if ($sdkVer) {
  $winmd = Join-Path $sdkVer.FullName 'Windows.winmd'
  & "$fx\csc.exe" /nologo /target:exe /out:$ocrDst "/r:$winmd" @baseRefs $ocr
  Write-Output ("OCR_CSC_EXIT=" + $LASTEXITCODE)
} else {
  Write-Output "OCR_SKIP: 未安装 Windows SDK UnionMetadata，ocr.exe 未编译（服务端会自动降级为口述题意）"
}
