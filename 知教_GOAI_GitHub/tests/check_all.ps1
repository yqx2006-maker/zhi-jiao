$ErrorActionPreference = 'Stop'

$h = Get-Content (Join-Path $PSScriptRoot '..\web\index.html') -Raw -Encoding UTF8
# Use the LAST <script> block (the main app script; the <head> has a tiny theme no-flash script too)
$blocks = [regex]::Matches($h, '(?s)<script>(.*?)</script>')
if ($blocks.Count -eq 0) {
    Write-Error 'No <script> blocks found'
    exit 1
}
$m = $blocks[$blocks.Count - 1]
$tmp = Join-Path $env:TEMP 'zj_check.js'
Set-Content -Path $tmp -Value $m.Groups[1].Value -Encoding UTF8

$node = Get-Command node -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $node) {
    Write-Error 'Node.js was not found. JavaScript syntax was not checked.'
    exit 1
}
& $node.Source --check $tmp
if ($LASTEXITCODE -ne 0) {
    Write-Output 'JS SYNTAX FAIL'
    exit $LASTEXITCODE
}
Write-Output 'JS SYNTAX OK'

$pythonArgs = @()
$python = Get-Command python -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $python) {
    $python = Get-Command py -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($python) { $pythonArgs = @('-3') }
}
if (-not $python) {
    $python = Get-Command python3 -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
}
if (-not $python) {
    Write-Error 'Python 3 was not found. server.py was not compiled.'
    exit 1
}
$pyc = Join-Path $env:TEMP 'zj_server_check.pyc'
& $python.Source @pythonArgs -c "import py_compile,sys; py_compile.compile(sys.argv[1], cfile=sys.argv[2], doraise=True)" (Join-Path $PSScriptRoot '..\server.py') $pyc
if ($LASTEXITCODE -ne 0) {
    Write-Output 'PY SYNTAX FAIL'
    exit $LASTEXITCODE
}
Write-Output 'PY SYNTAX OK'
