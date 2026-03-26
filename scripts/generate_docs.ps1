# IGUANA API Documentation Generator
# Automates the execution of edoc for the IGUANA swarm.

$ErlBin = "C:\PROGRA~1\ERLANG~1\bin"
$Erlc = Join-Path $ErlBin "erlc.exe"
$Erl = Join-Path $ErlBin "erl.exe"

# 1. Create output directory
$DocPath = Join-Path (Get-Location) "docs\html"
if (!(Test-Path $DocPath)) {
    New-Item -ItemType Directory -Path $DocPath
}

Write-Host "--- Generating IGUANA API Documentation ---" -ForegroundColor Cyan

# 2. Compile source files to ebin (required for some edoc analysis)
Write-Host "[1/2] Compiling source files..."
$SourceFiles = Get-ChildItem -Path "src\erlang\*.erl"
foreach ($file in $SourceFiles) {
    & $Erlc -I include -o ebin $file.FullName
}

# 3. Run edoc via erl
Write-Host "[2/2] Running edoc..."
$EdocCmd = "edoc:files(['src/erlang/iguana_meta_guard.erl', 'src/erlang/iguana_entropy_guard.erl', 'src/erlang/iguana_accelerator.erl'], [{dir, 'docs/html'}])."
& $Erl -noshell -pa ebin -eval "$EdocCmd" -s init stop

Write-Host "✅ Documentation generated at: $DocPath" -ForegroundColor Green
