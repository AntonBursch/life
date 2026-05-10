# Build the flow-wasm crate and run wasm-bindgen to produce ES module
# bindings the viewer can import. No wasm-pack needed.

$ErrorActionPreference = "Stop"

$repo = (Resolve-Path "$PSScriptRoot\..").Path
$core = Join-Path $repo "core"
$out = Join-Path $repo "viewer\pkg"

Write-Host "Building flow-wasm (release, wasm32-unknown-unknown)..." -ForegroundColor Cyan
Push-Location $core
try {
    cargo build --release --target wasm32-unknown-unknown -p flow-wasm
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
} finally {
    Pop-Location
}

$wasm = Join-Path $core "target\wasm32-unknown-unknown\release\flow_wasm.wasm"
if (-not (Test-Path $wasm)) {
    throw "expected wasm artefact not found at $wasm"
}

if (-not (Test-Path $out)) {
    New-Item -ItemType Directory -Path $out | Out-Null
}

Write-Host "Generating JS bindings into $out ..." -ForegroundColor Cyan
& "$env:USERPROFILE\.cargo\bin\wasm-bindgen.exe" `
    $wasm `
    --out-dir $out `
    --out-name flow_wasm `
    --target web `
    --no-typescript
if ($LASTEXITCODE -ne 0) { throw "wasm-bindgen failed" }

Write-Host "Done. Output:" -ForegroundColor Green
Get-ChildItem $out | Format-Table Name, Length -AutoSize
