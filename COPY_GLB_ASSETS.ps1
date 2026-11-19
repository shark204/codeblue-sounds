# Run this once inside F:\codeblue-sounds to copy ONLY heart & lungs 3D assets from F:\mrcs-bank
$src = "F:\mrcs-bank\public"
$dst = "$PSScriptRoot\public"

New-Item -ItemType Directory -Force -Path "$dst\anatomy" | Out-Null
New-Item -ItemType Directory -Force -Path "$dst\draco" | Out-Null

# Copy Draco decoder if present
if (Test-Path "$src\draco") {
  Copy-Item "$src\draco\*" "$dst\draco\" -Recurse -Force
}

# Copy Heart & Lungs / Respiratory GLB files only (keeps this site super fast!)
if (Test-Path "$src\anatomy") {
  Get-ChildItem "$src\anatomy" -File | Where-Object {
    $_.Name -match "(heart|cardio|cor|lung|pulmo|resp|trachea|bronch|diaphragm|manifest)"
  } | ForEach-Object {
    Copy-Item $_.FullName "$dst\anatomy\" -Force
    Write-Host "Copied $($_.Name)" -ForegroundColor Green
  }
}

Write-Host "Done! Now run: npm install && npm run dev" -ForegroundColor Cyan
