param([string]$Message = 'Update Amethyst blog')

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot

function Invoke-ProjectGit {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$GitArguments)
    & git -c "safe.directory=$projectRoot" -c http.sslBackend=openssl @GitArguments
    if ($LASTEXITCODE -ne 0) { throw "Git failed: $($GitArguments -join ' ')" }
}

$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python virtual environment.' }
}

& $pythonPath -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $pythonPath tools/content.py check
if ($LASTEXITCODE -ne 0) { throw 'Content checks failed.' }
& $pythonPath -m unittest discover -s tools -p 'test_*.py' -v
if ($LASTEXITCODE -ne 0) { throw 'Content management tests failed.' }
& $pythonPath -m mkdocs build --strict
if ($LASTEXITCODE -ne 0) { throw 'Site build failed.' }

Invoke-ProjectGit fetch origin main

# This workspace was imported with a read-only .git directory. The first run
# from the owner's terminal initializes the missing index without changing
# any working files. Refuse to replace a newer remote version silently.
$indexPath = Join-Path $projectRoot '.git\index'
if (-not (Test-Path -LiteralPath $indexPath)) {
    $migration = Get-Content -Raw -Encoding UTF8 config/migration.json | ConvertFrom-Json
    $remoteHead = (Invoke-ProjectGit rev-parse origin/main).Trim()
    if ($remoteHead -ne $migration.source_commit) {
        throw 'The remote changed since migration. Integrate those changes before publishing.'
    }
    Invoke-ProjectGit reset --mixed origin/main
}

$gitName = & git -c "safe.directory=$projectRoot" config user.name
if (-not $gitName) { Invoke-ProjectGit config user.name AmethystTim }
$gitEmail = & git -c "safe.directory=$projectRoot" config user.email
if (-not $gitEmail) { Invoke-ProjectGit config user.email '130140331+AmethystTim@users.noreply.github.com' }

# Stage only website source. Personal Markdown files in the workspace root,
# .local, .venv, drafts and the recycle bin are excluded.
Invoke-ProjectGit add -- docs config tools overrides archive .github README.md mkdocs.yml requirements.txt .gitignore publish.ps1 admin.cmd
& git -c "safe.directory=$projectRoot" diff --cached --quiet
if ($LASTEXITCODE -eq 1) {
    Invoke-ProjectGit commit -m $Message
} elseif ($LASTEXITCODE -ne 0) {
    throw 'Could not inspect staged changes.'
}
Invoke-ProjectGit push origin main
Write-Host 'Pushed successfully. GitHub Actions will build and publish the blog.'
Write-Host 'Actions: https://github.com/AmethystTim/AmethystTim.github.io/actions'
Write-Host 'Blog:    https://amethysttim.github.io/'
