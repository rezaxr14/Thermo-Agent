# Run ONE Kaggle CLI command as the project account (rezanadimi).
# The API token comes ONLY from repository-root .env (KAGGLE_API_TOKEN); never printed.
# The SOCKS proxy is scoped to this single command; non-Kaggle traffic is not proxied.
param([Parameter(ValueFromRemainingArguments=$true)][string[]]$KaggleArgs)
$token = ''
foreach ($line in Get-Content .env) {
    if ($line -match '^KAGGLE_API_TOKEN=(.*)$') { $token = $Matches[1].Trim().Trim('"').Trim("'") }
}
if (-not $token) { throw 'KAGGLE_API_TOKEN missing from .env; ask Reza, never switch accounts' }
$env:KAGGLE_API_TOKEN = $token
$env:HTTP_PROXY = 'socks5h://127.0.0.1:10808'
$env:HTTPS_PROXY = 'socks5h://127.0.0.1:10808'
& .venv/Scripts/kaggle.exe @KaggleArgs
exit $LASTEXITCODE
