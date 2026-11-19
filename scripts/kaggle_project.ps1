# Run ONE Kaggle CLI command with the project's own API token from .env.
# .env now holds exactly one KAGGLE_API_TOKEN: the project account that owns the
# rezanadimi datasets and the AI quota we have spent. Never use a second token.
# The token value is read from .env at call time and is never printed or saved.
param([Parameter(ValueFromRemainingArguments=$true)][string[]]$KaggleArgs)

$tokens = @()
foreach ($line in Get-Content .env) {
    if ($line -match '^KAGGLE_API_TOKEN=(.*)$') {
        $tokens += $Matches[1].Trim().Trim('"').Trim("'")
    }
}
if ($tokens.Count -ne 1) {
    throw "expected exactly one KAGGLE_API_TOKEN in .env; found $($tokens.Count)"
}
$env:KAGGLE_API_TOKEN = $tokens[0]
$env:HTTP_PROXY = 'socks5h://127.0.0.1:10808'
$env:HTTPS_PROXY = 'socks5h://127.0.0.1:10808'
& .venv/Scripts/kaggle.exe @KaggleArgs
exit $LASTEXITCODE
