param(
    [Parameter(Mandatory=$true)]
    [string]$RepoPath
)

$Source = Split-Path -Parent $MyInvocation.MyCommand.Path
$Items = Get-ChildItem -Path $Source -Force | Where-Object {
    $_.Name -notin @('apply_fix_batch.ps1','apply_fix_batch.sh','APPLY_FIXES.md','REPO_AUDIT_AND_FIX_PLAN.md')
}

foreach ($Item in $Items) {
    Copy-Item -Path $Item.FullName -Destination $RepoPath -Recurse -Force
}

Write-Host "Applied fix batch to $RepoPath"
