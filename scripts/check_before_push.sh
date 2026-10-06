#!/usr/bin/env bash
# Privacy scan before pushing. Exits non-zero if anything that looks like a credential, personal path,
# server address, email or large file would be committed. Run from the repository root.
set -uo pipefail
cd "$(dirname "$0")/.."
fail=0
if git rev-parse --git-dir >/dev/null 2>&1; then
  files=$(git ls-files --cached --others --exclude-standard)
else
  files=$(find . -type f -not -path "./.git/*" -not -path "./.venv/*" | sed 's|^\./||')
fi
check() {  # label, extended regex
  hits=$(echo "$files" | grep -v '^scripts/check_before_push.sh$' | tr '\n' '\0' | xargs -0 grep -I -l -E "$2" 2>/dev/null)
  if [[ -n "$hits" ]]; then echo "FOUND $1 in:"; echo "$hits" | sed 's/^/  /'; fail=1; fi
}
check "Hugging Face token"        'hf_[A-Za-z0-9]{30,}'
check "API key or secret"         '(api[_-]?key|secret[_-]?key|password|passwd)[[:space:]]*[:=][[:space:]]*[^[:space:]]{6,}'
check "private key"               'BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY'
check "GitHub or AWS token"       '(ghp_[A-Za-z0-9]{30,}|github_pat_|AKIA[0-9A-Z]{16})'
check "home-directory path"       '(/Users/[A-Za-z0-9._-]+|/home/[A-Za-z0-9._-]+)'
check "private IP address"        '\b(10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})\b'
check "email address"             '[A-Za-z0-9._%+-]+@(gmail|yahoo|hotmail|outlook|icloud)\.[a-z]{2,}'
check "SSH key reference"         'id_(rsa|ed25519)'
for name in .env id_rsa id_ed25519; do
  if echo "$files" | grep -q -E "(^|/)$name$"; then echo "FOUND sensitive file: $name"; fail=1; fi
done
big=$(echo "$files" | tr '\n' '\0' | xargs -0 -I{} find {} -maxdepth 0 -type f -size +50M 2>/dev/null)
if [[ -n "$big" ]]; then echo "FOUND files over 50 MB (GitHub rejects >100 MB):"; echo "$big" | sed 's/^/  /'; fail=1; fi
if [[ $fail -eq 0 ]]; then echo "OK: no credentials, personal paths, private IPs, emails or large files found."; fi
exit $fail
