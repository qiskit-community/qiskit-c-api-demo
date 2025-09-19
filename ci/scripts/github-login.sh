#!/bin/bash
set -euo pipefail

GITHUB_TOKEN="$(get_env git-token)"

echo -e "machine github.ibm.com\n  login $GITHUB_TOKEN" > ~/.netrc
git config user.name "ibmqops"
git config user.email "ibmqops@ibm.com"

echo '::info::Git is configured'
