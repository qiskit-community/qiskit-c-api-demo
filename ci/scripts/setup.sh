#!/usr/bin/env bash
set -euo pipefail

echo "Running ----------setup script----------"

# Enable/disable specific compliance checks
set_env branch-protection-check 0
set_env peer-review-compliance 0
set_env sbom-validation-collect-evidence 0
set_env evidence-reuse 0

# We're not enabling evidence collection yet, but we can turn on some settings now!
set_env batched-evidence-collection 1

# Make sure that we're interacting with the inventory as little as possible
set_env skip-inventory-update-on-failure 1
