#!/usr/bin/env bash
set -euo pipefail

config_file="/etc/default/safee-dms-power"
if [[ ! -r "${config_file}" ]]; then
  echo "CPU ceiling disabled: ${config_file} is absent"
  exit 0
fi

# shellcheck disable=SC1091
source "${config_file}"
ceiling_mhz="${SAFEE_CPU_MAX_MHZ:-0}"
case "${ceiling_mhz}" in
  0)
    echo "CPU ceiling disabled"
    exit 0
    ;;
  1800|2000|2200|2400) ;;
  *)
    echo "SAFEE_CPU_MAX_MHZ must be 0, 1800, 2000, 2200, or 2400" >&2
    exit 2
    ;;
esac

ceiling_khz="$((ceiling_mhz * 1000))"
found=0
for policy in /sys/devices/system/cpu/cpufreq/policy[0-9]*; do
  [[ -d "${policy}" ]] || continue
  printf '%s\n' "${ceiling_khz}" > "${policy}/scaling_max_freq"
  found=1
done
if [[ "${found}" -ne 1 ]]; then
  echo "No CPU frequency policies found" >&2
  exit 1
fi
echo "Applied transient Safee CPU ceiling: ${ceiling_mhz} MHz"
