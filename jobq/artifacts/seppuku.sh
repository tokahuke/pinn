#!/bin/bash
# Installed and launched by setup.sh. Placeholders are substituted before
# upload.
#
# Idle means NO SSH SESSIONS, which works because jobq run is attached: a
# training job holds its session for its whole life, so zero sessions really
# does mean nothing is running. Detached work started by hand is invisible to
# this and will be killed under it.
set -u

IDLE_MINUTES=@@IDLE_MINUTES@@
LOG=/var/log/jobq-seppuku.log

# RunPod's injected variables (RUNPOD_POD_ID, RUNPOD_API_KEY, ...) are NOT in
# the environment of a full ssh-over-tcp session, which is how jobq connects,
# so this script cannot read its own. PID 1 is the real entrypoint and has
# them. Verified 2026-08-11: without this the pod id is empty and the delete
# silently addresses /v1/pods/ forever.
injected() {
    tr '\0' '\n' </proc/1/environ | sed -n "s/^$1=//p" | head -1
}

POD_ID=$(injected RUNPOD_POD_ID)
API_KEY=$(injected RUNPOD_API_KEY)

# Both come from the pod itself. jobq deliberately ships NEITHER: a runpod
# key cannot be scoped to a single pod, so one written here would let anything
# with root on a rented box spend the whole account. Confirmed injected
# 2026-08-11.
if [ -z "$POD_ID" ] || [ -z "$API_KEY" ]; then
    echo "$(date -Is) no pod id or key in /proc/1/environ, refusing to arm" >>"$LOG"
    exit 1
fi

sessions() {
    # sshd forks one process per session, titled "sshd: root@pts/0" (or
    # "@notty" for rsync). Counting those needs no iproute2 and no utmp,
    # neither of which a container reliably has.
    pgrep -c -f 'sshd:.*@' || true
}

echo "$(date -Is) armed for $POD_ID, ${IDLE_MINUTES}m" >>"$LOG"

idle=0

while true; do
    sleep 60

    if [ "$(sessions)" -gt 0 ]; then
        idle=0
        continue
    fi

    idle=$((idle + 1))
    echo "$(date -Is) idle ${idle}/${IDLE_MINUTES}m" >>"$LOG"

    if [ "$idle" -lt "$IDLE_MINUTES" ]; then
        continue
    fi

    echo "$(date -Is) terminating $POD_ID" >>"$LOG"
    curl -sS -X DELETE \
        -H "Authorization: Bearer $API_KEY" \
        "https://rest.runpod.io/v1/pods/$POD_ID" >>"$LOG" 2>&1
    idle=0
done
