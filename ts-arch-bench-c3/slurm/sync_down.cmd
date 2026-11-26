@echo off
REM Daily pull of the cluster's irreplaceable artifacts to this machine.
REM Registered as the Windows Scheduled Task "ClaudeTSArchBenchSync".
REM
REM Why a task and not Claude's own cron: Claude's CronCreate jobs are
REM session-only and expire after 7 days, so they stop silently. $SCRATCH on
REM the cluster is purged 15 days after last modification and is never
REM backed up, so a backup that quietly stops is worse than none.
REM
REM Remove with:  schtasks /Delete /TN "ClaudeTSArchBenchSync" /F
set BASH="C:\Program Files\Git\bin\bash.exe"
%BASH% -lc "cd /d/TS-ARCH-BENCH-main/ts-arch-bench && bash slurm/sync_down.sh /d/TS-ARCH-BENCH-main/cluster-sync >> /d/TS-ARCH-BENCH-main/cluster-sync/sync_cron.log 2>&1"
