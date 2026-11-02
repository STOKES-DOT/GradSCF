# Status: 2026-10-22 10:39 +08:00

- Current diagnostic job: **103475**, PENDING (Resources), n005 / A800 requested.
- Resources: one GPU, 32 CPUs, 512 GiB, maximum 8 hours.
- Source checkpoint: immutable accepted update 16; SHA256
  `978423a317fa62e6ede9d3c49c45fbeda8083687cd86ae2c479341ad1d377cc7`.
- Its parameter-array bytes equal those in the original failed update-17 snapshot.
- Stop at update 40; no finite differences, extra scoring or startup replay.
- M-TZP 102765 now waits on `afterok:103475`; M-DZP 102760 continues running.
- Initial attempt 103473 exited before Python because `/usr/bin/time` is absent.
  The launcher now uses Python directly; timing/resource sampling is in the driver.
- Local verification: shell/Python syntax checks and 27 focused CPU tests passed.
  Scientific-source and campaign-input hashes match the frozen original campaign.
- **No successful GPU comparison or memory-stability result yet.** Check
  `execution.json`, `run/progress.json`, `resources.jsonl`, and `result.json` after
  the job starts. Original S-TZP training output is unchanged.
