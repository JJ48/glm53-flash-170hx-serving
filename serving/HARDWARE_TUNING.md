# Hardware setup & tuning

Three hardware-side steps beyond a stock 170HX box are needed to reach the numbers in the README.
None of this tooling is bundled here — the unlock and its fork are GPLv2, this repo is Apache-2.0 —
so what follows is links and commands, not redistributed code. All of it is third-party
firmware/driver-level tooling governed by its own license and NVIDIA's terms; obtain and use it at
your own discretion.

## 1. Unlock the cards — cmpunlocker

The CMP 170HX clamps SM compute, PCIe generation, GPU-to-GPU P2P, and HBM geometry in firmware.
[cmpunlocker](https://github.com/asm64-hooligan/cmpunlocker) (GPLv2, by asm64-hooligan) restores them:

```bash
sudo ./install.sh --p2p        # full SM compute + PCIe Gen2 + GPU-to-GPU P2P
```

The big win is restoring **full SM compute** (the stock card is heavily gimped). `--p2p` additionally
enables GPU-to-GPU transfers over the PCIe mailbox, which the pipeline hops use — on this box that was
worth only a small gain (~2 %), so we keep it on because it's free rather than for the throughput.

## 2. Per-card HBM clock — `--mclk-percard` + hbmtune

Upstream `--mclk-ndiv=N` applies one HBM multiplier to every card, but the cards don't all clock the
same (silicon and cooling vary). Our `--mclk-percard` extension plus the `tools/hbmtune` autotune
give each card its own HBM clock — searched, validated against a real kernel gate, and held under a
per-card memory-temperature limit; cards whose FBPA PLL registers don't answer are left at stock.

This is contributed to cmpunlocker; until it merges upstream it lives on the fork at
<https://github.com/JJ48/cmpunlocker> (branch `percard-hbm`):

```bash
sudo ./install.sh --mclk-percard
sudo hbmtune auto start        # search + hold per-card HBM clocks (see tools/hbmtune for options)
```

Validated on 8 GB (`0x20C2`) cards.

## 3. Power cap

The benchmark numbers were measured with each card capped at **175 W** (the stock default is 250 W).
The driver resets the cap on reboot, so re-apply it — e.g. a boot-time `systemd` oneshot running:

```bash
nvidia-smi -pl 175
```

Set the cap to your own power/thermal budget; it trades sustained clocks against draw and heat.
