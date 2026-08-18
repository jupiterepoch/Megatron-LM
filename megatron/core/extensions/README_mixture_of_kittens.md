# Mixture-of-Kittens (MoK) megakernel

`megatron/core/extensions/mixture_of_kittens` is a **git submodule**, not vendored source.
It backs the opt-in `moe_megakernel_backend="mok"` MoE path implemented in
`megatron/core/transformer/moe/mok_backend.py`.

| | |
|---|---|
| Submodule | https://github.com/jupiterepoch/mixture-of-kittens |
| Pinned branch | `megatron-integration` |
| Upstream | https://github.com/cursor/mixture-of-kittens (base `22fc95a`) |
| Licence | Apache-2.0 (MoK), Apache-2.0 (ThunderKittens, nested submodule) |

## Checkout and build

```bash
git submodule update --init --recursive megatron/core/extensions/mixture_of_kittens
cd megatron/core/extensions/mixture_of_kittens
make ARCH=SM100                       # GB200; use ARCH=SM103 for GB300
cuobjdump --list-elf mok/_C*.so       # must report the matching sm_XXXa
```

`--recursive` is required: MoK carries ThunderKittens as its own submodule and the build
includes those headers, so a non-recursive checkout fails to compile.

The SM target is the sharp edge. `make` defaults to `ARCH=SM103` and emits SASS with **no
PTX fallback**, so a default build cannot launch any kernel on GB200 and fails at the first
launch with `no kernel image is available for execution on the device`. Always pass `ARCH`
explicitly and verify with `cuobjdump`.

## Notes

- The built `mok/_C*.so` is a generated artefact and is ignored by the submodule.
- `csrc/` and `third_party/` are not Python modules, so a wheel build omits them unless
  `package_data`/`MANIFEST.in` are extended. Source installs and in-tree use are unaffected.
- MoK's own `tests/` package shadows Megatron's top-level `tests/` when a test harness runs
  with the submodule root as `sys.path[0]`; run MoK's tests from the submodule directory.
