# Twinly for Windows

The Windows build of [macOSTwinly](https://github.com/OjasMishra32/macOSTwinly),
compatible with [iOSTwinly](https://github.com/OjasMishra32/iOSTwinly). The UI is
Electron + React; the agent brain is C#. See [`CLAUDE.md`](CLAUDE.md) for the
architecture and [`desktop/BUILD.md`](desktop/BUILD.md) for the full build guide.

## Run (dev)

```bash
# 1) the agent core (headless, serves :4799), from the repo root:
dotnet run --project Twinly.Core.Host

# 2) the UI (Vite renderer + Electron shell):
cd desktop && npm install && npm run dev
```

## Build the installer

```bash
# publish the self-contained core, then package the Electron app (NSIS):
dotnet publish Twinly.Core.Host -c Release -r win-x64 --self-contained true -o publish/core
cd desktop && npm run package:win        # -> desktop/release/Twinly-Setup-<version>.exe
```

## Test

```bash
dotnet test Twinly.Windows.Tests/Twinly.Windows.Tests.csproj   # runs on any OS (net10.0)
```

## Repository map

- `desktop/` — the Electron + React UI (the shipping front end)
- `Twinly.Core/` — the agent brain: models, agent loop, tools, persistence, companion API
- `Twinly.Windows.Platform/` — Win32/WinRT/NAudio/SAPI tool implementations (no UI)
- `Twinly.Core.Host/` — the headless host the Electron shell spawns
- `Twinly.Windows/` — the legacy Avalonia UI (being retired; tests cover shared logic)
- `Twinly.Windows.Tests/` — unit and integration tests
- `docs/` — design specs and plans
- `scripts/` — packaging and audit automation
- `publish/`, `desktop/release/` — local build output; ignored by Git

macOS remains the source of truth. Backend file-by-file status: [`PARITY_AUDIT.md`](PARITY_AUDIT.md).
