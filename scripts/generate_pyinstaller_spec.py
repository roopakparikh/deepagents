#!/usr/bin/env python3
"""
Generate a PyInstaller spec file for the DeepAgents CLI.
This avoids committing a *.spec file (which is gitignored) while still
making it easy to (re)create locally.

Usage:
  python scripts/generate_pyinstaller_spec.py [OUTPUT_SPEC_PATH]

If OUTPUT_SPEC_PATH is not provided, defaults to ./deepagents.spec
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

SPEC_TEMPLATE = """# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_submodules

# Ensure project src is on path for analysis
if 'src' not in sys.path:
    sys.path.insert(0, 'src')

# Some dynamic imports in langchain/langgraph/mcp may require hiddenimports
hiddenimports = []
try:
    hiddenimports += collect_submodules('langchain')
except Exception:
    pass
try:
    hiddenimports += collect_submodules('langgraph')
except Exception:
    pass
try:
    hiddenimports += collect_submodules('langchain_community')
except Exception:
    pass
try:
    hiddenimports += collect_submodules('langchain_ollama')
except Exception:
    pass
try:
    hiddenimports += collect_submodules('langchain_mcp_adapters')
except Exception:
    pass

block_cipher = None

a = Analysis(
    ['src/deepagents/cli.py'],
    pathex=['src'],
    binaries=[],
    datas=[
        ('README.md', '.'),
        ('deep_agents.png', '.'),
        ('example-config.json', '.'),
        ('example-config-ollama.json', '.'),
        ('example-mcp-config.json', '.'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='deepagents-cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='deepagents-cli'
)
"""

def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('deepagents.spec')
    out.write_text(SPEC_TEMPLATE, encoding='utf-8')
    print(f"Wrote PyInstaller spec to: {out.resolve()}")

if __name__ == '__main__':
    main()
