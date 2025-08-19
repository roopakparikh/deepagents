# -*- mode: python ; coding: utf-8 -*-
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
    a.binaries,
    a.zipfiles,
    a.datas,
    name='deepagents-cli',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    onefile=True,
)
