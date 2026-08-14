# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs


def _entry_text(entry):
    return "/" + "/".join(
        str(part).replace("\\", "/").lower() for part in entry
    ) + "/"


def _is_pruned_entry(entry, *, parts=(), suffixes=()):
    text = _entry_text(entry)
    if any(Path(str(part)).suffix.lower() in suffixes for part in entry):
        return True
    return any(f"/{part.strip('/').lower()}/" in text for part in parts)


def collect_package(name, *, prune_parts=(), prune_suffixes=()):
    package_datas, package_binaries, package_hiddenimports = collect_all(name)
    if prune_parts or prune_suffixes:
        package_datas = [
            entry
            for entry in package_datas
            if not _is_pruned_entry(
                entry,
                parts=prune_parts,
                suffixes=prune_suffixes,
            )
        ]
        package_binaries = [
            entry
            for entry in package_binaries
            if not _is_pruned_entry(
                entry,
                parts=prune_parts,
                suffixes=prune_suffixes,
            )
        ]
    return package_datas, package_binaries, package_hiddenimports


def prune_toc_entries(entries, *, parts=(), suffixes=()):
    return [
        entry
        for entry in entries
        if not _is_pruned_entry(entry, parts=parts, suffixes=suffixes)
    ]


SPEC_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
BARRAGE_GRAB_ROOT = (SPEC_ROOT.parent / 'Release_V2.8.0').resolve()

datas = [('src/live_frontend/resource', 'resource')]
BARRAGE_GRAB_FILES = (
    'WssBarrageServer.exe',
    'WssBarrageServer.exe.config',
    'rootCert.pfx',
)
if BARRAGE_GRAB_ROOT.exists():
    for filename in BARRAGE_GRAB_FILES:
        source = BARRAGE_GRAB_ROOT / filename
        if source.exists():
            datas.append((str(source), 'barrage_grab'))
    crts_root = BARRAGE_GRAB_ROOT / 'crts'
    if crts_root.exists():
        datas.append((str(crts_root), 'barrage_grab/crts'))
binaries = []
hiddenimports = [
    'websocket',
    'live_frontend.local_douyin_barrage',
    'open_llm_vtuber.douyin_link_payload',
    'uiautomation',
    'pywinauto',
    'pywinauto.controls.uia_controls',
    'pywinauto.findwindows',
]
binaries += collect_dynamic_libs('uiautomation')
torch_prune_parts = (
    'torch/include',
    'torch/testing',
)
torch_prune_suffixes = ('.lib',)

for package_name in ('live2d', 'pygame', 'silero_vad'):
    package_datas, package_binaries, package_hiddenimports = collect_package(
        package_name
    )
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

package_datas, package_binaries, package_hiddenimports = collect_package(
    'torch',
    prune_parts=torch_prune_parts,
    prune_suffixes=torch_prune_suffixes,
)
datas += package_datas
binaries += package_binaries
hiddenimports += package_hiddenimports

excludes = [
    # The PyQt display client does not call these packages directly; excluding
    # them prevents PyInstaller hooks from pulling large optional stacks.
    'aiohttp',
    'cryptography',
    'google',
    'matplotlib',
    'onnxruntime',
    'pandas',
    'PIL',
    'scipy',
    'sklearn',
]


a = Analysis(
    ['src\\live_frontend\\pyqt_live2d_window.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DreamMakerLiveFrontend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    prune_toc_entries(
        a.binaries,
        parts=torch_prune_parts,
        suffixes=torch_prune_suffixes,
    ),
    prune_toc_entries(
        a.datas,
        parts=torch_prune_parts,
        suffixes=torch_prune_suffixes,
    ),
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DreamMakerLiveFrontend',
)
