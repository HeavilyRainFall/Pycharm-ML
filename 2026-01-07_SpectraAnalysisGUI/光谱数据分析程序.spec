# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = []
datas += collect_data_files('matplotlib')
datas += collect_data_files('pandas')
datas += collect_data_files('numpy')
datas += collect_data_files('scipy')
datas += collect_data_files('tkinter')


a = Analysis(
    ['spectra_analysis.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['matplotlib.backends.backend_tkagg', 'scipy.special._ufuncs_cxx', 'scipy.special._ufuncs_cxx', 'scipy.integrate', 'scipy.linalg.cython_blas', 'scipy.linalg.cython_lapack', 'scipy.spatial.transform.rotation', 'scipy.special._ellip_harm_2', 'scipy.optimize._highs', 'scipy.optimize._highs', 'scipy.optimize._highs', 'scipy.optimize._highs'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='光谱数据分析程序',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
