"""
i18n API routes for serving translation files
FastAPI version
"""

import os
import json
import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

i18n_router = APIRouter(tags=["i18n"])

# This file lives at: .../backend/i18n_routes.py
# Translation files at: .../backend/data/i18n/*.json
# So relative to THIS file: ./data/i18n/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_HARDCODED_PATH = os.path.normpath(os.path.join(_SCRIPT_DIR, 'data', 'i18n'))

_data_dir = None


def _get_data_dir():
    """Find the i18n directory - resolved once on first call"""
    global _data_dir
    if _data_dir is not None:
        return _data_dir

    # Primary: relative to this script file (most reliable)
    # i18n_routes.py is in backend/, data/i18n/ is in backend/data/i18n/
    primary = _HARDCODED_PATH

    if os.path.isdir(primary):
        json_files = [f for f in os.listdir(primary) if f.endswith('.json')]
        if json_files:
            _data_dir = primary
            print(f"[i18n] Found i18n directory: {_data_dir} ({len(json_files)} files)")
            print(f"[i18n] Languages: {[f.replace('.json','') for f in sorted(json_files)]}")
            return _data_dir

    # If primary didn't work, try alternatives
    candidates = [
        primary,
        os.path.join(os.getcwd(), 'data', 'i18n'),
        os.path.join(os.getcwd(), 'backend', 'data', 'i18n'),
        os.path.join(_SCRIPT_DIR, '..', 'backend', 'data', 'i18n'),
        os.path.join(_SCRIPT_DIR, '..', 'data', 'i18n'),
    ]

    print(f"[i18n] Primary path not found: {primary}")
    print(f"[i18n] Script location: {_SCRIPT_DIR}")
    print(f"[i18n] Working directory: {os.getcwd()}")

    for candidate in candidates:
        resolved = os.path.normpath(os.path.abspath(candidate))
        try:
            if os.path.isdir(resolved):
                json_files = [f for f in os.listdir(resolved) if f.endswith('.json')]
                if json_files:
                    _data_dir = resolved
                    print(f"[i18n] Found i18n directory (fallback): {_data_dir} ({len(json_files)} files)")
                    return _data_dir
                print(f"[i18n]   ✗ {resolved} (exists but no .json files)")
            else:
                print(f"[i18n]   ✗ {resolved} (not found)")
        except Exception as e:
            print(f"[i18n]   ✗ {resolved} (error: {e})")

    # Last resort
    _data_dir = primary
    print(f"[i18n] WARNING: No i18n directory found! Defaulting to: {_data_dir}")
    return _data_dir


def _list_json_files(directory):
    """Safely list .json files in a directory"""
    if not os.path.isdir(directory):
        return []
    try:
        return sorted([f for f in os.listdir(directory) if f.endswith('.json')])
    except Exception:
        return []


def get_available_languages():
    """Scan i18n directory for available language files"""
    data_dir = _get_data_dir()
    languages = []

    json_files = _list_json_files(data_dir)
    if not json_files:
        return languages

    for filename in json_files:
        code = filename[:-5]  # strip .json

        # Skip template files
        if code in ('template',):
            continue

        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            meta = data.get('_meta', {})
            languages.append({
                'code': code,
                'name': meta.get('language', code),
                'native_name': meta.get('native_name', code),
                'file': filename
            })
        except Exception as e:
            print(f"[i18n] Could not read {filename}: {e}")
            languages.append({
                'code': code,
                'name': code,
                'native_name': code,
                'file': filename
            })

    return languages


@i18n_router.get('/i18n')
async def i18n_info():
    """Get i18n metadata and available languages"""
    data_dir = _get_data_dir()
    languages = get_available_languages()

    if not languages:
        print(f"[i18n] WARNING: No languages found!")
        print(f"[i18n]   data_dir: {data_dir}")
        print(f"[i18n]   exists: {os.path.isdir(data_dir)}")
        if os.path.isdir(data_dir):
            print(f"[i18n]   contents: {os.listdir(data_dir)}")

    return JSONResponse(content={
        'i18n_path': data_dir,
        'available_languages': languages,
        'default_language': 'en',
        'data_dir_exists': os.path.isdir(data_dir),
        'data_dir_resolved': os.path.abspath(data_dir)
    })


@i18n_router.get('/i18n/{lang_code}')
async def get_language(lang_code: str):
    """Get translation file for a specific language"""
    data_dir = _get_data_dir()

    # Sanitize
    safe_code = ''.join(c for c in lang_code if c.isalnum() or c in ('-', '_'))
    if safe_code != lang_code or '..' in lang_code:
        return JSONResponse(
            status_code=400,
            content={'error': 'Invalid language code'}
        )

    filepath = os.path.join(data_dir, f'{safe_code}.json')

    if not os.path.isfile(filepath):
        available = _list_json_files(data_dir)
        print(f"[i18n] 404: '{safe_code}' not found at {os.path.abspath(filepath)}")
        print(f"[i18n]   Available: {available}")
        return JSONResponse(
            status_code=404,
            content={
                'error': f'Language "{safe_code}" not found',
                'looked_at': os.path.abspath(filepath),
                'available_files': available,
                'data_dir': os.path.abspath(data_dir)
            }
        )

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        meta = data.pop('_meta', None)
        key_count = len(data)
        print(f"[i18n] Serving '{safe_code}': {key_count} translation keys")

        return JSONResponse(content={
            'translations': data,
            'meta': meta
        })
    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=500,
            content={'error': f'Invalid JSON in {safe_code}.json: {str(e)}'}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )