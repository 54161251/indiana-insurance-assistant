"""Restore the original, checked Chroma snapshot without loading pickle objects."""
from pathlib import Path
import hashlib
import io
import json
import zipfile

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'data/snapshot.json').read_text())
payload = b''.join((root / 'data' / name).read_bytes() for name in manifest['parts'])
if hashlib.sha256(payload).hexdigest() != manifest['sha256']:
    raise SystemExit('Snapshot checksum mismatch.')
target = root / 'mydb_insurance'
if target.exists():
    raise SystemExit('mydb_insurance already exists. Move it aside before restoring; no files overwritten.')
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    if z.testzip() is not None:
        raise SystemExit('Snapshot archive integrity check failed.')
    for member in z.infolist():
        path = (target / member.filename).resolve()
        if not path.is_relative_to(target.resolve()):
            raise SystemExit('Unsafe snapshot path.')
    z.extractall(target)
print('Restored original Chroma snapshot to', target)
