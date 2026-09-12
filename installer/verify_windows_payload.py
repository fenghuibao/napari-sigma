"""Verify every installed file matches the complete pre-compression runtime."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    files = json.loads(args.manifest.read_text(encoding='utf-8'))
    for name, expected in files.items():
        with (args.prefix / name).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Installed payload differs: {name}')
    args.output.write_text(json.dumps({'status':'passed', 'checked_files':len(files)}), encoding='utf-8')
    print(f'Verified all {len(files)} installed payload files byte-for-byte')


if __name__ == '__main__':
    main()
