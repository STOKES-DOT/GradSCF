"""Fetch pinned author data; no electronic-structure calculations."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent / 'raw'
API = 'https://api.github.com/repos/aoterodelaroza/refdata/'
DIRECTORIES = ['20_bh9', '20_s66x8', '20_hb300spx']
DIN = ['bh9.din', 'bh9_bh.din', 'bh9_re.din', 's66x8.din', 'hb300spx.din']


def get(url):
    with urllib.request.urlopen(url, timeout=40) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paths', type=Path)
    parser.add_argument('--bh9', action='store_true')
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / 'source-index.json'
    if manifest.exists():
        index = json.loads(manifest.read_text())
    else:
        commit = json.loads(get(API + 'commits/master'))
        root = json.loads(get(API + 'git/trees/' + commit['commit']['tree']['sha']))
        entries = {e['path']: e for e in root['tree']}
        files = {}
        for directory in DIRECTORIES + ['10_din']:
            tree = json.loads(get(API + 'git/trees/' + entries[directory]['sha']))
            for entry in tree['tree']:
                if entry['type'] == 'blob' and (directory != '10_din' or entry['path'] in DIN):
                    files[directory + '/' + entry['path']] = entry
        for name in ['README.md', 'LICENSE']:
            files[name] = entries[name]
        index = dict(repository='https://github.com/aoterodelaroza/refdata',
                     commit=commit['sha'], tree=root['sha'], files=files)
        manifest.write_text(json.dumps(index, indent=2) + '\n')
    paths = ['10_din/' + name for name in DIN] + ['README.md', 'LICENSE']
    paths += [p for p in index['files'] if p.split('/')[-1].lower().startswith('readme')]
    if args.bh9:
        paths += [p for p in index['files'] if p.startswith('20_bh9/') and p.endswith('.xyz')]
    if args.paths:
        paths += json.loads(args.paths.read_text())
    paths = sorted(set(paths))

    if args.archive:
        with tarfile.open(ROOT/'refdata.tar.gz', 'r:gz') as archive:
            members = {member.name.split('/', 1)[1]: member for member in archive
                       if member.isfile() and '/' in member.name}
            for path in paths:
                content = archive.extractfile(members[path]).read()
                actual = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
                if actual != index['files'][path]['sha']:
                    raise ValueError('Archive Git blob checksum mismatch: '+path)
                target = ROOT/path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        print(json.dumps(dict(commit=index['commit'], verified_archive_files=len(paths))), flush=True)
        return

    def fetch(path):
        target = ROOT / path
        expected = index['files'][path]['sha']
        def valid(data):
            return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == expected
        if target.exists() and valid(target.read_bytes()):
            return
        content = get('https://raw.githubusercontent.com/aoterodelaroza/refdata/' + index['commit'] + '/' + path)
        if not valid(content):
            raise ValueError('Git blob checksum mismatch: ' + path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, _ in enumerate(pool.map(fetch, paths), 1):
            if i % 100 == 0:
                print('downloaded', i, '/', len(paths), flush=True)
    print(json.dumps(dict(commit=index['commit'], downloaded=len(paths))), flush=True)


if __name__ == '__main__':
    main()
