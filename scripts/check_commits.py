"""Check conventional commit subjects or the title of a squash-merged PR."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

SUBJECT = re.compile(r"^(feat|fix|perf|docs|style|refactor|test|build|ci|chore|revert)(\([a-zA-Z0-9_./-]+\))?!?: \S.*$")
SHA = re.compile(r"^[0-9a-f]{40}$")


def valid(subject: str) -> bool:
    return bool(SUBJECT.fullmatch(subject))


def main() -> int:
    if len(sys.argv) > 1:
        subjects = sys.argv[1:]
    else:
        event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
        if 'pull_request' in event:
            subjects = [event['pull_request']['title']]
        else:
            after, before = event['after'], event['before']
            if not SHA.fullmatch(after) or not SHA.fullmatch(before):
                raise ValueError('Expected GitHub commit hashes')
            exists = before != '0' * 40 and subprocess.run(
                ['git', 'cat-file', '-e', before + '^{commit}'], capture_output=True).returncode == 0
            rev = f'{before}..{after}' if exists else after
            subjects = subprocess.check_output(['git', 'log', '--format=%s', rev], text=True).splitlines()
    invalid = [s for s in subjects if not valid(s)]
    for subject in invalid:
        print(f'Non-conventional subject: {subject}', file=sys.stderr)
    return bool(invalid)


if __name__ == '__main__':
    raise SystemExit(main())
