"""Builds kumo_relational_sqlite_tutorial.ipynb from tutorial_source.py.

The notebook is kept as a plain-text "percent" script so diffs stay readable.
Cells start with `# %%` (code) or `# %% [markdown]` (markdown; every line of
the cell body starts with `# `).

    python build_notebook.py            # write the .ipynb next to this file
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'tutorial_source.py'
OUT = HERE / 'kumo_relational_sqlite_tutorial.ipynb'


def parse_cells(text: str) -> list[dict]:
    cells: list[dict] = []
    kind, lines = None, []

    def flush() -> None:
        if kind is None:
            return
        body = '\n'.join(lines).strip('\n')
        if kind == 'markdown':
            body = '\n'.join(
                line[2:] if line.startswith('# ') else line.lstrip('#')
                for line in body.split('\n')
            )
        source = [line + '\n' for line in body.split('\n')]
        source[-1] = source[-1].rstrip('\n')
        cell = {'cell_type': kind, 'id': f'cell-{len(cells):03d}',
                'metadata': {}, 'source': source}
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    for line in text.split('\n'):
        if line.startswith('# %%'):
            flush()
            kind = 'markdown' if '[markdown]' in line else 'code'
            lines = []
        elif kind is not None:
            lines.append(line)
    flush()
    return cells


def main() -> None:
    notebook = {
        'cells': parse_cells(SRC.read_text(encoding='utf-8')),
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {'name': 'python'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
    OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n',
                   encoding='utf-8')
    print(f'wrote {OUT.name}: {len(notebook["cells"])} cells')


if __name__ == '__main__':
    main()
