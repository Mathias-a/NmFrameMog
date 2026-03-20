from __future__ import annotations

import argparse
import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, List, Optional, Set, Tuple, Union, cast

CellValue = Union[bool, int, float, str, None]
NormalizedGrid = Tuple[Tuple[CellValue, ...], ...]


@dataclass(frozen=True)
class ViewportQuery:
    step: int
    x: int
    y: int
    width: int
    height: int
    screenshot: Optional[NormalizedGrid] = None
    note: Optional[str] = None


@dataclass(frozen=True)
class DebugTrace:
    start_grid: NormalizedGrid
    queries: Tuple[ViewportQuery, ...]
    title: str = "Astar Island debug trace"


@dataclass(frozen=True)
class DebugArtifacts:
    output_dir: Path
    index_html: Path
    start_state_svg: Path
    query_overlay_svg: Path
    screenshot_svgs: Tuple[Path, ...]


@dataclass(frozen=True)
class _QueryArtifact:
    order: int
    query: ViewportQuery
    screenshot_svg: Optional[Path]


@dataclass(frozen=True)
class _DesignTokens:
    page_background: str = "#08111f"
    panel_background: str = "#101b31"
    panel_background_alt: str = "#162540"
    panel_border: str = "#314260"
    text_primary: str = "#f4f7fb"
    text_muted: str = "#9fb0cf"
    accent: str = "#74d8ff"
    accent_alt: str = "#8cf5cf"
    shadow: str = "rgba(5, 10, 20, 0.35)"
    cell_border: str = "#24324f"
    label_text: str = "#08111f"
    overlay_label_text: str = "#f4f7fb"
    empty_cell: str = "#1b2944"


TOKENS: Final = _DesignTokens()
CELL_SIZE_WORLD: Final = 18
CELL_SIZE_SCREENSHOT: Final = 34
CELL_GAP: Final = 1
SVG_PADDING: Final = 24
SVG_CORNER_RADIUS: Final = 8
SVG_STROKE_WIDTH: Final = 2
OVERLAY_FILL_OPACITY: Final = 0.18
OVERLAY_STROKE_OPACITY: Final = 0.95
VALUE_LABEL_LIMIT: Final = 6
LABEL_VISIBLE_LIMIT: Final = 15
CARD_IMAGE_WIDTH: Final = 320
VALUE_PALETTE: Final[Tuple[str, ...]] = (
    "#6fe3ff",
    "#8df6c7",
    "#ffd36f",
    "#ff9e7a",
    "#d7a6ff",
    "#94b5ff",
    "#f59ac2",
    "#c0f07a",
)
OVERLAY_PALETTE: Final[Tuple[str, ...]] = (
    "#74d8ff",
    "#8cf5cf",
    "#ffd36f",
    "#ff9e7a",
    "#d7a6ff",
    "#94b5ff",
)


def load_trace_file(path: Path) -> DebugTrace:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    return _parse_trace_payload(payload)


def render_debug_bundle(trace: DebugTrace, output_dir: Path) -> DebugArtifacts:
    _validate_trace(trace)
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = output_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    value_colors = _build_value_color_map(trace)

    start_state_svg = output_dir / "start-state.svg"
    start_state_svg.write_text(
        _render_grid_svg(
            grid=trace.start_grid,
            value_colors=value_colors,
            title="Start state",
            cell_size=CELL_SIZE_WORLD,
            overlays=(),
        ),
        encoding="utf-8",
    )

    query_overlay_svg = output_dir / "query-overlay.svg"
    query_overlay_svg.write_text(
        _render_grid_svg(
            grid=trace.start_grid,
            value_colors=value_colors,
            title="Queried viewports",
            cell_size=CELL_SIZE_WORLD,
            overlays=trace.queries,
        ),
        encoding="utf-8",
    )

    query_artifacts: List[_QueryArtifact] = []
    for order, query in enumerate(trace.queries):
        screenshot_svg: Optional[Path] = None
        if query.screenshot is not None:
            screenshot_svg = screenshots_dir / _screenshot_filename(
                order=order, query=query
            )
            screenshot_svg.write_text(
                _render_grid_svg(
                    grid=query.screenshot,
                    value_colors=value_colors,
                    title=f"Viewport {query.step}",
                    cell_size=CELL_SIZE_SCREENSHOT,
                    overlays=(),
                ),
                encoding="utf-8",
            )
        query_artifacts.append(
            _QueryArtifact(order=order, query=query, screenshot_svg=screenshot_svg)
        )

    index_html = output_dir / "index.html"
    index_html.write_text(
        _render_html_report(
            trace=trace,
            output_dir=output_dir,
            start_state_svg=start_state_svg,
            query_overlay_svg=query_overlay_svg,
            query_artifacts=tuple(query_artifacts),
            value_colors=value_colors,
        ),
        encoding="utf-8",
    )

    return DebugArtifacts(
        output_dir=output_dir,
        index_html=index_html,
        start_state_svg=start_state_svg,
        query_overlay_svg=query_overlay_svg,
        screenshot_svgs=tuple(
            artifact.screenshot_svg
            for artifact in query_artifacts
            if artifact.screenshot_svg is not None
        ),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate static Astar Island debug artifacts."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Path to a debug trace JSON file."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the HTML report and SVG artifacts will be written.",
    )
    raw_args = parser.parse_args(list(argv) if argv is not None else None)
    input_path = _read_path_namespace_value(raw_args, "input")
    output_dir = _read_path_namespace_value(raw_args, "output_dir")

    artifacts = render_debug_bundle(load_trace_file(input_path), output_dir)
    print(artifacts.index_html)
    return 0


def _parse_trace_payload(payload: object) -> DebugTrace:
    if not isinstance(payload, dict):
        raise ValueError("Trace payload must be a JSON object.")
    trace_mapping = cast(Mapping[str, object], payload)

    title = trace_mapping.get("title", "Astar Island debug trace")
    if not isinstance(title, str):
        raise ValueError("Trace field 'title' must be a string when provided.")

    start_grid = _normalize_grid(
        name="start_grid", grid=_require_field(trace_mapping, "start_grid")
    )
    raw_queries_obj = trace_mapping.get("queries")
    if raw_queries_obj is None:
        raw_queries_obj = []
    if not isinstance(raw_queries_obj, list):
        raise ValueError("Trace field 'queries' must be a list.")
    raw_queries: Sequence[object] = cast(Sequence[object], raw_queries_obj)

    queries = tuple(
        _parse_query_payload(raw_query=raw_query, default_step=index)
        for index, raw_query in enumerate(raw_queries)
    )
    return DebugTrace(start_grid=start_grid, queries=queries, title=title)


def _parse_query_payload(raw_query: object, default_step: int) -> ViewportQuery:
    if not isinstance(raw_query, dict):
        raise ValueError("Each query entry must be an object.")
    query_mapping = cast(Mapping[str, object], raw_query)

    step = _read_int(query_mapping, "step", default_step)
    x = _read_int(query_mapping, "x")
    y = _read_int(query_mapping, "y")
    width = _read_int(query_mapping, "width")
    height = _read_int(query_mapping, "height")

    note = query_mapping.get("note")
    if note is not None and not isinstance(note, str):
        raise ValueError("Query field 'note' must be a string when provided.")

    screenshot: Optional[NormalizedGrid] = None
    if "screenshot" in query_mapping:
        screenshot = _normalize_grid(
            name=f"queries[{step}].screenshot", grid=query_mapping["screenshot"]
        )

    return ViewportQuery(
        step=step,
        x=x,
        y=y,
        width=width,
        height=height,
        screenshot=screenshot,
        note=note,
    )


def _validate_trace(trace: DebugTrace) -> None:
    rows = len(trace.start_grid)
    columns = len(trace.start_grid[0])
    for query in trace.queries:
        if query.width <= 0 or query.height <= 0:
            raise ValueError(
                f"Query step {query.step} must have positive width and height."
            )
        if query.x < 0 or query.y < 0:
            raise ValueError(f"Query step {query.step} must stay inside the grid.")
        if query.x + query.width > columns or query.y + query.height > rows:
            raise ValueError(f"Query step {query.step} exceeds the start grid bounds.")
        if query.screenshot is not None:
            screenshot_rows = len(query.screenshot)
            screenshot_columns = len(query.screenshot[0])
            if screenshot_rows != query.height or screenshot_columns != query.width:
                raise ValueError(
                    f"Query step {query.step} screenshot shape must match viewport width and height."
                )


def _normalize_grid(name: str, grid: object) -> NormalizedGrid:
    if not isinstance(grid, Sequence) or isinstance(grid, (str, bytes, bytearray)):
        raise ValueError(f"Grid '{name}' must be a sequence of rows.")
    row_sequence = cast(Sequence[object], grid)

    rows: List[Tuple[CellValue, ...]] = []
    row_width: Optional[int] = None
    for row_index, row in enumerate(row_sequence):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ValueError(f"Grid '{name}' row {row_index} must be a sequence.")
        cell_sequence = cast(Sequence[object], row)

        normalized_row: List[CellValue] = []
        for cell in cell_sequence:
            if not _is_cell_value(cell):
                raise ValueError(
                    f"Grid '{name}' contains unsupported cell value {cell!r}."
                )
            normalized_row.append(cast(CellValue, cell))

        if not normalized_row:
            raise ValueError(f"Grid '{name}' row {row_index} may not be empty.")

        if row_width is None:
            row_width = len(normalized_row)
        elif row_width != len(normalized_row):
            raise ValueError(f"Grid '{name}' must be rectangular.")

        rows.append(tuple(normalized_row))

    if not rows:
        raise ValueError(f"Grid '{name}' may not be empty.")

    return tuple(rows)


def _build_value_color_map(trace: DebugTrace) -> Dict[str, str]:
    values_in_order: List[str] = []
    seen_values: Set[str] = set()
    for grid in (
        trace.start_grid,
        *(query.screenshot for query in trace.queries if query.screenshot is not None),
    ):
        for row in grid:
            for cell in row:
                key = _color_key(cell)
                if key not in seen_values:
                    seen_values.add(key)
                    values_in_order.append(key)

    color_map: Dict[str, str] = {}
    for index, key in enumerate(values_in_order):
        if key == "∅":
            color_map[key] = TOKENS.empty_cell
            continue
        color_map[key] = VALUE_PALETTE[index % len(VALUE_PALETTE)]
    return color_map


def _render_grid_svg(
    *,
    grid: NormalizedGrid,
    value_colors: Mapping[str, str],
    title: str,
    cell_size: int,
    overlays: Sequence[ViewportQuery],
) -> str:
    rows = len(grid)
    columns = len(grid[0])
    show_labels = rows <= LABEL_VISIBLE_LIMIT and columns <= LABEL_VISIBLE_LIMIT

    drawing_width = columns * cell_size + (columns - 1) * CELL_GAP
    drawing_height = rows * cell_size + (rows - 1) * CELL_GAP
    width = drawing_width + SVG_PADDING * 2
    height = drawing_height + SVG_PADDING * 2

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        "<defs>",
        "<style>",
        f".frame {{ fill: {TOKENS.panel_background}; stroke: {TOKENS.panel_border}; stroke-width: 1; }}",
        f".cell {{ stroke: {TOKENS.cell_border}; stroke-width: 1; }}",
        f".value {{ fill: {TOKENS.label_text}; font-family: ui-monospace, SFMono-Regular, monospace; font-size: {cell_size * 0.32:.2f}px; font-weight: 700; text-anchor: middle; dominant-baseline: central; }}",
        f".overlay-label {{ fill: {TOKENS.overlay_label_text}; font-family: ui-monospace, SFMono-Regular, monospace; font-size: {cell_size * 0.52:.2f}px; font-weight: 700; text-anchor: start; dominant-baseline: hanging; }}",
        "</style>",
        "</defs>",
        f'<rect class="frame" x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="{SVG_CORNER_RADIUS}" />',
    ]

    for row_index, row in enumerate(grid):
        for column_index, cell in enumerate(row):
            x = SVG_PADDING + column_index * (cell_size + CELL_GAP)
            y = SVG_PADDING + row_index * (cell_size + CELL_GAP)
            parts.append(
                f'<rect class="cell" x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="{value_colors[_color_key(cell)]}" rx="2" />'
            )
            if show_labels:
                label = html.escape(_format_cell_value(cell))
                label_x = x + cell_size / 2
                label_y = y + cell_size / 2
                parts.append(
                    f'<text class="value" x="{label_x}" y="{label_y}">{label}</text>'
                )

    for overlay_index, overlay in enumerate(overlays):
        x = SVG_PADDING + overlay.x * (cell_size + CELL_GAP)
        y = SVG_PADDING + overlay.y * (cell_size + CELL_GAP)
        overlay_width = overlay.width * cell_size + (overlay.width - 1) * CELL_GAP
        overlay_height = overlay.height * cell_size + (overlay.height - 1) * CELL_GAP
        color = OVERLAY_PALETTE[overlay_index % len(OVERLAY_PALETTE)]
        parts.append(
            f'<rect x="{x}" y="{y}" width="{overlay_width}" height="{overlay_height}" fill="{color}" fill-opacity="{OVERLAY_FILL_OPACITY}" stroke="{color}" stroke-width="{SVG_STROKE_WIDTH}" stroke-opacity="{OVERLAY_STROKE_OPACITY}" rx="3" />'
        )
        label_x = x + 4
        label_y = y + 4
        parts.append(
            f'<text class="overlay-label" x="{label_x}" y="{label_y}">Q{overlay.step:02d}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _render_html_report(
    *,
    trace: DebugTrace,
    output_dir: Path,
    start_state_svg: Path,
    query_overlay_svg: Path,
    query_artifacts: Sequence[_QueryArtifact],
    value_colors: Mapping[str, str],
) -> str:
    rows = len(trace.start_grid)
    columns = len(trace.start_grid[0])
    start_state_path = start_state_svg.relative_to(output_dir).as_posix()
    query_overlay_path = query_overlay_svg.relative_to(output_dir).as_posix()
    legend = _render_legend(value_colors)
    query_cards = "".join(
        _render_query_card(output_dir=output_dir, artifact=artifact)
        for artifact in query_artifacts
    )

    return f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{html.escape(trace.title)}</title>
    <style>
      :root {{
        --page-background: {TOKENS.page_background};
        --panel-background: {TOKENS.panel_background};
        --panel-background-alt: {TOKENS.panel_background_alt};
        --panel-border: {TOKENS.panel_border};
        --text-primary: {TOKENS.text_primary};
        --text-muted: {TOKENS.text_muted};
        --accent: {TOKENS.accent};
        --accent-alt: {TOKENS.accent_alt};
        --shadow: {TOKENS.shadow};
        --space-1: 0.25rem;
        --space-2: 0.5rem;
        --space-3: 0.75rem;
        --space-4: 1rem;
        --space-5: 1.5rem;
        --space-6: 2rem;
        --radius-sm: 10px;
        --radius-md: 16px;
        --border-width: 1px;
        --content-width: 88rem;
        --shadow-panel: 0 24px 80px var(--shadow);
        --font-display: ui-serif, Georgia, Cambria, \"Times New Roman\", serif;
        --font-body: ui-sans-serif, \"Segoe UI\", sans-serif;
        --font-mono: ui-monospace, SFMono-Regular, monospace;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        background:
          radial-gradient(circle at top left, rgba(116, 216, 255, 0.16), transparent 28%),
          radial-gradient(circle at top right, rgba(140, 245, 207, 0.1), transparent 22%),
          var(--page-background);
        color: var(--text-primary);
        font-family: var(--font-body);
      }}
      a {{ color: var(--accent); }}
      main {{
        width: min(calc(100% - var(--space-6)), var(--content-width));
        margin: 0 auto;
        padding: var(--space-6) 0 calc(var(--space-6) * 1.5);
      }}
      .hero {{
        display: grid;
        gap: var(--space-5);
        padding: var(--space-6);
        background: linear-gradient(135deg, rgba(16, 27, 49, 0.96), rgba(22, 37, 64, 0.92));
        border: var(--border-width) solid var(--panel-border);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-panel);
      }}
      .eyebrow {{
        margin: 0;
        color: var(--accent-alt);
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-size: 0.8rem;
      }}
      h1 {{
        margin: 0;
        font-family: var(--font-display);
        font-size: clamp(2rem, 4vw, 3.4rem);
        line-height: 0.95;
      }}
      .hero p {{
        margin: 0;
        max-width: 56rem;
        color: var(--text-muted);
        font-size: 1rem;
        line-height: 1.6;
      }}
      .stats {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(10rem, 1fr));
        gap: var(--space-3);
      }}
      .stat {{
        padding: var(--space-4);
        border-radius: var(--radius-sm);
        border: var(--border-width) solid var(--panel-border);
        background: rgba(8, 17, 31, 0.42);
      }}
      .stat strong {{
        display: block;
        font-family: var(--font-mono);
        font-size: 1.5rem;
      }}
      .stat span {{ color: var(--text-muted); }}
      .grid {{
        display: grid;
        gap: var(--space-5);
        grid-template-columns: repeat(auto-fit, minmax(22rem, 1fr));
        margin-top: var(--space-6);
      }}
      .panel {{
        padding: var(--space-4);
        background: rgba(16, 27, 49, 0.9);
        border: var(--border-width) solid var(--panel-border);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-panel);
      }}
      .panel h2,
      .panel h3 {{ margin: 0 0 var(--space-3); }}
      .panel p {{ margin: 0 0 var(--space-4); color: var(--text-muted); line-height: 1.5; }}
      .panel img {{
        display: block;
        width: 100%;
        height: auto;
        border-radius: calc(var(--radius-sm) - 2px);
        border: var(--border-width) solid var(--panel-border);
        background: var(--panel-background-alt);
      }}
      .legend {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(8rem, 1fr));
        gap: var(--space-2);
        list-style: none;
        padding: 0;
        margin: 0;
      }}
      .legend li {{
        display: flex;
        align-items: center;
        gap: var(--space-2);
        color: var(--text-muted);
        font-family: var(--font-mono);
        font-size: 0.9rem;
      }}
      .swatch {{
        width: 1rem;
        height: 1rem;
        border-radius: 999px;
        border: var(--border-width) solid var(--panel-border);
      }}
      .queries {{
        display: grid;
        gap: var(--space-4);
        grid-template-columns: repeat(auto-fit, minmax(18rem, 1fr));
      }}
      .query-card {{
        display: grid;
        gap: var(--space-3);
        padding: var(--space-4);
        border-radius: var(--radius-md);
        border: var(--border-width) solid var(--panel-border);
        background: linear-gradient(180deg, rgba(16, 27, 49, 0.98), rgba(8, 17, 31, 0.95));
      }}
      .query-card dl {{
        display: grid;
        grid-template-columns: auto 1fr;
        gap: var(--space-2) var(--space-3);
        margin: 0;
        color: var(--text-muted);
      }}
      .query-card dt {{ font-family: var(--font-mono); }}
      .query-card dd {{ margin: 0; }}
      .query-card code {{
        font-family: var(--font-mono);
        font-size: 0.9rem;
      }}
      .query-card .empty {{
        padding: var(--space-4);
        border-radius: var(--radius-sm);
        border: var(--border-width) dashed var(--panel-border);
        color: var(--text-muted);
      }}
      .artifact-link {{
        font-family: var(--font-mono);
        font-size: 0.85rem;
        color: var(--accent-alt);
      }}
    </style>
  </head>
  <body>
    <main>
      <section class=\"hero\">
        <p class=\"eyebrow\">Astar Island / solver debug view</p>
        <div>
          <h1>{html.escape(trace.title)}</h1>
          <p>Static, local-first debug output for inspecting the start grid, every queried viewport, and the saved screenshot artifacts without adding a frontend stack.</p>
        </div>
        <div class=\"stats\">
          <div class=\"stat\"><strong>{columns}×{rows}</strong><span>start grid</span></div>
          <div class=\"stat\"><strong>{len(trace.queries)}</strong><span>queried viewports</span></div>
          <div class=\"stat\"><strong>{sum(artifact.screenshot_svg is not None for artifact in query_artifacts)}</strong><span>saved screenshots</span></div>
        </div>
      </section>

      <section class=\"grid\">
        <article class=\"panel\">
          <h2>Start state</h2>
          <p>The raw grid is rendered without overlays so you can inspect the starting world state before queries accumulate.</p>
          <img src=\"{start_state_path}\" alt=\"Start state grid\" />
        </article>
        <article class=\"panel\">
          <h2>Viewport overlay</h2>
          <p>Each query is drawn on top of the same grid. Labels match the query step number used in the screenshot artifact names.</p>
          <img src=\"{query_overlay_path}\" alt=\"Queried viewport overlay\" />
        </article>
      </section>

      <section class=\"grid\">
        <article class=\"panel\">
          <h3>Cell legend</h3>
          <p>Values are colored consistently across the start grid and all viewport screenshot artifacts.</p>
          <ul class=\"legend\">{legend}</ul>
        </article>
        <article class=\"panel\">
          <h3>Saved screenshot artifacts</h3>
          <p>Viewport screenshots are written as SVG files with stable filenames that include the query order, step, and rectangle coordinates.</p>
          <div class=\"queries\">{query_cards}</div>
        </article>
      </section>
    </main>
  </body>
</html>
"""


def _render_legend(value_colors: Mapping[str, str]) -> str:
    legend_items = []
    for key, color in value_colors.items():
        label = html.escape(key)
        legend_items.append(
            f'<li><span class="swatch" style="background:{color};"></span><span>{label}</span></li>'
        )
    return "".join(legend_items)


def _render_query_card(*, output_dir: Path, artifact: _QueryArtifact) -> str:
    query = artifact.query
    note = html.escape(query.note) if query.note is not None else "No note recorded."
    metadata = f"{query.width}×{query.height} at ({query.x}, {query.y})"

    screenshot_markup: str
    if artifact.screenshot_svg is None:
        screenshot_markup = (
            '<div class="empty">No screenshot data was provided for this query.</div>'
        )
    else:
        relative_path = artifact.screenshot_svg.relative_to(output_dir).as_posix()
        screenshot_markup = (
            f'<img src="{relative_path}" width="{CARD_IMAGE_WIDTH}" alt="Viewport screenshot for query {query.step}" />'
            f'<a class="artifact-link" href="{relative_path}">{relative_path}</a>'
        )

    return (
        '<article class="query-card">'
        f"<h3>Q{query.step:02d}</h3>"
        f"<p>{note}</p>"
        "<dl>"
        "<dt>viewport</dt>"
        f"<dd><code>{html.escape(metadata)}</code></dd>"
        "<dt>order</dt>"
        f"<dd><code>{artifact.order:03d}</code></dd>"
        "<dt>filename</dt>"
        f"<dd><code>{html.escape(_screenshot_filename(order=artifact.order, query=query))}</code></dd>"
        "</dl>"
        f"{screenshot_markup}"
        "</article>"
    )


def _screenshot_filename(*, order: int, query: ViewportQuery) -> str:
    return (
        f"query-{order:03d}_step-{query.step:03d}_"
        f"x-{query.x:03d}_y-{query.y:03d}_w-{query.width:03d}_h-{query.height:03d}.svg"
    )


def _format_cell_value(value: CellValue) -> str:
    if value is None:
        return "∅"
    if isinstance(value, float):
        return f"{value:.3g}"[:VALUE_LABEL_LIMIT]
    return str(value)[:VALUE_LABEL_LIMIT]


def _color_key(value: CellValue) -> str:
    return _format_cell_value(value)


def _read_int(
    mapping: Mapping[str, object], key: str, default: Optional[int] = None
) -> int:
    raw_value = mapping.get(key, default)
    if type(raw_value) is not int:
        raise ValueError(f"Field '{key}' must be an integer.")
    return raw_value


def _require_field(mapping: Mapping[str, object], key: str) -> object:
    if key not in mapping:
        raise ValueError(f"Missing required field '{key}'.")
    return mapping[key]


def _is_cell_value(value: object) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _read_path_namespace_value(namespace: argparse.Namespace, field_name: str) -> Path:
    value = getattr(namespace, field_name)
    if not isinstance(value, Path):
        raise ValueError(f"Argument '{field_name}' must resolve to a path.")
    return value
