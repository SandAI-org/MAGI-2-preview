#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
json_to_markdown_readable.py

Convert JSON into readable Markdown while preserving all non-empty information.

Default behavior:
- Accepts ordinary json.load/json.loads output: dict, list, str, int, float,
  bool, None.
- CLI parsing uses parse_json_preserve(), which can preserve duplicate keys and
  exact JSON number spellings when possible.
- Recursively drops no-information values by default: null/None, [], {}, "",
  and whitespace-only strings. Parent containers that become empty are dropped.
- Keeps meaningful falsy values: false, 0, 0.0.
- Renders top-level/root keys as Markdown headings.
- Omits only direct primitive leaf keys under each top-level section by default.
  Example: global_layer.context -> a plain paragraph containing its value.
- Keeps all nested/intermediate leaf keys.
  Example: aesthetics.style -> style: photograph.
- Renders arrays of objects as indexed Markdown subsections.
- Does not append raw JSON by default, so dropped empty keys do not reappear.

Library use:

    import json
    from json_to_markdown_readable import render_markdown

    with open("input.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    markdown = render_markdown(data)

CLI use:

    python json_to_markdown_readable.py input.json -o output.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Tuple


@dataclass(frozen=True)
class JsonNumber:
    """Stores the original lexical spelling of a JSON number, such as 11.0."""

    raw: str


@dataclass(frozen=True)
class JsonObject:
    """Stores JSON object pairs without collapsing duplicate keys."""

    pairs: List[Tuple[str, Any]]


def parse_json_preserve(text: str) -> Any:
    """
    Parse JSON while preserving key order, duplicate keys, and number spelling.

    render_markdown() also accepts ordinary json.load()/json.loads() output.
    Use this parser only when duplicate keys or exact number spelling matter.
    """

    parse_text = text[1:] if text.startswith("\ufeff") else text
    return json.loads(
        parse_text,
        object_pairs_hook=lambda pairs: JsonObject(pairs=list(pairs)),
        parse_int=lambda s: JsonNumber(s),
        parse_float=lambda s: JsonNumber(s),
    )


def is_object(value: Any) -> bool:
    return isinstance(value, JsonObject) or isinstance(value, Mapping)


def iter_object_pairs(value: Any) -> Iterable[Tuple[str, Any]]:
    if isinstance(value, JsonObject):
        yield from value.pairs
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
        return
    raise TypeError(f"Expected JSON object, got {type(value).__name__}")


def make_object_like(original: Any, pairs: List[Tuple[str, Any]]) -> Any:
    if isinstance(original, JsonObject):
        return JsonObject(pairs)
    return dict(pairs)


def is_primitive(value: Any) -> bool:
    return not is_object(value) and not isinstance(value, list)


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def inline_string(text: str, *, collapse_string_whitespace: bool) -> str:
    if collapse_string_whitespace:
        return collapse_ws(text)
    return (
        text.replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def primitive_to_text(
    value: Any,
    *,
    quote_strings: bool = False,
    collapse_string_whitespace: bool = False,
) -> str:
    """Render a JSON primitive as text."""

    if isinstance(value, JsonNumber):
        return value.raw
    if isinstance(value, str):
        rendered = inline_string(
            value,
            collapse_string_whitespace=collapse_string_whitespace,
        )
        if quote_strings:
            return json.dumps(rendered, ensure_ascii=False)
        return rendered
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    return json.dumps(value, ensure_ascii=False)


def filter_empty(value: Any, *, keep_empty_string: bool = False) -> Tuple[bool, Any]:
    """
    Recursively remove no-information values.

    Dropped by default:
    - None / JSON null
    - []
    - {}
    - "" and whitespace-only strings
    - parent objects/arrays that become empty after child filtering

    Preserved:
    - False / JSON false
    - 0
    - 0.0
    - non-empty strings
    - non-empty objects/lists after recursive filtering
    """

    if value is None:
        return False, value

    if isinstance(value, str):
        if keep_empty_string:
            return True, value
        return value.strip() != "", value

    if is_object(value):
        kept_pairs: List[Tuple[str, Any]] = []
        for key, child in iter_object_pairs(value):
            keep_child, filtered_child = filter_empty(
                child,
                keep_empty_string=keep_empty_string,
            )
            if keep_child:
                kept_pairs.append((key, filtered_child))
        if not kept_pairs:
            return False, make_object_like(value, [])
        return True, make_object_like(value, kept_pairs)

    if isinstance(value, list):
        kept_items: List[Any] = []
        for item in value:
            keep_item, filtered_item = filter_empty(
                item,
                keep_empty_string=keep_empty_string,
            )
            if keep_item:
                kept_items.append(filtered_item)
        if not kept_items:
            return False, []
        return True, kept_items

    return True, value


def heading(level: int, title: str, *, max_heading_level: int = 6) -> str:
    """Return a Markdown heading, falling back to bold labels beyond H6."""

    safe_level = max(1, min(level, max_heading_level))
    if level <= max_heading_level:
        return f"{'#' * safe_level} {title}"
    return f"*{title}*"


def render_inline_list(
    values: List[Any],
    *,
    quote_strings: bool,
    collapse_string_whitespace: bool,
    primitive_list_separator: str,
) -> str:
    """Render a list inline. Intended for primitive or simple lists."""

    parts: List[str] = []
    for item in values:
        if is_primitive(item):
            parts.append(
                primitive_to_text(
                    item,
                    quote_strings=quote_strings,
                    collapse_string_whitespace=collapse_string_whitespace,
                )
            )
        elif isinstance(item, list):
            parts.append(
                "["
                + render_inline_list(
                    item,
                    quote_strings=quote_strings,
                    collapse_string_whitespace=collapse_string_whitespace,
                    primitive_list_separator=primitive_list_separator,
                )
                + "]"
            )
        elif is_object(item):
            inner = []
            for key, child in iter_object_pairs(item):
                if is_primitive(child):
                    inner.append(
                        f"{key}:"
                        + primitive_to_text(
                            child,
                            quote_strings=quote_strings,
                            collapse_string_whitespace=collapse_string_whitespace,
                        )
                    )
                else:
                    inner.append(f"{key}: ...")
            parts.append("{" + ";".join(inner) + "}")
    return primitive_list_separator.join(parts)


def list_is_all_primitives(value: List[Any]) -> bool:
    return all(is_primitive(item) for item in value)


def list_is_simple(value: List[Any]) -> bool:
    """Simple lists are compact enough to render on one line."""

    if list_is_all_primitives(value):
        return True
    return False


def render_value(
    key: str | None,
    value: Any,
    *,
    level: int,
    root_depth: int,
    lines: List[str],
    omit_section_leaf_keys: bool,
    quote_strings: bool,
    collapse_string_whitespace: bool,
    primitive_list_separator: str,
    max_heading_level: int,
    group_scalar_fields_first: bool,
) -> None:
    """
    Render a JSON value into Markdown lines.

    root_depth is the object depth of the current value relative to the document:
    - 0: root document
    - 1: direct child of the root, usually a large section such as global_layer
    - 2+: nested content
    """

    if is_primitive(value):
        text = primitive_to_text(
            value,
            quote_strings=quote_strings,
            collapse_string_whitespace=collapse_string_whitespace,
        )
        if key is None:
            lines.append(text)
        elif omit_section_leaf_keys and root_depth == 1:
            # Only direct primitive leaves under top-level sections omit their keys.
            lines.append(text)
        else:
            lines.append(f"{key}:{text}")
        return

    if isinstance(value, list):
        if not value:
            return
        if key is not None:
            if list_is_simple(value):
                inline = render_inline_list(
                    value,
                    quote_strings=quote_strings,
                    collapse_string_whitespace=collapse_string_whitespace,
                    primitive_list_separator=primitive_list_separator,
                )
                lines.append(f"{key}:{inline}")
                return
            lines.append(heading(level, key, max_heading_level=max_heading_level))
            lines.append("")
        for idx, item in enumerate(value, start=1):
            item_title = f"{key}{idx}" if key else f"item{idx}"
            if is_primitive(item):
                text = primitive_to_text(
                    item,
                    quote_strings=quote_strings,
                    collapse_string_whitespace=collapse_string_whitespace,
                )
                lines.append(f"{item_title}:{text}")
            elif isinstance(item, list):
                lines.append(heading(level + 1, item_title, max_heading_level=max_heading_level))
                lines.append("")
                render_value(
                    None,
                    item,
                    level=level + 2,
                    root_depth=root_depth + 1,
                    lines=lines,
                    omit_section_leaf_keys=omit_section_leaf_keys,
                    quote_strings=quote_strings,
                    collapse_string_whitespace=collapse_string_whitespace,
                    primitive_list_separator=primitive_list_separator,
                    max_heading_level=max_heading_level,
                    group_scalar_fields_first=group_scalar_fields_first,
                )
            elif is_object(item):
                lines.append(heading(level + 1, item_title, max_heading_level=max_heading_level))
                lines.append("")
                render_object_contents(
                    item,
                    level=level + 2,
                    root_depth=root_depth + 1,
                    lines=lines,
                    omit_section_leaf_keys=omit_section_leaf_keys,
                    quote_strings=quote_strings,
                    collapse_string_whitespace=collapse_string_whitespace,
                    primitive_list_separator=primitive_list_separator,
                    max_heading_level=max_heading_level,
                    group_scalar_fields_first=group_scalar_fields_first,
                )
                lines.append("")
        return

    if is_object(value):
        if key is not None:
            lines.append(heading(level, key, max_heading_level=max_heading_level))
            lines.append("")
        render_object_contents(
            value,
            level=level + (1 if key is not None else 0),
            root_depth=root_depth + (1 if key is not None else 0),
            lines=lines,
            omit_section_leaf_keys=omit_section_leaf_keys,
            quote_strings=quote_strings,
            collapse_string_whitespace=collapse_string_whitespace,
            primitive_list_separator=primitive_list_separator,
            max_heading_level=max_heading_level,
            group_scalar_fields_first=group_scalar_fields_first,
        )
        return


def child_is_simple_for_line(child: Any) -> bool:
    """Return True when a child can be safely rendered as one key/value line."""

    return is_primitive(child) or (isinstance(child, list) and list_is_simple(child))


def render_object_contents(
    obj: Any,
    *,
    level: int,
    root_depth: int,
    lines: List[str],
    omit_section_leaf_keys: bool,
    quote_strings: bool,
    collapse_string_whitespace: bool,
    primitive_list_separator: str,
    max_heading_level: int,
    group_scalar_fields_first: bool,
) -> None:
    """Render only the children of an object.

    By default, scalar/simple fields are rendered before nested sections. This
    keeps Markdown hierarchy clear: a scalar field that appears after a nested
    heading in the original JSON will not look as if it belongs to that nested
    heading. The field values and keys are preserved; only local presentation
    order is changed for readability.
    """

    pairs = list(iter_object_pairs(obj))
    if group_scalar_fields_first:
        simple_pairs = [(key, child) for key, child in pairs if child_is_simple_for_line(child)]
        complex_pairs = [(key, child) for key, child in pairs if not child_is_simple_for_line(child)]
        ordered_pairs = simple_pairs + complex_pairs
    else:
        ordered_pairs = pairs

    first = True
    for key, child in ordered_pairs:
        if not first and (is_object(child) or isinstance(child, list)):
            if lines and lines[-1] != "":
                lines.append("")
        render_value(
            key,
            child,
            level=level,
            root_depth=root_depth,
            lines=lines,
            omit_section_leaf_keys=omit_section_leaf_keys,
            quote_strings=quote_strings,
            collapse_string_whitespace=collapse_string_whitespace,
            primitive_list_separator=primitive_list_separator,
            max_heading_level=max_heading_level,
            group_scalar_fields_first=group_scalar_fields_first,
        )
        first = False


def normalize_blank_lines(lines: List[str], *, max_blank_lines: int = 1) -> str:
    out: List[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= max_blank_lines:
                out.append(line)
        else:
            blank_count = 0
            out.append(line.rstrip())
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


def render_markdown(
    data: Any,
    *,
    drop_empty: bool = True,
    keep_empty_string: bool = False,
    omit_section_leaf_keys: bool = True,
    quote_strings: bool = False,
    collapse_string_whitespace: bool = False,
    primitive_list_separator: str = " | ",
    max_heading_level: int = 6,
    group_scalar_fields_first: bool = True,
) -> str:
    """
    Convert JSON-like data into readable Markdown.

    Args:
        data: JSON-like value, usually returned by json.load().
        drop_empty: Recursively drop null, empty arrays/objects, and empty strings.
        keep_empty_string: Preserve empty strings if drop_empty is enabled.
        omit_section_leaf_keys: Omit only direct primitive leaf keys under top-level
            sections. Turn off to preserve every key label.
        quote_strings: Render strings as JSON string literals.
        collapse_string_whitespace: Collapse internal whitespace in strings.
        primitive_list_separator: Separator used inside inline primitive lists.
        max_heading_level: Maximum Markdown heading depth.
        group_scalar_fields_first: Render scalar/simple fields before nested sections
            in each object to avoid Markdown hierarchy ambiguity.
    """

    if isinstance(data, str):
        data = json.loads(data.strip())

    value = data
    if drop_empty:
        keep, filtered = filter_empty(value, keep_empty_string=keep_empty_string)
        if not keep:
            return ""
        value = filtered

    lines: List[str] = []

    if is_object(value):
        # Root object: render each root key as a top-level section.
        for idx, (key, child) in enumerate(iter_object_pairs(value)):
            if idx > 0:
                lines.append("")
            render_value(
                key,
                child,
                level=1,
                root_depth=0,
                lines=lines,
                omit_section_leaf_keys=omit_section_leaf_keys,
                quote_strings=quote_strings,
                collapse_string_whitespace=collapse_string_whitespace,
                primitive_list_separator=primitive_list_separator,
                max_heading_level=max_heading_level,
                group_scalar_fields_first=group_scalar_fields_first,
            )
    else:
        render_value(
            None,
            value,
            level=1,
            root_depth=0,
            lines=lines,
            omit_section_leaf_keys=omit_section_leaf_keys,
            quote_strings=quote_strings,
            collapse_string_whitespace=collapse_string_whitespace,
            primitive_list_separator=primitive_list_separator,
            max_heading_level=max_heading_level,
            group_scalar_fields_first=group_scalar_fields_first,
        )

    return normalize_blank_lines(lines)


def load_input(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    return parse_json_preserve(text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert JSON to readable Markdown while preserving non-empty information."
    )
    parser.add_argument("input", type=Path, help="Path to a JSON or text file containing JSON.")
    parser.add_argument("-o", "--output", type=Path, help="Output Markdown path. Defaults to stdout.")
    parser.add_argument(
        "--keep-empty-string",
        action="store_true",
        help="Keep empty strings instead of dropping them.",
    )
    parser.add_argument(
        "--no-drop-empty",
        action="store_true",
        help="Do not drop null, empty arrays, empty objects, or empty strings.",
    )
    parser.add_argument(
        "--keep-section-leaf-keys",
        action="store_true",
        help="Keep direct primitive leaf keys under top-level sections, such as context: ...",
    )
    parser.add_argument(
        "--quote-strings",
        action="store_true",
        help="Render strings as JSON string literals.",
    )
    parser.add_argument(
        "--collapse-string-whitespace",
        action="store_true",
        help="Collapse repeated whitespace inside string values.",
    )
    parser.add_argument(
        "--primitive-list-separator",
        default=" | ",
        help="Separator for inline primitive lists. Default: ' | '.",
    )
    parser.add_argument(
        "--max-heading-level",
        type=int,
        default=6,
        help="Maximum Markdown heading level. Default: 6.",
    )
    parser.add_argument(
        "--preserve-object-order",
        action="store_true",
        help="Keep original local object order instead of grouping scalar fields before nested sections.",
    )

    args = parser.parse_args()

    try:
        data = load_input(args.input)
        markdown = render_markdown(
            data,
            drop_empty=not args.no_drop_empty,
            keep_empty_string=args.keep_empty_string,
            omit_section_leaf_keys=not args.keep_section_leaf_keys,
            quote_strings=args.quote_strings,
            collapse_string_whitespace=args.collapse_string_whitespace,
            primitive_list_separator=args.primitive_list_separator,
            max_heading_level=args.max_heading_level,
            group_scalar_fields_first=not args.preserve_object_order,
        )
    except Exception as exc:  # noqa: BLE001 - make CLI failures readable.
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
