"""Inspect GPT-2 tokenization used by data/Osho/prepare.py."""

import argparse
import html
from pathlib import Path

from transformers import GPT2TokenizerFast


DEFAULT_TEXT = "The mind becomes quiet when we observe each thought without judgment."


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize GPT-2 BPE tokenization.")
    parser.add_argument(
        "--tokenizer",
        default="gpt2",
        help="Local tokenizer directory or Hugging Face model name (default: gpt2).",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Paragraph to tokenize.",
    )
    parser.add_argument(
        "--text_file",
        type=Path,
        help="Read the paragraph from this UTF-8 text file instead of --text.",
    )
    parser.add_argument(
        "--html",
        type=Path,
        help="Also write a browser-friendly colored visualization to this HTML file.",
    )
    return parser.parse_args()


def color_token(token, index):
    colors = ("\033[96m", "\033[92m", "\033[93m", "\033[95m", "\033[94m")
    visible = token.replace(" ", "·").replace("\n", "↵\\n")
    return f"{colors[index % len(colors)]}{visible}\033[0m"


def write_html(path, tokens, token_ids):
    spans = []
    for index, (token, token_id) in enumerate(zip(tokens, token_ids)):
        text = token.replace(" ", "·").replace("\n", "↵\\n")
        spans.append(
            f'<span class="token c{index % 5}" title="ID: {token_id}">'
            f"{html.escape(text)}</span>"
        )
    document = """<!doctype html>
<meta charset="utf-8">
<title>GPT-2 tokenization</title>
<style>
body { background: #101418; color: #e8edf2; font: 16px/1.8 monospace; padding: 2rem; }
.token { padding: .2rem .35rem; margin: .12rem; border-radius: .25rem; }
.c0 { background: #164e63; } .c1 { background: #166534; }
.c2 { background: #854d0e; } .c3 { background: #701a75; }
.c4 { background: #1e40af; }
</style>
<p>Hover over a token to see its token ID.</p>
<p>""" + "".join(spans) + "</p>\n"
    path.write_text(document, encoding="utf-8")


def main():
    args = parse_args()
    text = args.text_file.read_text(encoding="utf-8") if args.text_file else args.text
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    print(f"Tokenizer: {args.tokenizer}")
    print(f"Characters: {len(text)} | Tokens: {len(token_ids)}")
    print("\nColored token sequence (· represents a space):")
    print(" ".join(color_token(token, index) for index, token in enumerate(tokens)))
    print("\nToken table:")
    print(f"{'#':>4} {'ID':>6}  TOKEN")
    for index, (token, token_id) in enumerate(zip(tokens, token_ids)):
        print(f"{index:>4} {token_id:>6}  {token!r}")

    if args.html:
        write_html(args.html, tokens, token_ids)
        print(f"\nWrote HTML visualization to {args.html}")


if __name__ == "__main__":
    main()
