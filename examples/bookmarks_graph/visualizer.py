"""Rich terminal output helpers."""

from typing import Dict, List

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def header(n_tweets: int, n_categories: int, n_edges: int):
    body = (
        f"[bold white]{n_tweets}[/bold white] bookmarks  •  "
        f"[bold white]{n_categories}[/bold white] categories  •  "
        f"[bold white]{n_edges}[/bold white] connections"
    )
    console.print(Panel(body, title="[bold cyan]Twitter Bookmarks Graph[/bold cyan]", border_style="cyan"))


def print_categories(categories: List[dict]):
    """Print category table sorted by tweet count."""
    if not categories:
        console.print("[yellow]No categories found. Run: bookmarks build[/yellow]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=4)
    table.add_column("Top Keywords", style="cyan")
    table.add_column("Tweets", justify="right", style="bold white", width=8)
    table.add_column("", min_width=22)

    sorted_cats = sorted(categories, key=lambda c: c["count"], reverse=True)
    max_count = sorted_cats[0]["count"] if sorted_cats else 1

    for i, cat in enumerate(sorted_cats):
        filled = int(cat["count"] / max_count * 20)
        bar = "[green]" + "█" * filled + "[/green]" + "[dim]" + "░" * (20 - filled) + "[/dim]"
        table.add_row(str(i + 1), cat["label"], str(cat["count"]), bar)

    console.print(table)


def print_search_results(results: List[dict]):
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Score", style="green", width=6)
    table.add_column("Author", style="cyan", width=20)
    table.add_column("Tweet")
    table.add_column("Category", style="dim", width=22)

    for r in results:
        score = f"{r.get('score', 0):.2f}"
        author = f"@{r.get('author_username', '?')}"
        text = r.get("text", "")[:160]
        category = r.get("category_label", "")
        table.add_row(score, author, text, category)

    console.print(table)


def print_related(tweet: dict, neighbors: List[dict]):
    src_text = tweet.get("text", "")[:220]
    src_author = f"@{tweet.get('author_username', '?')}"
    console.print(
        Panel(
            f"[bold cyan]{src_author}[/bold cyan]  {src_text}",
            title="[cyan]Source Tweet[/cyan]",
            border_style="dim",
        )
    )

    if not neighbors:
        console.print("[yellow]No related tweets found.[/yellow]")
        return

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Sim", style="green", width=6)
    table.add_column("Author", style="cyan", width=20)
    table.add_column("Tweet")

    for nb in neighbors:
        sim = f"{nb.get('similarity', 0):.2f}"
        author = f"@{nb.get('author_username', '?')}"
        text = nb.get("text", "")[:160]
        table.add_row(sim, author, text)

    console.print(table)


def print_stats(stats: dict):
    table = Table(box=box.SIMPLE, show_header=False)
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold white")
    for k, v in stats.items():
        label = k.replace("_", " ").title()
        table.add_row(label, str(v))
    console.print(Panel(table, title="[cyan]Graph Statistics[/cyan]", border_style="cyan"))
