"""CLI entry point for Twitter Bookmarks Semantic Graph.

Usage:
    python -m examples.bookmarks_graph.cli --help

    bookmarks auth                        # OAuth2 PKCE → browser → token saved
    bookmarks fetch                       # Pull bookmarks from Twitter API
    bookmarks build                       # Embed + graph + categories
    bookmarks categories                  # List detected topic clusters
    bookmarks search "query text"         # Semantic search
    bookmarks related <tweet_id>          # Graph neighbors of a tweet
    bookmarks stats                       # Graph statistics
    bookmarks export --format json|md|gexf
"""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

app = typer.Typer(
    name="bookmarks",
    help="Organize your Twitter bookmarks into a categorized semantic graph.",
    add_completion=False,
)
console = Console()


# ─── auth ─────────────────────────────────────────────────────────────────────

@app.command()
def auth(
    client_id: Optional[str] = typer.Option(None, "--client-id", help="Twitter OAuth2 Client ID"),
):
    """Authenticate with Twitter via OAuth 2.0 PKCE. Opens your browser once."""
    from . import auth as auth_module

    config = auth_module.load_config()
    cid = client_id or config.get("client_id")

    if not cid:
        console.print(
            "\n[bold]Setup:[/bold] You need a free Twitter Developer account.\n"
            "  1. Go to https://developer.twitter.com → create a project + app\n"
            "  2. Enable OAuth 2.0, set redirect URI: [cyan]http://localhost:8080/callback[/cyan]\n"
            "  3. Copy your [bold]Client ID[/bold] (no secret needed)\n"
        )
        cid = typer.prompt("Enter your Twitter OAuth2 Client ID")

    try:
        auth_module.authenticate(cid)
        console.print("[green]✓ Authenticated. Token saved to ~/.bookmarks_graph.json[/green]")
    except Exception as e:
        console.print(f"[red]Authentication failed: {e}[/red]")
        raise typer.Exit(1)


# ─── fetch ────────────────────────────────────────────────────────────────────

@app.command()
def fetch(
    max_results: int = typer.Option(800, "--max", help="Maximum bookmarks to fetch (API limit: 800)"),
):
    """Fetch your Twitter bookmarks and store them in ChromaDB with embeddings."""
    from . import auth as auth_module, twitter, storage

    config = auth_module.load_config()
    if not config.get("access_token"):
        console.print("[red]Not authenticated. Run: bookmarks auth[/red]")
        raise typer.Exit(1)

    token = auth_module.get_valid_token()

    with console.status("Fetching your Twitter user ID..."):
        try:
            user_id = twitter.get_user_id(token)
        except Exception as e:
            console.print(f"[red]API error: {e}[/red]")
            raise typer.Exit(1)

    tweets = []
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching bookmarks...", total=None)
            for tweet in twitter.fetch_bookmarks(token, user_id, max_total=max_results):
                tweets.append(tweet)
                progress.update(task, description=f"Fetched [bold]{len(tweets)}[/bold] bookmarks...")
    except Exception as e:
        console.print(f"\n[red]Error fetching bookmarks: {e}[/red]")
        raise typer.Exit(1)

    if not tweets:
        console.print("[yellow]No bookmarks found.[/yellow]")
        return

    with console.status(f"Generating embeddings and storing {len(tweets)} tweets..."):
        storage.upsert_tweets(tweets)

    console.print(f"[green]✓ Stored {len(tweets)} bookmarks in ChromaDB[/green]")


# ─── build ────────────────────────────────────────────────────────────────────

@app.command()
def build(
    threshold: float = typer.Option(0.45, "--threshold", help="Min cosine similarity for graph edges"),
    top_k: int = typer.Option(15, "--top-k", help="Neighbors per tweet to query from ChromaDB"),
):
    """Build the semantic graph and auto-detect topic categories."""
    from . import storage, graph as graph_module, categorizer

    total = storage.count()
    if total == 0:
        console.print("[red]No bookmarks found. Run: bookmarks fetch first.[/red]")
        raise typer.Exit(1)

    console.print(f"Building semantic graph from [bold]{total}[/bold] bookmarks...")
    console.print(f"  Similarity threshold: [cyan]{threshold}[/cyan]  |  Neighbors per tweet: [cyan]{top_k}[/cyan]")

    all_ids = storage.get_all_ids()
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(all_ids)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Building edges...", total=len(all_ids))
        for tweet_id in all_ids:
            for nb in storage.get_neighbors(tweet_id, n=top_k, threshold=threshold):
                if not G.has_edge(tweet_id, nb["id"]):
                    G.add_edge(tweet_id, nb["id"], weight=nb["similarity"])
            progress.advance(task)

    graph_module.save_graph(G)
    n_edges = G.number_of_edges()
    console.print(f"  Graph: [bold]{total}[/bold] nodes, [bold]{n_edges}[/bold] edges")

    with console.status("Detecting topic communities..."):
        partition = graph_module.detect_communities(G)

    with console.status("Labeling categories with TF-IDF..."):
        tweets_map = {t["id"]: t["text"] for t in storage.get_all_tweets()}
        labels = categorizer.label_clusters(partition, tweets_map)
        categorizer.save_categories(partition, labels)

    console.print(
        f"[green]✓ Done — {len(labels)} categories detected.[/green]\n"
        "Run [cyan]bookmarks categories[/cyan] to browse them."
    )


# ─── categories ───────────────────────────────────────────────────────────────

@app.command()
def categories():
    """List all auto-detected topic categories and their tweet counts."""
    from . import storage, categorizer, visualizer, graph as graph_module

    try:
        data = categorizer.load_categories()
    except FileNotFoundError:
        console.print("[red]Categories not built yet. Run: bookmarks build[/red]")
        raise typer.Exit(1)

    labels = data["labels"]
    counts = data["counts"]

    try:
        G = graph_module.load_graph()
        n_edges = G.number_of_edges()
    except FileNotFoundError:
        n_edges = 0

    visualizer.header(storage.count(), len(labels), n_edges)

    cats = [
        {"id": int(k), "label": labels[k], "count": int(counts.get(k, 0))}
        for k in labels
    ]
    visualizer.print_categories(cats)


# ─── search ───────────────────────────────────────────────────────────────────

@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language or keyword query"),
    top: int = typer.Option(10, "--top", "-n", help="Number of results to return"),
):
    """Semantic search across your bookmarks by meaning, not just keywords."""
    from . import storage, categorizer, visualizer

    results = storage.search(query, n_results=top)
    if not results:
        console.print("[yellow]No results.[/yellow]")
        return

    # Annotate with category label if available
    try:
        data = categorizer.load_categories()
        partition = data["partition"]
        labels = data["labels"]
        for r in results:
            comm = partition.get(r["id"])
            if comm is not None:
                r["category_label"] = labels.get(str(comm), "")
    except FileNotFoundError:
        pass

    console.print(f'\nTop [bold]{len(results)}[/bold] results for "[cyan]{query}[/cyan]"\n')
    visualizer.print_search_results(results)


# ─── related ──────────────────────────────────────────────────────────────────

@app.command()
def related(
    tweet_id: str = typer.Argument(..., help="Tweet ID to find neighbors for"),
    top: int = typer.Option(10, "--top", "-n", help="Number of related tweets"),
):
    """Find semantically related tweets via graph traversal."""
    from . import storage, graph as graph_module, visualizer

    tweet = storage.get_tweet(tweet_id)
    if not tweet:
        console.print(f"[red]Tweet {tweet_id} not found in local store.[/red]")
        raise typer.Exit(1)

    try:
        G = graph_module.load_graph()
    except FileNotFoundError:
        console.print("[red]Graph not built. Run: bookmarks build[/red]")
        raise typer.Exit(1)

    raw_neighbors = graph_module.get_neighbors_in_graph(G, tweet_id, top_n=top)
    neighbors = []
    for nid, sim in raw_neighbors:
        nb = storage.get_tweet(nid)
        if nb:
            nb["similarity"] = sim
            neighbors.append(nb)

    visualizer.print_related(tweet, neighbors)


# ─── stats ────────────────────────────────────────────────────────────────────

@app.command()
def stats():
    """Show graph and collection statistics."""
    from . import storage, graph as graph_module, visualizer

    s = {"total_bookmarks": storage.count()}
    try:
        G = graph_module.load_graph()
        s.update(graph_module.graph_stats(G))
    except FileNotFoundError:
        s["graph"] = "not built (run: bookmarks build)"

    visualizer.print_stats(s)


# ─── export ───────────────────────────────────────────────────────────────────

@app.command()
def export(
    fmt: str = typer.Option("json", "--format", "-f", help="Output format: json | md | gexf"),
    output: str = typer.Option("bookmarks_graph", "--output", "-o", help="Output filename (no extension)"),
):
    """Export the graph to JSON (D3.js), Markdown, or GEXF (Gephi)."""
    from . import storage, categorizer, graph as graph_module
    import networkx as nx

    tweets = {t["id"]: t for t in storage.get_all_tweets()}

    try:
        data = categorizer.load_categories()
        partition = data["partition"]
        labels = data["labels"]
    except FileNotFoundError:
        partition, labels = {}, {}

    if fmt == "json":
        out = {
            "tweets": list(tweets.values()),
            "categories": [
                {
                    "id": k,
                    "label": v,
                    "tweet_ids": [tid for tid, c in partition.items() if str(c) == k],
                }
                for k, v in labels.items()
            ],
        }
        Path(f"{output}.json").write_text(json.dumps(out, indent=2))
        console.print(f"[green]✓ Exported to {output}.json[/green]")

    elif fmt == "md":
        lines = ["# Twitter Bookmarks Graph\n\n"]
        for k, label in sorted(labels.items(), key=lambda x: int(x[0])):
            lines.append(f"## {label}\n\n")
            for tid, comm in partition.items():
                if str(comm) == k and tid in tweets:
                    t = tweets[tid]
                    url = t.get("url", "#")
                    author = t.get("author_username", "?")
                    text = t.get("text", "")[:160]
                    lines.append(f"- [@{author}]({url}): {text}\n")
            lines.append("\n")
        Path(f"{output}.md").write_text("".join(lines))
        console.print(f"[green]✓ Exported to {output}.md[/green]")

    elif fmt == "gexf":
        try:
            G = graph_module.load_graph()
        except FileNotFoundError:
            console.print("[red]Graph not built. Run: bookmarks build[/red]")
            raise typer.Exit(1)
        for node in G.nodes():
            if node in tweets:
                t = tweets[node]
                G.nodes[node]["label"] = t.get("text", "")[:60]
                G.nodes[node]["author"] = t.get("author_username", "")
                comm = partition.get(node)
                if comm is not None:
                    G.nodes[node]["category"] = labels.get(str(comm), "")
        nx.write_gexf(G, f"{output}.gexf")
        console.print(f"[green]✓ Exported to {output}.gexf (open in Gephi for visualization)[/green]")

    else:
        console.print(f"[red]Unknown format '{fmt}'. Use: json, md, or gexf[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
