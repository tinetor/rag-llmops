    parser.add_argument(
        "--chapter",
        nargs="+",
        default=None,
        help="Manually filter chapters, e.g. --chapter II III.",
    )
    parser.add_argument(
        "--route",
        action="store_true",
        help="Automatically select chapters before retrieving chunks.",
    )
    parser.add_argument(
        "--top-chapters",
        type=int,
        default=2,
        help="Number of chapters selected by routing (default: 2).",
    )
    parser.add_argument(
        "--route-pool",
        type=int,
        default=30,
        help="Number of global dense results used for chapter routing (default: 30).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of final chunks to return.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full chunk text instead of the first 500 characters.",
    )
    args = parser.parse_args()

    if args.route and args.chapter:
        parser.error("Use either --route or --chapter, not both.")
    if args.top_chapters < 1:
        parser.error("--top-chapters must be at least 1.")
    if args.route_pool < 1:
        parser.error("--route-pool must be at least 1.")

    config = load_config()
    mode = args.mode or config["retrieval"]["mode"]
    top_k = args.top_k if args.top_k is not None else config["retrieval"]["top_k"]
    persist_dir = config["index"]["persist_dir"] + "_qdrant"
    collection_name = config["index"]["collection_name"]

    routed: list[tuple[str, float]] | None = None
    chapters = args.chapter
    results: list[dict] = []

    try:
        index = load_index(
            persist_dir=persist_dir,
            collection_name=collection_name,
            model_name=config["embedding"]["model_name"],
        )

        if args.route:
            routed = route_chapters(
                index=index,
                query=args.query,
                top_n=args.top_chapters,
                pool=args.route_pool,
            )
            chapters = [chapter for chapter, _ in routed]
            if not chapters:
                raise ValueError(
                    "Could not identify a chapter from the initial dense search. "
                    "Try a larger --route-pool or run without --route."
                )

        retriever = build_retriever(
            index=index,
            persist_dir=persist_dir,
            collection_name=collection_name,
            k=top_k,
            mode=mode,
            candidate_k=config["retrieval"]["candidate_k"],
            chapter=chapters,
        )
        results = retrieve(args.query, retriever)
    finally:
        get_client(persist_dir).close()

    print(f"\nQuery: {args.query}")
    print(f"Mode: {mode}")
    if routed is not None:
        print("Routed chapters: " + ", ".join(f"{chapter} ({score:.3f})" for chapter, score in routed))
    else:
        print(f"Chapter filter: {chapters or 'None (global search)'}")
    print(f"Results: {len(results)}")

    for i, result in enumerate(results, start=1):
        text = result["text"] if args.full else result["text"][:500]
        print(f"\n--- Result {i} ---")
        print(f"Chapter: {result['chapter']}")
        print(f"Article: {result['article']}")
        print(f"Chapter title: {result['chapter_title']}")
        print(f"Score: {result['score']}")
        print(text)


if __name__ == "__main__":
    main()