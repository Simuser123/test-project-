from dataclasses import dataclass


@dataclass
class SourceRecord:
    name: str
    url: str
    license: str
    allowed: bool


def is_source_allowed(license_name: str) -> bool:
    """Gate sources by license/policy before ingestion."""
    allow_list = {
        "public-domain",
        "cc-by",
        "cc-by-sa",
        "permission-granted",
    }
    return license_name.strip().lower() in allow_list


def register_source(name: str, url: str, license_name: str) -> SourceRecord:
    allowed = is_source_allowed(license_name)
    return SourceRecord(name=name, url=url, license=license_name, allowed=allowed)


def main() -> None:
    # Example usage; replace with source adapters (archive, digital libraries, etc.)
    seeds = [
        ("Public Domain Corpus", "https://example.org/corpus", "public-domain"),
        ("Unknown Site", "https://example.org/unknown", "unknown"),
    ]

    records = [register_source(*seed) for seed in seeds]

    for r in records:
        status = "ALLOWED" if r.allowed else "BLOCKED"
        print(f"[{status}] {r.name} | {r.url} | license={r.license}")


if __name__ == "__main__":
    main()
