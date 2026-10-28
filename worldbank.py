"""World Bank Open Data lookups that ground the debate in real figures."""
import datetime
import difflib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

API = "https://api.worldbank.org/v2"

INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "GDP growth (annual %)",
    "FP.CPI.TOTL.ZG": "Inflation, consumer prices (annual %)",
    "SL.UEM.TOTL.ZS": "Unemployment (% of labor force, ILO modeled)",
    "NY.GDP.PCAP.CD": "GDP per capita (current US$)",
    "GC.DOD.TOTL.GD.ZS": "Central government debt (% of GDP)",
    "BN.CAB.XOKA.GD.ZS": "Current account balance (% of GDP)",
    "SH.XPD.CHEX.GD.ZS": "Health expenditure (% of GDP)",
    "SE.XPD.TOTL.GD.ZS": "Education expenditure (% of GDP)",
    "SP.DYN.LE00.IN": "Life expectancy at birth (years)",
    "SI.POV.GINI": "Gini index (inequality)",
    "SP.POP.TOTL": "Population",
}

# Common names that differ from the World Bank's official ones.
ALIASES = {
    "russia": "russian federation", "turkey": "turkiye", "south korea": "korea, rep.",
    "north korea": "korea, dem. people's rep.", "syria": "syrian arab republic",
    "vietnam": "viet nam", "egypt": "egypt, arab rep.", "iran": "iran, islamic rep.",
    "venezuela": "venezuela, rb", "uk": "united kingdom", "britain": "united kingdom",
    "usa": "united states", "us": "united states", "america": "united states",
    "uae": "united arab emirates", "yemen": "yemen, rep.", "laos": "lao pdr",
    "slovakia": "slovak republic", "kyrgyzstan": "kyrgyz republic",
    "gambia": "gambia, the", "bahamas": "bahamas, the", "congo": "congo, dem. rep.",
}

YEARS_BACK = 10      # older figures are ignored
POINTS_PER_SERIES = 5

_countries = None


@dataclass
class Stats:
    country: str                                  # official World Bank name
    series: dict = field(default_factory=dict)    # label -> [(year, value)], oldest first
    missing: list = field(default_factory=list)   # labels with no usable data

    def rows(self):
        return [(label, describe(points)) for label, points in self.series.items()]

    def as_prompt(self):
        return "\n".join(f"- {label}: {text}" for label, text in self.rows())


def fmt(value):
    return f"{value:,.2f}" if abs(value) < 1000 else f"{value:,.0f}"


def describe(points):
    """'2022: 9.28 → 2023: 4.59 (2022→2023: fell by 4.69)' for one series."""
    text = " → ".join(f"{year}: {fmt(value)}" for year, value in points)
    if len(points) > 1:
        change = points[-1][1] - points[0][1]
        span = f"{points[0][0]}→{points[-1][0]}"
        if abs(change) < 0.005:
            text += f" ({span}: unchanged)"
        else:
            text += f" ({span}: {'rose' if change > 0 else 'fell'} by {fmt(abs(change))})"
    return text


def _get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "gov-vs-opposition/1.0"})
    with urllib.request.urlopen(request, timeout=12) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_countries():
    global _countries
    if _countries is None:
        rows = _get(f"{API}/country?format=json&per_page=400")[1]
        _countries = [{"id": c["id"], "name": c["name"]} for c in rows
                      if c.get("region", {}).get("value") != "Aggregates"]
    return _countries


def find_country(text):
    """Match user input such as 'egypt' or 'UK' to a World Bank country."""
    query = ALIASES.get(text.strip().lower(), text.strip().lower())
    countries = _load_countries()
    by_name = {c["name"].lower(): c for c in countries}
    if query in by_name:
        return by_name[query]
    for name, country in by_name.items():
        if name.startswith(query):
            return country
    short = {name.split(",")[0]: country for name, country in by_name.items()}
    close = difflib.get_close_matches(query, list(short), n=1, cutoff=0.8)
    return short[close[0]] if close else None


def _fetch_series(code, iso):
    """Recent (year, value) points, oldest first. Returns None if the download failed."""
    oldest = datetime.date.today().year - YEARS_BACK
    for attempt in range(3):
        try:
            rows = _get(f"{API}/country/{iso}/indicator/{code}?format=json&mrv=10&per_page=20")[1] or []
            points = [(int(r["date"]), r["value"]) for r in rows
                      if r["value"] is not None and int(r["date"]) >= oldest]
            return sorted(points)[-POINTS_PER_SERIES:]
        except Exception:
            time.sleep(1 + attempt)
    return None


def fetch(country_name):
    """Download the indicators for a country. Returns Stats, or None if it can't be found."""
    try:
        country = find_country(country_name)
    except Exception:
        return None
    if not country:
        return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda code: _fetch_series(code, country["id"]), INDICATORS))
    stats = Stats(country["name"])
    for (code, label), points in zip(INDICATORS.items(), results):
        if points:
            stats.series[label] = points
        else:
            stats.missing.append(label + (" (download failed)" if points is None else " (no recent data)"))
    return stats
