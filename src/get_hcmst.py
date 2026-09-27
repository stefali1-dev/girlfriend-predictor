"""Download HCMST 2017-2022 (Stanford): data file, user's guide, codebook."""
import urllib.request
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "hcmst"
BASE_URL = "https://stacks.stanford.edu/file/druid:hg921sg6829/"

FILES = {
    "HCMST_2017_to_2022_small_public_version_2.2.dta": BASE_URL
    + "HCMST%202017%20to%202022%20small%20public%20version%202.2.dta",
    "HCMST_2017-2022_users_guide_v2.3.pdf": "https://stacks.stanford.edu/file/druid:tq903pj6286/"
    + "HCMST%202017-%202022%20user%27s%20guide%20v2.3.pdf",
    "HCMST_2017_to_2022_v2.2_codebook.pdf": BASE_URL + "HCMST%202017%20to%202022%20v2.2%20codebook.pdf",
}


def download(name: str, url: str) -> None:
    dest = RAW_DIR / name
    if dest.exists():
        print(f"skip {dest.name} (already downloaded)")
        return
    print(f"downloading {url}")
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        download(name, url)
