"""
Text normalization and abbreviation dictionary for business entity resolution.

Design choices:
- Keep BOTH raw and normalized fields — raw for typo-distance features,
  normalized for exact-match features.
- Legal suffixes include French forms (SARL, SAS, SA) proactively, since France
  appears only at test time. These are static linguistic knowledge, not registry data.
- Country is kept as a soft feature, never used as a hard filter.
"""

import re
from typing import Dict, Tuple
from unidecode import unidecode


# ── Legal suffix / abbreviation dictionary ──────────────────────────────────
# Bidirectional: we normalize all forms to the canonical (first) form.
LEGAL_SUFFIX_GROUPS = [
    # English
    ['corporation', 'corp', 'corp.'],
    ['incorporated', 'inc', 'inc.'],
    ['limited', 'ltd', 'ltd.', 'ltda'],
    ['company', 'co', 'co.'],
    ['private limited', 'pvt ltd', 'pvt. ltd.', 'pvt. ltd', 'pvt ltd.', 'private ltd'],
    ['public limited company', 'plc', 'plc.'],
    ['limited liability company', 'llc', 'llc.', 'l.l.c.', 'l.l.c'],
    ['limited liability partnership', 'llp', 'llp.', 'l.l.p.'],
    ['limited partnership', 'lp', 'lp.', 'l.p.'],
    ['general partnership', 'gp', 'gp.'],
    ['international', 'intl', "int'l", 'intl.'],
    ['enterprises', 'enterprise', 'ent', 'ent.'],
    ['industries', 'ind', 'ind.'],
    ['services', 'svc', 'svcs', 'svc.'],
    ['solutions', 'soln', 'solns'],
    ['technologies', 'tech', 'techs'],
    ['associates', 'assoc', 'assoc.'],
    ['brothers', 'bros', 'bros.'],
    ['manufacturing', 'mfg', 'mfg.'],
    ['holdings', 'hldg', 'hldgs'],
    ['group', 'grp', 'grp.'],
    ['and', '&'],

    # Indian
    ['nagar', 'ngr'],

    # French (proactive — France is unseen in training)
    ['société à responsabilité limitée', 'sarl', 's.a.r.l.', 's.a.r.l'],
    ['société par actions simplifiée', 'sas', 's.a.s.', 's.a.s'],
    ['société anonyme', 'sa', 's.a.', 's.a'],
    ['société en nom collectif', 'snc', 's.n.c.'],
    ['société civile immobilière', 'sci', 's.c.i.'],
    ['groupement dintérêt économique', 'gie', 'g.i.e.'],
    ['établissement', 'ets', 'ets.'],
    ['rue', 'r.'],
    ['avenue', 'ave', 'av', 'ave.', 'av.'],
    ['boulevard', 'blvd', 'blvd.'],

    # German (if encountered)
    ['gesellschaft mit beschränkter haftung', 'gmbh', 'g.m.b.h.'],
    ['aktiengesellschaft', 'ag', 'a.g.'],
]

# Build lookup: every variant → canonical form
ABBREVIATION_MAP: Dict[str, str] = {}
for group in LEGAL_SUFFIX_GROUPS:
    canonical = group[0]
    for variant in group:
        ABBREVIATION_MAP[variant.lower()] = canonical.lower()


# ── Address abbreviation dictionary ─────────────────────────────────────────
ADDRESS_ABBREV = {
    'rd': 'road', 'rd.': 'road',
    'st': 'street', 'st.': 'street',
    'ave': 'avenue', 'ave.': 'avenue', 'av': 'avenue', 'av.': 'avenue',
    'blvd': 'boulevard', 'blvd.': 'boulevard',
    'dr': 'drive', 'dr.': 'drive',
    'ln': 'lane', 'ln.': 'lane',
    'ct': 'court', 'ct.': 'court',
    'pl': 'place', 'pl.': 'place',
    'sq': 'square', 'sq.': 'square',
    'hwy': 'highway', 'hwy.': 'highway',
    'pkwy': 'parkway', 'pkwy.': 'parkway',
    'cir': 'circle', 'cir.': 'circle',
    'apt': 'apartment', 'apt.': 'apartment',
    'ste': 'suite', 'ste.': 'suite',
    'fl': 'floor', 'fl.': 'floor',
    'bldg': 'building', 'bldg.': 'building',
    'no': 'number', 'no.': 'number',
    'dist': 'district', 'dist.': 'district',
    'nr': 'near', 'nr.': 'near',
    'opp': 'opposite', 'opp.': 'opposite',
    'mkt': 'market', 'mkt.': 'market',
    'nagar': 'nagar',
    'ngr': 'nagar',
    'ext': 'extension', 'ext.': 'extension', 'extn': 'extension', 'extn.': 'extension',
}


# Landmark indicator phrases (to be split into a separate field)
LANDMARK_PATTERNS = [
    r'\bnear\s+',
    r'\bnr\.?\s+',
    r'\bopp\.?\s+to\s+',
    r'\bopp\.?\s+',
    r'\bbehind\s+',
    r'\bnext\s+to\s+',
    r'\bbeside\s+',
    r'\babove\s+',
    r'\bbelow\s+',
    r'\bin\s+front\s+of\s+',
]
LANDMARK_REGEX = re.compile('|'.join(LANDMARK_PATTERNS), re.IGNORECASE)


def normalize_text_basic(text: str) -> str:
    """Basic text normalization: lowercase, transliterate, collapse whitespace."""
    if not text:
        return ''
    text = text.lower().strip()
    text = unidecode(text)  # Transliterate accented chars → ASCII
    text = re.sub(r'[^\w\s&]', ' ', text)  # Keep & for "and" normalization
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_abbreviations(text: str, abbrev_map: Dict[str, str]) -> str:
    """
    Replace known abbreviations with canonical forms.

    Works on word/token boundaries to avoid partial matches.
    Handles multi-word abbreviations by trying longest match first.
    """
    if not text:
        return ''

    # Sort by length descending so multi-word patterns match first
    sorted_abbrevs = sorted(abbrev_map.keys(), key=len, reverse=True)

    for abbrev in sorted_abbrevs:
        # Build regex pattern with word boundaries
        pattern = r'\b' + re.escape(abbrev) + r'\b'
        canonical = abbrev_map[abbrev]
        text = re.sub(pattern, canonical, text, flags=re.IGNORECASE)

    return re.sub(r'\s+', ' ', text).strip()


def extract_landmark(address: str) -> Tuple[str, str]:
    """
    Split an address into (address_without_landmark, landmark_text).

    Landmark phrases like "Near X", "Opp. Y" are extracted into a separate field
    so they don't pollute address matching but can still be used as features.
    """
    if not address:
        return '', ''

    match = LANDMARK_REGEX.search(address)
    if match:
        landmark = address[match.start():].strip()
        address_clean = address[:match.start()].strip()
        return address_clean, landmark

    return address, ''


def normalize_business_name(name: str) -> str:
    """
    Full normalization pipeline for business names.

    Steps:
    1. Basic normalization (lowercase, transliterate, collapse whitespace)
    2. Abbreviation expansion (Corp → Corporation, etc.)
    """
    name = normalize_text_basic(name)
    name = normalize_abbreviations(name, ABBREVIATION_MAP)
    return name


def normalize_address(address: str) -> Tuple[str, str]:
    """
    Full normalization pipeline for addresses.

    Returns (normalized_address, landmark_text).
    """
    address = normalize_text_basic(address)
    address, landmark = extract_landmark(address)
    address = normalize_abbreviations(address, ADDRESS_ABBREV)
    return address, landmark


def extract_address_tokens(normalized_address: str) -> Dict[str, str]:
    """
    Extract structured tokens from a normalized address.

    Returns a dict with keys like 'numbers', 'words', 'postal_codes'.
    """
    tokens = normalized_address.split()

    # Separate numeric tokens (street numbers, PIN codes) from word tokens
    numbers = [t for t in tokens if re.match(r'^\d+$', t)]
    words = [t for t in tokens if not re.match(r'^\d+$', t)]

    # Postal code heuristic: 5-6 digit numbers
    postal_candidates = [n for n in numbers if 5 <= len(n) <= 6]

    return {
        'all_tokens': tokens,
        'numbers': numbers,
        'words': words,
        'postal_codes': postal_candidates,
    }


def normalize_country(country: str) -> str:
    """Normalize country field — lowercase and strip, nothing more."""
    if not country:
        return ''
    return country.lower().strip()


def normalize_record(record: dict) -> dict:
    """
    Normalize a single record, producing both raw and normalized fields.

    Input keys: entity_id, business_name, business_address, country
    Output adds: name_normalized, address_normalized, landmark, country_normalized
    """
    result = dict(record)  # Keep all raw fields

    result['name_normalized'] = normalize_business_name(record.get('business_name', ''))
    addr_norm, landmark = normalize_address(record.get('business_address', ''))
    result['address_normalized'] = addr_norm
    result['landmark'] = landmark
    result['country_normalized'] = normalize_country(record.get('country', ''))

    return result


def normalize_dataframe(df) -> 'pd.DataFrame':
    """
    Normalize all records in a DataFrame.

    Adds normalized columns in-place and returns the DataFrame.
    """
    import pandas as pd

    df = df.copy()
    df['name_normalized'] = df['business_name'].apply(normalize_business_name)

    addr_results = df['business_address'].apply(normalize_address)
    df['address_normalized'] = addr_results.apply(lambda x: x[0])
    df['landmark'] = addr_results.apply(lambda x: x[1])

    df['country_normalized'] = df['country'].apply(normalize_country)

    return df
