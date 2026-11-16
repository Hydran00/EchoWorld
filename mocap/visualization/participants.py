"""Doctor/subject labels for viewers; never rewrites training groups or splits."""
import json
import re
from pathlib import Path

DEFAULT_MAPPING = Path(__file__).resolve().parents[2] / 'annotations' / 'recording_participants.json'


def load_mapping(path=DEFAULT_MAPPING):
    return json.loads(Path(path).read_text())


def recording_identity(episode_id, mapping):
    """Apply explicit prefix/range rules, detecting overlaps and unresolved assignments."""
    bag = Path(episode_id).name
    token = bag.removeprefix('bag_').split('_')[0]
    match = re.fullmatch(r'([A-Za-z]+)([0-9]*)([A-Za-z]*)', token)
    if not match:
        raise ValueError(f'Cannot parse recording prefix: {bag}')
    prefix, digits, _suffix = match.groups()
    prefix = prefix.lower()
    number = int(digits) if digits else None
    matches = []
    for rule in mapping['rules']:
        if rule['prefix'].lower() != prefix:
            continue
        if 'first' in rule or 'last' in rule:
            if number is None or not rule.get('first', 0) <= number <= rule.get('last', float('inf')):
                continue
        matches.append(rule)
    if len(matches) > 1:
        raise ValueError(f'Conflicting participant rules for {bag}')
    if matches:
        doctor, subject = matches[0].get('doctor'), matches[0].get('subject')
        if not doctor or not subject:
            raise ValueError(f'Participant assignment still needs clarification: {bag}')
    else:
        if prefix in mapping.get('require_explicit_prefixes', []):
            raise ValueError(f'Missing explicit participant rule for {bag}')
        doctor = mapping['default_doctor']
        subject = prefix.capitalize()
    return dict(doctor=doctor, subject=subject, prefix=prefix, number=number, bag=bag)
