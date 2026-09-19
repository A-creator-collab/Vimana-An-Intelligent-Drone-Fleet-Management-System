"""nlp_command_parser.py - Rule/regex natural-language command parser (no LLM, no API).

'send 2 drones to sector B and return when battery hits 20%'
   -> Command(action='dispatch', params={'count': 2, 'sector': 'B', 'return_battery': 20})
The parser only PROPOSES a structured command; SafetyValidator approves it.
"""
import re
from dataclasses import dataclass, field
from typing import Dict

WORDNUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
           "seven": 7, "eight": 8, "nine": 9, "ten": 10}


class CommandParseError(ValueError):
    pass


@dataclass
class Command:
    action: str
    params: Dict = field(default_factory=dict)
    raw: str = ""


def _to_int(tok: str, default: int = 1) -> int:
    tok = tok.strip().lower()
    return int(tok) if tok.isdigit() else WORDNUM.get(tok, default)


def _loc(tok: str):
    """'12' -> ('node', 12); 'b' -> ('sector', 'B')."""
    return ("node", int(tok)) if tok.isdigit() else ("sector", tok.upper())


class NLPCommandParser:
    def parse(self, text: str) -> Command:
        s = text.lower().strip()
        for rule in (self._recall, self._status, self._nofly, self._deliver, self._dispatch):
            cmd = rule(s)
            if cmd:
                cmd.raw = text
                return cmd
        raise CommandParseError(f"could not understand: {text!r}")

    @staticmethod
    def _recall(s):
        m = re.search(r"\brecall\b\s*(?:all|everyone|drone\s*#?\s*(\d+))", s)
        return Command("recall", {"drone_id": int(m.group(1)) if m.group(1) else None}) if m else None

    @staticmethod
    def _status(s):
        m = re.search(r"\b(?:status|report|check)\b.*?drone\s*#?\s*(\d+)", s)
        return Command("status", {"drone_id": int(m.group(1))}) if m else None

    @staticmethod
    def _nofly(s):
        m = re.search(r"\b(?:no[\s-]?fly|block|close)\b.*?sector\s+([a-d])\b", s)
        if not m:
            return None
        d = re.search(r"for\s+(\d+)\s*(?:ticks?|steps?|min(?:ute)?s?)", s)
        return Command("nofly", {"sector": m.group(1).upper(), "duration": int(d.group(1)) if d else 20})

    @staticmethod
    def _deliver(s):
        m = re.search(r"\b(?:deliver|carry|transport)\b.*?from\s+(?:node\s+|sector\s+)?(\w+)\s+to\s+(?:node\s+|sector\s+)?(\w+)", s)
        if not m:
            return None
        pr = re.search(r"priority\s*(?:level\s*)?(\d)", s)
        wt = re.search(r"(\d+(?:\.\d+)?)\s*kg", s)
        return Command("deliver", {"from": _loc(m.group(1)), "to": _loc(m.group(2)),
                                   "priority": int(pr.group(1)) if pr else 3,
                                   "weight": float(wt.group(1)) if wt else 1.0})

    @staticmethod
    def _dispatch(s):
        if not re.search(r"\b(?:send|dispatch|deploy|inspect|survey|patrol)\b", s):
            return None
        sec = re.search(r"sector\s+([a-d])\b", s)
        if not sec:
            return None
        cnt = re.search(r"(\w+)\s+drones?\b", s)
        bat = (re.search(r"battery\s+(?:hits|reaches|drops?\s+to|falls?\s+to|is|at|below)\s*(?:at\s+)?(\d+)", s)
               or re.search(r"(\d+)\s*%\s*battery", s))
        p = {"count": _to_int(cnt.group(1)) if cnt else 1, "sector": sec.group(1).upper()}
        if bat:
            p["return_battery"] = int(bat.group(1))
        return Command("dispatch", p)
