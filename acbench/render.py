# Antechamber Benchmark — Copyright (c) 2026 House of Solance. PolyForm Noncommercial 1.0.0. Required Notice: Copyright (c) 2026 House of Solance (https://github.com/SolanceLab)
"""The one plain rendering of a recall payload given to the answering model.

Each winner is its date, then one line per message: 'sender: message'.
A winner with no lines (a note, a memory without source lines) is one line, 'title: text'.
A winner with lines whose text is not one of them (a memory with source lines) shows that text first.
"""


def _same_line(text, ln):
    """True when `text` is one of the winner's own lines at another cap: one is a prefix of the other."""
    if not isinstance(ln, dict):
        return False
    a, b = text.strip(), (ln.get("message") or "").strip()
    if not b:
        return False
    a, b = a.rstrip(".… "), b.rstrip(".… ")
    return a.startswith(b) or b.startswith(a)


def render(payload):
    winners = []
    if isinstance(payload, dict):
        winners = payload.get("winners") or []
    blocks = []
    for w in winners:
        if not isinstance(w, dict):
            continue
        rows = []
        when = w.get("date")
        if isinstance(when, str) and when:
            rows.append(when)
        lines = w.get("lines") or []
        if lines:
            # A memory winner carries its source lines AND its own text (topic: summary, excerpt). The text is
            # shown above the lines unless it is already one of them (a chat winner's text is its hit line, capped
            # at a different length, so chat rendering is unchanged). A memory's own text must not be dropped.
            text = w.get("text")
            if isinstance(text, str) and text.strip() and not any(_same_line(text, ln) for ln in lines):
                rows.append(text.strip())
            for ln in lines:
                if not isinstance(ln, dict):
                    continue
                rows.append("%s: %s" % (ln.get("sender") or "", ln.get("message") or ""))
        else:
            who = ""
            for key in ("title", "source", "topic"):
                if isinstance(w.get(key), str) and w.get(key):
                    who = w[key]
                    break
            msg = ""
            for key in ("text", "snippet", "summary", "excerpt"):
                if isinstance(w.get(key), str) and w.get(key):
                    msg = w[key]
                    break
            if who or msg:
                rows.append("%s: %s" % (who, msg))
        if rows:
            blocks.append("\n".join(rows))
    return "\n\n".join(blocks)
