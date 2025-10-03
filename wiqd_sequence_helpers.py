from wiqd_constants import AA


def clean_pep(p):
    return "".join(ch for ch in str(p).strip().upper() if ch in AA)


def collapse_xle(s: str) -> str:
    # Collapse I/L into J for terminal k-mer comparisons
    return s.replace("I", "J").replace("L", "J")


def collapse_seq_mode(seq: str, collapse: str) -> str:
    s = seq.upper()
    if collapse in ("xle", "xle_de_qn", "xle+deqn"):
        s = collapse_xle(s)
        if collapse in ("xle_de_qn", "xle+deqn"):
            s = (
                s.replace("D", "B")
                .replace("E", "B")
                .replace("Q", "Z")
                .replace("N", "Z")
            )
    return s
