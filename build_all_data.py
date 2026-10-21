import os, re, csv, pickle
import numpy as np

HIST = 10
MSG_LEN = 40          # characters per log message (OneLog)
OUT = "revision_id/unified"
MIA = "/home/yihanliao4/projects/LogMIA"

# --- character vocabulary: 0=pad, 1=unknown, 2..128 = ord+2 ---
NUM_CHARS = 130


def encode_chars(text, msg_len=MSG_LEN):
    v = np.zeros(msg_len, dtype=np.int16)
    for i, c in enumerate(text[:msg_len]):
        o = ord(c)
        v[i] = o + 2 if o < 126 else 1
    return v


MASKERS = [
    (re.compile(r"0x[0-9a-fA-F]+"), "<*>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<*>"),
    (re.compile(r"blk_-?\d+"), "blk_<*>"),
    (re.compile(r"core\.\d+"), "core.<*>"),
    (re.compile(r"\b\d+\b"), "<*>"),
]


def mask(s):
    for p, r in MASKERS:
        s = p.sub(r, s)
    return s


class Drain:
    """Standard online Drain: length layer -> prefix-token layers -> groups."""

    def __init__(self, st=0.4, depth=4, max_children=100):
        self.st = st
        self.prefix_depth = depth - 2     # token layers below the length node
        self.max_children = max_children
        self.root = {"__len__": {}}
        self.templates = []

    @staticmethod
    def _sim(seq, tpl):
        # only called when len(seq) == len(tpl)
        same = sum(1 for a, b in zip(seq, tpl) if a == b)
        return same / len(seq)

    def _search(self, seq):
        length_node = self.root["__len__"].get(len(seq))
        if length_node is None:
            return None, None
        node = length_node
        for d in range(self.prefix_depth):
            key = seq[d] if d < len(seq) else "<END>"
            node = node["__children__"].get(key)
            if node is None:
                return None, None
        best, bestsim = None, self.st
        for cl in node["__groups__"]:
            sim = self._sim(seq, cl["tpl"])
            if sim > bestsim:
                best, bestsim = cl, sim
        return best, node

    def parse(self, toks):
        seq = list(toks)
        cl, _leaf = self._search(seq)
        if cl is not None:
            cl["tpl"] = ["<*>" if a != b else a
                         for a, b in zip(cl["tpl"], seq)]
            self.templates[cl["cid"]] = cl["tpl"]
            return cl["cid"]

        length_node = self.root["__len__"].setdefault(
            len(seq), {"__children__": {}})
        node = length_node
        for d in range(self.prefix_depth):
            key = seq[d] if d < len(seq) else "<END>"
            ch = node["__children__"]
            if key not in ch:
                ch[key] = {"__children__": {}}
            node = ch[key]
        cid = len(self.templates)
        node.setdefault("__groups__", []).append({"tpl": list(seq), "cid": cid})
        self.templates.append(list(seq))
        return cid


def samples_from_flow(flow, chars, labels, step):
    X, Xc, yn, yl, yp = [], [], [], [], []
    for i in range(0, len(flow) - HIST - 1, step):
        X.append(flow[i:i+HIST])
        Xc.append(chars[i:i+HIST])
        yn.append(flow[i+HIST])
        yl.append(int(any(labels[i:i+HIST+1])))
        yp.append(int(labels[i+HIST]))
    X, Xc = np.array(X), np.array(Xc, dtype=np.int16)
    yn, yl, yp = np.array(yn), np.array(yl), np.array(yp)
    # continuous-flow logs: each window is its own detection group
    group = np.arange(len(X))
    return X, Xc, yn, yl, yp, group, yp, None


def build_supercomputer(name, files, content_off, step, max_per_file=None, st=0.4):
    drain = Drain(st=st)
    flows, chars, labs = [], [], []
    for path in files:
        ev, cv, lb = [], [], []
        with open(path, errors="ignore") as f:
            for nline, line in enumerate(f):
                if max_per_file is not None and nline >= max_per_file:
                    break
                parts = line.strip().split()
                if len(parts) <= content_off:
                    continue
                anomal = 1 if parts[0] != "-" else 0
                content = mask(" ".join(parts[content_off:]))
                ev.append(drain.parse(content.split()))
                cv.append(encode_chars(content))
                lb.append(anomal)
        flows += ev; chars += cv; labs += lb
    n = len(drain.templates)
    X, Xc, yn, yl, yp, group, glabel, gcounts = samples_from_flow(flows, chars, labs, step)
    return n, X, Xc, yn, yl, yp, group, glabel, gcounts


def build_hdfs():
    tpath = f"{MIA}/datasets/HDFS/preprocessed/HDFS.log_templates.csv"
    templates = {}
    with open(tpath) as f:
        for row in csv.DictReader(f):
            templates[row["EventId"]] = row["EventTemplate"]

    path = f"{MIA}/datasets/HDFS/preprocessed/Event_traces.csv"
    raw, labels = [], []
    eids = set()
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            evs = re.findall(r"E\d+", row["Features"])
            if len(evs) < HIST + 1:
                continue
            eids.update(evs)
            raw.append(evs); labels.append(1 if row["Label"] == "Fail" else 0)
    eids = sorted(eids)
    mp = {e: i for i, e in enumerate(eids)}

    X, Xc, yn, yl, yp, group = [], [], [], [], [], []
    glabel, gcounts, gid = [], [], 0
    for evs, l in zip(raw, labels):
        ids = [mp[e] for e in evs]
        cvecs = [encode_chars(templates.get(e, "")) for e in evs]
        added = 0
        for i in range(0, len(ids) - HIST, 2):
            X.append(ids[i:i+HIST])
            Xc.append(cvecs[i:i+HIST])
            yn.append(ids[i+HIST]); yl.append(l); yp.append(l)
            group.append(gid); added += 1
        if added:
            cnt = np.zeros(len(eids), dtype=np.float32)
            for x in ids:
                cnt[x] += 1
            glabel.append(l); gcounts.append(cnt); gid += 1
    return (len(eids), np.array(X), np.array(Xc, dtype=np.int16),
            np.array(yn), np.array(yl), np.array(yp),
            np.array(group), np.array(glabel), np.asarray(gcounts))


def save(name, n, X, Xc, yn, yl, yp, group, glabel, gcounts=None):
    pickle.dump({"num_event_ids": n, "X": X, "Xchar": Xc,
                 "ynext": yn, "ylab": yl, "ypoint": yp,
                 "group": group, "group_label": np.asarray(glabel),
                 "group_counts": (None if gcounts is None else np.asarray(gcounts)),
                 "num_chars": NUM_CHARS},
                open(f"{OUT}/{name}.pkl", "wb"))
    print(f"{name:11s}: events={n} samples={len(X)} groups={len(glabel)} "
          f"anomaly={yl.sum()} ({100*yl.mean():.2f}%)")


def main():
    os.makedirs(OUT, exist_ok=True)

    bgl_files = [
        f"{MIA}/datasets/BGL/shadow/member.log",
        f"{MIA}/datasets/BGL/shadow/non_member.log",
        f"{MIA}/datasets/BGL/target/member.log",
        f"{MIA}/datasets/BGL/target/non_member.log",
    ]
    save("BGL", *build_supercomputer("BGL", bgl_files, 6, 2, st=0.7))

    tb_files = [
        f"{MIA}/datasets/Thunderbird/shadow/member.log",
        f"{MIA}/datasets/Thunderbird/shadow/non_member.log",
        f"{MIA}/datasets/Thunderbird/target/member.log",
        f"{MIA}/datasets/Thunderbird/target/non_member.log",
    ]
    save("Thunderbird", *build_supercomputer("Thunderbird", tb_files, 8, 1, max_per_file=400000))

    save("HDFS", *build_hdfs())


if __name__ == "__main__":
    main()
