#!/usr/bin/env python3
"""Minimal eBPF sampling profiler (no bcc/bpftrace needed).

Loads a BPF_PROG_TYPE_PERF_EVENT program that, on every cpu-clock sample,
stores the user-space stack (frame-pointer walk) in a BPF_MAP_TYPE_STACK_TRACE
map and counts it in a hash map. One perf event per thread of the target
process. Afterwards the counts are symbolized against the binary's symbol
table and printed as a flat profile (self time) plus a few hot stacks.

usage: bpfprof.py <pid> <seconds> [--binary PATH] [--hz 997]
"""
import ctypes, os, struct, sys, subprocess, time, argparse, collections, fcntl

libc = ctypes.CDLL(None, use_errno=True)
SYS_bpf = 321
SYS_perf_event_open = 298

BPF_MAP_CREATE, BPF_MAP_LOOKUP_ELEM, BPF_MAP_GET_NEXT_KEY, BPF_PROG_LOAD = 0, 1, 4, 5
BPF_MAP_TYPE_HASH, BPF_MAP_TYPE_STACK_TRACE = 1, 7
BPF_PROG_TYPE_PERF_EVENT = 7
PERF_EVENT_IOC_ENABLE = 0x2400
PERF_EVENT_IOC_SET_BPF = 0x40042408
MAX_DEPTH = 127


def bpf(cmd, attr):
    r = libc.syscall(SYS_bpf, cmd, ctypes.byref(attr), ctypes.sizeof(attr))
    if r < 0:
        raise OSError(ctypes.get_errno(), f"bpf({cmd}): {os.strerror(ctypes.get_errno())}")
    return r


class MapAttr(ctypes.Structure):
    _fields_ = [("map_type", ctypes.c_uint), ("key_size", ctypes.c_uint), ("value_size", ctypes.c_uint),
                ("max_entries", ctypes.c_uint), ("map_flags", ctypes.c_uint)]


class ElemAttr(ctypes.Structure):
    _fields_ = [("map_fd", ctypes.c_uint), ("pad", ctypes.c_uint), ("key", ctypes.c_ulonglong),
                ("value", ctypes.c_ulonglong), ("flags", ctypes.c_ulonglong)]


class ProgAttr(ctypes.Structure):
    _fields_ = [("prog_type", ctypes.c_uint), ("insn_cnt", ctypes.c_uint), ("insns", ctypes.c_ulonglong),
                ("license", ctypes.c_ulonglong), ("log_level", ctypes.c_uint), ("log_size", ctypes.c_uint),
                ("log_buf", ctypes.c_ulonglong), ("kern_version", ctypes.c_uint), ("prog_flags", ctypes.c_uint),
                ("rest", ctypes.c_ubyte * 112)]


def map_create(t, ks, vs, n):
    return bpf(BPF_MAP_CREATE, MapAttr(t, ks, vs, n, 0))


# ---- tiny BPF assembler
def insn(code, dst=0, src=0, off=0, imm=0):
    return struct.pack("<BBhi", code, (src << 4) | dst, off, imm)


def ld_map_fd(dst, fd):  # BPF_LD | BPF_DW | BPF_IMM with src = BPF_PSEUDO_MAP_FD(1)
    return insn(0x18, dst, 1, 0, fd) + insn(0, 0, 0, 0, 0)


def prog_bytes(stack_fd, count_fd):
    p = b""
    p += insn(0xbf, 6, 1)                    # r6 = r1 (ctx)
    p += ld_map_fd(2, stack_fd)              # r2 = &stack_map
    p += insn(0xb7, 3, 0, 0, 256)            # r3 = BPF_F_USER_STACK
    p += insn(0x85, 0, 0, 0, 27)             # call bpf_get_stackid
    p += insn(0xc5, 0, 0, 20, 0)             # if r0 s< 0 goto exit (+20)
    p += insn(0x63, 10, 0, -4, 0)            # *(u32 *)(r10-4) = r0
    p += insn(0xbf, 2, 10)                   # r2 = r10
    p += insn(0x07, 2, 0, 0, -4)             # r2 += -4
    p += ld_map_fd(1, count_fd)              # r1 = &count_map
    p += insn(0x85, 0, 0, 0, 1)              # call map_lookup_elem
    p += insn(0x15, 0, 0, 3, 0)              # if r0 == 0 goto insert (+3)
    p += insn(0xb7, 1, 0, 0, 1)              # r1 = 1
    p += insn(0xdb, 0, 1, 0, 0)              # lock *(u64 *)(r0) += r1   (atomic add)
    p += insn(0x05, 0, 0, 10, 0)             # goto exit (+10)
    # insert:
    p += insn(0xb7, 1, 0, 0, 1)              # r1 = 1
    p += insn(0x7b, 10, 1, -16, 0)           # *(u64 *)(r10-16) = r1
    p += ld_map_fd(1, count_fd)              # r1 = &count_map
    p += insn(0xbf, 2, 10); p += insn(0x07, 2, 0, 0, -4)    # r2 = r10 - 4
    p += insn(0xbf, 3, 10); p += insn(0x07, 3, 0, 0, -16)   # r3 = r10 - 16
    p += insn(0xb7, 4, 0, 0, 0)              # r4 = BPF_ANY
    p += insn(0x85, 0, 0, 0, 2)              # call map_update_elem
    # exit:
    p += insn(0xb7, 0, 0, 0, 0)              # r0 = 0
    p += insn(0x95)                          # exit
    return p


def prog_load(stack_fd, count_fd):
    code = prog_bytes(stack_fd, count_fd)
    buf = ctypes.create_string_buffer(code)
    lic = ctypes.create_string_buffer(b"GPL")
    log = ctypes.create_string_buffer(1 << 16)
    a = ProgAttr()
    a.prog_type = BPF_PROG_TYPE_PERF_EVENT
    a.insn_cnt = len(code) // 8
    a.insns = ctypes.addressof(buf)
    a.license = ctypes.addressof(lic)
    a.log_level = 1
    a.log_size = len(log)
    a.log_buf = ctypes.addressof(log)
    r = libc.syscall(SYS_bpf, BPF_PROG_LOAD, ctypes.byref(a), ctypes.sizeof(a))
    if r < 0:
        sys.stderr.write(log.value.decode(errors="replace"))
        raise OSError(ctypes.get_errno(), "BPF_PROG_LOAD failed: " + os.strerror(ctypes.get_errno()))
    return r


class PerfAttr(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint), ("size", ctypes.c_uint), ("config", ctypes.c_ulonglong),
                ("sample_period", ctypes.c_ulonglong), ("sample_type", ctypes.c_ulonglong),
                ("read_format", ctypes.c_ulonglong), ("flags", ctypes.c_ulonglong),
                ("wakeup_events", ctypes.c_uint), ("bp_type", ctypes.c_uint), ("config1", ctypes.c_ulonglong),
                ("config2", ctypes.c_ulonglong), ("branch_sample_type", ctypes.c_ulonglong),
                ("sample_regs_user", ctypes.c_ulonglong), ("sample_stack_user", ctypes.c_uint),
                ("clockid", ctypes.c_int), ("sample_regs_intr", ctypes.c_ulonglong),
                ("aux_watermark", ctypes.c_uint), ("sample_max_stack", ctypes.c_ushort), ("pad", ctypes.c_ushort),
                ("rest", ctypes.c_ubyte * 32)]


def perf_open(tid, hz):
    a = PerfAttr()
    a.type = 1          # PERF_TYPE_SOFTWARE
    a.size = ctypes.sizeof(PerfAttr)
    a.config = 0        # PERF_COUNT_SW_CPU_CLOCK
    a.sample_period = int(1e9 / hz)
    a.flags = 1 | (1 << 5)  # disabled, exclude_kernel
    fd = libc.syscall(SYS_perf_event_open, ctypes.byref(a), tid, -1, -1, 8)  # PERF_FLAG_FD_CLOEXEC
    if fd < 0:
        raise OSError(ctypes.get_errno(), f"perf_event_open(tid={tid}): {os.strerror(ctypes.get_errno())}")
    return fd


def map_iter(fd, ksize):
    key = ctypes.create_string_buffer(ksize)
    nxt = ctypes.create_string_buffer(ksize)
    a = ElemAttr(fd, 0, 0, ctypes.addressof(nxt), 0)
    r = libc.syscall(SYS_bpf, BPF_MAP_GET_NEXT_KEY, ctypes.byref(a), ctypes.sizeof(a))
    while r == 0:
        yield nxt.raw[:ksize]
        key.raw = nxt.raw
        a = ElemAttr(fd, 0, ctypes.addressof(key), ctypes.addressof(nxt), 0)
        r = libc.syscall(SYS_bpf, BPF_MAP_GET_NEXT_KEY, ctypes.byref(a), ctypes.sizeof(a))


def map_lookup(fd, key, vsize):
    k = ctypes.create_string_buffer(key)
    v = ctypes.create_string_buffer(vsize)
    a = ElemAttr(fd, 0, ctypes.addressof(k), ctypes.addressof(v), 0)
    r = libc.syscall(SYS_bpf, BPF_MAP_LOOKUP_ELEM, ctypes.byref(a), ctypes.sizeof(a))
    return v.raw if r == 0 else None


def load_symbols(binary, base):
    out = subprocess.run(["nm", "-C", "--defined-only", "-n", binary], capture_output=True, text=True).stdout
    syms = []
    for line in out.splitlines():
        parts = line.split(" ", 2)
        if len(parts) == 3 and parts[1] in "tTwW":
            syms.append((int(parts[0], 16) + base, parts[2]))
    syms.sort()
    return syms


def symbolize(syms, addr):
    import bisect
    i = bisect.bisect_right(syms, (addr, "\xff")) - 1
    if i < 0:
        return "?"
    name = syms[i][1]
    # strip hash suffix of Rust mangled names
    if "::h" in name and len(name) - name.rfind("::h") == 20:
        name = name[: name.rfind("::h")]
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pid", type=int)
    ap.add_argument("seconds", type=float)
    ap.add_argument("--binary")
    ap.add_argument("--hz", type=int, default=997)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    pid = args.pid
    binary = args.binary or os.readlink(f"/proc/{pid}/exe")
    # load base of the main binary (PIE)
    base = None
    for line in open(f"/proc/{pid}/maps"):
        parts = line.split()
        if len(parts) >= 6 and parts[5] == binary:
            lo = int(parts[0].split("-")[0], 16)
            off = int(parts[2], 16)
            base = lo - off
            break
    if base is None:
        sys.exit("could not find binary mapping")

    stack_fd = map_create(BPF_MAP_TYPE_STACK_TRACE, 4, MAX_DEPTH * 8, 20000)
    count_fd = map_create(BPF_MAP_TYPE_HASH, 4, 8, 20000)
    prog_fd = prog_load(stack_fd, count_fd)
    tids = [int(t) for t in os.listdir(f"/proc/{pid}/task")]
    fds = []
    for tid in tids:
        fd = perf_open(tid, args.hz)
        fcntl.ioctl(fd, PERF_EVENT_IOC_SET_BPF, prog_fd)
        fcntl.ioctl(fd, PERF_EVENT_IOC_ENABLE, 0)
        fds.append(fd)
    print(f"profiling pid {pid} ({len(tids)} threads) for {args.seconds}s at {args.hz} Hz ...", file=sys.stderr)
    time.sleep(args.seconds)
    for fd in fds:
        os.close(fd)

    syms = load_symbols(binary, base)
    self_counts = collections.Counter()
    incl_counts = collections.Counter()
    stacks = []
    total = 0
    for key in map_iter(count_fd, 4):
        cnt = struct.unpack("<Q", map_lookup(count_fd, key, 8))[0]
        raw = map_lookup(stack_fd, key, MAX_DEPTH * 8)
        if raw is None:
            continue
        ips = [a for a in struct.unpack(f"<{MAX_DEPTH}Q", raw) if a]
        names = [symbolize(syms, a) for a in ips]
        total += cnt
        if names:
            self_counts[names[0]] += cnt
            for n in set(names):
                incl_counts[n] += cnt
        stacks.append((cnt, names))
    if total == 0:
        sys.exit("no samples (is the process still running? frame pointers enabled?)")
    print(f"\n{total} samples ({total / args.hz:.1f} thread-seconds)\n")
    print("SELF time (where the CPU actually is):")
    for name, c in self_counts.most_common(args.top):
        print(f"  {100.0 * c / total:5.1f}%  {name}")
    print("\nINCLUSIVE time (function or anything it calls):")
    for name, c in incl_counts.most_common(args.top):
        print(f"  {100.0 * c / total:5.1f}%  {name}")
    print("\nhottest stacks:")
    stacks.sort(reverse=True)
    for cnt, names in stacks[:8]:
        print(f"  {100.0 * cnt / total:5.1f}%  " + " <- ".join(names[:7]))


if __name__ == "__main__":
    main()
