//! Minimal ELF32 (RISC-V, little-endian) loader: PT_LOAD segments + entry.

use crate::Image;
use anyhow::{bail, ensure, Result};

fn u16le(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn u32le(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

pub fn load_elf(bytes: &[u8]) -> Result<Image> {
    ensure!(bytes.len() >= 52, "ELF too short");
    ensure!(&bytes[0..4] == b"\x7fELF", "not an ELF file");
    ensure!(bytes[4] == 1, "not ELF32");
    ensure!(bytes[5] == 1, "not little-endian");
    let e_type = u16le(bytes, 16);
    let e_machine = u16le(bytes, 18);
    ensure!(e_machine == 0xf3, "not a RISC-V ELF (machine={e_machine:#x})");
    ensure!(e_type == 2, "not an executable ELF (type={e_type})");
    let entry = u32le(bytes, 24);
    let phoff = u32le(bytes, 28) as usize;
    let phentsize = u16le(bytes, 42) as usize;
    let phnum = u16le(bytes, 44) as usize;
    ensure!(phentsize >= 32, "bad phentsize");
    let mut segments = Vec::new();
    for i in 0..phnum {
        let o = phoff + i * phentsize;
        ensure!(o + 32 <= bytes.len(), "program header out of range");
        let p_type = u32le(bytes, o);
        if p_type != 1 {
            continue; // PT_LOAD only
        }
        let p_offset = u32le(bytes, o + 4) as usize;
        let p_vaddr = u32le(bytes, o + 8);
        let p_filesz = u32le(bytes, o + 16) as usize;
        let p_memsz = u32le(bytes, o + 20) as usize;
        ensure!(p_filesz <= p_memsz, "filesz > memsz");
        ensure!(p_offset + p_filesz <= bytes.len(), "segment data out of range");
        if p_filesz == 0 {
            continue;
        }
        // Word-align the segment start by padding from an aligned base.
        let base = p_vaddr & !3;
        let pad = (p_vaddr - base) as usize;
        let mut data = vec![0u8; pad];
        data.extend_from_slice(&bytes[p_offset..p_offset + p_filesz]);
        segments.push((base, data));
    }
    if segments.is_empty() {
        bail!("ELF has no loadable segments");
    }
    Image::from_bytes(entry, &segments)
}
