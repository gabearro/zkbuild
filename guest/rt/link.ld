ENTRY(_start)
MEMORY { RAM (rwx) : ORIGIN = 0x1000, LENGTH = 0x00FF0000 }
SECTIONS {
  . = 0x1000;
  .text : { KEEP(*(.text._start)) *(.text .text.*) } > RAM
  .rodata : { *(.rodata .rodata.* .srodata .srodata.*) } > RAM
  .data : { *(.data .data.* .sdata .sdata.*) } > RAM
  .bss (NOLOAD) : { __bss_start = .; *(.bss .bss.* .sbss .sbss.* COMMON) __bss_end = .; } > RAM
  /DISCARD/ : { *(.eh_frame) *(.comment) *(.riscv.attributes) }
}
