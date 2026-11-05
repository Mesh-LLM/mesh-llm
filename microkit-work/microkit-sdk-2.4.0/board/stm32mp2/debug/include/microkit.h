/*
 * Copyright 2021, Breakaway Consulting Pty. Ltd.
 * Copyright 2025, UNSW
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */

/* Microkit interface */

#pragma once

#include <sel4/sel4.h>
#ifdef CONFIG_VTX
#include <sel4/arch/vmenter.h>
#endif /* CONFIG_VTX */

typedef unsigned int microkit_channel;
typedef unsigned int microkit_child;
typedef unsigned int microkit_ioport;
typedef seL4_MessageInfo_t microkit_msginfo;

#define MONITOR_EP 5
/* Only valid in the 'benchmark' configuration */
#define TCB_CAP 6
/* Only valid when the PD has been configured to make SMC calls */
#define ARM_SMC_CAP 7
#define BASE_OUTPUT_NOTIFICATION_CAP 10
#define BASE_ENDPOINT_CAP 74
#define BASE_IRQ_CAP 138
#define BASE_TCB_CAP 202
#define BASE_VM_TCB_CAP 266
#define BASE_VCPU_CAP 330
#define BASE_IOPORT_CAP 394
#define BASE_USER_CAPS 458

/* This should be kept in sync with `PD_ROOT_CAP_BITS` in capdl/builder.rs */
#define PD_ROOT_CAP_BITS 6
#define PD_ROOT_CAP_SIZE (1ULL << PD_ROOT_CAP_BITS)

#define MICROKIT_MAX_USER_CAPS 128
#define MICROKIT_MAX_CHANNELS 62
#define MICROKIT_MAX_CHANNEL_ID (MICROKIT_MAX_CHANNELS - 1)
#define MICROKIT_MAX_IOPORT_ID MICROKIT_MAX_CHANNELS
#define MICROKIT_PD_NAME_LENGTH 64

/* User provided functions */
void init(void);
void notified(microkit_channel ch);
microkit_msginfo protected(microkit_channel ch, microkit_msginfo msginfo);
seL4_Bool fault(microkit_child child, microkit_msginfo msginfo, microkit_msginfo *reply_msginfo);

extern char microkit_name[MICROKIT_PD_NAME_LENGTH];
/* These next three variables are so our PDs can combine a signal with the next Recv syscall */
extern seL4_Bool microkit_have_signal;
extern seL4_CPtr microkit_signal_cap;
extern seL4_MessageInfo_t microkit_signal_msg;
#if defined(CONFIG_VTX)
struct microkit_x86_vcpu_state {
    seL4_Bool is_on;
    seL4_Word rip;
    seL4_Word prim_proc_ctl;
    seL4_Word irq_info;
};
extern struct microkit_x86_vcpu_state microkit_x86_vcpu_state;
#endif

/* Symbols for error checking libmicrokit API calls. Patched by the Microkit tool
 * to set bits corresponding to valid channels for this PD. */
extern seL4_Word microkit_irqs;
extern seL4_Word microkit_notifications;
extern seL4_Word microkit_pps;
extern seL4_Word microkit_ioports;

/*
 * Output a single character on the debug console.
 */
void microkit_dbg_putc(int c);

/*
 * Output a NUL terminated string to the debug console.
 */
void microkit_dbg_puts(const char *s);

/*
 * Output the decimal representation of an 8-bit integer to the debug console.
 */
void microkit_dbg_put8(seL4_Uint8 x);

/*
 * Output the decimal representation of an 32-bit integer to the debug console.
 */
void microkit_dbg_put32(seL4_Uint32 x);

static inline void microkit_internal_crash(seL4_Error err)
{
    /*
     * Currently crash be dereferencing NULL page
     *
     * Actually dereference 'err' which means the crash reporting will have
     * `err` as the fault address. A bit of a cute hack. Not a good long term
     * solution but good for now.
     */
    int *x = (int *)(seL4_Word) err;
    *x = 0;
}

static inline void microkit_notify(microkit_channel ch)
{
    if (ch > MICROKIT_MAX_CHANNEL_ID || (microkit_notifications & (1ULL << ch)) == 0) {
        microkit_dbg_puts(microkit_name);
        microkit_dbg_puts(" microkit_notify: invalid channel given '");
        microkit_dbg_put32(ch);
        microkit_dbg_puts("'\n");
        return;
    }
    seL4_Signal(BASE_OUTPUT_NOTIFICATION_CAP + ch);
}

static inline void microkit_irq_ack(microkit_channel ch)
{
    if (ch > MICROKIT_MAX_CHANNEL_ID || (microkit_irqs & (1ULL << ch)) == 0) {
        microkit_dbg_puts(microkit_name);
        microkit_dbg_puts(" microkit_irq_ack: invalid channel given '");
        microkit_dbg_put32(ch);
        microkit_dbg_puts("'\n");
        return;
    }
    seL4_IRQHandler_Ack(BASE_IRQ_CAP + ch);
}

static inline void microkit_pd_restart(microkit_child pd, seL4_Word entry_point)
{
    seL4_Error err;
    seL4_UserContext ctxt = {0};
#if defined(CONFIG_ARCH_X86_64)
    ctxt.rip = entry_point;
#elif defined(CONFIG_ARCH_AARCH64) || defined(CONFIG_ARCH_RISCV)
    ctxt.pc = entry_point;
#else
#error "Unsupported architecture for 'microkit_pd_restart'"
#endif
    err = seL4_TCB_WriteRegisters(
              BASE_TCB_CAP + pd,
              seL4_True,
              0, /* No flags */
              1, /* writing 1 register */
              &ctxt
          );

    if (err != seL4_NoError) {
        microkit_dbg_puts("microkit_pd_restart: error writing TCB registers\n");
        microkit_internal_crash(err);
    }
}

static inline void microkit_pd_stop(microkit_child pd)
{
    seL4_Error err;
    err = seL4_TCB_Suspend(BASE_TCB_CAP + pd);
    if (err != seL4_NoError) {
        microkit_dbg_puts("microkit_pd_stop: error stopping TCB '");
        microkit_dbg_put32(pd);
        microkit_dbg_puts("'\n");
        microkit_internal_crash(err);
    }
}

static inline void microkit_pd_resume(microkit_child pd)
{
    seL4_Error err;
    err = seL4_TCB_Resume(BASE_TCB_CAP + pd);
    if (err != seL4_NoError) {
        microkit_dbg_puts("microkit_pd_resume: error resuming TCB '");
        microkit_dbg_put32(pd);
        microkit_dbg_puts("'\n");
        microkit_internal_crash(err);
    }
}

static inline microkit_msginfo microkit_ppcall(microkit_channel ch, microkit_msginfo msginfo)
{
    if (ch > MICROKIT_MAX_CHANNEL_ID || (microkit_pps & (1ULL << ch)) == 0) {
        microkit_dbg_puts(microkit_name);
        microkit_dbg_puts(" microkit_ppcall: invalid channel given '");
        microkit_dbg_put32(ch);
        microkit_dbg_puts("'\n");
        return seL4_MessageInfo_new(0, 0, 0, 0);
    }
    return seL4_Call(BASE_ENDPOINT_CAP + ch, msginfo);
}

static inline microkit_msginfo microkit_msginfo_new(seL4_Word label, seL4_Uint16 count)
{
    return seL4_MessageInfo_new(label, 0, 0, count);
}

static inline seL4_Word microkit_msginfo_get_label(microkit_msginfo msginfo)
{
    return seL4_MessageInfo_get_label(msginfo);
}

static inline seL4_Word microkit_msginfo_get_count(microkit_msginfo msginfo)
{
    return seL4_MessageInfo_get_length(msginfo);
}

static inline void microkit_mr_set(seL4_Uint8 mr, seL4_Word value)
{
    seL4_SetMR(mr, value);
}

static inline seL4_Word microkit_mr_get(seL4_Uint8 mr)
{
    return seL4_GetMR(mr);
}

/* The following APIs are only available where the kernel is built as a hypervisor. */
#if defined(CONFIG_ARM_HYPERVISOR_SUPPORT)
static inline void microkit_vcpu_restart(microkit_child vcpu, seL4_Word entry_point)
{
    seL4_Error err;
    seL4_UserContext ctxt = {0};
    ctxt.pc = entry_point;

    err = seL4_TCB_WriteRegisters(
              BASE_VM_TCB_CAP + vcpu,
              seL4_True,
              0, /* No flags */
              1, /* writing 1 register */
              &ctxt
          );

    if (err != seL4_NoError) {
        microkit_dbg_puts("microkit_vcpu_restart: error writing registers\n");
        microkit_internal_crash(err);
    }
}

static inline void microkit_vcpu_stop(microkit_child vcpu)
{
    seL4_Error err;
    err = seL4_TCB_Suspend(BASE_VM_TCB_CAP + vcpu);
    if (err != seL4_NoError) {
        microkit_dbg_puts("microkit_vcpu_stop: error suspending TCB\n");
        microkit_internal_crash(err);
    }
}

static inline void microkit_vcpu_arm_inject_irq(microkit_child vcpu, seL4_Uint16 irq, seL4_Uint8 priority,
                                                seL4_Uint8 group, seL4_Uint8 index)
{
    seL4_Error err;
    err = seL4_ARM_VCPU_InjectIRQ(BASE_VCPU_CAP + vcpu, irq, priority, group, index);
    if (err != seL4_NoError) {
        microkit_dbg_puts("microkit_vcpu_arm_inject_irq: error injecting IRQ\n");
        microkit_internal_crash(err);
    }
}

static inline void microkit_vcpu_arm_ack_vppi(microkit_child vcpu, seL4_Word irq)
{
    seL4_Error err;
    err = seL4_ARM_VCPU_AckVPPI(BASE_VCPU_CAP + vcpu, irq);
    if (err != seL4_NoError) {
        microkit_dbg_puts("microkit_vcpu_arm_ack_vppi: error acking VPPI\n");
        microkit_internal_crash(err);
    }
}

static inline seL4_Word microkit_vcpu_arm_read_reg(microkit_child vcpu, seL4_Word reg)
{
    seL4_ARM_VCPU_ReadRegs_t ret;
    ret = seL4_ARM_VCPU_ReadRegs(BASE_VCPU_CAP + vcpu, reg);
    if (ret.error != seL4_NoError) {
        microkit_dbg_puts("microkit_vcpu_arm_read_reg: error reading vCPU register\n");
        microkit_internal_crash(ret.error);
    }

    return ret.value;
}

static inline void microkit_vcpu_arm_write_reg(microkit_child vcpu, seL4_Word reg, seL4_Word value)
{
    seL4_Error err;
    err = seL4_ARM_VCPU_WriteRegs(BASE_VCPU_CAP + vcpu, reg, value);
    if (err != seL4_NoError) {
        microkit_dbg_puts("microkit_vcpu_arm_write_reg: error writing vCPU register\n");
        microkit_internal_crash(err);
    }
}
#endif /* CONFIG_ARM_HYPERVISOR_SUPPORT */

#if defined(CONFIG_ALLOW_SMC_CALLS)
static inline void microkit_arm_smc_call(seL4_ARM_SMCContext *args, seL4_ARM_SMCContext *response)
{
    seL4_Error err;
    err = seL4_ARM_SMC_Call(ARM_SMC_CAP, args, response);
    if (err != seL4_NoError) {
        microkit_dbg_puts("microkit_arm_smc_call: error making SMC call\n");
        microkit_internal_crash(err);
    }
}
#endif /* CONFIG_ALLOW_SMC_CALLS */

#if defined(CONFIG_ARCH_X86_64)
static inline void microkit_x86_ioport_write_8(microkit_ioport ioport_id, seL4_Word port_addr, seL4_Word data)
{
    if (ioport_id > MICROKIT_MAX_IOPORT_ID || (microkit_ioports & (1ULL << ioport_id)) == 0) {
        microkit_dbg_puts(microkit_name);
        microkit_dbg_puts(" microkit_x86_ioport_write_8: invalid I/O Port ID given '");
        microkit_dbg_put32(ioport_id);
        microkit_dbg_puts("'\n");
        return;
    }

    seL4_Error err;
    err = seL4_X86_IOPort_Out8(BASE_IOPORT_CAP + ioport_id, port_addr, data);
    if (err != seL4_NoError) {
        microkit_dbg_puts("microkit_x86_ioport_write_8: error writing data\n");
        microkit_internal_crash(err);
    }
}

static inline void microkit_x86_ioport_write_16(microkit_ioport ioport_id, seL4_Word port_addr, seL4_Word data)
{
    if (ioport_id > MICROKIT_MAX_IOPORT_ID || (microkit_ioports & (1ULL << ioport_id)) == 0) {
        microkit_dbg_puts(microkit_name);
        microkit_dbg_puts(" microkit_x86_ioport_write_16: invalid I/O Port ID given '");
        microkit_dbg_put32(ioport_id);
        microkit_dbg_puts("'\n");
        return;
    }

    seL4_Error err;
    err = seL4_X86_IOPort_Out16(BASE_IOPORT_CAP + ioport_id, port_addr, data);
    if (err != seL4_NoError) {
        microkit_dbg_puts("microkit_x86_ioport_write_16: error writing data\n");
        microkit_internal_crash(err);
    }
}

static inline void microkit_x86_ioport_write_32(microkit_ioport ioport_id, seL4_Word port_addr, seL4_Word data)
{
    if (ioport_id > MICROKIT_MAX_IOPORT_ID || (microkit_ioports & (1ULL << ioport_id)) == 0) {
        microkit_dbg_puts(microkit_name);
        microkit_dbg_puts(" microkit_x86_ioport_write_32: invalid I/O Port ID given '");
        microkit_dbg_put32(ioport_id);
        microkit_dbg_puts("'\n");
        return;
    }

    seL4_Error err;
    err = seL4_X86_IOPort_Out32(BASE_IOPORT_CAP + ioport_id, port_addr, data);
    if (err != seL4_NoError) {
        microkit_dbg_puts("microkit_x86_ioport_write_32: error writing data\n");
        microkit_internal_crash(err);
    }
}

static inline seL4_Uint8 microkit_x86_ioport_read_8(microkit_ioport ioport_id, seL4_Word port_addr)
{
    if (ioport_id > MICROKIT_MAX_IOPORT_ID || (microkit_ioports & (1ULL << ioport_id)) == 0) {
        microkit_dbg_puts(microkit_name);
        microkit_dbg_puts(" microkit_x86_ioport_read_8: invalid I/O Port ID given '");
        microkit_dbg_put32(ioport_id);
        microkit_dbg_puts("'\n");
        return 0;
    }

    seL4_X86_IOPort_In8_t ret;
    ret = seL4_X86_IOPort_In8(BASE_IOPORT_CAP + ioport_id, port_addr);
    if (ret.error != seL4_NoError) {
        microkit_dbg_puts("microkit_x86_ioport_read_8: error reading data\n");
        microkit_internal_crash(ret.error);
    }

    return ret.result;
}

static inline seL4_Uint16 microkit_x86_ioport_read_16(microkit_ioport ioport_id, seL4_Word port_addr)
{
    if (ioport_id > MICROKIT_MAX_IOPORT_ID || (microkit_ioports & (1ULL << ioport_id)) == 0) {
        microkit_dbg_puts(microkit_name);
        microkit_dbg_puts(" microkit_x86_ioport_read_16: invalid I/O Port ID given '");
        microkit_dbg_put32(ioport_id);
        microkit_dbg_puts("'\n");
        return 0;
    }

    seL4_X86_IOPort_In16_t ret;
    ret = seL4_X86_IOPort_In16(BASE_IOPORT_CAP + ioport_id, port_addr);
    if (ret.error != seL4_NoError) {
        microkit_dbg_puts("microkit_x86_ioport_read_16: error reading data\n");
        microkit_internal_crash(ret.error);
    }

    return ret.result;
}

static inline seL4_Uint32 microkit_x86_ioport_read_32(microkit_ioport ioport_id, seL4_Word port_addr)
{
    if (ioport_id > MICROKIT_MAX_IOPORT_ID || (microkit_ioports & (1ULL << ioport_id)) == 0) {
        microkit_dbg_puts(microkit_name);
        microkit_dbg_puts(" microkit_x86_ioport_read_32: invalid I/O Port ID given '");
        microkit_dbg_put32(ioport_id);
        microkit_dbg_puts("'\n");
        return 0;
    }

    seL4_X86_IOPort_In32_t ret;
    ret = seL4_X86_IOPort_In32(BASE_IOPORT_CAP + ioport_id, port_addr);
    if (ret.error != seL4_NoError) {
        microkit_dbg_puts("microkit_x86_ioport_read_32: error reading data\n");
        microkit_internal_crash(ret.error);
    }

    return ret.result;
}

#if defined(CONFIG_VTX)
/* Architecturally defined identifiers for a x86 VCPU's VMCS fields,
 * see seL4 source: `include/arch/x86/arch/object/vcpu.h`
 * or Intel® 64 and IA-32 Architectures Software Developer’s Manual
 * Combined Volumes: 1, 2A, 2B, 2C, 2D, 3A, 3B, 3C, 3D, and 4
 * Order Number: 325462-084US June 2024
 * "Table B-8. Encodings for 32-Bit Control Fields (0100_00xx_xxxx_xxx0B)" and
 * "Table B-14. Encodings for Natural-Width Guest-State Fields (0110_10xx_xxxx_xxx0B) (Contd.)".
 * */
#define MICROKIT_VMX_GUEST_RIP 0x0000681E
#define MICROKIT_VMX_CONTROL_PRIMARY_PROCESSOR_CONTROLS 0x00004002
#define MICROKIT_VMX_CONTROL_ENTRY_INTERRUPTION_INFO 0x00004016

static inline seL4_Word microkit_vcpu_x86_read_vmcs(microkit_child vcpu, seL4_Word field)
{
    /* This function assumes that a PD would only have access to 1 VCPU object.
     * For these fields:
     * - Guest RIP,
     * - Primary Processor Based VM Execution Controls, and
     * - VM Entry Interruption-Information
     * they will be read from local variables that were written
     * to by `microkit_vcpu_x86_write_vmcs()`. The kernel will service
     * reading from other fields.
     */

    seL4_Word value;
    switch (field) {
    case MICROKIT_VMX_GUEST_RIP:
        value = microkit_x86_vcpu_state.rip;
        break;
    case MICROKIT_VMX_CONTROL_PRIMARY_PROCESSOR_CONTROLS:
        value = microkit_x86_vcpu_state.prim_proc_ctl;
        break;
    case MICROKIT_VMX_CONTROL_ENTRY_INTERRUPTION_INFO:
        value = microkit_x86_vcpu_state.irq_info;
        break;
    default: {
        seL4_X86_VCPU_ReadVMCS_t ret = seL4_X86_VCPU_ReadVMCS(BASE_VCPU_CAP + vcpu, field);
        if (ret.error != seL4_NoError) {
            microkit_dbg_puts("microkit_x86_read_vmcs: error reading data\n");
            microkit_internal_crash(ret.error);
        }
        value = ret.value;
        break;
    }
    }

    return value;
}

static inline void microkit_vcpu_x86_write_vmcs(microkit_child vcpu, seL4_Word field, seL4_Word value)
{
    /* Assumes that a PD would only have access to 1 VCPU object.
     * These fields will be written to local variables rather than the actual VMCS:
     * - RIP,
     * - Primary Processor Based VM Execution Controls, and
     * - VM Entry Interruption-Information.
     * Because they will be passed to the kernel on `seL4_VMEnter()`,
     * then the kernel will write them to the VMCS for us.
     * Writing other VMCS fields will go to the real VMCS immediately via a syscall.
     */
    switch (field) {
    case MICROKIT_VMX_GUEST_RIP:
        microkit_x86_vcpu_state.rip = value;
        break;
    case MICROKIT_VMX_CONTROL_PRIMARY_PROCESSOR_CONTROLS:
        microkit_x86_vcpu_state.prim_proc_ctl = value;
        break;
    case MICROKIT_VMX_CONTROL_ENTRY_INTERRUPTION_INFO:
        microkit_x86_vcpu_state.irq_info = value;
        break;
    default: {
        seL4_X86_VCPU_WriteVMCS_t ret = seL4_X86_VCPU_WriteVMCS(BASE_VCPU_CAP + vcpu, field, value);
        if (ret.error != seL4_NoError) {
            microkit_dbg_puts("microkit_x86_write_vmcs: error writing data\n");
            microkit_internal_crash(ret.error);
        }
        break;
    }
    }
}

static inline seL4_Word microkit_vcpu_x86_read_msr(microkit_child vcpu, seL4_Word field)
{
    seL4_X86_VCPU_ReadMSR_t ret;
    ret = seL4_X86_VCPU_ReadMSR(BASE_VCPU_CAP + vcpu, field);
    if (ret.error != seL4_NoError) {
        microkit_dbg_puts("microkit_x86_read_msr: error reading data\n");
        microkit_internal_crash(ret.error);
    }

    return ret.value;
}

static inline void microkit_vcpu_x86_write_msr(microkit_child vcpu, seL4_Word field, seL4_Word value)
{
    seL4_X86_VCPU_WriteMSR_t ret;
    ret = seL4_X86_VCPU_WriteMSR(BASE_VCPU_CAP + vcpu, field, value);
    if (ret.error != seL4_NoError) {
        microkit_dbg_puts("microkit_x86_write_msr: error writing data\n");
        microkit_internal_crash(ret.error);
    }
}

static inline void microkit_vcpu_x86_enable_ioport(microkit_child vcpu, microkit_ioport ioport_id, seL4_Word port_addr,
                                                   seL4_Word length)
{
    if (ioport_id > MICROKIT_MAX_IOPORT_ID || (microkit_ioports & (1ULL << ioport_id)) == 0) {
        microkit_dbg_puts(microkit_name);
        microkit_dbg_puts(" microkit_vcpu_x86_enable_ioport: invalid I/O Port ID given '");
        microkit_dbg_put32(ioport_id);
        microkit_dbg_puts("'\n");
        return;
    }

    int ret;
    ret = seL4_X86_VCPU_EnableIOPort(BASE_VCPU_CAP + vcpu, BASE_IOPORT_CAP + ioport_id, port_addr, port_addr + length - 1);
    if (ret != seL4_NoError) {
        microkit_dbg_puts("microkit_vcpu_x86_enable_ioport: error enabling I/O Port\n");
        microkit_internal_crash(ret);
    }
}

static inline void microkit_vcpu_x86_disable_ioport(microkit_child vcpu, seL4_Word port_addr, seL4_Word length)
{
    int ret;
    ret = seL4_X86_VCPU_DisableIOPort(BASE_VCPU_CAP + vcpu, port_addr, port_addr + length - 1);
    if (ret != seL4_NoError) {
        microkit_dbg_puts("microkit_vcpu_x86_disable_ioport: error disabling I/O Port\n");
        microkit_internal_crash(ret);
    }
}

static inline void microkit_vcpu_x86_write_regs(microkit_child vcpu, seL4_VCPUContext *regs)
{
    int ret;
    ret = seL4_X86_VCPU_WriteRegisters(BASE_VCPU_CAP + vcpu, regs);
    if (ret != seL4_NoError) {
        microkit_dbg_puts("microkit_vcpu_x86_write_regs: error writing vCPU registers\n");
        microkit_internal_crash(ret);
    }
}

static inline void microkit_vcpu_x86_on(void)
{
    /* On x86, a TCB can only have one bound VCPU at any given time.
     * So we don't take a `microkit_child vcpu` here. */
    microkit_x86_vcpu_state.is_on = seL4_True;
}

static inline void microkit_vcpu_x86_off(void)
{
    microkit_x86_vcpu_state.is_on = seL4_False;
}

#endif /* CONFIG_VTX */
#endif /* CONFIG_ARCH_X86_64 */

static inline void microkit_deferred_notify(microkit_channel ch)
{
    if (ch > MICROKIT_MAX_CHANNEL_ID || (microkit_notifications & (1ULL << ch)) == 0) {
        microkit_dbg_puts(microkit_name);
        microkit_dbg_puts(" microkit_deferred_notify: invalid channel given '");
        microkit_dbg_put32(ch);
        microkit_dbg_puts("'\n");
        return;
    }
    microkit_have_signal = seL4_True;
    microkit_signal_msg = seL4_MessageInfo_new(0, 0, 0, 0);
    microkit_signal_cap = (BASE_OUTPUT_NOTIFICATION_CAP + ch);
}

static inline void microkit_deferred_irq_ack(microkit_channel ch)
{
    if (ch > MICROKIT_MAX_CHANNEL_ID || (microkit_irqs & (1ULL << ch)) == 0) {
        microkit_dbg_puts(microkit_name);
        microkit_dbg_puts(" microkit_deferred_irq_ack: invalid channel given '");
        microkit_dbg_put32(ch);
        microkit_dbg_puts("'\n");
        return;
    }
    microkit_have_signal = seL4_True;
    microkit_signal_msg = seL4_MessageInfo_new(IRQAckIRQ, 0, 0, 0);
    microkit_signal_cap = (BASE_IRQ_CAP + ch);
}

/**
 * Convert the "slot" identifier from the system file for the extra user caps
 * <cspace> element into the seL4_CPtr at runtime.
 *
 * If the slot exceeds the valid range of inputs (0 <= slot < MICROKIT_MAX_USER_CAPS),
 * it returns the value `seL4_CapNull`.
 **/
static inline seL4_CPtr microkit_cspace_root_slot_to_cptr(seL4_Word slot)
{
    if (slot == 0 || slot >= PD_ROOT_CAP_SIZE) {
        return seL4_CapNull;
    }

    return slot << (seL4_WordBits - PD_ROOT_CAP_BITS);
}
