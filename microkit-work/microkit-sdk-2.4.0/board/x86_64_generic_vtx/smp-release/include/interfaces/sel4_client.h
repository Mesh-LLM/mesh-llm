
/*
 * Automatically generated system call stubs.
 */

#pragma once

#include <sel4/config.h>
#include <sel4/types.h>
#include <sel4/sel4_arch/constants.h>

/*
 * The following code generates a compile-time error if the system call
 * stub generator has an incorrect understanding of how large a type is.
 *
 * If you receive a compile-time error here, you will need to adjust
 * the type information in the stub generator.
 */
#define assert_size_correct(type, expected_bytes) \
        typedef unsigned long __type_##type##_size_incorrect[ \
                (sizeof(type) == expected_bytes) ? 1 : -1]

assert_size_correct(int, 4);
assert_size_correct(long, 8);
assert_size_correct(seL4_Uint8, 1);
assert_size_correct(seL4_Uint16, 2);
assert_size_correct(seL4_Uint32, 4);
assert_size_correct(seL4_Uint64, 8);
assert_size_correct(seL4_Time, 8);
assert_size_correct(seL4_Word, 8);
assert_size_correct(seL4_Bool, 1);
assert_size_correct(seL4_CapRights_t, 8);
assert_size_correct(seL4_CPtr, 8);
assert_size_correct(seL4_CNode, 8);
assert_size_correct(seL4_IRQHandler, 8);
assert_size_correct(seL4_IRQControl, 8);
assert_size_correct(seL4_TCB, 8);
assert_size_correct(seL4_Untyped, 8);
assert_size_correct(seL4_DomainSet, 8);
assert_size_correct(seL4_SchedContext, 8);
assert_size_correct(seL4_SchedControl, 8);
assert_size_correct(seL4_X86_VMAttributes, 8);
assert_size_correct(seL4_X86_EPT_VMAttributes, 8);
assert_size_correct(seL4_X86_IOPort, 8);
assert_size_correct(seL4_X86_IOPortControl, 8);
assert_size_correct(seL4_X86_ASIDControl, 8);
assert_size_correct(seL4_X86_ASIDPool, 8);
assert_size_correct(seL4_X86_IOSpace, 8);
assert_size_correct(seL4_X86_Page, 8);
assert_size_correct(seL4_X64_PML4, 8);
assert_size_correct(seL4_X86_PDPT, 8);
assert_size_correct(seL4_X86_PageDirectory, 8);
assert_size_correct(seL4_X86_PageTable, 8);
assert_size_correct(seL4_X86_IOPageTable, 8);
assert_size_correct(seL4_X86_VCPU, 8);
assert_size_correct(seL4_X86_EPTPML4, 8);
assert_size_correct(seL4_X86_EPTPDPT, 8);
assert_size_correct(seL4_X86_EPTPD, 8);
assert_size_correct(seL4_X86_EPTPT, 8);
assert_size_correct(seL4_VCPUContext, 120);
assert_size_correct(seL4_UserContext, 160);

/*
 * Return types for generated methods.
 */
struct seL4_X86_VCPU_ReadMSR {
	int error;
	seL4_Word value;
};
typedef struct seL4_X86_VCPU_ReadMSR seL4_X86_VCPU_ReadMSR_t;

struct seL4_X86_VCPU_WriteMSR {
	int error;
	seL4_Word written;
};
typedef struct seL4_X86_VCPU_WriteMSR seL4_X86_VCPU_WriteMSR_t;

struct seL4_X86_PageDirectory_GetStatusBits {
	int error;
	seL4_Word accessed;
	seL4_Word dirty;
};
typedef struct seL4_X86_PageDirectory_GetStatusBits seL4_X86_PageDirectory_GetStatusBits_t;

struct seL4_X86_Page_GetAddress {
	int error;
	seL4_Word paddr;
};
typedef struct seL4_X86_Page_GetAddress seL4_X86_Page_GetAddress_t;

struct seL4_X86_IOPort_In8 {
	int error;
	seL4_Uint8 result;
};
typedef struct seL4_X86_IOPort_In8 seL4_X86_IOPort_In8_t;

struct seL4_X86_IOPort_In16 {
	int error;
	seL4_Uint16 result;
};
typedef struct seL4_X86_IOPort_In16 seL4_X86_IOPort_In16_t;

struct seL4_X86_IOPort_In32 {
	int error;
	seL4_Uint32 result;
};
typedef struct seL4_X86_IOPort_In32 seL4_X86_IOPort_In32_t;

struct seL4_X86_VCPU_ReadVMCS {
	int error;
	seL4_Word value;
};
typedef struct seL4_X86_VCPU_ReadVMCS seL4_X86_VCPU_ReadVMCS_t;

struct seL4_X86_VCPU_WriteVMCS {
	int error;
	seL4_Word written;
};
typedef struct seL4_X86_VCPU_WriteVMCS seL4_X86_VCPU_WriteVMCS_t;

struct seL4_TCB_GetBreakpoint {
	int error;
	seL4_Word vaddr;
	seL4_Word type;
	seL4_Word size;
	seL4_Word rw;
	seL4_Bool is_enabled;
};
typedef struct seL4_TCB_GetBreakpoint seL4_TCB_GetBreakpoint_t;

struct seL4_TCB_ConfigureSingleStepping {
	int error;
	seL4_Bool bp_was_consumed;
};
typedef struct seL4_TCB_ConfigureSingleStepping seL4_TCB_ConfigureSingleStepping_t;

struct seL4_TCB_SetFlags {
	int error;
	seL4_Word flags;
};
typedef struct seL4_TCB_SetFlags seL4_TCB_SetFlags_t;

struct seL4_SchedContext_Consumed {
	int error;
	seL4_Time consumed;
};
typedef struct seL4_SchedContext_Consumed seL4_SchedContext_Consumed_t;

struct seL4_SchedContext_YieldTo {
	int error;
	seL4_Time consumed;
};
typedef struct seL4_SchedContext_YieldTo seL4_SchedContext_YieldTo_t;

/*
 * Generated stubs.
 */
/**
 * @xmlonly <manual name="Map" label="x86_64_pdpt_map"/> @endxmlonly
 * @brief @xmlonly Map a page directory page table. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the PDPT being operated on.
 * @param[in] pml4 Capability to the VSpace which will contain the mapping. 
 * @param[in] vaddr Virtual address at which to map page. 
 * @param[in] attr VM attributes for the mapping.  @xmlonly <docref>Possible values for this type are given in <autoref label="ch:vspace"/>.</docref> @endxmlonly 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst A mapping already exists for this level in  @xmlonly <texttt text="vspace"/> @endxmlonly  at  @xmlonly <texttt text="vaddr"/> @endxmlonly . 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="pml4"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="vaddr"/> @endxmlonly  is in the kernel virtual address range. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="pml4"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="pml4"/> @endxmlonly  is not assigned to an ASID pool.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is already mapped in a VSpace. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_PDPT_Map(seL4_X86_PDPT _service, seL4_X64_PML4 pml4, seL4_Word vaddr, seL4_X86_VMAttributes attr)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86PDPTMap, 0, 1, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, pml4);

	/* Marshal and initialise parameters. */
	mr0 = vaddr;
	mr1 = attr;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Unmap" label="x86_64_pdpt_unmap"/> @endxmlonly
 * @brief @xmlonly Unmap a page directory page table. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the PDPT being operated on.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RevokeFirst A copy of the  @xmlonly <texttt text="_service"/> @endxmlonly  capability exists. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_PDPT_Unmap(seL4_X86_PDPT _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86PDPTUnmap, 0, 0, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#if defined(CONFIG_X86_64_VTX_64BIT_GUESTS)
/**
 * @xmlonly <manual name="Read MSR" label="x86_64_vcpu_read_msr"/> @endxmlonly
 * @brief @xmlonly Read 64-bit specific MSR field from the hardware @endxmlonly
 * 
 * @xmlonly
 * Thin wrapper around the <texttt text="rdmsr"/> instruction that is performed on
 * specific, needed registers. Certain registers might simply be cached and restored
 * later.
 * @endxmlonly
 * 
 * @param[in] _service VCPU object to operate on
 * @param[in] field Field to give to `rdmsr` instruction 
 * @return @xmlonly 
 *           A <texttt text="seL4_X86_VCPU_ReadMSR_t"/> struct that contains a
 *           <texttt text="seL4_Word value"/>, which holds the return result of the
 *           <texttt text="rdmsr"/> instruction, and <texttt text="int error"/>.
 *           <docref>See <autoref label="sec:errors"/> for a description
 *           of the message register and tag contents upon error.</docref>
 *          @endxmlonly
 */
LIBSEL4_INLINE seL4_X86_VCPU_ReadMSR_t
seL4_X86_VCPU_ReadMSR(seL4_X86_VCPU _service, seL4_Word field)
{
	seL4_X86_VCPU_ReadMSR_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86VCPUReadMSR, 0, 0, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = field;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result.error = seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result.error != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
		return result;
	}

	/* Unmarshal result. */
	result.value = mr0;
	return result;
}

#endif
#if defined(CONFIG_X86_64_VTX_64BIT_GUESTS)
/**
 * @xmlonly <manual name="Write MSR" label="x86_64_vcpu_write_msr"/> @endxmlonly
 * @brief @xmlonly Write 64-bit specific MSR field to the hardware @endxmlonly
 * 
 * @xmlonly
 * Thin wrapper around the <texttt text="wrmsr"/> instruction that is performed on
 * specific, needed registers. As well as validating that
 * a legal field is requested, the value may be modified to ensure any
 * bits that are fixed in the hardware are correct, and that any features
 * required for kernel correctness are not disabled <docref>(see <autoref label="sec:virt"/>)</docref>.
 * 
 * The final value written to the hardware is returned and can be compared
 * to the input parameter to determine what bits the kernel changed.
 * @endxmlonly
 * 
 * @param[in] _service VCPU object to operate on
 * @param[in] field Field to give to `wrsmr` instruction 
 * @param[in] value Value to write using `wrsmr` instruction 
 * @return @xmlonly 
 *           A <texttt text="seL4_X86_VCPU_WriteMSR_t"/> struct that contains a
 *           <texttt text="seL4_Word written"/>, which holds the final value written with the <texttt text="wrmsr"/> instruction,
 *           and <texttt text="int error"/>. <docref>See <autoref label="sec:errors"/> for a description
 *           of the message register and tag contents upon error.</docref>
 *          @endxmlonly
 */
LIBSEL4_INLINE seL4_X86_VCPU_WriteMSR_t
seL4_X86_VCPU_WriteMSR(seL4_X86_VCPU _service, seL4_Word field, seL4_Word value)
{
	seL4_X86_VCPU_WriteMSR_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86VCPUWriteMSR, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = field;
	mr1 = value;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result.error = seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result.error != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
		return result;
	}

	/* Unmarshal result. */
	result.written = mr0;
	return result;
}

#endif
/**
 * @xmlonly <manual name="Map" label="x86_page_directory_map"/> @endxmlonly
 * @brief @xmlonly Map a page directory. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page directory being operated on.
 * @param[in] vspace Capability to the VSpace which will contain the mapping 
 * @param[in] vaddr Virtual address to map the page into. 
 * @param[in] attr VM attributes for the mapping.  @xmlonly <docref>Possible values for this type are given in <autoref label="ch:vspace"/></docref> @endxmlonly 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst A mapping already exists for this level in  @xmlonly <texttt text="vspace"/> @endxmlonly  at  @xmlonly <texttt text="vaddr"/> @endxmlonly . 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="vspace"/> @endxmlonly  does not have a PDPT mapped at  @xmlonly <texttt text="vaddr"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="vspace"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="vaddr"/> @endxmlonly  is in the kernel virtual address range. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="vspace"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="vspace"/> @endxmlonly  is not assigned to an ASID pool.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is already mapped in a VSpace. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_PageDirectory_Map(seL4_X86_PageDirectory _service, seL4_CPtr vspace, seL4_Word vaddr, seL4_X86_VMAttributes attr)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86PageDirectoryMap, 0, 1, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, vspace);

	/* Marshal and initialise parameters. */
	mr0 = vaddr;
	mr1 = attr;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Unmap" label="x86_page_directory_unmap"/> @endxmlonly
 * @brief @xmlonly Unmap a page directory. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page directory being operated on.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RevokeFirst A copy of the  @xmlonly <texttt text="_service"/> @endxmlonly  capability exists. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_PageDirectory_Unmap(seL4_X86_PageDirectory _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86PageDirectoryUnmap, 0, 0, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#if defined(CONFIG_ARCH_IA32)
/**
 * @xmlonly <manual name="Get Status Bits" label="x86_page_directory_get_status_bits"/> @endxmlonly
 * @brief @xmlonly Retrieve the accessed and dirty bits of a page mapped into an address space. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page directory being operated on.Capability to the address space to query.
 * @param[in] vaddr Virtual address of the page to query 
 * @return @xmlonly 
 *                 A <texttt text="seL4_X86_PageDirectory_GetStatusBits_t"/> structure.
 *              @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="_service"/> @endxmlonly  does not have a mapping at  @xmlonly <texttt text="vaddr"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="vaddr"/> @endxmlonly  is in the kernel virtual address range. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_X86_PageDirectory_GetStatusBits_t
seL4_X86_PageDirectory_GetStatusBits(seL4_X86_PageDirectory _service, seL4_Word vaddr)
{
	seL4_X86_PageDirectory_GetStatusBits_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86PageDirectoryGetStatusBits, 0, 0, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = vaddr;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result.error = seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result.error != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
		return result;
	}

	/* Unmarshal result. */
	result.accessed = mr0;
	result.dirty = mr1;
	return result;
}

#endif
/**
 * @xmlonly <manual name="Map" label="x86_pagetable_map"/> @endxmlonly
 * @brief @xmlonly Map a page table into an address space. @endxmlonly
 * 
 * @xmlonly
 * Takes a <texttt text="PageDirectory"/> capability as an argument,
 * and installs a reference to the invoked
 * <texttt text="PageTable"/> in a specified slot in the <texttt text="PageDirectory"/>.
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page table being operated on.
 * @param[in] vspace Capability to the VSpace which will contain the mapping 
 * @param[in] vaddr Virtual address to map the page into. 
 * @param[in] attr VM attributes for the mapping.  @xmlonly <docref>Possible values for this type are given in <autoref label="ch:vspace"/></docref> @endxmlonly 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst A mapping already exists for this level in  @xmlonly <texttt text="vspace"/> @endxmlonly  at  @xmlonly <texttt text="vaddr"/> @endxmlonly . 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="vspace"/> @endxmlonly  does not have a Page Directory mapped at  @xmlonly <texttt text="vaddr"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="vspace"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="vaddr"/> @endxmlonly  is in the kernel virtual address range. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="vspace"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="vspace"/> @endxmlonly  is not assigned to an ASID pool.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is already mapped in a VSpace. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_PageTable_Map(seL4_X86_PageTable _service, seL4_CPtr vspace, seL4_Word vaddr, seL4_X86_VMAttributes attr)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86PageTableMap, 0, 1, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, vspace);

	/* Marshal and initialise parameters. */
	mr0 = vaddr;
	mr1 = attr;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Unmap" label="x86_pagetable_unmap"/> @endxmlonly
 * @brief @xmlonly Unmap a page table from its address space and zero it out. @endxmlonly
 * 
 * @xmlonly
 * Removes the reference to the invoked <texttt text="PageTable"/> from its containing
 * <texttt text="PageDirectory"/>.
 * <docref>See <autoref label="ch:vspace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page table being operated on.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RevokeFirst A copy of the  @xmlonly <texttt text="_service"/> @endxmlonly  capability exists. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_PageTable_Unmap(seL4_X86_PageTable _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86PageTableUnmap, 0, 0, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#if defined(CONFIG_IOMMU)
/**
 * @xmlonly <manual name="Map" label="x86_io_page_table_map"/> @endxmlonly
 * @brief @xmlonly Map an IO page table into an IOSpace. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:iospace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the I/O page table being operated on.
 * @param[in] iospace The IOSpace to map the page table into. 
 * @param[in] ioaddr The address to map the page table at. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst All required page tables are already mapped in  @xmlonly <texttt text="iospace"/> @endxmlonly  at  @xmlonly <texttt text="ioaddr"/> @endxmlonly . 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="iospace"/> @endxmlonly  does not have a paging structure at the required level mapped at  @xmlonly <texttt text="ioaddr"/> @endxmlonly . 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="iospace"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="iospace"/> @endxmlonly  is not assigned to a PCI device.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is already mapped in an IOSpace. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_IOPageTable_Map(seL4_X86_IOPageTable _service, seL4_X86_IOSpace iospace, seL4_Word ioaddr)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86IOPageTableMap, 0, 1, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, iospace);

	/* Marshal and initialise parameters. */
	mr0 = ioaddr;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_IOMMU)
/**
 * @xmlonly <manual name="Unmap" label="x86_io_page_table_unmap"/> @endxmlonly
 * @brief @xmlonly Unmap an IO page table from an IOSpace. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:iospace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the I/O page table being operated on.The page table to unmap.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RevokeFirst A copy of the  @xmlonly <texttt text="_service"/> @endxmlonly  capability exists. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_IOPageTable_Unmap(seL4_X86_IOPageTable _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86IOPageTableUnmap, 0, 0, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
/**
 * @xmlonly <manual name="Map" label="x86_page_map"/> @endxmlonly
 * @brief @xmlonly Map a page into an address space or update the mapping attributes. @endxmlonly
 * 
 * @xmlonly
 * Takes a VSpace capability, as an
 * argument and installs a reference
 * to the given <texttt text="Page"/> in the lowest-level unmapped paging structure
 * corresponding to the given address, or updates the mapping attributes if the page is already mapped at this address. If the required paging structures are not present
 * this operation will fail, returning a seL4_FailedLookup error.
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page being operated on.
 * @param[in] vspace Capability to the VSpace which will contain the mapping 
 * @param[in] vaddr Virtual address to map the page into. 
 * @param[in] rights Rights for the mapping.  @xmlonly <docref>Possible values for this type are given in <autoref label="sec:cap_rights"/></docref> @endxmlonly 
 * @param[in] attr VM attributes for the mapping.  @xmlonly <docref>Possible values for this type are given in <autoref label="ch:vspace"/></docref> @endxmlonly 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_AlignmentError The  @xmlonly <texttt text="vaddr"/> @endxmlonly  is not aligned to the page size of  @xmlonly <texttt text="_service"/> @endxmlonly . 
 * @retval seL4_DeleteFirst A mapping already exists in  @xmlonly <texttt text="vspace"/> @endxmlonly  at  @xmlonly <texttt text="vaddr"/> @endxmlonly . 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="vspace"/> @endxmlonly  does not have a paging structure at the required level mapped at  @xmlonly <texttt text="vaddr"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="vspace"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is already mapped in an IOSpace. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="_service"/> @endxmlonly  is already mapped in  @xmlonly <texttt text="vspace"/> @endxmlonly  at a different virtual address.
 * Or,  @xmlonly <texttt text="vaddr"/> @endxmlonly  is in the kernel virtual address range. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="vspace"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="vspace"/> @endxmlonly  is not assigned to an ASID pool.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is already mapped in a different VSpace. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_Page_Map(seL4_X86_Page _service, seL4_CPtr vspace, seL4_Word vaddr, seL4_CapRights_t rights, seL4_X86_VMAttributes attr)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86PageMap, 0, 1, 3);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, vspace);

	/* Marshal and initialise parameters. */
	mr0 = vaddr;
	mr1 = rights.words[0];
	mr2 = attr;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Unmap" label="x86_page_unmap"/> @endxmlonly
 * @brief @xmlonly Unmap a page. @endxmlonly
 * 
 * @xmlonly
 * Removes an existing mapping.
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page being operated on.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_Page_Unmap(seL4_X86_Page _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86PageUnmap, 0, 0, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#if defined(CONFIG_IOMMU)
/**
 * @xmlonly <manual name="Map I/O" label="x86_page_map_io"/> @endxmlonly
 * @brief @xmlonly Map a page into an IOSpace. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page being operated on.
 * @param[in] iospace The IOSpace that the frame is being mapped into 
 * @param[in] rights Rights for the mapping.  @xmlonly <docref>Possible values for this type are given in <autoref label="sec:cap_rights"/></docref> @endxmlonly 
 * @param[in] ioaddr The address that the frame is being mapped at. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst A mapping already exists in  @xmlonly <texttt text="iospace"/> @endxmlonly  at  @xmlonly <texttt text="ioaddr"/> @endxmlonly . 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="iospace"/> @endxmlonly  does not have a sufficient number of IO Page Tables mapped at  @xmlonly <texttt text="ioaddr"/> @endxmlonly . 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument No rights were specified in  @xmlonly <texttt text="rights"/> @endxmlonly .
 * Or, the rights in the  @xmlonly <texttt text="_service"/> @endxmlonly  capability do not include  @xmlonly <texttt text="rights"/> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="iospace"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is already mapped.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is not a page of size 4 KiB.
 * Or,  @xmlonly <texttt text="iospace"/> @endxmlonly  is not assigned to a PCI device. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_Page_MapIO(seL4_X86_Page _service, seL4_X86_IOSpace iospace, seL4_CapRights_t rights, seL4_Word ioaddr)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86PageMapIO, 0, 1, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, iospace);

	/* Marshal and initialise parameters. */
	mr0 = rights.words[0];
	mr1 = ioaddr;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
/**
 * @xmlonly <manual name="Get Address" label="x86_page_getaddress"/> @endxmlonly
 * @brief @xmlonly Get the physical address of the underlying frame. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page being operated on.
 * @return @xmlonly 
 *                 A <texttt text="seL4_IA32_Page_GetAddress_t"/> struct that contains a
 *                 <texttt text="seL4_Word paddr"/>, which holds the physical address of the page,
 *                 and <texttt text="int error"/>. <docref>See <autoref label="sec:errors"/> for a description
 *                 of the message register and tag contents upon error.</docref>
 *              @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_X86_Page_GetAddress_t
seL4_X86_Page_GetAddress(seL4_X86_Page _service)
{
	seL4_X86_Page_GetAddress_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86PageGetAddress, 0, 0, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result.error = seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result.error != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
		return result;
	}

	/* Unmarshal result. */
	result.paddr = mr0;
	return result;
}

#if defined(CONFIG_VTX)
/**
 * @xmlonly <manual name="Map EPT" label="x86_page_map_ept"/> @endxmlonly
 * @brief @xmlonly Map a page into a VCPU Extended Page Table (EPT). @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page being operated on.
 * @param[in] eptpml4 Capability to the EPT root which will contain the mapping. 
 * @param[in] gpa Guest physical address to map the page into. 
 * @param[in] rights Rights for the mapping. If these rights, masked with the rights of  @xmlonly <texttt text="_service"/> @endxmlonly , do not include read access, the mapping grants the guest no
 * access to the page.
 * @xmlonly <docref>Possible values for this type are given in <autoref label="sec:cap_rights"/>.</docref> @endxmlonly 
 * @param[in] attr VM attributes for the mapping. @xmlonly <docref> Possible values for this type are given in <autoref label="ch:vspace"/>.</docref> @endxmlonly 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_AlignmentError The  @xmlonly <texttt text="gpa"/> @endxmlonly  is not aligned to the page size of  @xmlonly <texttt text="_service"/> @endxmlonly . 
 * @retval seL4_DeleteFirst A mapping already exists in  @xmlonly <texttt text="eptpml4"/> @endxmlonly  at  @xmlonly <texttt text="gpa"/> @endxmlonly . 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="eptpml4"/> @endxmlonly  does not have a paging structure at the required level mapped at  @xmlonly <texttt text="gpa"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="eptpml4"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="eptpml4"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="eptpml4"/> @endxmlonly  is not assigned to an ASID pool.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is already mapped.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  has an unsupported page size. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_Page_MapEPT(seL4_X86_Page _service, seL4_X86_EPTPML4 eptpml4, seL4_Word gpa, seL4_CapRights_t rights, seL4_X86_EPT_VMAttributes attr)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86PageMapEPT, 0, 1, 3);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, eptpml4);

	/* Marshal and initialise parameters. */
	mr0 = gpa;
	mr1 = rights.words[0];
	mr2 = attr;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
/**
 * @xmlonly <manual name="Make Pool" label="x86_ASID_controlmakepool"/> @endxmlonly
 * @brief @xmlonly Create an X86 ASID pool. @endxmlonly
 * 
 * @xmlonly
 * Together with a capability to <texttt text="Untyped Memory"/>, which is passed as an argument,
 * create an <texttt text="ASID Pool"/>. The untyped capability must represent a
 * 4K memory object. This will create an ASID pool with enough space for 1024 VSpaces.
 * @endxmlonly
 * 
 * @param[in] _service The master ASIDControl capability.
 * @param[in] untyped Capability to an untyped memory object that will become the pool. Must be 4K bytes. 
 * @param[in] root CPtr to the CNode that forms the root of the destination CSpace. Must be at a depth equivalent to the wordsize. 
 * @param[in] index CPtr to the destination slot. Resolved from the root of the destination CSpace. 
 * @param[in] depth Number of bits of index to resolve to find the destination slot. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The destination slot contains a capability.
 * Or, there are no more ASID pools available. 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="index"/> @endxmlonly  or  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly .
 * Or,  @xmlonly <texttt text="root"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="untyped"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="untyped"/> @endxmlonly  is not the exact size of an ASID pool object.
 * Or,  @xmlonly <texttt text="untyped"/> @endxmlonly  is a device untyped  @xmlonly <docref>(see <autoref label="sec:kernmemalloc"/>)</docref> @endxmlonly . 
 * @retval seL4_RangeError The  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 * @retval seL4_RevokeFirst The  @xmlonly <texttt text="untyped"/> @endxmlonly  has been used to retype an object.
 * Or, a copy of the  @xmlonly <texttt text="untyped"/> @endxmlonly  capability exists. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_ASIDControl_MakePool(seL4_X86_ASIDControl _service, seL4_Untyped untyped, seL4_CNode root, seL4_Word index, seL4_Uint8 depth)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86ASIDControlMakePool, 0, 2, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, untyped);
	seL4_SetCap(1, root);

	/* Marshal and initialise parameters. */
	mr0 = index;
	mr1 = (depth & 0xffull);
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Assign" label="x86_asidpool_assign"/> @endxmlonly
 * @brief @xmlonly Assign an ASID pool. @endxmlonly
 * 
 * @xmlonly
 * Assigns an ASID to the VSpace associated with the <texttt text="Page Directory"/> passed in as an argument.
 * @endxmlonly
 * 
 * @param[in] _service The ASID pool which is being assigned to. Must not be full. Each ASID pool can contain 1024 entries.
 * @param[in] vspace The page directory that is being assigned to an ASID pool. Must not already be assigned to an ASID pool. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst There are no more ASIDs available in  @xmlonly <texttt text="_service"/> @endxmlonly . 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="vspace"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="vspace"/> @endxmlonly  is already assigned to an ASID pool. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_ASIDPool_Assign(seL4_X86_ASIDPool _service, seL4_CPtr vspace)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86ASIDPoolAssign, 0, 1, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, vspace);

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Issue" label="x86_ioport_issue"/> @endxmlonly
 * @brief @xmlonly Issue an IO port sub range. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:ioports"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Control capability for I/O ports.
 * @param[in] first_port First port of the range of the issued capability. 
 * @param[in] last_port Last port of the range of the issued capability. 
 * @param[in] root CPtr to the CNode that forms the root of the destination CSpace. 
 * @param[in] index CPtr to the destination slot. Resolved from the root of the destination CSpace. 
 * @param[in] depth Number of bits of index to resolve to find the destination slot. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The destination slot contains a capability. 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="index"/> @endxmlonly  or  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly .
 * Or,  @xmlonly <texttt text="root"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="last_port"/> @endxmlonly  is less than  @xmlonly <texttt text="first_port"/> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 * @retval seL4_RevokeFirst One or more ports in the requested range have already been issued. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_IOPortControl_Issue(seL4_X86_IOPortControl _service, seL4_Word first_port, seL4_Word last_port, seL4_CNode root, seL4_Word index, seL4_Uint8 depth)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86IOPortControlIssue, 0, 1, 4);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, root);

	/* Marshal and initialise parameters. */
	mr0 = first_port;
	mr1 = last_port;
	mr2 = index;
	mr3 = (depth & 0xffull);

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="In8" label="x86_io_port_in8"/> @endxmlonly
 * @brief @xmlonly Read 8 bits from an IO port. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:ioports"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service An I/O Port capability.
 * @param[in] port The port to read from. 
 * @return @xmlonly 
 *                 A <texttt text="seL4_X86_IOPort_In8_t"/> structure <docref>as described in <autoref label="sec:ioports"/>.</docref>
 *              @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, reading from  @xmlonly <texttt text="port"/> @endxmlonly  is not authorized by the capability. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_X86_IOPort_In8_t
seL4_X86_IOPort_In8(seL4_X86_IOPort _service, seL4_Uint16 port)
{
	seL4_X86_IOPort_In8_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86IOPortIn8, 0, 0, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = (port & 0xffffull);
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result.error = seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result.error != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
		return result;
	}

	/* Unmarshal result. */
	result.result = (mr0 & 0xff);
	return result;
}

/**
 * @xmlonly <manual name="In16" label="x86_io_port_in16"/> @endxmlonly
 * @brief @xmlonly Read 16 bits from an IO port. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:ioports"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service An I/O Port capability.
 * @param[in] port The port to read from. 
 * @return @xmlonly 
 *                 A <texttt text="seL4_X86_IOPort_In16_t"/> structure <docref>as described in <autoref label="sec:ioports"/>.</docref>
 *              @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, reading from  @xmlonly <texttt text="port"/> @endxmlonly  and  @xmlonly <texttt text="port+1"/> @endxmlonly  is not authorized by the capability. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_X86_IOPort_In16_t
seL4_X86_IOPort_In16(seL4_X86_IOPort _service, seL4_Uint16 port)
{
	seL4_X86_IOPort_In16_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86IOPortIn16, 0, 0, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = (port & 0xffffull);
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result.error = seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result.error != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
		return result;
	}

	/* Unmarshal result. */
	result.result = (mr0 & 0xffff);
	return result;
}

/**
 * @xmlonly <manual name="In32" label="x86_io_port_in32"/> @endxmlonly
 * @brief @xmlonly Read 32 bits from an IO port. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:ioports"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service An I/O Port capability.
 * @param[in] port The port to read from. 
 * @return @xmlonly 
 *                 A <texttt text="seL4_X86_IOPort_In32_t"/> structure <docref>as described in <autoref label="sec:ioports"/>.</docref>
 *              @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, reading from ports  @xmlonly <texttt text="port"/> @endxmlonly  through  @xmlonly <texttt text="port+3"/> @endxmlonly  is not authorized by the capability. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_X86_IOPort_In32_t
seL4_X86_IOPort_In32(seL4_X86_IOPort _service, seL4_Uint16 port)
{
	seL4_X86_IOPort_In32_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86IOPortIn32, 0, 0, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = (port & 0xffffull);
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result.error = seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result.error != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
		return result;
	}

	/* Unmarshal result. */
	result.result = (mr0 & 0xffffffff);
	return result;
}

/**
 * @xmlonly <manual name="Out8" label="x86_io_port_out8"/> @endxmlonly
 * @brief @xmlonly Write 8 bits to an IO port. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:ioports"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service An I/O Port capability.
 * @param[in] port The port to write to. 
 * @param[in] data Data to write to the IO port. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, writing to  @xmlonly <texttt text="port"/> @endxmlonly  is not authorized by the capability. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_IOPort_Out8(seL4_X86_IOPort _service, seL4_Word port, seL4_Word data)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86IOPortOut8, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = port;
	mr1 = data;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Out16" label="x86_io_port_out16"/> @endxmlonly
 * @brief @xmlonly Write 16 bits to an IO port. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:ioports"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service An I/O Port capability.
 * @param[in] port The port to write to. 
 * @param[in] data Data to write to the IO port. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, writing to  @xmlonly <texttt text="port"/> @endxmlonly  and  @xmlonly <texttt text="port+1"/> @endxmlonly  is not authorized by the capability. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_IOPort_Out16(seL4_X86_IOPort _service, seL4_Word port, seL4_Word data)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86IOPortOut16, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = port;
	mr1 = data;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Out32" label="x86_io_port_out32"/> @endxmlonly
 * @brief @xmlonly Write 32 bits to an IO port. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:ioports"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service An I/O Port capability.
 * @param[in] port The port to write to. 
 * @param[in] data Data to write to the IO port. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, writing to ports  @xmlonly <texttt text="port"/> @endxmlonly  through  @xmlonly <texttt text="port+3"/> @endxmlonly  is not authorized by the capability. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_IOPort_Out32(seL4_X86_IOPort _service, seL4_Word port, seL4_Word data)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86IOPortOut32, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = port;
	mr1 = data;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Get I/O APIC Handler" label="x86_irq_control_get_io_apic_handler"/> @endxmlonly
 * @brief @xmlonly Create an IRQ handler capability for an interrupt from an IOAPIC. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:interrupts"/> and <autoref label="sec:x86_interrupts"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service An IRQControl capability. This gives you the authority to make this call.
 * @param[in] root CPtr to the CNode that forms the root of the destination CSpace. Must be at a depth equivalent to the wordsize. 
 * @param[in] index CPtr to the destination slot. Resolved from the root of the destination CSpace. 
 * @param[in] depth Number of bits of index to resolve to find the destination slot. 
 * @param[in] ioapic Zero based index of IOAPIC to get interrupt from,                              ordered the same as in ACPI tables 
 * @param[in] pin IOAPIC pin that generates the interrupt. 
 * @param[in] level Indicates whether the IOAPIC should be programmed to treat this interrupt as level triggered. 
 * @param[in] polarity Indicates whether the IOAPIC should be programmed to treat this interrupt as high or                      low triggered 
 * @param[in] vector CPU vector to deliver the interrupt to. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The destination slot contains a capability. 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="index"/> @endxmlonly  or  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly .
 * Or,  @xmlonly <texttt text="root"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, an IOAPIC is not in use. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="vector"/> @endxmlonly ,  @xmlonly <texttt text="ioapic"/> @endxmlonly , or  @xmlonly <texttt text="pin"/> @endxmlonly  is invalid.
 * Or,  @xmlonly <texttt text="level"/> @endxmlonly  or  @xmlonly <texttt text="polarity"/> @endxmlonly  is not 0 or 1.
 * Or,  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 * @retval seL4_RevokeFirst An IRQ handler capability for  @xmlonly <texttt text="vector"/> @endxmlonly  has already been created. 
 */
LIBSEL4_INLINE seL4_Error
seL4_IRQControl_GetIOAPIC(seL4_IRQControl _service, seL4_CNode root, seL4_Word index, seL4_Uint8 depth, seL4_Word ioapic, seL4_Word pin, seL4_Word level, seL4_Word polarity, seL4_Word vector)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86IRQIssueIRQHandlerIOAPIC, 0, 1, 7);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, root);

	/* Marshal and initialise parameters. */
	mr0 = index;
	mr1 = (depth & 0xffull);
	mr2 = ioapic;
	mr3 = pin;
	seL4_SetMR(4, level);
	seL4_SetMR(5, polarity);
	seL4_SetMR(6, vector);

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Get MSI Handler" label="x86_irq_control_get_msi_handler"/> @endxmlonly
 * @brief @xmlonly Create an IRQ handler capability for an interrupt from an MSI. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:interrupts"/> and <autoref label="sec:x86_interrupts"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service An IRQControl capability. This gives you the authority to make this call.
 * @param[in] root CPtr to the CNode that forms the root of the destination CSpace. Must be at a depth equivalent to the wordsize. 
 * @param[in] index CPtr to the destination slot. Resolved from the root of the destination CSpace. 
 * @param[in] depth Number of bits of index to resolve to find the destination slot. 
 * @param[in] pci_bus PCI bus ID of the device that will generate the interrupt. 
 * @param[in] pci_dev PCI device ID of the device that will generate the interrupt. 
 * @param[in] pci_func PCI function ID of the device that will generate the interrupt. 
 * @param[in] handle Value of the handle programmed into the data portion of the MSI. 
 * @param[in] vector CPU vector to deliver the interrupt to. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The destination slot contains a capability. 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="index"/> @endxmlonly  or  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly .
 * Or,  @xmlonly <texttt text="root"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, an IOAPIC is not in use. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="vector"/> @endxmlonly ,  @xmlonly <texttt text="pic_bus"/> @endxmlonly ,  @xmlonly <texttt text="pci_dev"/> @endxmlonly , or  @xmlonly <texttt text="pci_func"/> @endxmlonly  is invalid.
 * Or, the  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 * @retval seL4_RevokeFirst An IRQ handler capability for  @xmlonly <texttt text="vector"/> @endxmlonly  has already been created. 
 */
LIBSEL4_INLINE seL4_Error
seL4_IRQControl_GetMSI(seL4_IRQControl _service, seL4_CNode root, seL4_Word index, seL4_Uint8 depth, seL4_Word pci_bus, seL4_Word pci_dev, seL4_Word pci_func, seL4_Word handle, seL4_Word vector)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86IRQIssueIRQHandlerMSI, 0, 1, 7);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, root);

	/* Marshal and initialise parameters. */
	mr0 = index;
	mr1 = (depth & 0xffull);
	mr2 = pci_bus;
	mr3 = pci_dev;
	seL4_SetMR(4, pci_func);
	seL4_SetMR(5, handle);
	seL4_SetMR(6, vector);

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#if defined(CONFIG_VTX)
/**
 * @xmlonly <manual name="Set EPT Root" label="x86_set_eptroot"/> @endxmlonly
 * @brief @xmlonly Set the EPT root of a thread @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:virt"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] eptpml4 CPtr to an EPT PML4 object to act as the guest mode vspace root 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="eptpml4"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="eptpml4"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_SetEPTRoot(seL4_TCB _service, seL4_X86_EPTPML4 eptpml4)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBSetEPTRoot, 0, 1, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, eptpml4);

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_VTX)
/**
 * @xmlonly <manual name="Set TCB" label="x86_vcpu_set_tcb"/> @endxmlonly
 * @brief @xmlonly Bind TCB to VCPU @endxmlonly
 * 
 * @xmlonly
 * Configures the one-to-one binding of a VCPU and TCB, overwriting any previous binding
 * in both. <docref>See <autoref label="sec:virt"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service VCPU object to operate on
 * @param[in] tcb CPtr of the TCB to bind to 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="tcb"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_VCPU_SetTCB(seL4_X86_VCPU _service, seL4_TCB tcb)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86VCPUSetTCB, 0, 1, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, tcb);

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_VTX)
/**
 * @xmlonly <manual name="Read VMCS" label="x86_vcpu_readvmcs"/> @endxmlonly
 * @brief @xmlonly Read VMCS field from the hardware @endxmlonly
 * 
 * @xmlonly
 * Thin wrapper around the <texttt text="vmread"/> instruction that is performed on the
 * VMCS region that is part of the VCPU object. After validating that a legal
 * field is requested the value of `vmread` is returned directly in the result.
 * @endxmlonly
 * 
 * @param[in] _service VCPU object to operate on
 * @param[in] field Field to give to `vmread` instruction 
 * @return @xmlonly 
 *                 A <texttt text="seL4_X86_VCPU_ReadVMCS_t"/> struct that contains a
 *                 <texttt text="seL4_Word value"/>, which holds the return result of the <texttt text="vmread"/> instruction,
 *                 and <texttt text="int error"/>. <docref>See <autoref label="sec:errors"/> for a description
 *                 of the message register and tag contents upon error.</docref>
 *              @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="field"/> @endxmlonly  is invalid or unsupported. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_X86_VCPU_ReadVMCS_t
seL4_X86_VCPU_ReadVMCS(seL4_X86_VCPU _service, seL4_Word field)
{
	seL4_X86_VCPU_ReadVMCS_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86VCPUReadVMCS, 0, 0, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = field;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result.error = seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result.error != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
		return result;
	}

	/* Unmarshal result. */
	result.value = mr0;
	return result;
}

#endif
#if defined(CONFIG_VTX)
/**
 * @xmlonly <manual name="Write VMCS" label="x86_vcpu_writevmcs"/> @endxmlonly
 * @brief @xmlonly Write VMCS field to the hardware @endxmlonly
 * 
 * @xmlonly
 * Thin wrapper around the `vmwrite` instruction that is performed on the
 * VMCS region that is part of the VCPU object. As well as validating that
 * a legal field is requested, the value may be modified to ensure any
 * bits that are fixed in the hardware are correct, and that any features
 * required for kernel correctness are not disabled <docref>(see <autoref label="sec:virt"/>)</docref>.
 * 
 * The final value written to the hardware is returned and can be compared
 * to the input parameter to determine what bits the kernel changed.
 * @endxmlonly
 * 
 * @param[in] _service VCPU object to operate on
 * @param[in] field Field to give to `vmwrite` instruction 
 * @param[in] value Value to write using `vmwrite` instruction 
 * @return @xmlonly 
 *                 A <texttt text="seL4_X86_VCPU_WriteVMCS_t"/> struct that contains a
 *                 <texttt text="seL4_Word written"/>, which holds the final value written with the <texttt text="vmwrite"/> instruction,
 *                 and <texttt text="int error"/>. <docref>See <autoref label="sec:errors"/> for a description
 *                 of the message register and tag contents upon error.</docref>
 *              @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="field"/> @endxmlonly  is invalid or unsupported. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_X86_VCPU_WriteVMCS_t
seL4_X86_VCPU_WriteVMCS(seL4_X86_VCPU _service, seL4_Word field, seL4_Word value)
{
	seL4_X86_VCPU_WriteVMCS_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86VCPUWriteVMCS, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = field;
	mr1 = value;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result.error = seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result.error != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
		return result;
	}

	/* Unmarshal result. */
	result.written = mr0;
	return result;
}

#endif
#if defined(CONFIG_VTX)
/**
 * @xmlonly <manual name="Enable I/O Port" label="x86_vcpu_enableioport"/> @endxmlonly
 * @brief @xmlonly Enable I/O port range in guest execution @endxmlonly
 * 
 * @xmlonly
 * Enables a range of I/O ports for direct access by the execution mode in
 * the <texttt text="VCPU"/>. The requested port range must be a sub range
 * of the provided I/O port capability.
 * 
 * This also establishes a link between the provided I/O port capability and
 * the <texttt text="VCPU"/><docref>, see <autoref label="sec:virt"/> for details.</docref>
 * @endxmlonly
 * 
 * @param[in] _service VCPU object to operate on
 * @param[in] ioPort I/O port capability whose authority is being delegating 
 * @param[in] low Start of the I/O port range to enable 
 * @param[in] high Last I/O port in the range to enable 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="ioPort"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="low"/> @endxmlonly  or  @xmlonly <texttt text="high"/> @endxmlonly  IO port exceeds the range authorized by  @xmlonly <texttt text="ioPort"/> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_VCPU_EnableIOPort(seL4_X86_VCPU _service, seL4_X86_IOPort ioPort, seL4_Word low, seL4_Word high)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86VCPUEnableIOPort, 0, 1, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, ioPort);

	/* Marshal and initialise parameters. */
	mr0 = low;
	mr1 = high;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_VTX)
/**
 * @xmlonly <manual name="Disable I/O Port" label="x86_vcpu_disable_io_port"/> @endxmlonly
 * @brief @xmlonly Disable I/O port range in privileged execution @endxmlonly
 * 
 * @xmlonly
 * Disable a range of I/O ports for direct access by the execution mode in
 * the <texttt text="VCPU"/>.
 * @endxmlonly
 * 
 * @param[in] _service VCPU object to operate on
 * @param[in] low Start of the I/O port range to disable 
 * @param[in] high Last I/O port in the range to disable 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_VCPU_DisableIOPort(seL4_X86_VCPU _service, seL4_Word low, seL4_Word high)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86VCPUDisableIOPort, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = low;
	mr1 = high;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_VTX)
/**
 * @xmlonly <manual name="Write Registers" label="x86_vcpu_write_registers"/> @endxmlonly
 * @brief @xmlonly Set guest mode registers to the fields of a given <texttt text="seL4_VCPUContext"/> @endxmlonly
 * 
 * @xmlonly
 * Sets the guest mode registers, which is any registers not already part of the VMCS.
 * @endxmlonly
 * 
 * @param[in] _service VCPU object to operate on
 * @param[in] regs Data structure containing the new register values. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_VCPU_WriteRegisters(seL4_X86_VCPU _service, seL4_VCPUContext *regs)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86VCPUWriteRegisters, 0, 0, 15);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = regs->eax;
	mr1 = regs->ebx;
	mr2 = regs->ecx;
	mr3 = regs->edx;
	seL4_SetMR(4, regs->esi);
	seL4_SetMR(5, regs->edi);
	seL4_SetMR(6, regs->ebp);
	seL4_SetMR(7, regs->r8);
	seL4_SetMR(8, regs->r9);
	seL4_SetMR(9, regs->r10);
	seL4_SetMR(10, regs->r11);
	seL4_SetMR(11, regs->r12);
	seL4_SetMR(12, regs->r13);
	seL4_SetMR(13, regs->r14);
	seL4_SetMR(14, regs->r15);

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_VTX)
/**
 * @xmlonly <manual name="Map" label="x86_extended_page_table_page-directory_pointers_table_map"/> @endxmlonly
 * @brief @xmlonly Map a Page-Directory Pointers Table (PDPT) into a VCPU Extended Page Table. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the EPT PDPT being operated on.
 * @param[in] eptpml4 Capability to the EPT root which will contain the mapping 
 * @param[in] gpa Guest physical address to map the page into. 
 * @param[in] attr VM attributes for the mapping.  @xmlonly <docref>Possible values for this type are given in <autoref label="ch:vspace"/></docref> @endxmlonly 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst A mapping already exists for this level in  @xmlonly <texttt text="eptpml4"/> @endxmlonly  at  @xmlonly <texttt text="gpa"/> @endxmlonly . 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="eptpml4"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="eptpml4"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is already mapped in a VSpace.
 * Or,  @xmlonly <texttt text="eptpml4"/> @endxmlonly  is not assigned to an ASID pool. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_EPTPDPT_Map(seL4_X86_EPTPDPT _service, seL4_X86_EPTPML4 eptpml4, seL4_Word gpa, seL4_X86_EPT_VMAttributes attr)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86EPTPDPTMap, 0, 1, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, eptpml4);

	/* Marshal and initialise parameters. */
	mr0 = gpa;
	mr1 = attr;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_VTX)
/**
 * @xmlonly <manual name="Unmap" label="x86_extended_page_table_page-directory_pointers_table_unmap"/> @endxmlonly
 * @brief @xmlonly Unmap a Page-Directory Pointer Table (PDPT) into a VCPU Extended Page Table. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the EPT PDPT being operated on.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RevokeFirst A copy of the  @xmlonly <texttt text="_service"/> @endxmlonly  capability exists. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_EPTPDPT_Unmap(seL4_X86_EPTPDPT _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86EPTPDPTUnmap, 0, 0, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_VTX)
/**
 * @xmlonly <manual name="Map" label="x86_extended_page_table_page_directory_map"/> @endxmlonly
 * @brief @xmlonly Map a Page Directory (PD) into a VCPU Extended Page Table. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the EPT PD being operated on.
 * @param[in] eptpml4 Capability to the EPT root which will contain the mapping 
 * @param[in] gpa Guest physical address to map the page into. 
 * @param[in] attr VM attributes for the mapping.  @xmlonly <docref>Possible values for this type are given in <autoref label="ch:vspace"/></docref> @endxmlonly 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst A mapping already exists for this level in  @xmlonly <texttt text="eptpml4"/> @endxmlonly  at  @xmlonly <texttt text="gpa"/> @endxmlonly . 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="eptpml4"/> @endxmlonly  is not assigned to an ASID pool.
 * Or,  @xmlonly <texttt text="eptpml4"/> @endxmlonly  does not have an EPTPDPT mapped at  @xmlonly <texttt text="gpa"/> @endxmlonly . 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="eptpml4"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is already mapped in a VSpace.
 * Or,  @xmlonly <texttt text="eptpml4"/> @endxmlonly  is not assigned to an ASID pool. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_EPTPD_Map(seL4_X86_EPTPD _service, seL4_X86_EPTPML4 eptpml4, seL4_Word gpa, seL4_X86_EPT_VMAttributes attr)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86EPTPDMap, 0, 1, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, eptpml4);

	/* Marshal and initialise parameters. */
	mr0 = gpa;
	mr1 = attr;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_VTX)
/**
 * @xmlonly <manual name="Unmap" label="x86_extended_page_table_page_directory_unmap"/> @endxmlonly
 * @brief @xmlonly Unmap a Page Directory (PD) from a VCPU Extended Page Table. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the EPT PD being operated on.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RevokeFirst A copy of the  @xmlonly <texttt text="_service"/> @endxmlonly  capability exists. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_EPTPD_Unmap(seL4_X86_EPTPD _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86EPTPDUnmap, 0, 0, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_VTX)
/**
 * @xmlonly <manual name="Map" label="x86_extended_page_table_page_table_map"/> @endxmlonly
 * @brief @xmlonly Map a Page Table (PT) into a VCPU Extended Page Table. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the EPT PT being operated on.
 * @param[in] eptpml4 Capability to the EPT root which will contain the mapping 
 * @param[in] gpa Guest physical address to map the page into. 
 * @param[in] attr VM attributes for the mapping.  @xmlonly <docref>Possible values for this type are given in <autoref label="ch:vspace"/></docref> @endxmlonly 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst A mapping already exists for this level in  @xmlonly <texttt text="eptpml4"/> @endxmlonly  at  @xmlonly <texttt text="gpa"/> @endxmlonly . 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="eptpml4"/> @endxmlonly  is not assigned to an ASID pool.
 * Or,  @xmlonly <texttt text="eptpml4"/> @endxmlonly  does not have an EPTPD mapped at  @xmlonly <texttt text="gpa"/> @endxmlonly . 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="eptpml4"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is already mapped in a VSpace.
 * Or,  @xmlonly <texttt text="eptpml4"/> @endxmlonly  is not assigned to an ASID pool. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_EPTPT_Map(seL4_X86_EPTPT _service, seL4_X86_EPTPML4 eptpml4, seL4_Word gpa, seL4_X86_EPT_VMAttributes attr)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86EPTPTMap, 0, 1, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, eptpml4);

	/* Marshal and initialise parameters. */
	mr0 = gpa;
	mr1 = attr;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_VTX)
/**
 * @xmlonly <manual name="Unmap" label="x86_extended_page_table_page_table_unmap"/> @endxmlonly
 * @brief @xmlonly Unmap a Page Table (PT) from a VCPU Extended Page Table. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the EPT PT being operated on.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RevokeFirst A copy of the  @xmlonly <texttt text="_service"/> @endxmlonly  capability exists. 
 */
LIBSEL4_INLINE seL4_Error
seL4_X86_EPTPT_Unmap(seL4_X86_EPTPT _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(X86EPTPTUnmap, 0, 0, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
/**
 * @xmlonly <manual name="Retype" label="untyped_retype"/> @endxmlonly
 * @brief @xmlonly Retype an untyped object @endxmlonly
 * 
 * @xmlonly
 * Given a capability, <texttt text="_service"/>, to an untyped object,
 * creates <texttt text="num_objects"/> of the requested type. Creates
 * <texttt text="num_objects"/> capabilities to the new objects starting
 * at <texttt text="node_offset"/> in the CNode specified by
 * <texttt text="root"/>, <texttt text="node_index"/>, and
 * <texttt text="node_depth"/>.
 * 
 * For variable-sized
 * kernel objects, the <texttt text="size_bits"/> argument is used to
 * determine the size of objects to create. The relationship between
 * <texttt text="size_bits"/> and object size depends on the type of object
 * being created. <docref>See <autoref label="sec:object_sizes"/> for more information
 * about object sizes.</docref>
 * 
 * <docref>See <autoref label="sec:kernmemalloc"/> for more information about how untyped
 * memory is retyped.</docref>
 * 
 * <docref>See <autoref label="sec:caps_to_new_objects"/> for more information about the
 * placement of capabilities to created objects.</docref>
 * @endxmlonly
 * 
 * @param[in] _service CPtr to an untyped object.
 * @param[in] type The seL4 object type that we are retyping to. 
 * @param[in] size_bits Used to determine the size of variable-sized objects. 
 * @param[in] root CPtr to the CNode at the root of the destination CSpace. 
 * @param[in] node_index CPtr to the destination CNode. Resolved relative to the root parameter. 
 * @param[in] node_depth Number of bits of node_index to translate when addressing the destination CNode. 
 * @param[in] node_offset Number of slots into the node at which capabilities start being placed. 
 * @param[in] num_objects Number of capabilities to create. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst A capability exists in the destination window of the CNode. 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="root"/> @endxmlonly ,  @xmlonly <texttt text="node_index"/> @endxmlonly , or  @xmlonly <texttt text="node_depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="size_bits"/> @endxmlonly  is too big or too small for the requested object type.
 * Or,  @xmlonly <texttt text="type"/> @endxmlonly  cannot be created from a device untyped  @xmlonly <docref>(see <autoref label="sec:kernmemalloc"/>)</docref> @endxmlonly .
 * Or, the requested object  @xmlonly <texttt text="type"/> @endxmlonly  does not exist. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_NotEnoughMemory The total size of the new objects exceeds the space available. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="num_objects"/> @endxmlonly  do not fit in the destination CNode at  @xmlonly <texttt text="node_offset"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="num_objects"/> @endxmlonly  is greater than  @xmlonly <texttt text="CONFIG_RETYPE_FAN_OUT_LIMIT"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="size_bits"/> @endxmlonly  is too large. 
 */
LIBSEL4_INLINE seL4_Error
seL4_Untyped_Retype(seL4_Untyped _service, seL4_Word type, seL4_Word size_bits, seL4_CNode root, seL4_Word node_index, seL4_Word node_depth, seL4_Word node_offset, seL4_Word num_objects)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(UntypedRetype, 0, 1, 6);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, root);

	/* Marshal and initialise parameters. */
	mr0 = type;
	mr1 = size_bits;
	mr2 = node_index;
	mr3 = node_depth;
	seL4_SetMR(4, node_offset);
	seL4_SetMR(5, num_objects);

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Read Registers" label="tcb_readregisters"/> @endxmlonly
 * @brief @xmlonly Read a thread's registers into the first <texttt text="count"/> fields of a given
 * seL4_UserContext @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:read_write_registers"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] suspend_source The invocation should also suspend the source thread. 
 * @param[in] arch_flags Architecture dependent flags. These have no meaning on x86, ARM, and RISC-V. 
 * @param[in] count The number of registers to read. 
 * @param[out] regs The structure to read the registers into. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is the current thread's TCB. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="count"/> @endxmlonly  requested too few or too many registers. 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_ReadRegisters(seL4_TCB _service, seL4_Bool suspend_source, seL4_Uint8 arch_flags, seL4_Word count, seL4_UserContext *regs)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBReadRegisters, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = (suspend_source & 0x1ull) | ((arch_flags & 0xffull) << 8);
	mr1 = count;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	/* Unmarshal result. */
	regs->rip = mr0;
	regs->rsp = mr1;
	regs->rflags = mr2;
	regs->rax = mr3;
	for (seL4_Word i = 4; i < count; i++) {
		((seL4_Word *) &regs->rbx)[i - 4] = seL4_GetMR(i);
	}
	return result;
}

/**
 * @xmlonly <manual name="Write Registers" label="tcb_writeregisters"/> @endxmlonly
 * @brief @xmlonly Set a thread's registers to the first <texttt text="count"/> fields of a given seL4_UserContext @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:read_write_registers"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] resume_target The invocation should also resume the destination thread. 
 * @param[in] arch_flags Architecture dependent flags. These have no meaning on x86, ARM, and RISC-V. 
 * @param[in] count The number of registers to be set. 
 * @param[in] regs Data structure containing the new register values. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is the current thread's TCB. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_WriteRegisters(seL4_TCB _service, seL4_Bool resume_target, seL4_Uint8 arch_flags, seL4_Word count, seL4_UserContext *regs)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBWriteRegisters, 0, 0, 22);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = (resume_target & 0x1ull) | ((arch_flags & 0xffull) << 8);
	mr1 = count;
	mr2 = regs->rip;
	mr3 = regs->rsp;
	for (seL4_Word i = 2; i < count; i++) {
		seL4_SetMR(i + 2, ((seL4_Word*)&regs->rflags)[i - 2]);
	}

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Copy Registers" label="tcb_copyregisters"/> @endxmlonly
 * @brief @xmlonly Copy the registers from one thread to another @endxmlonly
 * 
 * @xmlonly
 * In the context of this function, frame registers are those that are read, modified or preserved by a
 * system call and integer registers are those that are not. Refer to the seL4 userland library source for specifics.
 * <docref><autoref label="sec:thread_deactivation"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on. This is the destination TCB.
 * @param[in] source Cap to the source TCB. 
 * @param[in] suspend_source The invocation should also suspend the source thread. 
 * @param[in] resume_target The invocation should also resume the destination thread. 
 * @param[in] transfer_frame Frame registers should be transferred. 
 * @param[in] transfer_integer Integer registers should be transferred. 
 * @param[in] arch_flags Architecture dependent flags. These have no meaning on x86, ARM, and RISC-V. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="source"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_CopyRegisters(seL4_TCB _service, seL4_TCB source, seL4_Bool suspend_source, seL4_Bool resume_target, seL4_Bool transfer_frame, seL4_Bool transfer_integer, seL4_Uint8 arch_flags)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBCopyRegisters, 0, 1, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, source);

	/* Marshal and initialise parameters. */
	mr0 = (suspend_source & 0x1ull) | ((resume_target & 0x1ull) << 1) | ((transfer_frame & 0x1ull) << 2) | ((transfer_integer & 0x1ull) << 3) | ((arch_flags & 0xffull) << 8);
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#if !defined(CONFIG_KERNEL_MCS)
/**
 * @xmlonly <manual name="Configure" label="tcb_configure"/> @endxmlonly
 * @brief @xmlonly Set the parameters of a TCB @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:threads"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] fault_ep CPtr to the endpoint which receives IPCs when this thread faults. This capability is in the CSpace of the thread being configured. 
 * @param[in] cspace_root The new CSpace root. 
 * @param[in] cspace_root_data Optionally set the guard and guard size of the new root CNode. If set to zero, this parameter has no effect. 
 * @param[in] vspace_root The new VSpace root. 
 * @param[in] vspace_root_data Has no effect on x86 or ARM processors. 
 * @param[in] buffer Location of the thread's IPC buffer. Must be aligned to  @xmlonly <texttt text="seL4_IPCBufferSizeBits"/> @endxmlonly . The IPC buffer may not cross a page boundary. 
 * @param[in] bufferFrame Capability to a page containing the thread's IPC buffer. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly ,  @xmlonly <texttt text="bufferFrame"/> @endxmlonly ,  @xmlonly <texttt text="cspace_root"/> @endxmlonly , or  @xmlonly <texttt text="vspace_root"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="vspace_root"/> @endxmlonly  is not assigned to an ASID pool.
 * Or,  @xmlonly <texttt text="cspace_root_data"/> @endxmlonly  is invalid.
 * Or,  @xmlonly <texttt text="buffer"/> @endxmlonly  is not aligned.
 * Or,  @xmlonly <texttt text="bufferFrame"/> @endxmlonly  is retyped from a device untyped  @xmlonly <docref>(see <autoref label="sec:kernmemalloc"/>)</docref> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RevokeFirst The  @xmlonly <texttt text="bufferFrame"/> @endxmlonly ,  @xmlonly <texttt text="cspace_root"/> @endxmlonly , or  @xmlonly <texttt text="vspace_root"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_Configure(seL4_TCB _service, seL4_Word fault_ep, seL4_CNode cspace_root, seL4_Word cspace_root_data, seL4_CPtr vspace_root, seL4_Word vspace_root_data, seL4_Word buffer, seL4_CPtr bufferFrame)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBConfigure, 0, 3, 4);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, cspace_root);
	seL4_SetCap(1, vspace_root);
	seL4_SetCap(2, bufferFrame);

	/* Marshal and initialise parameters. */
	mr0 = fault_ep;
	mr1 = cspace_root_data;
	mr2 = vspace_root_data;
	mr3 = buffer;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_KERNEL_MCS)
/**
 * @xmlonly <manual name="Configure (MCS)" label="tcb_configure_mcs"/> @endxmlonly
 * @brief @xmlonly Set the parameters of a TCB @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:threads"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] cspace_root The new CSpace root. 
 * @param[in] cspace_root_data Optionally set the guard and guard size of the new root CNode. If set to zero, this parameter has no effect. 
 * @param[in] vspace_root The new VSpace root. 
 * @param[in] vspace_root_data Has no effect on x86 or ARM processors. 
 * @param[in] buffer Location of the thread's IPC buffer. Must be aligned to  @xmlonly <texttt text="seL4_IPCBufferSizeBits"/> @endxmlonly . The IPC buffer may not cross a page boundary. 
 * @param[in] bufferFrame Capability to a page containing the thread's IPC buffer. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_AlignmentError The  @xmlonly <texttt text="buffer"/> @endxmlonly  is not aligned. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly ,  @xmlonly <texttt text="bufferFrame"/> @endxmlonly ,  @xmlonly <texttt text="cspace_root"/> @endxmlonly , or  @xmlonly <texttt text="vspace_root"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="vspace_root"/> @endxmlonly  is not assigned to an ASID pool.
 * Or,  @xmlonly <texttt text="cspace_root_data"/> @endxmlonly  is invalid.
 * Or,  @xmlonly <texttt text="bufferFrame"/> @endxmlonly  is retyped from a device untyped  @xmlonly <docref>(see <autoref label="sec:kernmemalloc"/>)</docref> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RevokeFirst The  @xmlonly <texttt text="bufferFrame"/> @endxmlonly ,  @xmlonly <texttt text="cspace_root"/> @endxmlonly , or  @xmlonly <texttt text="vspace_root"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_Configure(seL4_TCB _service, seL4_CNode cspace_root, seL4_Word cspace_root_data, seL4_CPtr vspace_root, seL4_Word vspace_root_data, seL4_Word buffer, seL4_CPtr bufferFrame)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBConfigure, 0, 3, 3);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, cspace_root);
	seL4_SetCap(1, vspace_root);
	seL4_SetCap(2, bufferFrame);

	/* Marshal and initialise parameters. */
	mr0 = cspace_root_data;
	mr1 = vspace_root_data;
	mr2 = buffer;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
/**
 * @xmlonly <manual name="Set Priority" label="tcb_setpriority"/> @endxmlonly
 * @brief @xmlonly Change a thread's priority @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:sched"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] authority Capability to the TCB to use the MCP from when setting the priority. 
 * @param[in] priority The thread's new priority. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="authority"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="priority"/> @endxmlonly  is greater than the maximum controlled priority of  @xmlonly <texttt text="authority"/> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_SetPriority(seL4_TCB _service, seL4_TCB authority, seL4_Word priority)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBSetPriority, 0, 1, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, authority);

	/* Marshal and initialise parameters. */
	mr0 = priority;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Set Maximum Controlled Priority" label="tcb_setmcpriority"/> @endxmlonly
 * @brief @xmlonly Change a thread's maximum controlled priority @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:sched"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] authority Capability to the TCB to use the MCP from when setting the MCP. 
 * @param[in] mcp The thread's new maximum controlled priority. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="authority"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="mcp"/> @endxmlonly  is greater than the maximum controlled priority of  @xmlonly <texttt text="authority"/> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_SetMCPriority(seL4_TCB _service, seL4_TCB authority, seL4_Word mcp)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBSetMCPriority, 0, 1, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, authority);

	/* Marshal and initialise parameters. */
	mr0 = mcp;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#if !defined(CONFIG_KERNEL_MCS)
/**
 * @xmlonly <manual name="Set Sched Params" label="tcb_setschedparams"/> @endxmlonly
 * @brief @xmlonly Change a thread's priority and maximum controlled priority. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:sched"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] authority Capability to the TCB to use the MCP from when setting the priority and MCP. 
 * @param[in] mcp The thread's new maximum controlled priority. 
 * @param[in] priority The thread's new priority. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="authority"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="mcp"/> @endxmlonly  is greater than the maximum controlled priority of  @xmlonly <texttt text="authority"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="priority"/> @endxmlonly  is greater than the maximum controlled priority of  @xmlonly <texttt text="authority"/> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_SetSchedParams(seL4_TCB _service, seL4_TCB authority, seL4_Word mcp, seL4_Word priority)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBSetSchedParams, 0, 1, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, authority);

	/* Marshal and initialise parameters. */
	mr0 = mcp;
	mr1 = priority;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_KERNEL_MCS)
/**
 * @xmlonly <manual name="Set Sched Params (MCS)" label="tcb_setschedparams_mcs"/> @endxmlonly
 * @brief @xmlonly Change a thread's priority, maximum controlled priority, scheduling context
 * and fault handler. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:sched"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] authority Capability to the TCB to use the MCP from when setting the priority and MCP. 
 * @param[in] mcp The thread's new maximum controlled priority. 
 * @param[in] priority The thread's new priority. 
 * @param[in] sched_context Capability to the scheduling context that the TCB should run on. If the scheduling context is already bound to a notification or TCB that is not this TCB this operation will fail. Similarly, if this TCB is already bound to a scheduling context that is not this scheduling context, this will also fail. 
 * @param[in] fault_ep CPtr to the endpoint which receives IPCs when this thread faults. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="sched_context"/> @endxmlonly  is already bound.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is the current thread's TCB.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is a TCB in the blocked state and  @xmlonly <texttt text="sched_context"/> @endxmlonly  is not schedulable. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly ,  @xmlonly <texttt text="authority"/> @endxmlonly ,  @xmlonly <texttt text="sched_context"/> @endxmlonly , or  @xmlonly <texttt text="fault_ep"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="fault_ep"/> @endxmlonly  does not have both Write rights and either Grant or GrantReply rights to the Endpoint  @xmlonly <docref>(see <autoref label="sec:cap_rights"/>)</docref> @endxmlonly . 
 * @retval seL4_RangeError The  @xmlonly <texttt text="mcp"/> @endxmlonly  is greater than the maximum controlled priority of  @xmlonly <texttt text="authority"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="priority"/> @endxmlonly  is greater than the maximum controlled priority of  @xmlonly <texttt text="authority"/> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_SetSchedParams(seL4_TCB _service, seL4_TCB authority, seL4_Word mcp, seL4_Word priority, seL4_CPtr sched_context, seL4_CPtr fault_ep)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBSetSchedParams, 0, 3, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, authority);
	seL4_SetCap(1, sched_context);
	seL4_SetCap(2, fault_ep);

	/* Marshal and initialise parameters. */
	mr0 = mcp;
	mr1 = priority;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_KERNEL_MCS)
/**
 * @xmlonly <manual name="Set Timeout Endpoint" label="tcb_settimeoutendpoint"/> @endxmlonly
 * @brief @xmlonly Set a thread's timeout endpoint. @endxmlonly
 * 
 * @xmlonly
 * Timeout exception messages will be delivered to this endpoint if it is not a null capability.
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] timeout_fault_ep CPtr to the endpoint which receives IPCs when this thread triggers timeout faults. Can be null. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="timeout_fault_ep"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="timeout_fault_ep"/> @endxmlonly  does not have both Write rights and either Grant or GrantReply rights to the Endpoint  @xmlonly <docref>(see <autoref label="sec:cap_rights"/>)</docref> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_SetTimeoutEndpoint(seL4_TCB _service, seL4_CPtr timeout_fault_ep)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBSetTimeoutEndpoint, 0, 1, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, timeout_fault_ep);

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
/**
 * @xmlonly <manual name="Set IPC Buffer" label="tcb_setipcbuffer"/> @endxmlonly
 * @brief @xmlonly Set a thread's IPC buffer @endxmlonly
 * 
 * @xmlonly
 * See Sections <shortref sec="threads"/> and <shortref sec="messageinfo"/>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] buffer Location of the thread's IPC buffer. Must be aligned to  @xmlonly <texttt text="seL4_IPCBufferSizeBits"/> @endxmlonly . The IPC buffer may not cross a page boundary. 
 * @param[in] bufferFrame Capability to a page containing the thread's IPC buffer. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_AlignmentError The  @xmlonly <texttt text="buffer"/> @endxmlonly  is not aligned. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="bufferFrame"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="bufferFrame"/> @endxmlonly  is retyped from a device untyped  @xmlonly <docref>(see <autoref label="sec:kernmemalloc"/>)</docref> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RevokeFirst The  @xmlonly <texttt text="bufferFrame"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_SetIPCBuffer(seL4_TCB _service, seL4_Word buffer, seL4_CPtr bufferFrame)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBSetIPCBuffer, 0, 1, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, bufferFrame);

	/* Marshal and initialise parameters. */
	mr0 = buffer;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#if !defined(CONFIG_KERNEL_MCS)
/**
 * @xmlonly <manual name="Set Space" label="tcb_setspace"/> @endxmlonly
 * @brief @xmlonly Set the fault endpoint, CSpace and VSpace of a thread @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:threads"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] fault_ep CPtr to the endpoint which receives IPCs when this thread faults. This capability is in the CSpace of the thread being configured. 
 * @param[in] cspace_root The new CSpace root. 
 * @param[in] cspace_root_data Optionally set the guard and guard size of the new root CNode. If set to zero, this parameter has no effect. 
 * @param[in] vspace_root The new VSpace root. 
 * @param[in] vspace_root_data Has no effect on x86 or ARM processors. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly ,  @xmlonly <texttt text="cspace_root"/> @endxmlonly , or  @xmlonly <texttt text="vspace_root"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="vspace_root"/> @endxmlonly  is not assigned to an ASID pool.
 * Or,  @xmlonly <texttt text="cspace_root_data"/> @endxmlonly  is invalid. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RevokeFirst The  @xmlonly <texttt text="cspace_root"/> @endxmlonly  or  @xmlonly <texttt text="vspace_root"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_SetSpace(seL4_TCB _service, seL4_Word fault_ep, seL4_CNode cspace_root, seL4_Word cspace_root_data, seL4_CPtr vspace_root, seL4_Word vspace_root_data)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBSetSpace, 0, 2, 3);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, cspace_root);
	seL4_SetCap(1, vspace_root);

	/* Marshal and initialise parameters. */
	mr0 = fault_ep;
	mr1 = cspace_root_data;
	mr2 = vspace_root_data;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_KERNEL_MCS)
/**
 * @xmlonly <manual name="Set Space (MCS)" label="tcb_setspace_mcs"/> @endxmlonly
 * @brief @xmlonly Set the fault endpoint, CSpace and VSpace of a thread @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:threads"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] fault_ep CPtr to the endpoint which receives IPCs when this thread faults. On MCS this cap gets copied into the TCB. 
 * @param[in] cspace_root The new CSpace root. 
 * @param[in] cspace_root_data Optionally set the guard and guard size of the new root CNode. If set to zero, this parameter has no effect. 
 * @param[in] vspace_root The new VSpace root. 
 * @param[in] vspace_root_data Has no effect on x86 or ARM processors. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly ,  @xmlonly <texttt text="cspace_root"/> @endxmlonly , or  @xmlonly <texttt text="vspace_root"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="vspace_root"/> @endxmlonly  is not assigned to an ASID pool.
 * Or,  @xmlonly <texttt text="cspace_root_data"/> @endxmlonly  is invalid. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="fault_ep"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="fault_ep"/> @endxmlonly  does not have both Write rights and either Grant or GrantReply rights to the Endpoint  @xmlonly <docref>(see <autoref label="sec:cap_rights"/>)</docref> @endxmlonly . 
 * @retval seL4_RevokeFirst The  @xmlonly <texttt text="cspace_root"/> @endxmlonly  or  @xmlonly <texttt text="vspace_root"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_SetSpace(seL4_TCB _service, seL4_CPtr fault_ep, seL4_CNode cspace_root, seL4_Word cspace_root_data, seL4_CPtr vspace_root, seL4_Word vspace_root_data)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBSetSpace, 0, 3, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, fault_ep);
	seL4_SetCap(1, cspace_root);
	seL4_SetCap(2, vspace_root);

	/* Marshal and initialise parameters. */
	mr0 = cspace_root_data;
	mr1 = vspace_root_data;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
/**
 * @xmlonly <manual name="Suspend" label="tcb_suspend"/> @endxmlonly
 * @brief @xmlonly Suspend a thread @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:thread_deactivation"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_Suspend(seL4_TCB _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBSuspend, 0, 0, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Resume" label="tcb_resume"/> @endxmlonly
 * @brief @xmlonly Resume a thread @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:thread_deactivation"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_Resume(seL4_TCB _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBResume, 0, 0, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Bind Notification" label="tcb_bindnotification"/> @endxmlonly
 * @brief @xmlonly Binds a notification object to a <obj name="TCB"/> @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:notification-binding"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] notification Notification to bind. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="notification"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="notification"/> @endxmlonly  is already bound.
 * Or,  @xmlonly <texttt text="notification"/> @endxmlonly  does not have Read rights to the Notification  @xmlonly <docref>(see <autoref label="sec:cap_rights"/>)</docref> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_BindNotification(seL4_TCB _service, seL4_CPtr notification)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBBindNotification, 0, 1, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, notification);

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Unbind Notification" label="tcb_unbindnotification"/> @endxmlonly
 * @brief @xmlonly Unbinds any notification object from a <obj name="TCB"/> @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:notification-binding"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is not bound to a notification. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_UnbindNotification(seL4_TCB _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBUnbindNotification, 0, 0, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#if (!defined(CONFIG_KERNEL_MCS) && defined(CONFIG_ENABLE_SMP_SUPPORT))
/**
 * @xmlonly <manual name="Set CPU Affinity" label="tcb_setaffinity"/> @endxmlonly
 * @brief @xmlonly Change a thread's current CPU in multicore machine @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:thread_creation"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] affinity The thread's new CPU to run. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="affinity"/> @endxmlonly  is not a valid CPU number. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_SetAffinity(seL4_TCB _service, seL4_Word affinity)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBSetAffinity, 0, 0, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = affinity;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_HARDWARE_DEBUG_API)
/**
 * @xmlonly <manual name="Set Breakpoint" label="tcb_setbreakpoint"/> @endxmlonly
 * @brief @xmlonly Set or modify a thread's breakpoints or watchpoints. Calls to this function
 * overwrite previous configurations for the target breakpoint. Do not use this
 * with seL4_SingleStep: the API will reject the call and return an error.
 * Instead, use seL4_TCB_ConfigureSingleStepping to configure single-stepping. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:debug_exceptions"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] bp_num The API-ID of a target breakpoint. This ID will be a positive integer, with values ranging from 0 to seL4_NumHWBreakpoints - 1. 
 * @param[in] vaddr A virtual address which forms part of the match conditions for the triggering of the breakpoint. 
 * @param[in] type One of: seL4_InstructionBreakpoint, which specifies that the breakpoint should occur on instruction execution at the specified vaddr or seL4_DataBreakpoint, which states that the breakpoint should occur on data access at the specified vaddr. 
 * @param[in] size A positive integer indicating the trigger-span of the watchpoint. Must be zero when 'type' is seL4_InstructionBreakpoint. 
 * @param[in] rw One of seL4_BreakOnRead, meaning the breakpoint will only be triggered on read-access; seL4_BreakOnWrite meaning the breakpoint will only be triggered on write-access, and seL4_BreakOnReadWrite meaning the breakpoint will be triggered on any access. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_AlignmentError The  @xmlonly <texttt text="vaddr"/> @endxmlonly  is not aligned to  @xmlonly <texttt text="size"/> @endxmlonly  bytes. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="bp_num"/> @endxmlonly ,  @xmlonly <texttt text="size"/> @endxmlonly , or  @xmlonly <texttt text="rw"/> @endxmlonly  is not valid for the given  @xmlonly <texttt text="type"/> @endxmlonly .
 * Or, argument values are inappropriate for the target architecture.
 * Or,  @xmlonly <texttt text="vaddr"/> @endxmlonly  is in the kernel virtual address range. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The argument values are inappropriate for the target architecture. 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_SetBreakpoint(seL4_TCB _service, seL4_Uint16 bp_num, seL4_Word vaddr, seL4_Word type, seL4_Word size, seL4_Word rw)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBSetBreakpoint, 0, 0, 5);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = (bp_num & 0xffffull);
	mr1 = vaddr;
	mr2 = type;
	mr3 = size;
	seL4_SetMR(4, rw);

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_HARDWARE_DEBUG_API)
/**
 * @xmlonly <manual name="Get Breakpoint" label="tcb_getbreakpoint"/> @endxmlonly
 * @brief @xmlonly Read a breakpoint or watchpoint's current configuration. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:debug_exceptions"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] bp_num The API-ID of a target breakpoint. This ID will be a positive integer, with values ranging from 0 to seL4_NumHWBreakpoints - 1. 
 * @return @xmlonly 
 *                 A <texttt text="seL4_TCB_GetBreakpoint_t"/>: Struct that contains
 *                 <texttt text="seL4_Error error"/>, an seL4 API error value,
 *                 <texttt text="seL4_Word vaddr"/>, the virtual address at which the breakpoint will currently
 *                 be triggered;
 *                 <texttt text="seL4_Word type"/>, the type of operation which will currently trigger the
 *                 breakpoint, whether instruction execution, or data access;
 *                 <texttt text="seL4_Word size"/>, integer value for the span-size of the breakpoint.
 *                 Usually a power of two (1, 2, 4, etc.);
 *                 <texttt text="seL4_Word rw"/>, the access direction that will currently trigger the breakpoint,
 *                 whether read, write, or both and
 *                 <texttt text="seL4_Bool is_enabled"/>, which indicates whether or not the breakpoint
 *                 will currently be triggered if the match conditions are met.
 *              @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The argument values are inappropriate for the target architecture. 
 */
LIBSEL4_INLINE seL4_TCB_GetBreakpoint_t
seL4_TCB_GetBreakpoint(seL4_TCB _service, seL4_Uint16 bp_num)
{
	seL4_TCB_GetBreakpoint_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBGetBreakpoint, 0, 0, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = (bp_num & 0xffffull);
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result.error = seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result.error != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
		return result;
	}

	/* Unmarshal result. */
	result.vaddr = mr0;
	result.type = mr1;
	result.size = mr2;
	result.rw = mr3;
	result.is_enabled = (seL4_GetMR(4) & 0x1);
	return result;
}

#endif
#if defined(CONFIG_HARDWARE_DEBUG_API)
/**
 * @xmlonly <manual name="Unset Breakpoint" label="tcb_unsetbreakpoint"/> @endxmlonly
 * @brief @xmlonly Disables a hardware breakpoint or watchpoint. The caller should assume that
 * the underlying configuration of the hardware registers has also been cleared.
 * Do not use this to clear single-stepping: the API will reject the call and
 * return an error. Instead, use seL4_TCB_ConfigureSingleStepping to disable
 * single-stepping. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:debug_exceptions"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] bp_num The API-ID of a target breakpoint. This ID will be a positive integer, with values ranging from 0 to seL4_NumHWBreakpoints - 1. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, the argument values are inappropriate for the target architecture. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The argument values are inappropriate for the target architecture. 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_UnsetBreakpoint(seL4_TCB _service, seL4_Uint16 bp_num)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBUnsetBreakpoint, 0, 0, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = (bp_num & 0xffffull);
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_HARDWARE_DEBUG_API)
/**
 * @xmlonly <manual name="Configure Single Stepping" label="tcb_configuresinglestepping"/> @endxmlonly
 * @brief @xmlonly Set or modify single stepping options for the target TCB. Subsequent calls to this
 * function overwrite previous configuration. Depending on your processor architecture,
 * this may or may not require the consumption of a hardware register. @endxmlonly
 * 
 * @xmlonly
 * <docref>See Sections <shortref sec="single_stepping_debug_exception"/> and <shortref sec="debug_exceptions"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] bp_num The API-ID of a target breakpoint. This ID will be a positive integer, with values ranging from 0 to seL4_NumHWBreakpoints - 1. This value is unused on AARCH64 
 * @param[in] num_instructions Number of instructions to step over before delivering a fault to the target thread's fault endpoint. Setting this to 0 disables single-stepping. 
 * @return @xmlonly 
 *                 A <texttt text="seL4_TCB_ConfigureSingleStepping_t"/>: Struct that contains
 *                 <texttt text="seL4_Error error"/>, an seL4 API error value,
 *                 <texttt text="seL4_Bool bp_was_consumed"/>, a boolean which indicates whether or not the <texttt text="bp_num"/>
 *                 breakpoint ID that was passed to the function, was consumed in the setup of the single-stepping
 *                 functionality: if this is <texttt text="true"/>, the caller should not attempt to re-use <texttt text="bp_num"/>
 *                 until it has disabled the single-stepping functionality via a subsequent call to
 *                 seL4_TCB_ConfigureSingleStepping with an <texttt text="num_instructions"/> argument of 0.
 *              @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, the argument values are inappropriate for the target architecture. 
 * @retval seL4_InvalidArgument The argument values are inappropriate for the target architecture. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_TCB_ConfigureSingleStepping_t
seL4_TCB_ConfigureSingleStepping(seL4_TCB _service, seL4_Uint16 bp_num, seL4_Word num_instructions)
{
	seL4_TCB_ConfigureSingleStepping_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBConfigureSingleStepping, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = (bp_num & 0xffffull);
	mr1 = num_instructions;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result.error = seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result.error != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
		return result;
	}

	/* Unmarshal result. */
	result.bp_was_consumed = (mr0 & 0x1);
	return result;
}

#endif
/**
 * @xmlonly <manual name="Set TLS Base" label="tcb_settlsbase"/> @endxmlonly
 * @brief @xmlonly Set the TLS base of the target TCB. @endxmlonly
 * 
 * @xmlonly
 * An invocation for setting the Thread Local Storage (TLS) base address. This ensures that across all platforms, the TLSBase register is viewed as being completely mutable, just like all of the general purpose registers, even on platforms where modification is a privileged operation. The value is subject to the same architecture-specific sanitisation as <texttt text="seL4_TCB_WriteRegisters"/>. In particular, on x86_64 non-canonical addresses are replaced by 0.
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] tls_base The TLS base to set. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_TCB_SetTLSBase(seL4_TCB _service, seL4_Word tls_base)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBSetTLSBase, 0, 0, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = tls_base;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Set Feature Flags" label="tcb_setflags"/> @endxmlonly
 * @brief @xmlonly Set or clear <texttt text="seL4_TCBFlag"/> feature flags of the target TCB. @endxmlonly
 * 
 * @xmlonly
 * A newly created TCB has all flags cleared.
 * Currently the only flag supported is <texttt text="seL4_TCBFlag_fpuDisabled"/>.
 * The flags are cleared and set in the given order, i.e. when a flag is both cleared and set, it will be set.
 * Unknown flags are ignored. Use zero for both clear and set to retrieve the current flags value.
 * @endxmlonly
 * 
 * @param[in] _service Capability to the TCB which is being operated on.
 * @param[in] clear Bitwise OR'd set of seL4_TCBFlag to clear. 
 * @param[in] set Bitwise OR'd set of seL4_TCBFlag to set. 
 * @return @xmlonly 
 *                 The resulting TCB flags value is returned in the first message register.
 *              @endxmlonly
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_TCB_SetFlags_t
seL4_TCB_SetFlags(seL4_TCB _service, seL4_Word clear, seL4_Word set)
{
	seL4_TCB_SetFlags_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBSetFlags, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = clear;
	mr1 = set;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result.error = seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result.error != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
		return result;
	}

	/* Unmarshal result. */
	result.flags = mr0;
	return result;
}

/**
 * @xmlonly <manual name="Revoke" label="cnode_revoke"/> @endxmlonly
 * @brief @xmlonly Delete all child capabilities of a capability @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:cnode-ops"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service CPtr to the CNode at the root of the CSpace where the capability will be found. Must be at a depth equivalent to the wordsize.
 * @param[in] index CPtr to the capability. Resolved from the root of the _service parameter. 
 * @param[in] depth Number of bits of index to resolve to find the capability being operated on. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="index"/> @endxmlonly  or  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_CNode_Revoke(seL4_CNode _service, seL4_Word index, seL4_Uint8 depth)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(CNodeRevoke, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = index;
	mr1 = (depth & 0xffull);
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Delete" label="cnode_delete"/> @endxmlonly
 * @brief @xmlonly Delete a capability @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:cnode-ops"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service CPtr to the CNode at the root of the CSpace where the capability will be found. Must be at a depth equivalent to the wordsize.
 * @param[in] index CPtr to the capability. Resolved from the root of the _service parameter. 
 * @param[in] depth Number of bits of index to resolve to find the capability being operated on. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="index"/> @endxmlonly  or  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_CNode_Delete(seL4_CNode _service, seL4_Word index, seL4_Uint8 depth)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(CNodeDelete, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = index;
	mr1 = (depth & 0xffull);
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Cancel Badged Sends" label="cnode_cancelbadgedsends"/> @endxmlonly
 * @brief @xmlonly The cancel badged sends method is intended to allow for the reuse of badges by an
 * authority. When used with a badged endpoint capability it
 * will cancel any outstanding send operations for that endpoint and badge.
 * This operation has no effect on un-badged or other objects. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:cnode-ops"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service CPtr to the CNode at the root of the CSpace where the capability will be found. Must be at a depth equivalent to the wordsize.
 * @param[in] index CPtr to the capability. Resolved from the root of the _service parameter. 
 * @param[in] depth Number of bits of index to resolve to find the capability being operated on. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="index"/> @endxmlonly  or  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, the capability does not have full rights to the Endpoint  @xmlonly <docref>(see <autoref label="sec:cap_rights"/>)</docref> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_CNode_CancelBadgedSends(seL4_CNode _service, seL4_Word index, seL4_Uint8 depth)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(CNodeCancelBadgedSends, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = index;
	mr1 = (depth & 0xffull);
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Copy" label="cnode_copy"/> @endxmlonly
 * @brief @xmlonly Copy a capability, setting its access rights whilst doing so @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:cnode-ops"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service CPtr to the CNode that forms the root of the destination CSpace. Must be at a depth equivalent to the wordsize.
 * @param[in] dest_index CPtr to the destination slot. Resolved from the root of the destination CSpace. 
 * @param[in] dest_depth Number of bits of dest_index to resolve to find the destination slot. 
 * @param[in] src_root CPtr to the CNode that forms the root of the source CSpace. Must be at a depth equivalent to the wordsize. 
 * @param[in] src_index CPtr to the source slot. Resolved from the root of the source CSpace. 
 * @param[in] src_depth Number of bits of src_index to resolve to find the source slot. 
 * @param[in] rights The rights inherited by the new capability. @xmlonly <docref> Possible values for this type are given in <autoref label="sec:cap_rights"/>  .</docref> @endxmlonly 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The destination slot contains a capability. 
 * @retval seL4_FailedLookup The index or depth of the source or destination is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly .
 * Or,  @xmlonly <texttt text="src_root"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, the source slot is empty. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, the source capability cannot be derived @xmlonly <docref> (see <autoref label="sec:cap_derivation"/>)</docref> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="dest_depth"/> @endxmlonly  or  @xmlonly <texttt text="src_depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 * @retval seL4_RevokeFirst The source capability cannot be derived @xmlonly <docref> (see <autoref label="sec:cap_derivation"/>)</docref> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_CNode_Copy(seL4_CNode _service, seL4_Word dest_index, seL4_Uint8 dest_depth, seL4_CNode src_root, seL4_Word src_index, seL4_Uint8 src_depth, seL4_CapRights_t rights)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(CNodeCopy, 0, 1, 5);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, src_root);

	/* Marshal and initialise parameters. */
	mr0 = dest_index;
	mr1 = (dest_depth & 0xffull);
	mr2 = src_index;
	mr3 = (src_depth & 0xffull);
	seL4_SetMR(4, rights.words[0]);

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Mint" label="cnode_mint"/> @endxmlonly
 * @brief @xmlonly Copy a capability, setting its access rights and badge whilst doing so @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:cnode-ops"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service CPtr to the CNode that forms the root of the destination CSpace. Must be at a depth equivalent to the wordsize.
 * @param[in] dest_index CPtr to the destination slot. Resolved from the root of the destination CSpace. 
 * @param[in] dest_depth Number of bits of dest_index to resolve to find the destination slot. 
 * @param[in] src_root CPtr to the CNode that forms the root of the source CSpace. Must be at a depth equivalent to the wordsize. 
 * @param[in] src_index CPtr to the source slot. Resolved from the root of the source CSpace. 
 * @param[in] src_depth Number of bits of src_index to resolve to find the source slot. 
 * @param[in] rights The rights inherited by the new capability. @xmlonly <docref> Possible values for this type are given in <autoref label="sec:cap_rights"/>  .</docref> @endxmlonly 
 * @param[in] badge Badge or guard to be applied to the new capability. For badges on 32-bit platforms, the high 4 bits are ignored. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The destination slot contains a capability. 
 * @retval seL4_FailedLookup The index or depth of the source or destination is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly .
 * Or,  @xmlonly <texttt text="src_root"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, the source slot is empty. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, the source capability cannot be derived  @xmlonly <docref>(see <autoref label="sec:cap_derivation"/>)</docref> @endxmlonly .
 * Or, the badge or guard value is invalid. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="dest_depth"/> @endxmlonly  or  @xmlonly <texttt text="src_depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 * @retval seL4_RevokeFirst The source capability cannot be derived  @xmlonly <docref>(see <autoref label="sec:cap_derivation"/>)</docref> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_CNode_Mint(seL4_CNode _service, seL4_Word dest_index, seL4_Uint8 dest_depth, seL4_CNode src_root, seL4_Word src_index, seL4_Uint8 src_depth, seL4_CapRights_t rights, seL4_Word badge)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(CNodeMint, 0, 1, 6);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, src_root);

	/* Marshal and initialise parameters. */
	mr0 = dest_index;
	mr1 = (dest_depth & 0xffull);
	mr2 = src_index;
	mr3 = (src_depth & 0xffull);
	seL4_SetMR(4, rights.words[0]);
	seL4_SetMR(5, badge);

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Move" label="cnode_move"/> @endxmlonly
 * @brief @xmlonly Move a capability @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:cnode-ops"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service CPtr to the CNode that forms the root of the destination CSpace. Must be at a depth equivalent to the wordsize.
 * @param[in] dest_index CPtr to the destination slot. Resolved from the root of the destination CSpace. 
 * @param[in] dest_depth Number of bits of dest_index to resolve to find the destination slot. 
 * @param[in] src_root CPtr to the CNode that forms the root of the source CSpace. Must be at a depth equivalent to the wordsize. 
 * @param[in] src_index CPtr to the source slot. Resolved from the root of the source CSpace. 
 * @param[in] src_depth Number of bits of src_index to resolve to find the source slot. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The destination slot contains a capability. 
 * @retval seL4_FailedLookup The index or depth of the source or destination is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly .
 * Or,  @xmlonly <texttt text="src_root"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, the source slot is empty. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="dest_depth"/> @endxmlonly  or  @xmlonly <texttt text="src_depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_CNode_Move(seL4_CNode _service, seL4_Word dest_index, seL4_Uint8 dest_depth, seL4_CNode src_root, seL4_Word src_index, seL4_Uint8 src_depth)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(CNodeMove, 0, 1, 4);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, src_root);

	/* Marshal and initialise parameters. */
	mr0 = dest_index;
	mr1 = (dest_depth & 0xffull);
	mr2 = src_index;
	mr3 = (src_depth & 0xffull);

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Mutate" label="cnode_mutate"/> @endxmlonly
 * @brief @xmlonly Move a capability, setting its guard in the process. This
 * operation is mostly useful for setting the guard of a CNode
 * capability without losing revokability of that CNode capability.
 * All other uses can be replaced by a combination of Mint and
 * Delete. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:cnode-ops"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service CPtr to the CNode that forms the root of the destination CSpace. Must be at a depth equivalent to the wordsize.
 * @param[in] dest_index CPtr to the destination slot. Resolved from the root of the destination CSpace. 
 * @param[in] dest_depth Number of bits of dest_index to resolve to find the destination slot. 
 * @param[in] src_root CPtr to the CNode that forms the root of the source CSpace. Must be at a depth equivalent to the wordsize. 
 * @param[in] src_index CPtr to the source slot. Resolved from the root of the source CSpace. 
 * @param[in] src_depth Number of bits of src_index to resolve to find the source slot. 
 * @param[in] badge Guard to be applied to the new capability. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The destination slot contains a capability. 
 * @retval seL4_FailedLookup The index or depth of the source or destination is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly .
 * Or,  @xmlonly <texttt text="src_root"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, the source slot is empty. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, the guard value is invalid. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="dest_depth"/> @endxmlonly  or  @xmlonly <texttt text="src_depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_CNode_Mutate(seL4_CNode _service, seL4_Word dest_index, seL4_Uint8 dest_depth, seL4_CNode src_root, seL4_Word src_index, seL4_Uint8 src_depth, seL4_Word badge)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(CNodeMutate, 0, 1, 5);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, src_root);

	/* Marshal and initialise parameters. */
	mr0 = dest_index;
	mr1 = (dest_depth & 0xffull);
	mr2 = src_index;
	mr3 = (src_depth & 0xffull);
	seL4_SetMR(4, badge);

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Rotate" label="cnode_rotate"/> @endxmlonly
 * @brief @xmlonly Given 3 capability slots - a destination, pivot and source - move the capability in the
 * pivot slot to the destination slot and the capability in the source slot to the pivot slot @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:cnode-ops"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service CPtr to the CNode at the root of the CSpace where the destination slot will be found. Must be at a depth equivalent to the wordsize.
 * @param[in] dest_index CPtr to the destination slot. Resolved relative to _service. Must be empty unless it refers to the same slot as the source slot. 
 * @param[in] dest_depth Depth to resolve dest_index to. 
 * @param[in] dest_badge The new capdata for the capability that ends up in the destination slot. 
 * @param[in] pivot_root CPtr to the CNode at the root of the CSpace where the pivot slot will be found. Must be at a depth equivalent to the wordsize. 
 * @param[in] pivot_index CPtr to the pivot slot. Resolved relative to pivot_root. The resolved slot must not refer to the source or destination slots. 
 * @param[in] pivot_depth Depth to resolve pivot_index to. 
 * @param[in] pivot_badge The new capdata for the capability that ends up in the pivot slot. 
 * @param[in] src_root CPtr to the CNode at the root of the CSpace where the source slot will be found. Must be at a depth equivalent to the wordsize. 
 * @param[in] src_index CPtr to the source slot. Resolved relative to src_root. 
 * @param[in] src_depth Depth to resolve src_index to. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst If the destination is not the same slot as the source and the destination slot contains a capability. 
 * @retval seL4_FailedLookup The index or depth of the source, destination, or pivot is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly .
 * Or,  @xmlonly <texttt text="src_root"/> @endxmlonly  or  @xmlonly <texttt text="pivot_root"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, the source or pivot slot is empty. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, the pivot is the same slot as the source or destination.
 * Or, the guard value on the destination or pivot is invalid. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="dest_depth"/> @endxmlonly ,  @xmlonly <texttt text="src_depth"/> @endxmlonly , or  @xmlonly <texttt text="pivot_depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_CNode_Rotate(seL4_CNode _service, seL4_Word dest_index, seL4_Uint8 dest_depth, seL4_Word dest_badge, seL4_CNode pivot_root, seL4_Word pivot_index, seL4_Uint8 pivot_depth, seL4_Word pivot_badge, seL4_CNode src_root, seL4_Word src_index, seL4_Uint8 src_depth)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(CNodeRotate, 0, 2, 8);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, pivot_root);
	seL4_SetCap(1, src_root);

	/* Marshal and initialise parameters. */
	mr0 = dest_index;
	mr1 = (dest_depth & 0xffull);
	mr2 = dest_badge;
	mr3 = pivot_index;
	seL4_SetMR(4, (pivot_depth & 0xffull));
	seL4_SetMR(5, pivot_badge);
	seL4_SetMR(6, src_index);
	seL4_SetMR(7, (src_depth & 0xffull));

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#if !defined(CONFIG_KERNEL_MCS)
/**
 * @xmlonly <manual name="Save Caller" label="cnode_savecaller"/> @endxmlonly
 * @brief @xmlonly Save the reply capability from the last time the thread was called in the given CSpace so that it can be invoked later @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:cnode-ops"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service CPtr to the CNode at the root of the CSpace where the capability is to be saved. Must be at a depth equivalent to the wordsize.
 * @param[in] index CPtr to the slot in which to save the capability. Resolved from the root of the _service parameter. 
 * @param[in] depth Number of bits of index to resolve to find the slot being targeted. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The destination slot contains a capability. 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="index"/> @endxmlonly  or  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_CNode_SaveCaller(seL4_CNode _service, seL4_Word index, seL4_Uint8 depth)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(CNodeSaveCaller, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = index;
	mr1 = (depth & 0xffull);
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
/**
 * @xmlonly <manual name="Get IRQ Handler" label="irq_controlget"/> @endxmlonly
 * @brief @xmlonly Create an IRQ handler capability @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:interrupts"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service An IRQControl capability. This gives you the authority to make this call.
 * @param[in] irq The IRQ that you want this capability to handle. 
 * @param[in] root CPtr to the CNode that forms the root of the destination CSpace. Must be at a depth equivalent to the wordsize. 
 * @param[in] index CPtr to the destination slot. Resolved from the root of the destination CSpace. 
 * @param[in] depth Number of bits of index to resolve to find the destination slot. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The destination slot contains a capability. 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="root"/> @endxmlonly ,  @xmlonly <texttt text="index"/> @endxmlonly , or  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, on x86, an IOAPIC is being used. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="irq"/> @endxmlonly  is invalid for the target architecture.
 * Or, on x86,  @xmlonly <texttt text="irq"/> @endxmlonly  is not in the ISA IRQ range.
 * Or,  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 * @retval seL4_RevokeFirst An IRQ handler capability for  @xmlonly <texttt text="irq"/> @endxmlonly  has already been created. 
 */
LIBSEL4_INLINE seL4_Error
seL4_IRQControl_Get(seL4_IRQControl _service, seL4_Word irq, seL4_CNode root, seL4_Word index, seL4_Uint8 depth)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(IRQIssueIRQHandler, 0, 1, 3);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, root);

	/* Marshal and initialise parameters. */
	mr0 = irq;
	mr1 = index;
	mr2 = (depth & 0xffull);
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Acknowledge" label="irq_handleracknowledge"/> @endxmlonly
 * @brief @xmlonly Acknowledge the receipt of an interrupt and re-enable it @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:interrupts"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service The IRQ handler capability.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_IRQHandler_Ack(seL4_IRQHandler _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(IRQAckIRQ, 0, 0, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Set Notification" label="irq_handlersetnotification"/> @endxmlonly
 * @brief @xmlonly Set the notification which the kernel will signal on interrupts
 * controlled by the supplied IRQ handler capability @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:interrupts"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service The IRQ handler capability.
 * @param[in] notification The notification which the IRQs will signal. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="notification"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="notification"/> @endxmlonly  does not have the Write right  @xmlonly <docref>(see <autoref label="sec:cap_rights"/>)</docref> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_IRQHandler_SetNotification(seL4_IRQHandler _service, seL4_CPtr notification)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(IRQSetIRQHandler, 0, 1, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, notification);

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Clear" label="irq_handlerclear"/> @endxmlonly
 * @brief @xmlonly Clear the handler capability from the IRQ slot @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:interrupts"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service The IRQ handler capability.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_IRQHandler_Clear(seL4_IRQHandler _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(IRQClearIRQHandler, 0, 0, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="Set" label="domainset_set"/> @endxmlonly
 * @brief @xmlonly Change the domain of a thread. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:domains"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability allowing domain configuration.
 * @param[in] domain The thread's new domain. 
 * @param[in] thread Capability to the TCB which is being operated on. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="domain"/> @endxmlonly  is not less than  @xmlonly <texttt text="KernelNumDomains"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="thread"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_DomainSet_Set(seL4_DomainSet _service, seL4_Uint8 domain, seL4_TCB thread)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(DomainSetSet, 0, 1, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, thread);

	/* Marshal and initialise parameters. */
	mr0 = (domain & 0xffull);
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="ScheduleConfigure" label="domain_schedule_configure"/> @endxmlonly
 * @brief @xmlonly Modify a domain scheduling entry. @endxmlonly
 * 
 * @xmlonly
 * Configure the domain and duration of a scheduling entry.
 * A zero domain and duration means it is an end marker.
 * The domain scheduler will go back to the starting index when reaching
 * an end marker.
 * This makes it possible to configure multiple independent domain schedules
 * and switch between them atomically.
 * 
 * The scheduling entry will be modified immediately. However, if the currently active
 * index is modified, the change will not apply until the next time this entry is reached
 * by the domain scheduler.
 * 
 * After boot the first scheduling entry has domain zero and maximum duration, all other
 * entries will be end markers.
 * 
 * Hence, after initial configuration a call to <texttt text="seL4_DomainSet_ScheduleSetStart"/>
 * is needed to update the current execution, even when there is no need to actually
 * change the starting index.
 * 
 * <docref>See <autoref label="sec:domains"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability allowing domain configuration.
 * @param[in] index The scheduling entry to modify. 
 * @param[in] domain The scheduling entry's new domain. 
 * @param[in] duration Timeslice in ticks (kernel ticks for non-MCS, timer ticks for MCS). 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="duration"/> @endxmlonly  is zero, but  @xmlonly <texttt text="domain"/> @endxmlonly  is not.
 * Or,  @xmlonly <texttt text="index"/> @endxmlonly  points to the starting index, but  @xmlonly <texttt text="duration"/> @endxmlonly  is zero.
 * Or,  @xmlonly <texttt text="duration"/> @endxmlonly  does not fit in 56 bits. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="index"/> @endxmlonly  is not less than  @xmlonly <texttt text="KernelNumDomainSchedules"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="domain"/> @endxmlonly  is not less than  @xmlonly <texttt text="KernelNumDomains"/> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_DomainSet_ScheduleConfigure(seL4_DomainSet _service, seL4_Word index, seL4_Uint8 domain, seL4_Time duration)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(DomainScheduleConfigure, 0, 0, 3);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = index;
	mr1 = (domain & 0xffull);
	mr2 = duration;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

/**
 * @xmlonly <manual name="ScheduleSetStart" label="domain_schedule_set_start"/> @endxmlonly
 * @brief @xmlonly Change the starting index of the domain scheduler. @endxmlonly
 * 
 * @xmlonly
 * The domain scheduler starts with the schedule entry indicated by the starting index.
 * When reaching an end marker or the end of the schedules, the scheduler will go back
 * to the starting entry again. The starting index is zero after boot.
 * 
 * The currently active scheduling index will be changed to the new starting index
 * immediately, before returning from this system call. This relinquishes all remaining
 * duration of the last schedule.
 * 
 * <docref>See <autoref label="sec:domains"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability allowing domain configuration.
 * @param[in] index The new starting index. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="index"/> @endxmlonly  points to an end marker entry. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="index"/> @endxmlonly  is not less than  @xmlonly <texttt text="KernelNumDomainSchedules"/> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_DomainSet_ScheduleSetStart(seL4_DomainSet _service, seL4_Word index)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(DomainScheduleSetStart, 0, 0, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = index;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#if defined(CONFIG_KERNEL_MCS)
/**
 * @xmlonly <manual name="Configure Flags" label="schedcontrol_configureflags"/> @endxmlonly
 * @brief @xmlonly Set the parameters of a scheduling context by invoking the scheduling control capability. If the scheduling context is bound to a currently running thread, the parameters will take effect immediately: that is the current budget will be increased or reduced by the difference between the new and previous budget and the replenishment time will be updated according to any difference in the period. This can result in active threads being post-poned or released depending on the nature of the parameter change and the state of the thread. Additionally, if the scheduling context was previously empty (no budget) but bound to a runnable thread, this can result in a thread running for the first time since it now has access to CPU time. This call will return seL4 Invalid Argument if the parameters are too small (smaller than the kernel WCET for this platform) or too large (will overflow the timer). @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:threads"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to a scheduling control object.
 * @param[in] schedcontext Capability to the scheduling context which is being operated on. 
 * @param[in] budget Timeslice in microseconds, when the budget expires the thread will be pre-empted. 
 * @param[in] period Period in microseconds, if equal to budget, this thread will be treated as a round-robin thread. Otherwise, sporadic servers will be used to assure the scheduling context does not exceed the budget over the specified period. 
 * @param[in] extra_refills Number of extra sporadic replenishments this scheduling context should use. Ignored for round-robin threads. 
 * @param[in] badge Identifier for this scheduling context. Delivered to timeout exception handler. Can be used to determine which scheduling context triggered the timeout. 
 * @param[in] flags Bitwise OR'd set of seL4_SchedContextFlag. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="schedcontext"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="budget"/> @endxmlonly  or  @xmlonly <texttt text="period"/> @endxmlonly  or  @xmlonly <texttt text="extra_refills"/> @endxmlonly  is too big or too small.
 * Or,  @xmlonly <texttt text="budget"/> @endxmlonly  is greater than  @xmlonly <texttt text="period"/> @endxmlonly . 
 */
LIBSEL4_INLINE seL4_Error
seL4_SchedControl_ConfigureFlags(seL4_SchedControl _service, seL4_SchedContext schedcontext, seL4_Time budget, seL4_Time period, seL4_Word extra_refills, seL4_Word badge, seL4_Word flags)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(SchedControlConfigureFlags, 0, 1, 5);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, schedcontext);

	/* Marshal and initialise parameters. */
	mr0 = budget;
	mr1 = period;
	mr2 = extra_refills;
	mr3 = badge;
	seL4_SetMR(4, flags);

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_KERNEL_MCS)
/**
 * @xmlonly <manual name="Bind" label="schedcontext_bind"/> @endxmlonly
 * @brief @xmlonly Bind an object to a scheduling context. The object can be a notification object or a
 * thread.
 * 
 * If the object is a thread and the thread is in a runnable state and the scheduling
 * context has available budget, this will start the thread running.
 * 
 * If the object is a notification, when passive threads wait on the notification object and
 * a signal arrives, the passive thread will receive the scheduling context and possess it
 * until it waits on the notification object again.
 * 
 * This operation will fail for notification objects if the scheduling context is already
 * bound to a notification object, and for thread objects if the scheduling context is
 * already bound to a thread. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:threads"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the scheduling context which is being operated on.
 * @param[in] cap Capability to a TCB or a notification object 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="cap"/> @endxmlonly  is already bound to the same type of object.
 * Or,  @xmlonly <texttt text="cap"/> @endxmlonly  is a TCB in the blocked state and  @xmlonly <texttt text="_service"/> @endxmlonly  is not schedulable. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="cap"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_SchedContext_Bind(seL4_SchedContext _service, seL4_CPtr cap)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(SchedContextBind, 0, 1, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, cap);

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_KERNEL_MCS)
/**
 * @xmlonly <manual name="Unbind" label="schedcontext_unbind"/> @endxmlonly
 * @brief @xmlonly Unbind any objects (threads or notification objects) from a scheduling context. This
 * will render the bound thread passive, see Section 6.1.5. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:threads"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the scheduling context which is being operated on.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, the current thread's TCB is bound to  @xmlonly <texttt text="_service"/> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_SchedContext_Unbind(seL4_SchedContext _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(SchedContextUnbind, 0, 0, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_KERNEL_MCS)
/**
 * @xmlonly <manual name="Unbind Object" label="schedcontext_unbindobject"/> @endxmlonly
 * @brief @xmlonly Unbind an object from a scheduling context. The object can be either a thread or a
 * notification.
 * 
 * If the thread being unbound is the thread that is bound to this scheduling context,
 * this will render the thread passive. However if the thread being
 * unbound received the scheduling context via scheduling context donation over IPC,
 * the scheduling context will be returned to the thread that it was originally bound to.
 * 
 * If the object is a notification and it is bound to the scheduling context, unbind it. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:passive"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the scheduling context which is being operated on.
 * @param[in] cap Capability to a notification that is bound to the scheduling context or capability to a TCB that is bound to this scheduling context or has received it through scheduling context donation. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="cap"/> @endxmlonly  is not bound to  @xmlonly <texttt text="_service"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="cap"/> @endxmlonly  is the current thread's TCB. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="cap"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_SchedContext_UnbindObject(seL4_SchedContext _service, seL4_CPtr cap)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(SchedContextUnbindObject, 0, 1, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, cap);

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result = (seL4_Error) seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
	}

	return result;
}

#endif
#if defined(CONFIG_KERNEL_MCS)
/**
 * @xmlonly <manual name="Consumed" label="schedcontext_consumed"/> @endxmlonly
 * @brief @xmlonly Return the amount of time used by this scheduling context since this function was last called or a timeout exception triggered. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:threads"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the scheduling context which is being operated on.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_SchedContext_Consumed_t
seL4_SchedContext_Consumed(seL4_SchedContext _service)
{
	seL4_SchedContext_Consumed_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(SchedContextConsumed, 0, 0, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result.error = seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result.error != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
		return result;
	}

	/* Unmarshal result. */
	result.consumed = mr0;
	return result;
}

#endif
#if defined(CONFIG_KERNEL_MCS)
/**
 * @xmlonly <manual name="Yield To" label="schedcontext_yieldto"/> @endxmlonly
 * @brief @xmlonly If a thread is currently runnable and running on this scheduling context and the scheduling context has available budget, place it at the head of the scheduling queue.
 * If the caller is at an equal priority to the thread this will result in the thread being scheduled.
 * If the caller is at a higher priority the thread will not run until the threads priority is the highest priority in the system.
 * The caller must have a maximum control priority greater than or equal to the threads priority. @endxmlonly
 * 
 * @xmlonly
 * Capability to the scheduling context which is being operated on.
 * @endxmlonly
 * 
 * @param[in] _service Capability to the scheduling context which is being operated on.
 * @return @xmlonly 
 *             <docref>See <autoref label="sec:scheduling_contexts"/></docref>
 *            @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is not bound to a TCB or is bound to the current thread's TCB.
 * Or, the target thread's priority is greater than the current thread's maximum controlled priority @xmlonly <docref> (see <autoref label="sec:sched"/>)</docref> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_SchedContext_YieldTo_t
seL4_SchedContext_YieldTo(seL4_SchedContext _service)
{
	seL4_SchedContext_YieldTo_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(SchedContextYieldTo, 0, 0, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = 0;
	mr1 = 0;
	mr2 = 0;
	mr3 = 0;

	/* Perform the call, passing in-register arguments directly. */
	output_tag = seL4_CallWithMRs(_service, tag,
		&mr0, &mr1, &mr2, &mr3);
	result.error = seL4_MessageInfo_get_label(output_tag);

	/* Unmarshal registers into IPC buffer on error. */
	if (result.error != seL4_NoError) {
		seL4_SetMR(0, mr0);
		seL4_SetMR(1, mr1);
		seL4_SetMR(2, mr2);
		seL4_SetMR(3, mr3);
#ifdef CONFIG_KERNEL_INVOCATION_REPORT_ERROR_IPC
		if (seL4_CanPrintError()) {
			seL4_DebugPutString(seL4_GetDebugError());
		}
#endif
		return result;
	}

	/* Unmarshal result. */
	result.consumed = mr0;
	return result;
}

#endif