
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
assert_size_correct(seL4_ARM_VMAttributes, 8);
assert_size_correct(seL4_ARM_Page, 8);
assert_size_correct(seL4_ARM_PageTable, 8);
assert_size_correct(seL4_ARM_VSpace, 8);
assert_size_correct(seL4_ARM_ASIDControl, 8);
assert_size_correct(seL4_ARM_ASIDPool, 8);
assert_size_correct(seL4_ARM_VCPU, 8);
assert_size_correct(seL4_ARM_IOSpace, 8);
assert_size_correct(seL4_ARM_IOPageTable, 8);
assert_size_correct(seL4_ARM_SMC, 8);
assert_size_correct(seL4_UserContext, 288);
assert_size_correct(seL4_ARM_SMCContext, 64);
assert_size_correct(seL4_VCPUReg, 8);
assert_size_correct(seL4_ARM_SIDControl, 8);
assert_size_correct(seL4_ARM_SID, 8);
assert_size_correct(seL4_ARM_CBControl, 8);
assert_size_correct(seL4_ARM_CB, 8);

/*
 * Return types for generated methods.
 */
struct seL4_ARM_Page_GetAddress {
	int error;
	seL4_Word paddr;
};
typedef struct seL4_ARM_Page_GetAddress seL4_ARM_Page_GetAddress_t;

struct seL4_ARM_VCPU_ReadRegs {
	int error;
	seL4_Word value;
};
typedef struct seL4_ARM_VCPU_ReadRegs seL4_ARM_VCPU_ReadRegs_t;

struct seL4_ARM_SIDControl_GetFault {
	int error;
	seL4_Word status;
	seL4_Word syndrome_0;
	seL4_Word syndrome_1;
};
typedef struct seL4_ARM_SIDControl_GetFault seL4_ARM_SIDControl_GetFault_t;

struct seL4_ARM_CB_CBGetFault {
	int error;
	seL4_Word status;
	seL4_Word address;
};
typedef struct seL4_ARM_CB_CBGetFault seL4_ARM_CB_CBGetFault_t;

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
 * @xmlonly <manual name="Clean Data" label="aarch64_vspace_clean"/> @endxmlonly
 * @brief @xmlonly Clean cached pages within a top level translation table @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the top level translation table being operated on.
 * @param[in] start Start address 
 * @param[in] end End address 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="_service"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="end"/> @endxmlonly  is in the kernel virtual address range.
 * Or, if hypervisor support is configured, the requested range is not entirely
 * within the kernel physical memory window. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="start"/> @endxmlonly  is greater than or equal to  @xmlonly <texttt text="end"/> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_RangeError The specified range crosses a page boundary. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_VSpace_Clean_Data(seL4_ARM_VSpace _service, seL4_Word start, seL4_Word end)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMVSpaceClean_Data, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = start;
	mr1 = end;
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
 * @xmlonly <manual name="Invalidate Data" label="aarch64_vspace_invalidate"/> @endxmlonly
 * @brief @xmlonly Invalidate cached pages within a top level translation table @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the top level translation table being operated on.
 * @param[in] start Start address 
 * @param[in] end End address 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="_service"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="end"/> @endxmlonly  is in the kernel virtual address range.
 * Or, if hypervisor support is configured, the requested range is not entirely
 * within the kernel physical memory window. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="start"/> @endxmlonly  is greater than or equal to  @xmlonly <texttt text="end"/> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_RangeError The specified range crosses a page boundary. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_VSpace_Invalidate_Data(seL4_ARM_VSpace _service, seL4_Word start, seL4_Word end)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMVSpaceInvalidate_Data, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = start;
	mr1 = end;
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
 * @xmlonly <manual name="Clean and Invalidate Data" label="aarch64_vspace_clean_invalidate"/> @endxmlonly
 * @brief @xmlonly Clean and invalidate cached pages within a top level translation table @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the top level translation table being operated on.
 * @param[in] start Start address 
 * @param[in] end End address 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="_service"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="end"/> @endxmlonly  is in the kernel virtual address range.
 * Or, if hypervisor support is configured, the requested range is not entirely
 * within the kernel physical memory window. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="start"/> @endxmlonly  is greater than or equal to  @xmlonly <texttt text="end"/> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_RangeError The specified range crosses a page boundary. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_VSpace_CleanInvalidate_Data(seL4_ARM_VSpace _service, seL4_Word start, seL4_Word end)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMVSpaceCleanInvalidate_Data, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = start;
	mr1 = end;
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
 * @xmlonly <manual name="Unify Instruction" label="aarch64_vspace_unify_instruction"/> @endxmlonly
 * @brief @xmlonly Clean and invalidate cached instruction pages to point of unification @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the top level translation table being operated on.
 * @param[in] start Start address 
 * @param[in] end End address 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="_service"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="end"/> @endxmlonly  is in the kernel virtual address range.
 * Or, if hypervisor support is configured, the requested range is not entirely
 * within the kernel physical memory window. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="start"/> @endxmlonly  is greater than or equal to  @xmlonly <texttt text="end"/> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_RangeError The specified range crosses a page boundary. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_VSpace_Unify_Instruction(seL4_ARM_VSpace _service, seL4_Word start, seL4_Word end)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMVSpaceUnify_Instruction, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = start;
	mr1 = end;
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
 * @xmlonly <manual name="SMC Call" label="aarch64_smc_call"/> @endxmlonly
 * @brief @xmlonly Tell the kernel to make the real SMC call. @endxmlonly
 * 
 * @xmlonly
 * Takes x0-x7 as arguments to an SMC call which are defined as a seL4_ARM_SMCContext
 * struct. The kernel makes the SMC call and then returns the results as a
 * new seL4_ARM_SMCContext.
 * @endxmlonly
 * 
 * @param[in] _service Capability to allow threads to make Secure Monitor Calls.
 * @param[in] smc_args The structure that has the provided arguments. 
 * @param[out] smc_response The structure to capture the responses. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_SMC_Call(seL4_ARM_SMC _service, seL4_ARM_SMCContext *smc_args, seL4_ARM_SMCContext *smc_response)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMSMCCall, 0, 0, 8);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = smc_args->x0;
	mr1 = smc_args->x1;
	mr2 = smc_args->x2;
	mr3 = smc_args->x3;
	seL4_SetMR(4, smc_args->x4);
	seL4_SetMR(5, smc_args->x5);
	seL4_SetMR(6, smc_args->x6);
	seL4_SetMR(7, smc_args->x7);

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
	smc_response->x0 = mr0;
	smc_response->x1 = mr1;
	smc_response->x2 = mr2;
	smc_response->x3 = mr3;
	smc_response->x4 = seL4_GetMR(4);
	smc_response->x5 = seL4_GetMR(5);
	smc_response->x6 = seL4_GetMR(6);
	smc_response->x7 = seL4_GetMR(7);
	return result;
}

/**
 * @xmlonly <manual name="Map" label="arm_pagetable_map"/> @endxmlonly
 * @brief @xmlonly Map a page table into an address space. @endxmlonly
 * 
 * @xmlonly
 * Takes a VSpace capability as an argument, and installs a
 * reference to the page table in the VSpace at the provided
 * virtual address.
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page table being operated on.
 * @param[in] vspace Capability to the VSpace which will contain the mapping.                 Must be assigned to an ASID pool. 
 * @param[in] vaddr Virtual address to map the page into. 
 * @param[in] attr VM Attributes for the mapping.  @xmlonly <docref>Possible values for this type are given in <autoref label="ch:vspace"/>  .</docref> @endxmlonly 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst A mapping already exists for this level in  @xmlonly <texttt text="vspace"/> @endxmlonly  at  @xmlonly <texttt text="vaddr."/> @endxmlonly 
 * @retval seL4_FailedLookup On aarch64,  @xmlonly <texttt text="vspace"/> @endxmlonly  does not have a Page Directory mapped at  @xmlonly <texttt text="vaddr"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="vspace"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="vaddr"/> @endxmlonly  is in the kernel virtual address range. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="vspace"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="vspace"/> @endxmlonly  is not assigned to an ASID pool.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is already mapped in a VSpace. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_PageTable_Map(seL4_ARM_PageTable _service, seL4_CPtr vspace, seL4_Word vaddr, seL4_ARM_VMAttributes attr)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMPageTableMap, 0, 1, 2);
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
 * @xmlonly <manual name="Unmap" label="arm_pagetable_unmap"/> @endxmlonly
 * @brief @xmlonly Unmap a page table from its <texttt text="Page Directory"/> and zero it out. @endxmlonly
 * 
 * @xmlonly
 * Removes the reference to the invoked <texttt text="Page Table"/> from its
 * containing <texttt text="Page Directory"/>.
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page table being operated on.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RevokeFirst A copy of the  @xmlonly <texttt text="_service"/> @endxmlonly  capability exists. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_PageTable_Unmap(seL4_ARM_PageTable _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMPageTableUnmap, 0, 0, 0);
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

#if defined(CONFIG_TK1_SMMU)
/**
 * @xmlonly <manual name="Map" label="arm_io_page_table_map"/> @endxmlonly
 * @brief @xmlonly Map an IO page table into an IOSpace. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:iospace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the I/O page table being operated on.
 * @param[in] iospace The IOSpace to map the page table into. 
 * @param[in] ioaddr Virtual address at which to map the page table. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst All required page tables are already mapped in  @xmlonly <texttt text="iospace"/> @endxmlonly  at  @xmlonly <texttt text="ioaddr"/> @endxmlonly . 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="iospace"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is already mapped in an IOSpace. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_IOPageTable_Map(seL4_ARM_IOPageTable _service, seL4_ARM_IOSpace iospace, seL4_Word ioaddr)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMIOPageTableMap, 0, 1, 1);
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
#if defined(CONFIG_TK1_SMMU)
/**
 * @xmlonly <manual name="Unmap" label="arm_io_page_table_unmap"/> @endxmlonly
 * @brief @xmlonly Unmap an IO page table from an IOSpace. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the I/O page table being operated on.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RevokeFirst A copy of the  @xmlonly <texttt text="_service"/> @endxmlonly  capability exists. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_IOPageTable_Unmap(seL4_ARM_IOPageTable _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMIOPageTableUnmap, 0, 0, 0);
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
 * @xmlonly <manual name="Map" label="arm_page_map"/> @endxmlonly
 * @brief @xmlonly Map a page into an address space or update the mapping attributes. @endxmlonly
 * 
 * @xmlonly
 * Takes a VSpace capability as an argument and installs a reference
 * to the given <texttt text="Page"/> in the lowest-level unmapped paging structure
 * corresponding to the given address, or updates the mapping attributes if the page is
 * already mapped at this address. The page must not already be mapped through this
 * capability in a different VSpace or at a different address; the page may be mapped
 * in multiple VSpaces by copying the capability.
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page being operated on.
 * @param[in] vspace Capability to the VSpace which will contain the mapping.                 Must be assigned to an ASID pool. 
 * @param[in] vaddr Virtual address to map the page into. 
 * @param[in] rights Rights for the mapping. @xmlonly <docref> Possible values for this type are given in <autoref label="sec:cap_rights"/>  .</docref> @endxmlonly 
 * @param[in] attr VM Attributes for the mapping. @xmlonly <docref> Possible values for this type are given in <autoref label="ch:vspace"/>  .</docref> @endxmlonly 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_AlignmentError The  @xmlonly <texttt text="vaddr"/> @endxmlonly  is not aligned to the page size of  @xmlonly <texttt text="_service"/> @endxmlonly . 
 * @retval seL4_DeleteFirst A mapping already exists in  @xmlonly <texttt text="vspace"/> @endxmlonly  at  @xmlonly <texttt text="vaddr"/> @endxmlonly . 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="vspace"/> @endxmlonly  does not have a paging structure at the required level mapped at  @xmlonly <texttt text="vaddr"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="vspace"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="_service"/> @endxmlonly  is already mapped in  @xmlonly <texttt text="vspace"/> @endxmlonly  at a different virtual address.
 * Or,  @xmlonly <texttt text="vaddr"/> @endxmlonly  is in the kernel virtual address range. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="vspace"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="vspace"/> @endxmlonly  is not assigned to an ASID pool.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is already mapped in a different VSpace. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_Page_Map(seL4_ARM_Page _service, seL4_CPtr vspace, seL4_Word vaddr, seL4_CapRights_t rights, seL4_ARM_VMAttributes attr)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMPageMap, 0, 1, 3);
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
 * @xmlonly <manual name="Unmap" label="arm_page_unmap"/> @endxmlonly
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
seL4_ARM_Page_Unmap(seL4_ARM_Page _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMPageUnmap, 0, 0, 0);
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

#if defined(CONFIG_TK1_SMMU)
/**
 * @xmlonly <manual name="Map I/O" label="arm_page_map_io"/> @endxmlonly
 * @brief @xmlonly Map a page into an IOSpace. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/></docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page being operated on.
 * @param[in] iospace The IOSpace to map the page into. 
 * @param[in] rights Rights for the mapping. @xmlonly <docref> Possible values for this type are given in <autoref label="sec:cap_rights"/>  .</docref> @endxmlonly 
 * @param[in] ioaddr Virtual address at which to map page. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst A mapping already exists in  @xmlonly <texttt text="iospace"/> @endxmlonly  at  @xmlonly <texttt text="ioaddr"/> @endxmlonly . 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="iospace"/> @endxmlonly  does not have a sufficient number of IO Page Tables mapped at  @xmlonly <texttt text="ioaddr"/> @endxmlonly . 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument No rights were specified in  @xmlonly <texttt text="rights"/> @endxmlonly .
 * Or, the rights in the  @xmlonly <texttt text="_service"/> @endxmlonly  capability do not include  @xmlonly <texttt text="rights"/> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="iospace"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is already mapped.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is not a page of size 4 KiB. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_Page_MapIO(seL4_ARM_Page _service, seL4_ARM_IOSpace iospace, seL4_CapRights_t rights, seL4_Word ioaddr)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMPageMapIO, 0, 1, 2);
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
 * @xmlonly <manual name="Clean Data" label="arm_page_clean_data"/> @endxmlonly
 * @brief @xmlonly Cleans the data cache out to RAM. The start and end are relative to the page being serviced. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page being operated on.
 * @param[in] start_offset The offset, relative to the start of the page inclusive. 
 * @param[in] end_offset The offset, relative to the start of the page exclusive. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_FailedLookup The VSpace of  @xmlonly <texttt text="_service"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is not mapped in a VSpace.
 * Or, if hypervisor support is configured, the requested range is not entirely
 * within the kernel physical memory window. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="start_offset"/> @endxmlonly  is greater than or equal to  @xmlonly <texttt text="end_offset"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="start_offset"/> @endxmlonly  or  @xmlonly <texttt text="end_offset"/> @endxmlonly  exceeds the page size of  @xmlonly <texttt text="_service"/> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type or
 * represents a stale mapping. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_Page_Clean_Data(seL4_ARM_Page _service, seL4_Word start_offset, seL4_Word end_offset)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMPageClean_Data, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = start_offset;
	mr1 = end_offset;
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
 * @xmlonly <manual name="Invalidate Data" label="arm_page_invalidate_data"/> @endxmlonly
 * @brief @xmlonly Invalidates the cache range within the given page. The start and end offsets are relative to the page being serviced
 * and should be aligned to a cache line boundary where possible.
 * An additional clean is performed on the outer cache lines if the start and end are
 * not aligned, to clean out the bytes between the requested and the cache line boundary. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page being operated on.
 * @param[in] start_offset The offset, relative to the start of the page inclusive. 
 * @param[in] end_offset The offset, relative to the start of the page exclusive. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_FailedLookup The VSpace of  @xmlonly <texttt text="_service"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is not mapped in a VSpace.
 * Or, if hypervisor support is configured, the requested range is not entirely
 * within the kernel physical memory window. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="start_offset"/> @endxmlonly  is greater than or equal to  @xmlonly <texttt text="end_offset"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="start_offset"/> @endxmlonly  or  @xmlonly <texttt text="end_offset"/> @endxmlonly  exceeds the page size of  @xmlonly <texttt text="_service"/> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type or
 * represents a stale mapping. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_Page_Invalidate_Data(seL4_ARM_Page _service, seL4_Word start_offset, seL4_Word end_offset)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMPageInvalidate_Data, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = start_offset;
	mr1 = end_offset;
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
 * @xmlonly <manual name="Clean and Invalidate Data" label="arm_page_clean_and_invalidate_data"/> @endxmlonly
 * @brief @xmlonly Clean and invalidates the cache range within the given page. The range will be flushed out to RAM.
 * The start and end offsets are relative to the page being serviced. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page being operated on.
 * @param[in] start_offset The offset, relative to the start of the page inclusive. 
 * @param[in] end_offset The offset, relative to the start of the page exclusive. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_FailedLookup The VSpace of  @xmlonly <texttt text="_service"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is not mapped in a VSpace.
 * Or, if hypervisor support is configured, the requested range is not entirely
 * within the kernel physical memory window. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="start_offset"/> @endxmlonly  is greater than or equal to  @xmlonly <texttt text="end_offset"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="start_offset"/> @endxmlonly  or  @xmlonly <texttt text="end_offset"/> @endxmlonly  exceeds the page size of  @xmlonly <texttt text="_service"/> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type or
 * represents a stale mapping. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_Page_CleanInvalidate_Data(seL4_ARM_Page _service, seL4_Word start_offset, seL4_Word end_offset)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMPageCleanInvalidate_Data, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = start_offset;
	mr1 = end_offset;
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
 * @xmlonly <manual name="Unify Instruction" label="arm_page_unify_instruction"/> @endxmlonly
 * @brief @xmlonly Unify Instruction Cache. Cleans data lines to point of unification, invalidate
 * corresponding instruction lines to point of unification, then invalidates branch
 * predictors. The start and end offsets are relative to the page being
 * serviced. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page being operated on.
 * @param[in] start_offset The offset, relative to the start of the page inclusive. 
 * @param[in] end_offset The offset, relative to the start of the page exclusive. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_FailedLookup The VSpace of  @xmlonly <texttt text="_service"/> @endxmlonly  is not assigned to an ASID pool. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is not mapped in a VSpace.
 * Or, if hypervisor support is configured, the requested range is not entirely
 * within the kernel physical memory window. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="start_offset"/> @endxmlonly  is greater than or equal to  @xmlonly <texttt text="end_offset"/> @endxmlonly .
 * Or,  @xmlonly <texttt text="start_offset"/> @endxmlonly  or  @xmlonly <texttt text="end_offset"/> @endxmlonly  exceeds the page size of  @xmlonly <texttt text="_service"/> @endxmlonly . 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type or
 * represents a stale mapping. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_Page_Unify_Instruction(seL4_ARM_Page _service, seL4_Word start_offset, seL4_Word end_offset)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMPageUnify_Instruction, 0, 0, 2);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = start_offset;
	mr1 = end_offset;
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
 * @xmlonly <manual name="Get Address" label="arm_page_get_address"/> @endxmlonly
 * @brief @xmlonly Get the physical address of the underlying frame. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="ch:vspace"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service Capability to the page being operated on.
 * @return @xmlonly 
 *                 A <texttt text="seL4_ARM_Page_GetAddress_t"/> struct that contains a
 *                 <texttt text="seL4_Word paddr"/>, which holds the physical address of the page,
 *                 and <texttt text="int error"/>.<docref> See <autoref label="sec:errors"/> for a description
 *                 of the message register and tag contents upon error.</docref>
 *              @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_ARM_Page_GetAddress_t
seL4_ARM_Page_GetAddress(seL4_ARM_Page _service)
{
	seL4_ARM_Page_GetAddress_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMPageGetAddress, 0, 0, 0);
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

/**
 * @xmlonly <manual name="Make Pool" label="arm_asid_control_make_pool"/> @endxmlonly
 * @brief @xmlonly Create an ASID Pool. @endxmlonly
 * 
 * @xmlonly
 * Together with a capability to <texttt text="Untyped Memory"/>, which is passed as an argument,
 * create an <texttt text="ASID Pool"/>. The untyped capability must represent a
 * 4K memory object. This will create an ASID pool with enough space for 1024 VSpaces.
 * @endxmlonly
 * 
 * @param[in] _service The master ASIDControl capability being operated on.
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
seL4_ARM_ASIDControl_MakePool(seL4_ARM_ASIDControl _service, seL4_Untyped untyped, seL4_CNode root, seL4_Word index, seL4_Uint8 depth)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMASIDControlMakePool, 0, 2, 2);
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
 * @xmlonly <manual name="ASID Pool Assign" label="arm_asidpool_assign"/> @endxmlonly
 * @brief @xmlonly Assign an ASID Pool. @endxmlonly
 * 
 * @xmlonly
 * Assigns an ASID to the VSpace passed in as an argument.
 * @endxmlonly
 * 
 * @param[in] _service The ASID pool which is being assigned to. Must not be full. Each ASID pool can contain 1024 entries.
 * @param[in] vspace The VSpace that is being assigned to an ASID pool. Must not already be assigned to an ASID pool. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst There are no more ASIDs available in  @xmlonly <texttt text="_service"/> @endxmlonly . 
 * @retval seL4_FailedLookup The ASID pool of  @xmlonly <texttt text="_service"/> @endxmlonly  is no longer assigned. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="vspace"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="vspace"/> @endxmlonly  is already assigned to an ASID pool. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_ASIDPool_Assign(seL4_ARM_ASIDPool _service, seL4_CPtr vspace)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMASIDPoolAssign, 0, 1, 0);
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

#if defined(CONFIG_ARM_HYPERVISOR_SUPPORT)
/**
 * @xmlonly <manual name="Set TCB" label="arm_vcpu_set_tcb"/> @endxmlonly
 * @brief @xmlonly Bind a TCB to a virtual CPU. @endxmlonly
 * 
 * @xmlonly
 * There is a 1:1 relationship between a virtual CPU and a TCB. If either (or both) of them is
 * associated with another one, they will be dissociated, and then associated to the
 * ones called in this system calls.
 * @endxmlonly
 * 
 * @param[in] _service Capability to the VCPU being operated on.
 * @param[in] tcb Capability to TCB to bind to a virtual CPU 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="tcb"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_VCPU_SetTCB(seL4_ARM_VCPU _service, seL4_TCB tcb)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMVCPUSetTCB, 0, 1, 0);
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
#if defined(CONFIG_ARM_HYPERVISOR_SUPPORT)
/**
 * @xmlonly <manual name="Inject IRQ" label="arm_vcpu_inject_irq"/> @endxmlonly
 * @brief @xmlonly Inject an IRQ to a virtual CPU. @endxmlonly
 * 
 * @xmlonly
 * Used to queue IRQs towards the VCPU.
 * Writes <texttt text="ICH_LRn_EL2"/> for GICv3 or <texttt text="LRn"/> for GICv2,
 * where <texttt text="n"/> is determined by <texttt text="index"/>.
 * The list register becomes available again when the guest acknowledges
 * the injected interrupt.
 * @endxmlonly
 * 
 * @param[in] _service Capability to the VCPU being operated on.
 * @param[in] virq Virtual IRQ ID 
 * @param[in] priority Priority of the IRQ to be injected 
 * @param[in] group IRQ group 
 * @param[in] index VGIC list register 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The  @xmlonly <texttt text="index"/> @endxmlonly  is in use and not yet handled by the guest. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="virq"/> @endxmlonly ,  @xmlonly <texttt text="priority"/> @endxmlonly ,  @xmlonly <texttt text="group"/> @endxmlonly , or  @xmlonly <texttt text="index"/> @endxmlonly  is invalid. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_VCPU_InjectIRQ(seL4_ARM_VCPU _service, seL4_Uint16 virq, seL4_Uint8 priority, seL4_Uint8 group, seL4_Uint8 index)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMVCPUInjectIRQ, 0, 0, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = (virq & 0xffffull) | ((priority & 0xffull) << 16) | ((group & 0xffull) << 24) | ((index & 0xffull) << 32);
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
#if defined(CONFIG_ARM_HYPERVISOR_SUPPORT)
/**
 * @xmlonly <manual name="Read Registers" label="arm_vcpu_read_registers"/> @endxmlonly
 * @brief @xmlonly Read a virtual CPU register. @endxmlonly
 * 
 * @xmlonly
 * Provides a way to read EL1 system registers, as well as <texttt text="VMPIDR_EL2"/>.
 * @endxmlonly
 * 
 * @param[in] _service Capability to the VCPU being operated on.
 * @param[in] field Register to read from a VCPU 
 * @return @xmlonly 
 *                 A <texttt text="seL4_ARM_VCPU_ReadRegs_t"/> struct that contains a
 *                 <texttt text="seL4_Word value"/>, which holds the value of the system register,
 *                 and <texttt text="int error"/>, which will be non-zero when an error occurred.
 *                 <docref>See <autoref label="sec:errors"/> for a description
 *                 of the message register and tag contents upon error.</docref>
 *              @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="field"/> @endxmlonly  is invalid. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_ARM_VCPU_ReadRegs_t
seL4_ARM_VCPU_ReadRegs(seL4_ARM_VCPU _service, seL4_VCPUReg field)
{
	seL4_ARM_VCPU_ReadRegs_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMVCPUReadReg, 0, 0, 1);
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
#if defined(CONFIG_ARM_HYPERVISOR_SUPPORT)
/**
 * @xmlonly <manual name="Write Registers" label="arm_vcpu_write_registers"/> @endxmlonly
 * @brief @xmlonly Write a virtual CPU register. @endxmlonly
 * 
 * @xmlonly
 * Provides a way to write EL1 system registers, as well as <texttt text="VMPIDR_EL2"/>.
 * @endxmlonly
 * 
 * @param[in] _service Capability to the VCPU being operated on.
 * @param[in] field Register ID to write to a VCPU 
 * @param[in] value Value to be written to the VCPU register 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="field"/> @endxmlonly  is invalid. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_VCPU_WriteRegs(seL4_ARM_VCPU _service, seL4_VCPUReg field, seL4_Word value)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMVCPUWriteReg, 0, 0, 2);
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
#if defined(CONFIG_ARM_HYPERVISOR_SUPPORT)
/**
 * @xmlonly <manual name="Acknowledge Virtual PPI IRQ" label="arm_vcpu_acknowledge_virtual_ppi_irq"/> @endxmlonly
 * @brief @xmlonly Acknowledge a PPI IRQ previously forwarded from a VPPIEvent fault. @endxmlonly
 * 
 * @xmlonly
 * Acknowledge and unmask the PPI interrupt so that further interrupts can be forwarded
 * through VPPIEvent faults.
 * @endxmlonly
 * 
 * @param[in] _service Capability to the VCPU being operated on.
 * @param[in] irq irq to ack. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="irq"/> @endxmlonly  is invalid. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_VCPU_AckVPPI(seL4_ARM_VCPU _service, seL4_Word irq)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMVCPUAckVPPI, 0, 0, 1);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = irq;
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
 * @xmlonly <manual name="Get IRQ Handler with Trigger Type" label="arm_irq_controlgettrigger"/> @endxmlonly
 * @brief @xmlonly Create an IRQ handler capability and specify the trigger method (edge or level). @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:interrupts"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service An IRQControl capability. This gives you the authority to make this call.
 * @param[in] irq The IRQ that you want this capability to handle. 
 * @param[in] trigger Indicates whether this IRQ is edge (1) or level (0) triggered. 
 * @param[in] root CPtr to the CNode that forms the root of the destination CSpace. Must be at a depth equivalent to the wordsize. 
 * @param[in] index CPtr to the destination slot. Resolved from the root of the destination CSpace. 
 * @param[in] depth Number of bits of index to resolve to find the destination slot. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The destination slot contains a capability. 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="index"/> @endxmlonly  or  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly .
 * Or,  @xmlonly <texttt text="root"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, the platform does not support setting the IRQ trigger. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="irq"/> @endxmlonly  is invalid.
 * Or,  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 * @retval seL4_RevokeFirst An IRQ handler capability for  @xmlonly <texttt text="irq"/> @endxmlonly  has already been created. 
 */
LIBSEL4_INLINE seL4_Error
seL4_IRQControl_GetTrigger(seL4_IRQControl _service, seL4_Word irq, seL4_Word trigger, seL4_CNode root, seL4_Word index, seL4_Uint8 depth)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMIRQIssueIRQHandlerTrigger, 0, 1, 4);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, root);

	/* Marshal and initialise parameters. */
	mr0 = irq;
	mr1 = trigger;
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

#if defined(CONFIG_ENABLE_SMP_SUPPORT)
/**
 * @xmlonly <manual name="Get IRQ Handler (SMP)" label="arm_irq_controlgettriggercore"/> @endxmlonly
 * @brief @xmlonly Create an IRQ handler capability and specify the trigger method (edge or level) and the target core. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:interrupts"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service An IRQControl capability. This gives you the authority to make this call.
 * @param[in] irq The IRQ that you want this capability to handle. 
 * @param[in] trigger Indicates whether this IRQ is edge (1) or level (0) triggered. 
 * @param[in] root CPtr to the CNode that forms the root of the destination CSpace. Must be at a depth equivalent to the wordsize. 
 * @param[in] index CPtr to the destination slot. Resolved from the root of the destination CSpace. 
 * @param[in] depth Number of bits of index to resolve to find the destination slot. 
 * @param[in] target Indicates the target core ID to which this IRQ will be sent. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The destination slot contains a capability. 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="index"/> @endxmlonly  or  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly .
 * Or,  @xmlonly <texttt text="root"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, SMP support is not enabled. 
 * @retval seL4_InvalidArgument The  @xmlonly <texttt text="target"/> @endxmlonly  is invalid. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 * @retval seL4_RevokeFirst An IRQ handler capability for  @xmlonly <texttt text="irq"/> @endxmlonly  has already been created. 
 */
LIBSEL4_INLINE seL4_Error
seL4_IRQControl_GetTriggerCore(seL4_IRQControl _service, seL4_Word irq, seL4_Word trigger, seL4_CNode root, seL4_Word index, seL4_Uint8 depth, seL4_Word target)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMIRQIssueIRQHandlerTriggerCore, 0, 1, 5);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, root);

	/* Marshal and initialise parameters. */
	mr0 = irq;
	mr1 = trigger;
	mr2 = index;
	mr3 = (depth & 0xffull);
	seL4_SetMR(4, target);

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
#if !defined(CONFIG_ENABLE_SMP_SUPPORT)
/**
 * @xmlonly <manual name="IssueSGISignal" label="arm_irq_controlissuesgisignal"/> @endxmlonly
 * @brief @xmlonly Create a software generated interrupt (SGI) signal capability. @endxmlonly
 * 
 * @xmlonly
 * Create an SGISignal capability and place it in the specific target CSpace slot. The
 * capability can be used to raise an SGI with a specific ID on a specific target core.
 * Currently this feature is supported on GICv2 and GICv3 hardware. Only available on
 * non-SMP configurations.
 * 
 * The resulting capability can be invoked like a notification capability that supports
 * only signal/send. SGIs can be received by IRQ notification objects on the target
 * core like other IRQs. See also <autoref label="sec:interrupts"/>.
 * 
 * Note that the kernel only checks architectural limits for SGI IRQ id and target. It
 * does not know whether the corresponding target core exists. Depending on hardware
 * implementation, signalling a non-existent target may create an unrecoverable SError.
 * This means it is the responsibility of the developer to not issue capabilities for
 * targets that do not exist.
 * @endxmlonly
 * 
 * @param[in] _service An IRQControl capability. This gives you the authority to make this call.
 * @param[in] irq The SGI INTID (0-15) that can be signalled. 
 * @param[in] target The node ID that will be targeted.             0-7 for GICv2 and the affinity value for GICv3 (concatenation of Aff3+Aff2+Aff1+Aff0 from MPIDR).             Targets within that range that are not supported by thde hardware will be not be rejected. For example,             on a GICv2 board with 4 CPUs, the capability for target 5 can be created. The result of later signalling             this target depends on the hardware implementation. 
 * @param[in] root CPtr to the CNode that forms the root of the destination CSpace. Must be at a depth equivalent to the wordsize. 
 * @param[in] index CPtr to the destination slot. Resolved from the root of the destination CSpace. 
 * @param[in] depth Number of bits of dest_index to resolve to find the destination slot. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The destination slot contains a capability. 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="index"/> @endxmlonly  or  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly .
 * Or,  @xmlonly <texttt text="root"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or, SGIs are not supported on this platform. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The value of  @xmlonly <texttt text="irq"/> @endxmlonly  or  @xmlonly <texttt text="target"/> @endxmlonly  is out of range.
 * Or,  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly . 
 * @retval seL4_InvalidArgument The SGI target is not supported on this GIC. Note that this only checks
 * architectural limits, not the presence of the target core on the current board. 
 */
LIBSEL4_INLINE seL4_Error
seL4_IRQControl_IssueSGISignal(seL4_IRQControl _service, seL4_Word irq, seL4_Word target, seL4_CNode root, seL4_Word index, seL4_Uint8 depth)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMIRQIssueSGISignal, 0, 1, 4);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, root);

	/* Marshal and initialise parameters. */
	mr0 = irq;
	mr1 = target;
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

#endif
#if defined(CONFIG_ARM_SMMU)
/**
 * @xmlonly <manual name="Get SID" label="arm_sid_controlgetsid"/> @endxmlonly
 * @brief @xmlonly Create a SID capability. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:smmuv2-creating-sel4-arm-sid-capabilities"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service A SIDControl capability. This gives you the authority to make this call.
 * @param[in] sid The SID that you want this capability to manage. 
 * @param[in] root CPtr to the CNode that forms the root of the destination CSpace. Must be at a depth equivalent to the wordsize. 
 * @param[in] index CPtr to the destination slot. Resolved from the root of the destination CSpace. 
 * @param[in] depth Number of bits of index to resolve to find the destination slot. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The destination slot contains a capability. 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="index"/> @endxmlonly  or  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly .
 * Or,  @xmlonly <texttt text="root"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly .
 * Or,  @xmlonly <texttt text="sid"/> @endxmlonly  is invalid. 
 * @retval seL4_RevokeFirst An SID capability for  @xmlonly <texttt text="sid"/> @endxmlonly  has already been created. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_SIDControl_GetSID(seL4_ARM_SIDControl _service, seL4_Word sid, seL4_CNode root, seL4_Word index, seL4_Uint8 depth)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMSIDIssueSIDManager, 0, 1, 3);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, root);

	/* Marshal and initialise parameters. */
	mr0 = sid;
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

#endif
#if defined(CONFIG_ARM_SMMU)
/**
 * @xmlonly <manual name="Get Fault" label="arm_sid_controlgetfault"/> @endxmlonly
 * @brief @xmlonly Get the fault status of the SMMU. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:smmuv2-fault-handling"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service A SIDControl capability. This gives you the authority to make this call.
 * @return @xmlonly 
 *                 A <texttt text="seL4_ARM_SMMU_GetFault_t"/> struct that contains a
 *                 <texttt text="seL4_Word status"/>, which holds the global fault status of the SMMU,
 *                 <texttt text="seL4_Word syndrome_0"/>, which holds the global fault syndrome 0 of the SMMU,
 *                 <texttt text="seL4_Word syndrome_1"/>, which holds the global fault syndrome 1 of the SMMU,
 *                 and <texttt text="int error"/>.<docref> See <autoref label="sec:errors"/> for a description
 *                 of the message register and tag contents upon error.</docref>
 *              @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_ARM_SIDControl_GetFault_t
seL4_ARM_SIDControl_GetFault(seL4_ARM_SIDControl _service)
{
	seL4_ARM_SIDControl_GetFault_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMSIDGetFault, 0, 0, 0);
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
	result.status = mr0;
	result.syndrome_0 = mr1;
	result.syndrome_1 = mr2;
	return result;
}

#endif
#if defined(CONFIG_ARM_SMMU)
/**
 * @xmlonly <manual name="Clear Fault" label="arm_sid_controlclearfault"/> @endxmlonly
 * @brief @xmlonly Clear the fault status of the SMMU. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:smmuv2-fault-handling"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service A SIDControl capability. This gives you the authority to make this call.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_SIDControl_ClearFault(seL4_ARM_SIDControl _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMSIDClearFault, 0, 0, 0);
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
#if defined(CONFIG_ARM_SMMU)
/**
 * @xmlonly <manual name="Bind CB" label="arm_sid_bindcb"/> @endxmlonly
 * @brief @xmlonly Binding a context bank to a stream ID. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:smmuv2-configuring-streams-transactions"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service A SID capability. This gives you the authority to make this call.
 * @param[in] cb The CB that is being binded to a stream ID. Must already has an assigned vspace. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The  @xmlonly <texttt text="_service"/> @endxmlonly  is already bound to a context bank. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="cb"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="cb"/> @endxmlonly  is not assigned to a VSpace. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_SID_BindCB(seL4_ARM_SID _service, seL4_CPtr cb)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMSIDBindCB, 0, 1, 0);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, cb);

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
#if defined(CONFIG_ARM_SMMU)
/**
 * @xmlonly <manual name="Unbind CB" label="arm_sid_unbindcb"/> @endxmlonly
 * @brief @xmlonly Unbinding a context bank from a stream ID. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:smmuv2-configuring-streams-transactions"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service A SID capability. This gives you the authority to make this call.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is not bound to a context block. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_SID_UnbindCB(seL4_ARM_SID _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMSIDUnbindCB, 0, 0, 0);
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
#if defined(CONFIG_ARM_SMMU)
/**
 * @xmlonly <manual name="Get CB" label="arm_cb_controlgetcb"/> @endxmlonly
 * @brief @xmlonly Create a CB capability. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:smmuv2-creating-sel4-arm-cb-capabilities"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service A CBControl capability. This gives you the authority to make this call.
 * @param[in] cb The CB that you want this capability to manage. 
 * @param[in] root CPtr to the CNode that forms the root of the destination CSpace. Must be at a depth equivalent to the wordsize. 
 * @param[in] index CPtr to the destination slot. Resolved from the root of the destination CSpace. 
 * @param[in] depth Number of bits of index to resolve to find the destination slot. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The destination slot contains a capability. 
 * @retval seL4_FailedLookup The  @xmlonly <texttt text="index"/> @endxmlonly  or  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly .
 * Or,  @xmlonly <texttt text="root"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_RangeError The  @xmlonly <texttt text="depth"/> @endxmlonly  is invalid  @xmlonly <docref>(see <autoref label="s:cspace-addressing"/>)</docref> @endxmlonly .
 * Or,  @xmlonly <texttt text="cb"/> @endxmlonly  is invalid. 
 * @retval seL4_RevokeFirst A CB capability for  @xmlonly <texttt text="cb"/> @endxmlonly  has already been created. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_CBControl_GetCB(seL4_ARM_CBControl _service, seL4_Word cb, seL4_CNode root, seL4_Word index, seL4_Uint8 depth)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMCBIssueCBManager, 0, 1, 3);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Setup input capabilities. */
	seL4_SetCap(0, root);

	/* Marshal and initialise parameters. */
	mr0 = cb;
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

#endif
#if defined(CONFIG_ARM_SMMU)
/**
 * @xmlonly <manual name="TLB Invalidate All" label="arm_cb_controltlbinvalidate"/> @endxmlonly
 * @brief @xmlonly Invalidate all TLB entries. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:smmuv2-tlb-invalidation"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service A CBControl capability. This gives you the authority to make this call.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_CBControl_TLBInvalidateAll(seL4_ARM_CBControl _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMCBTLBInvalidateAll, 0, 0, 0);
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
#if defined(CONFIG_ARM_SMMU)
/**
 * @xmlonly <manual name="Assign VSpace" label="arm_cb_assignvspace"/> @endxmlonly
 * @brief @xmlonly Assigning a VSpace to a context bank. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:smmuv2-configuring-context-banks"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service A CB capability. This gives you the authority to make this call.
 * @param[in] vspace The VSpace that is being assigned to a context bank. Must already has an assigned ASID. 
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_DeleteFirst The  @xmlonly <texttt text="_service"/> @endxmlonly  is already assigned to a VSpace. 
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  or  @xmlonly <texttt text="vspace"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="vspace"/> @endxmlonly  is not assigned to an ASID pool. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_CB_AssignVspace(seL4_ARM_CB _service, seL4_CPtr vspace)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMCBAssignVspace, 0, 1, 0);
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

#endif
#if defined(CONFIG_ARM_SMMU)
/**
 * @xmlonly <manual name="Unassign VSpace" label="arm_cb_unassignvspace"/> @endxmlonly
 * @brief @xmlonly Unassigning a VSpace to a context bank. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:smmuv2-configuring-context-banks"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service A CB capability. This gives you the authority to make this call.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is not assigned to a VSpace. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_CB_UnassignVspace(seL4_ARM_CB _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMCBUnassignVspace, 0, 0, 0);
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
#if defined(CONFIG_ARM_SMMU)
/**
 * @xmlonly <manual name="TLB Invalidate" label="arm_cb_tlbinvalidate"/> @endxmlonly
 * @brief @xmlonly Invalidating TLB entries used by the current ASID in this context bank. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:smmuv2-tlb-invalidation"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service A CB capability. This gives you the authority to make this call.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type.
 * Or,  @xmlonly <texttt text="_service"/> @endxmlonly  is not assigned to a VSpace. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_CB_TLBInvalidate(seL4_ARM_CB _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMCBTLBInvalidate, 0, 0, 0);
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
#if defined(CONFIG_ARM_SMMU)
/**
 * @xmlonly <manual name="CB Get Fault" label="arm_cb_getfault"/> @endxmlonly
 * @brief @xmlonly Get the fault status of the context bank. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:smmuv2-fault-handling"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service A CB capability. This gives you the authority to make this call.
 * @return @xmlonly 
 *                 A <texttt text="seL4_ARM_SMMU_CB_GetFault_t"/> struct that contains a
 *                 <texttt text="seL4_Word status"/>, which holds the fault status of the context bank,
 *                 <texttt text="seL4_Word address"/>, which holds the faulty address,
 *                 and <texttt text="int error"/>.<docref> See <autoref label="sec:errors"/> for a description
 *                 of the message register and tag contents upon error.</docref>
 *              @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_ARM_CB_CBGetFault_t
seL4_ARM_CB_CBGetFault(seL4_ARM_CB _service)
{
	seL4_ARM_CB_CBGetFault_t result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMCBGetFault, 0, 0, 0);
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
	result.status = mr0;
	result.address = mr1;
	return result;
}

#endif
#if defined(CONFIG_ARM_SMMU)
/**
 * @xmlonly <manual name="CB Clear Fault" label="arm_cb_clearfault"/> @endxmlonly
 * @brief @xmlonly Clear the fault status of the context bank. @endxmlonly
 * 
 * @xmlonly
 * <docref>See <autoref label="sec:smmuv2-fault-handling"/>.</docref>
 * @endxmlonly
 * 
 * @param[in] _service A CB capability. This gives you the authority to make this call.
 * @return @xmlonly <errorenumdesc/> @endxmlonly
 * @retval seL4_IllegalOperation The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 * @retval seL4_InvalidCapability The  @xmlonly <texttt text="_service"/> @endxmlonly  is a CPtr to a capability of the wrong type. 
 */
LIBSEL4_INLINE seL4_Error
seL4_ARM_CB_CBClearFault(seL4_ARM_CB _service)
{
	seL4_Error result;
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(ARMCBClearFault, 0, 0, 0);
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
	regs->pc = mr0;
	regs->sp = mr1;
	regs->spsr = mr2;
	regs->x0 = mr3;
	for (seL4_Word i = 4; i < count; i++) {
		((seL4_Word *) &regs->x1)[i - 4] = seL4_GetMR(i);
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
	seL4_MessageInfo_t tag = seL4_MessageInfo_new(TCBWriteRegisters, 0, 0, 38);
	seL4_MessageInfo_t output_tag;
	seL4_Word mr0;
	seL4_Word mr1;
	seL4_Word mr2;
	seL4_Word mr3;

	/* Marshal and initialise parameters. */
	mr0 = (resume_target & 0x1ull) | ((arch_flags & 0xffull) << 8);
	mr1 = count;
	mr2 = regs->pc;
	mr3 = regs->sp;
	for (seL4_Word i = 2; i < count; i++) {
		seL4_SetMR(i + 2, ((seL4_Word*)&regs->spsr)[i - 2]);
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