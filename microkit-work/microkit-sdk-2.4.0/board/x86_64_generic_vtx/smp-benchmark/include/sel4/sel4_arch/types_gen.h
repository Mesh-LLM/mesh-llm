/* generated from libsel4/sel4_arch_include/x86_64/sel4/sel4_arch/types.bf */

#pragma once

#include <sel4/config.h>
#include <sel4/simple_types.h>
#include <sel4/debug_assert.h>
struct seL4_X86_IOSpace_CapData {
    seL4_Uint64 words[1];
};
typedef struct seL4_X86_IOSpace_CapData seL4_X86_IOSpace_CapData_t;

LIBSEL4_INLINE_FUNC seL4_X86_IOSpace_CapData_t CONST
seL4_X86_IOSpace_CapData_new(seL4_Uint64 domainID, seL4_Uint64 pciBus, seL4_Uint64 pciDev, seL4_Uint64 pciFun) {
    seL4_X86_IOSpace_CapData_t seL4_X86_IOSpace_CapData;

    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert((domainID & ~0xffffull) == ((0 && (domainID & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((pciBus & ~0xffull) == ((0 && (pciBus & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((pciDev & ~0x1full) == ((0 && (pciDev & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((pciFun & ~0x7ull) == ((0 && (pciFun & (1ull << 63))) ? 0x0 : 0));

    seL4_X86_IOSpace_CapData.words[0] = 0
        | (domainID & 0xffffull) << 16
        | (pciBus & 0xffull) << 8
        | (pciDev & 0x1full) << 3
        | (pciFun & 0x7ull) << 0;

    return seL4_X86_IOSpace_CapData;
}

LIBSEL4_INLINE_FUNC void
seL4_X86_IOSpace_CapData_ptr_new(seL4_X86_IOSpace_CapData_t *seL4_X86_IOSpace_CapData_ptr, seL4_Uint64 domainID, seL4_Uint64 pciBus, seL4_Uint64 pciDev, seL4_Uint64 pciFun) {
    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert((domainID & ~0xffffull) == ((0 && (domainID & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((pciBus & ~0xffull) == ((0 && (pciBus & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((pciDev & ~0x1full) == ((0 && (pciDev & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((pciFun & ~0x7ull) == ((0 && (pciFun & (1ull << 63))) ? 0x0 : 0));

    seL4_X86_IOSpace_CapData_ptr->words[0] = 0
        | (domainID & 0xffffull) << 16
        | (pciBus & 0xffull) << 8
        | (pciDev & 0x1full) << 3
        | (pciFun & 0x7ull) << 0;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_X86_IOSpace_CapData_get_domainID(seL4_X86_IOSpace_CapData_t seL4_X86_IOSpace_CapData) {
    seL4_Uint64 ret;
    ret = (seL4_X86_IOSpace_CapData.words[0] & 0xffff0000ull) >> 16;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_X86_IOSpace_CapData_t CONST
seL4_X86_IOSpace_CapData_set_domainID(seL4_X86_IOSpace_CapData_t seL4_X86_IOSpace_CapData, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffff0000ull >> 16 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_X86_IOSpace_CapData.words[0] &= ~0xffff0000ull;
    seL4_X86_IOSpace_CapData.words[0] |= (v64 << 16) & 0xffff0000ull;
    return seL4_X86_IOSpace_CapData;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_X86_IOSpace_CapData_ptr_get_domainID(seL4_X86_IOSpace_CapData_t *seL4_X86_IOSpace_CapData_ptr) {
    seL4_Uint64 ret;
    ret = (seL4_X86_IOSpace_CapData_ptr->words[0] & 0xffff0000ull) >> 16;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_X86_IOSpace_CapData_ptr_set_domainID(seL4_X86_IOSpace_CapData_t *seL4_X86_IOSpace_CapData_ptr, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffff0000ull >> 16) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_X86_IOSpace_CapData_ptr->words[0] &= ~0xffff0000ull;
    seL4_X86_IOSpace_CapData_ptr->words[0] |= (v64 << 16) & 0xffff0000;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_X86_IOSpace_CapData_get_pciBus(seL4_X86_IOSpace_CapData_t seL4_X86_IOSpace_CapData) {
    seL4_Uint64 ret;
    ret = (seL4_X86_IOSpace_CapData.words[0] & 0xff00ull) >> 8;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_X86_IOSpace_CapData_t CONST
seL4_X86_IOSpace_CapData_set_pciBus(seL4_X86_IOSpace_CapData_t seL4_X86_IOSpace_CapData, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xff00ull >> 8 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_X86_IOSpace_CapData.words[0] &= ~0xff00ull;
    seL4_X86_IOSpace_CapData.words[0] |= (v64 << 8) & 0xff00ull;
    return seL4_X86_IOSpace_CapData;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_X86_IOSpace_CapData_ptr_get_pciBus(seL4_X86_IOSpace_CapData_t *seL4_X86_IOSpace_CapData_ptr) {
    seL4_Uint64 ret;
    ret = (seL4_X86_IOSpace_CapData_ptr->words[0] & 0xff00ull) >> 8;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_X86_IOSpace_CapData_ptr_set_pciBus(seL4_X86_IOSpace_CapData_t *seL4_X86_IOSpace_CapData_ptr, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xff00ull >> 8) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_X86_IOSpace_CapData_ptr->words[0] &= ~0xff00ull;
    seL4_X86_IOSpace_CapData_ptr->words[0] |= (v64 << 8) & 0xff00;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_X86_IOSpace_CapData_get_pciDev(seL4_X86_IOSpace_CapData_t seL4_X86_IOSpace_CapData) {
    seL4_Uint64 ret;
    ret = (seL4_X86_IOSpace_CapData.words[0] & 0xf8ull) >> 3;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_X86_IOSpace_CapData_t CONST
seL4_X86_IOSpace_CapData_set_pciDev(seL4_X86_IOSpace_CapData_t seL4_X86_IOSpace_CapData, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xf8ull >> 3 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_X86_IOSpace_CapData.words[0] &= ~0xf8ull;
    seL4_X86_IOSpace_CapData.words[0] |= (v64 << 3) & 0xf8ull;
    return seL4_X86_IOSpace_CapData;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_X86_IOSpace_CapData_ptr_get_pciDev(seL4_X86_IOSpace_CapData_t *seL4_X86_IOSpace_CapData_ptr) {
    seL4_Uint64 ret;
    ret = (seL4_X86_IOSpace_CapData_ptr->words[0] & 0xf8ull) >> 3;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_X86_IOSpace_CapData_ptr_set_pciDev(seL4_X86_IOSpace_CapData_t *seL4_X86_IOSpace_CapData_ptr, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xf8ull >> 3) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_X86_IOSpace_CapData_ptr->words[0] &= ~0xf8ull;
    seL4_X86_IOSpace_CapData_ptr->words[0] |= (v64 << 3) & 0xf8;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_X86_IOSpace_CapData_get_pciFun(seL4_X86_IOSpace_CapData_t seL4_X86_IOSpace_CapData) {
    seL4_Uint64 ret;
    ret = (seL4_X86_IOSpace_CapData.words[0] & 0x7ull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_X86_IOSpace_CapData_t CONST
seL4_X86_IOSpace_CapData_set_pciFun(seL4_X86_IOSpace_CapData_t seL4_X86_IOSpace_CapData, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0x7ull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_X86_IOSpace_CapData.words[0] &= ~0x7ull;
    seL4_X86_IOSpace_CapData.words[0] |= (v64 << 0) & 0x7ull;
    return seL4_X86_IOSpace_CapData;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_X86_IOSpace_CapData_ptr_get_pciFun(seL4_X86_IOSpace_CapData_t *seL4_X86_IOSpace_CapData_ptr) {
    seL4_Uint64 ret;
    ret = (seL4_X86_IOSpace_CapData_ptr->words[0] & 0x7ull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_X86_IOSpace_CapData_ptr_set_pciFun(seL4_X86_IOSpace_CapData_t *seL4_X86_IOSpace_CapData_ptr, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0x7ull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_X86_IOSpace_CapData_ptr->words[0] &= ~0x7ull;
    seL4_X86_IOSpace_CapData_ptr->words[0] |= (v64 << 0) & 0x7;
}

struct seL4_Fault {
    seL4_Uint64 words[20];
};
typedef struct seL4_Fault seL4_Fault_t;

enum seL4_Fault_tag {
    seL4_Fault_NullFault = 0,
    seL4_Fault_CapFault = 1,
    seL4_Fault_UnknownSyscall = 2,
    seL4_Fault_UserException = 3,
    seL4_Fault_Timeout = 5,
    seL4_Fault_VMFault = 6
};
typedef enum seL4_Fault_tag seL4_Fault_tag_t;

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_get_seL4_FaultType(seL4_Fault_t seL4_Fault) {
    return (seL4_Fault.words[0] >> 0) & 0xfull;
}

LIBSEL4_INLINE_FUNC int CONST
seL4_Fault_seL4_FaultType_equals(seL4_Fault_t seL4_Fault, seL4_Uint64 seL4_Fault_type_tag) {
    return ((seL4_Fault.words[0] >> 0) & 0xfull) == seL4_Fault_type_tag;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_ptr_get_seL4_FaultType(seL4_Fault_t *seL4_Fault_ptr) {
    return (seL4_Fault_ptr->words[0] >> 0) & 0xfull;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_NullFault_new(void) {
    seL4_Fault_t seL4_Fault;

    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert(((seL4_Uint64)seL4_Fault_NullFault & ~0xfull) == ((0 && ((seL4_Uint64)seL4_Fault_NullFault & (1ull << 63))) ? 0x0 : 0));

    seL4_Fault.words[0] = 0
        | ((seL4_Uint64)seL4_Fault_NullFault & 0xfull) << 0;
    seL4_Fault.words[1] = 0;
    seL4_Fault.words[2] = 0;
    seL4_Fault.words[3] = 0;
    seL4_Fault.words[4] = 0;
    seL4_Fault.words[5] = 0;
    seL4_Fault.words[6] = 0;
    seL4_Fault.words[7] = 0;
    seL4_Fault.words[8] = 0;
    seL4_Fault.words[9] = 0;
    seL4_Fault.words[10] = 0;
    seL4_Fault.words[11] = 0;
    seL4_Fault.words[12] = 0;
    seL4_Fault.words[13] = 0;
    seL4_Fault.words[14] = 0;
    seL4_Fault.words[15] = 0;
    seL4_Fault.words[16] = 0;
    seL4_Fault.words[17] = 0;
    seL4_Fault.words[18] = 0;
    seL4_Fault.words[19] = 0;

    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_NullFault_ptr_new(seL4_Fault_t *seL4_Fault_ptr) {
    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert(((seL4_Uint64)seL4_Fault_NullFault & ~0xfull) == ((0 && ((seL4_Uint64)seL4_Fault_NullFault & (1ull << 63))) ? 0x0 : 0));

    seL4_Fault_ptr->words[0] = 0
        | ((seL4_Uint64)seL4_Fault_NullFault & 0xfull) << 0;
    seL4_Fault_ptr->words[1] = 0;
    seL4_Fault_ptr->words[2] = 0;
    seL4_Fault_ptr->words[3] = 0;
    seL4_Fault_ptr->words[4] = 0;
    seL4_Fault_ptr->words[5] = 0;
    seL4_Fault_ptr->words[6] = 0;
    seL4_Fault_ptr->words[7] = 0;
    seL4_Fault_ptr->words[8] = 0;
    seL4_Fault_ptr->words[9] = 0;
    seL4_Fault_ptr->words[10] = 0;
    seL4_Fault_ptr->words[11] = 0;
    seL4_Fault_ptr->words[12] = 0;
    seL4_Fault_ptr->words[13] = 0;
    seL4_Fault_ptr->words[14] = 0;
    seL4_Fault_ptr->words[15] = 0;
    seL4_Fault_ptr->words[16] = 0;
    seL4_Fault_ptr->words[17] = 0;
    seL4_Fault_ptr->words[18] = 0;
    seL4_Fault_ptr->words[19] = 0;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_CapFault_new(seL4_Uint64 IP, seL4_Uint64 Addr, seL4_Uint64 InRecvPhase, seL4_Uint64 LookupFailureType, seL4_Uint64 MR4, seL4_Uint64 MR5, seL4_Uint64 MR6) {
    seL4_Fault_t seL4_Fault;

    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert(((seL4_Uint64)seL4_Fault_CapFault & ~0xfull) == ((0 && ((seL4_Uint64)seL4_Fault_CapFault & (1ull << 63))) ? 0x0 : 0));

    seL4_Fault.words[0] = 0
        | ((seL4_Uint64)seL4_Fault_CapFault & 0xfull) << 0;
    seL4_Fault.words[1] = 0
        | MR6 << 0;
    seL4_Fault.words[2] = 0
        | MR5 << 0;
    seL4_Fault.words[3] = 0
        | MR4 << 0;
    seL4_Fault.words[4] = 0
        | LookupFailureType << 0;
    seL4_Fault.words[5] = 0
        | InRecvPhase << 0;
    seL4_Fault.words[6] = 0
        | Addr << 0;
    seL4_Fault.words[7] = 0
        | IP << 0;
    seL4_Fault.words[8] = 0;
    seL4_Fault.words[9] = 0;
    seL4_Fault.words[10] = 0;
    seL4_Fault.words[11] = 0;
    seL4_Fault.words[12] = 0;
    seL4_Fault.words[13] = 0;
    seL4_Fault.words[14] = 0;
    seL4_Fault.words[15] = 0;
    seL4_Fault.words[16] = 0;
    seL4_Fault.words[17] = 0;
    seL4_Fault.words[18] = 0;
    seL4_Fault.words[19] = 0;

    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_CapFault_ptr_new(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 IP, seL4_Uint64 Addr, seL4_Uint64 InRecvPhase, seL4_Uint64 LookupFailureType, seL4_Uint64 MR4, seL4_Uint64 MR5, seL4_Uint64 MR6) {
    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert(((seL4_Uint64)seL4_Fault_CapFault & ~0xfull) == ((0 && ((seL4_Uint64)seL4_Fault_CapFault & (1ull << 63))) ? 0x0 : 0));

    seL4_Fault_ptr->words[0] = 0
        | ((seL4_Uint64)seL4_Fault_CapFault & 0xfull) << 0;
    seL4_Fault_ptr->words[1] = 0
        | MR6 << 0;
    seL4_Fault_ptr->words[2] = 0
        | MR5 << 0;
    seL4_Fault_ptr->words[3] = 0
        | MR4 << 0;
    seL4_Fault_ptr->words[4] = 0
        | LookupFailureType << 0;
    seL4_Fault_ptr->words[5] = 0
        | InRecvPhase << 0;
    seL4_Fault_ptr->words[6] = 0
        | Addr << 0;
    seL4_Fault_ptr->words[7] = 0
        | IP << 0;
    seL4_Fault_ptr->words[8] = 0;
    seL4_Fault_ptr->words[9] = 0;
    seL4_Fault_ptr->words[10] = 0;
    seL4_Fault_ptr->words[11] = 0;
    seL4_Fault_ptr->words[12] = 0;
    seL4_Fault_ptr->words[13] = 0;
    seL4_Fault_ptr->words[14] = 0;
    seL4_Fault_ptr->words[15] = 0;
    seL4_Fault_ptr->words[16] = 0;
    seL4_Fault_ptr->words[17] = 0;
    seL4_Fault_ptr->words[18] = 0;
    seL4_Fault_ptr->words[19] = 0;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_CapFault_get_IP(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    ret = (seL4_Fault.words[7] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_CapFault_set_IP(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_CapFault);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[7] &= ~0xffffffffffffffffull;
    seL4_Fault.words[7] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_CapFault_ptr_get_IP(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    ret = (seL4_Fault_ptr->words[7] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_CapFault_ptr_set_IP(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[7] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[7] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_CapFault_get_Addr(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    ret = (seL4_Fault.words[6] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_CapFault_set_Addr(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_CapFault);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[6] &= ~0xffffffffffffffffull;
    seL4_Fault.words[6] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_CapFault_ptr_get_Addr(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    ret = (seL4_Fault_ptr->words[6] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_CapFault_ptr_set_Addr(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[6] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[6] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_CapFault_get_InRecvPhase(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    ret = (seL4_Fault.words[5] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_CapFault_set_InRecvPhase(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_CapFault);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[5] &= ~0xffffffffffffffffull;
    seL4_Fault.words[5] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_CapFault_ptr_get_InRecvPhase(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    ret = (seL4_Fault_ptr->words[5] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_CapFault_ptr_set_InRecvPhase(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[5] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[5] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_CapFault_get_LookupFailureType(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    ret = (seL4_Fault.words[4] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_CapFault_set_LookupFailureType(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_CapFault);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[4] &= ~0xffffffffffffffffull;
    seL4_Fault.words[4] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_CapFault_ptr_get_LookupFailureType(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    ret = (seL4_Fault_ptr->words[4] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_CapFault_ptr_set_LookupFailureType(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[4] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[4] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_CapFault_get_MR4(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    ret = (seL4_Fault.words[3] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_CapFault_set_MR4(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_CapFault);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[3] &= ~0xffffffffffffffffull;
    seL4_Fault.words[3] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_CapFault_ptr_get_MR4(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    ret = (seL4_Fault_ptr->words[3] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_CapFault_ptr_set_MR4(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[3] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[3] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_CapFault_get_MR5(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    ret = (seL4_Fault.words[2] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_CapFault_set_MR5(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_CapFault);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[2] &= ~0xffffffffffffffffull;
    seL4_Fault.words[2] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_CapFault_ptr_get_MR5(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    ret = (seL4_Fault_ptr->words[2] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_CapFault_ptr_set_MR5(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[2] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[2] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_CapFault_get_MR6(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    ret = (seL4_Fault.words[1] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_CapFault_set_MR6(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_CapFault);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[1] &= ~0xffffffffffffffffull;
    seL4_Fault.words[1] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_CapFault_ptr_get_MR6(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    ret = (seL4_Fault_ptr->words[1] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_CapFault_ptr_set_MR6(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_CapFault);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[1] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[1] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_new(seL4_Uint64 RAX, seL4_Uint64 RBX, seL4_Uint64 RCX, seL4_Uint64 RDX, seL4_Uint64 RSI, seL4_Uint64 RDI, seL4_Uint64 RBP, seL4_Uint64 R8, seL4_Uint64 R9, seL4_Uint64 R10, seL4_Uint64 R11, seL4_Uint64 R12, seL4_Uint64 R13, seL4_Uint64 R14, seL4_Uint64 R15, seL4_Uint64 FaultIP, seL4_Uint64 RSP, seL4_Uint64 FLAGS, seL4_Uint64 Syscall) {
    seL4_Fault_t seL4_Fault;

    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert(((seL4_Uint64)seL4_Fault_UnknownSyscall & ~0xfull) == ((0 && ((seL4_Uint64)seL4_Fault_UnknownSyscall & (1ull << 63))) ? 0x0 : 0));

    seL4_Fault.words[0] = 0
        | ((seL4_Uint64)seL4_Fault_UnknownSyscall & 0xfull) << 0;
    seL4_Fault.words[1] = 0
        | Syscall << 0;
    seL4_Fault.words[2] = 0
        | FLAGS << 0;
    seL4_Fault.words[3] = 0
        | RSP << 0;
    seL4_Fault.words[4] = 0
        | FaultIP << 0;
    seL4_Fault.words[5] = 0
        | R15 << 0;
    seL4_Fault.words[6] = 0
        | R14 << 0;
    seL4_Fault.words[7] = 0
        | R13 << 0;
    seL4_Fault.words[8] = 0
        | R12 << 0;
    seL4_Fault.words[9] = 0
        | R11 << 0;
    seL4_Fault.words[10] = 0
        | R10 << 0;
    seL4_Fault.words[11] = 0
        | R9 << 0;
    seL4_Fault.words[12] = 0
        | R8 << 0;
    seL4_Fault.words[13] = 0
        | RBP << 0;
    seL4_Fault.words[14] = 0
        | RDI << 0;
    seL4_Fault.words[15] = 0
        | RSI << 0;
    seL4_Fault.words[16] = 0
        | RDX << 0;
    seL4_Fault.words[17] = 0
        | RCX << 0;
    seL4_Fault.words[18] = 0
        | RBX << 0;
    seL4_Fault.words[19] = 0
        | RAX << 0;

    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_new(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 RAX, seL4_Uint64 RBX, seL4_Uint64 RCX, seL4_Uint64 RDX, seL4_Uint64 RSI, seL4_Uint64 RDI, seL4_Uint64 RBP, seL4_Uint64 R8, seL4_Uint64 R9, seL4_Uint64 R10, seL4_Uint64 R11, seL4_Uint64 R12, seL4_Uint64 R13, seL4_Uint64 R14, seL4_Uint64 R15, seL4_Uint64 FaultIP, seL4_Uint64 RSP, seL4_Uint64 FLAGS, seL4_Uint64 Syscall) {
    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert(((seL4_Uint64)seL4_Fault_UnknownSyscall & ~0xfull) == ((0 && ((seL4_Uint64)seL4_Fault_UnknownSyscall & (1ull << 63))) ? 0x0 : 0));

    seL4_Fault_ptr->words[0] = 0
        | ((seL4_Uint64)seL4_Fault_UnknownSyscall & 0xfull) << 0;
    seL4_Fault_ptr->words[1] = 0
        | Syscall << 0;
    seL4_Fault_ptr->words[2] = 0
        | FLAGS << 0;
    seL4_Fault_ptr->words[3] = 0
        | RSP << 0;
    seL4_Fault_ptr->words[4] = 0
        | FaultIP << 0;
    seL4_Fault_ptr->words[5] = 0
        | R15 << 0;
    seL4_Fault_ptr->words[6] = 0
        | R14 << 0;
    seL4_Fault_ptr->words[7] = 0
        | R13 << 0;
    seL4_Fault_ptr->words[8] = 0
        | R12 << 0;
    seL4_Fault_ptr->words[9] = 0
        | R11 << 0;
    seL4_Fault_ptr->words[10] = 0
        | R10 << 0;
    seL4_Fault_ptr->words[11] = 0
        | R9 << 0;
    seL4_Fault_ptr->words[12] = 0
        | R8 << 0;
    seL4_Fault_ptr->words[13] = 0
        | RBP << 0;
    seL4_Fault_ptr->words[14] = 0
        | RDI << 0;
    seL4_Fault_ptr->words[15] = 0
        | RSI << 0;
    seL4_Fault_ptr->words[16] = 0
        | RDX << 0;
    seL4_Fault_ptr->words[17] = 0
        | RCX << 0;
    seL4_Fault_ptr->words[18] = 0
        | RBX << 0;
    seL4_Fault_ptr->words[19] = 0
        | RAX << 0;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_RAX(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[19] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_RAX(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[19] &= ~0xffffffffffffffffull;
    seL4_Fault.words[19] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_RAX(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[19] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_RAX(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[19] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[19] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_RBX(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[18] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_RBX(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[18] &= ~0xffffffffffffffffull;
    seL4_Fault.words[18] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_RBX(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[18] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_RBX(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[18] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[18] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_RCX(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[17] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_RCX(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[17] &= ~0xffffffffffffffffull;
    seL4_Fault.words[17] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_RCX(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[17] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_RCX(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[17] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[17] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_RDX(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[16] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_RDX(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[16] &= ~0xffffffffffffffffull;
    seL4_Fault.words[16] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_RDX(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[16] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_RDX(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[16] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[16] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_RSI(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[15] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_RSI(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[15] &= ~0xffffffffffffffffull;
    seL4_Fault.words[15] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_RSI(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[15] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_RSI(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[15] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[15] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_RDI(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[14] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_RDI(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[14] &= ~0xffffffffffffffffull;
    seL4_Fault.words[14] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_RDI(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[14] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_RDI(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[14] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[14] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_RBP(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[13] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_RBP(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[13] &= ~0xffffffffffffffffull;
    seL4_Fault.words[13] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_RBP(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[13] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_RBP(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[13] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[13] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_R8(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[12] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_R8(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[12] &= ~0xffffffffffffffffull;
    seL4_Fault.words[12] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_R8(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[12] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_R8(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[12] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[12] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_R9(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[11] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_R9(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[11] &= ~0xffffffffffffffffull;
    seL4_Fault.words[11] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_R9(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[11] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_R9(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[11] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[11] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_R10(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[10] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_R10(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[10] &= ~0xffffffffffffffffull;
    seL4_Fault.words[10] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_R10(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[10] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_R10(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[10] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[10] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_R11(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[9] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_R11(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[9] &= ~0xffffffffffffffffull;
    seL4_Fault.words[9] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_R11(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[9] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_R11(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[9] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[9] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_R12(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[8] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_R12(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[8] &= ~0xffffffffffffffffull;
    seL4_Fault.words[8] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_R12(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[8] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_R12(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[8] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[8] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_R13(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[7] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_R13(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[7] &= ~0xffffffffffffffffull;
    seL4_Fault.words[7] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_R13(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[7] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_R13(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[7] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[7] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_R14(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[6] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_R14(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[6] &= ~0xffffffffffffffffull;
    seL4_Fault.words[6] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_R14(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[6] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_R14(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[6] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[6] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_R15(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[5] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_R15(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[5] &= ~0xffffffffffffffffull;
    seL4_Fault.words[5] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_R15(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[5] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_R15(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[5] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[5] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_FaultIP(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[4] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_FaultIP(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[4] &= ~0xffffffffffffffffull;
    seL4_Fault.words[4] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_FaultIP(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[4] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_FaultIP(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[4] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[4] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_RSP(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[3] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_RSP(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[3] &= ~0xffffffffffffffffull;
    seL4_Fault.words[3] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_RSP(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[3] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_RSP(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[3] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[3] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_FLAGS(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[2] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_FLAGS(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[2] &= ~0xffffffffffffffffull;
    seL4_Fault.words[2] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_FLAGS(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[2] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_FLAGS(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[2] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[2] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UnknownSyscall_get_Syscall(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault.words[1] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UnknownSyscall_set_Syscall(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[1] &= ~0xffffffffffffffffull;
    seL4_Fault.words[1] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UnknownSyscall_ptr_get_Syscall(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    ret = (seL4_Fault_ptr->words[1] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UnknownSyscall_ptr_set_Syscall(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UnknownSyscall);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[1] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[1] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UserException_new(seL4_Uint64 FaultIP, seL4_Uint64 Stack, seL4_Uint64 FLAGS, seL4_Uint64 Number, seL4_Uint64 Code) {
    seL4_Fault_t seL4_Fault;

    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert(((seL4_Uint64)seL4_Fault_UserException & ~0xfull) == ((0 && ((seL4_Uint64)seL4_Fault_UserException & (1ull << 63))) ? 0x0 : 0));

    seL4_Fault.words[0] = 0
        | ((seL4_Uint64)seL4_Fault_UserException & 0xfull) << 0;
    seL4_Fault.words[1] = 0
        | Code << 0;
    seL4_Fault.words[2] = 0
        | Number << 0;
    seL4_Fault.words[3] = 0
        | FLAGS << 0;
    seL4_Fault.words[4] = 0
        | Stack << 0;
    seL4_Fault.words[5] = 0
        | FaultIP << 0;
    seL4_Fault.words[6] = 0;
    seL4_Fault.words[7] = 0;
    seL4_Fault.words[8] = 0;
    seL4_Fault.words[9] = 0;
    seL4_Fault.words[10] = 0;
    seL4_Fault.words[11] = 0;
    seL4_Fault.words[12] = 0;
    seL4_Fault.words[13] = 0;
    seL4_Fault.words[14] = 0;
    seL4_Fault.words[15] = 0;
    seL4_Fault.words[16] = 0;
    seL4_Fault.words[17] = 0;
    seL4_Fault.words[18] = 0;
    seL4_Fault.words[19] = 0;

    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UserException_ptr_new(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 FaultIP, seL4_Uint64 Stack, seL4_Uint64 FLAGS, seL4_Uint64 Number, seL4_Uint64 Code) {
    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert(((seL4_Uint64)seL4_Fault_UserException & ~0xfull) == ((0 && ((seL4_Uint64)seL4_Fault_UserException & (1ull << 63))) ? 0x0 : 0));

    seL4_Fault_ptr->words[0] = 0
        | ((seL4_Uint64)seL4_Fault_UserException & 0xfull) << 0;
    seL4_Fault_ptr->words[1] = 0
        | Code << 0;
    seL4_Fault_ptr->words[2] = 0
        | Number << 0;
    seL4_Fault_ptr->words[3] = 0
        | FLAGS << 0;
    seL4_Fault_ptr->words[4] = 0
        | Stack << 0;
    seL4_Fault_ptr->words[5] = 0
        | FaultIP << 0;
    seL4_Fault_ptr->words[6] = 0;
    seL4_Fault_ptr->words[7] = 0;
    seL4_Fault_ptr->words[8] = 0;
    seL4_Fault_ptr->words[9] = 0;
    seL4_Fault_ptr->words[10] = 0;
    seL4_Fault_ptr->words[11] = 0;
    seL4_Fault_ptr->words[12] = 0;
    seL4_Fault_ptr->words[13] = 0;
    seL4_Fault_ptr->words[14] = 0;
    seL4_Fault_ptr->words[15] = 0;
    seL4_Fault_ptr->words[16] = 0;
    seL4_Fault_ptr->words[17] = 0;
    seL4_Fault_ptr->words[18] = 0;
    seL4_Fault_ptr->words[19] = 0;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UserException_get_FaultIP(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UserException);

    ret = (seL4_Fault.words[5] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UserException_set_FaultIP(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UserException);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[5] &= ~0xffffffffffffffffull;
    seL4_Fault.words[5] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UserException_ptr_get_FaultIP(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UserException);

    ret = (seL4_Fault_ptr->words[5] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UserException_ptr_set_FaultIP(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UserException);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[5] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[5] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UserException_get_Stack(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UserException);

    ret = (seL4_Fault.words[4] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UserException_set_Stack(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UserException);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[4] &= ~0xffffffffffffffffull;
    seL4_Fault.words[4] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UserException_ptr_get_Stack(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UserException);

    ret = (seL4_Fault_ptr->words[4] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UserException_ptr_set_Stack(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UserException);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[4] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[4] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UserException_get_FLAGS(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UserException);

    ret = (seL4_Fault.words[3] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UserException_set_FLAGS(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UserException);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[3] &= ~0xffffffffffffffffull;
    seL4_Fault.words[3] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UserException_ptr_get_FLAGS(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UserException);

    ret = (seL4_Fault_ptr->words[3] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UserException_ptr_set_FLAGS(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UserException);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[3] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[3] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UserException_get_Number(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UserException);

    ret = (seL4_Fault.words[2] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UserException_set_Number(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UserException);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[2] &= ~0xffffffffffffffffull;
    seL4_Fault.words[2] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UserException_ptr_get_Number(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UserException);

    ret = (seL4_Fault_ptr->words[2] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UserException_ptr_set_Number(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UserException);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[2] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[2] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_UserException_get_Code(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UserException);

    ret = (seL4_Fault.words[1] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_UserException_set_Code(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_UserException);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[1] &= ~0xffffffffffffffffull;
    seL4_Fault.words[1] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_UserException_ptr_get_Code(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UserException);

    ret = (seL4_Fault_ptr->words[1] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_UserException_ptr_set_Code(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_UserException);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[1] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[1] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_Timeout_new(seL4_Uint64 data, seL4_Uint64 consumed) {
    seL4_Fault_t seL4_Fault;

    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert(((seL4_Uint64)seL4_Fault_Timeout & ~0xfull) == ((0 && ((seL4_Uint64)seL4_Fault_Timeout & (1ull << 63))) ? 0x0 : 0));

    seL4_Fault.words[0] = 0
        | ((seL4_Uint64)seL4_Fault_Timeout & 0xfull) << 0;
    seL4_Fault.words[1] = 0
        | consumed << 0;
    seL4_Fault.words[2] = 0
        | data << 0;
    seL4_Fault.words[3] = 0;
    seL4_Fault.words[4] = 0;
    seL4_Fault.words[5] = 0;
    seL4_Fault.words[6] = 0;
    seL4_Fault.words[7] = 0;
    seL4_Fault.words[8] = 0;
    seL4_Fault.words[9] = 0;
    seL4_Fault.words[10] = 0;
    seL4_Fault.words[11] = 0;
    seL4_Fault.words[12] = 0;
    seL4_Fault.words[13] = 0;
    seL4_Fault.words[14] = 0;
    seL4_Fault.words[15] = 0;
    seL4_Fault.words[16] = 0;
    seL4_Fault.words[17] = 0;
    seL4_Fault.words[18] = 0;
    seL4_Fault.words[19] = 0;

    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_Timeout_ptr_new(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 data, seL4_Uint64 consumed) {
    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert(((seL4_Uint64)seL4_Fault_Timeout & ~0xfull) == ((0 && ((seL4_Uint64)seL4_Fault_Timeout & (1ull << 63))) ? 0x0 : 0));

    seL4_Fault_ptr->words[0] = 0
        | ((seL4_Uint64)seL4_Fault_Timeout & 0xfull) << 0;
    seL4_Fault_ptr->words[1] = 0
        | consumed << 0;
    seL4_Fault_ptr->words[2] = 0
        | data << 0;
    seL4_Fault_ptr->words[3] = 0;
    seL4_Fault_ptr->words[4] = 0;
    seL4_Fault_ptr->words[5] = 0;
    seL4_Fault_ptr->words[6] = 0;
    seL4_Fault_ptr->words[7] = 0;
    seL4_Fault_ptr->words[8] = 0;
    seL4_Fault_ptr->words[9] = 0;
    seL4_Fault_ptr->words[10] = 0;
    seL4_Fault_ptr->words[11] = 0;
    seL4_Fault_ptr->words[12] = 0;
    seL4_Fault_ptr->words[13] = 0;
    seL4_Fault_ptr->words[14] = 0;
    seL4_Fault_ptr->words[15] = 0;
    seL4_Fault_ptr->words[16] = 0;
    seL4_Fault_ptr->words[17] = 0;
    seL4_Fault_ptr->words[18] = 0;
    seL4_Fault_ptr->words[19] = 0;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_Timeout_get_data(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_Timeout);

    ret = (seL4_Fault.words[2] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_Timeout_set_data(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_Timeout);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[2] &= ~0xffffffffffffffffull;
    seL4_Fault.words[2] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_Timeout_ptr_get_data(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_Timeout);

    ret = (seL4_Fault_ptr->words[2] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_Timeout_ptr_set_data(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_Timeout);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[2] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[2] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_Timeout_get_consumed(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_Timeout);

    ret = (seL4_Fault.words[1] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_Timeout_set_consumed(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_Timeout);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[1] &= ~0xffffffffffffffffull;
    seL4_Fault.words[1] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_Timeout_ptr_get_consumed(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_Timeout);

    ret = (seL4_Fault_ptr->words[1] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_Timeout_ptr_set_consumed(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_Timeout);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[1] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[1] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_VMFault_new(seL4_Uint64 IP, seL4_Uint64 Addr, seL4_Uint64 PrefetchFault, seL4_Uint64 FSR) {
    seL4_Fault_t seL4_Fault;

    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert(((seL4_Uint64)seL4_Fault_VMFault & ~0xfull) == ((0 && ((seL4_Uint64)seL4_Fault_VMFault & (1ull << 63))) ? 0x0 : 0));

    seL4_Fault.words[0] = 0
        | ((seL4_Uint64)seL4_Fault_VMFault & 0xfull) << 0;
    seL4_Fault.words[1] = 0
        | FSR << 0;
    seL4_Fault.words[2] = 0
        | PrefetchFault << 0;
    seL4_Fault.words[3] = 0
        | Addr << 0;
    seL4_Fault.words[4] = 0
        | IP << 0;
    seL4_Fault.words[5] = 0;
    seL4_Fault.words[6] = 0;
    seL4_Fault.words[7] = 0;
    seL4_Fault.words[8] = 0;
    seL4_Fault.words[9] = 0;
    seL4_Fault.words[10] = 0;
    seL4_Fault.words[11] = 0;
    seL4_Fault.words[12] = 0;
    seL4_Fault.words[13] = 0;
    seL4_Fault.words[14] = 0;
    seL4_Fault.words[15] = 0;
    seL4_Fault.words[16] = 0;
    seL4_Fault.words[17] = 0;
    seL4_Fault.words[18] = 0;
    seL4_Fault.words[19] = 0;

    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_VMFault_ptr_new(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 IP, seL4_Uint64 Addr, seL4_Uint64 PrefetchFault, seL4_Uint64 FSR) {
    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert(((seL4_Uint64)seL4_Fault_VMFault & ~0xfull) == ((0 && ((seL4_Uint64)seL4_Fault_VMFault & (1ull << 63))) ? 0x0 : 0));

    seL4_Fault_ptr->words[0] = 0
        | ((seL4_Uint64)seL4_Fault_VMFault & 0xfull) << 0;
    seL4_Fault_ptr->words[1] = 0
        | FSR << 0;
    seL4_Fault_ptr->words[2] = 0
        | PrefetchFault << 0;
    seL4_Fault_ptr->words[3] = 0
        | Addr << 0;
    seL4_Fault_ptr->words[4] = 0
        | IP << 0;
    seL4_Fault_ptr->words[5] = 0;
    seL4_Fault_ptr->words[6] = 0;
    seL4_Fault_ptr->words[7] = 0;
    seL4_Fault_ptr->words[8] = 0;
    seL4_Fault_ptr->words[9] = 0;
    seL4_Fault_ptr->words[10] = 0;
    seL4_Fault_ptr->words[11] = 0;
    seL4_Fault_ptr->words[12] = 0;
    seL4_Fault_ptr->words[13] = 0;
    seL4_Fault_ptr->words[14] = 0;
    seL4_Fault_ptr->words[15] = 0;
    seL4_Fault_ptr->words[16] = 0;
    seL4_Fault_ptr->words[17] = 0;
    seL4_Fault_ptr->words[18] = 0;
    seL4_Fault_ptr->words[19] = 0;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_VMFault_get_IP(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_VMFault);

    ret = (seL4_Fault.words[4] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_VMFault_set_IP(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_VMFault);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[4] &= ~0xffffffffffffffffull;
    seL4_Fault.words[4] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_VMFault_ptr_get_IP(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_VMFault);

    ret = (seL4_Fault_ptr->words[4] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_VMFault_ptr_set_IP(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_VMFault);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[4] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[4] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_VMFault_get_Addr(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_VMFault);

    ret = (seL4_Fault.words[3] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_VMFault_set_Addr(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_VMFault);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[3] &= ~0xffffffffffffffffull;
    seL4_Fault.words[3] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_VMFault_ptr_get_Addr(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_VMFault);

    ret = (seL4_Fault_ptr->words[3] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_VMFault_ptr_set_Addr(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_VMFault);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[3] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[3] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_VMFault_get_PrefetchFault(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_VMFault);

    ret = (seL4_Fault.words[2] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_VMFault_set_PrefetchFault(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_VMFault);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[2] &= ~0xffffffffffffffffull;
    seL4_Fault.words[2] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_VMFault_ptr_get_PrefetchFault(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_VMFault);

    ret = (seL4_Fault_ptr->words[2] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_VMFault_ptr_set_PrefetchFault(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_VMFault);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[2] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[2] |= (v64 << 0) & 0xffffffffffffffffull;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_Fault_VMFault_get_FSR(seL4_Fault_t seL4_Fault) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_VMFault);

    ret = (seL4_Fault.words[1] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_Fault_t CONST
seL4_Fault_VMFault_set_FSR(seL4_Fault_t seL4_Fault, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault.words[0] >> 0) & 0xf) == seL4_Fault_VMFault);
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault.words[1] &= ~0xffffffffffffffffull;
    seL4_Fault.words[1] |= (v64 << 0) & 0xffffffffffffffffull;
    return seL4_Fault;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_Fault_VMFault_ptr_get_FSR(seL4_Fault_t *seL4_Fault_ptr) {
    seL4_Uint64 ret;
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_VMFault);

    ret = (seL4_Fault_ptr->words[1] & 0xffffffffffffffffull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_Fault_VMFault_ptr_set_FSR(seL4_Fault_t *seL4_Fault_ptr, seL4_Uint64 v64) {
    /* fail if union does not have the expected tag */
    seL4_DebugAssert(((seL4_Fault_ptr->words[0] >> 0) & 0xf) == seL4_Fault_VMFault);

    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffffull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));

    seL4_Fault_ptr->words[1] &= ~0xffffffffffffffffull;
    seL4_Fault_ptr->words[1] |= (v64 << 0) & 0xffffffffffffffffull;
}

