/* generated from libsel4/mode_include/64/sel4/shared_types.bf */

#pragma once

#include <sel4/config.h>
#include <sel4/simple_types.h>
#include <sel4/debug_assert.h>
struct seL4_CNode_CapData {
    seL4_Uint64 words[1];
};
typedef struct seL4_CNode_CapData seL4_CNode_CapData_t;

LIBSEL4_INLINE_FUNC seL4_CNode_CapData_t CONST
seL4_CNode_CapData_new(seL4_Uint64 guard, seL4_Uint64 guardSize) {
    seL4_CNode_CapData_t seL4_CNode_CapData;

    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert((guard & ~0x3ffffffffffffffull) == ((0 && (guard & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((guardSize & ~0x3full) == ((0 && (guardSize & (1ull << 63))) ? 0x0 : 0));

    seL4_CNode_CapData.words[0] = 0
        | (guard & 0x3ffffffffffffffull) << 6
        | (guardSize & 0x3full) << 0;

    return seL4_CNode_CapData;
}

LIBSEL4_INLINE_FUNC void
seL4_CNode_CapData_ptr_new(seL4_CNode_CapData_t *seL4_CNode_CapData_ptr, seL4_Uint64 guard, seL4_Uint64 guardSize) {
    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert((guard & ~0x3ffffffffffffffull) == ((0 && (guard & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((guardSize & ~0x3full) == ((0 && (guardSize & (1ull << 63))) ? 0x0 : 0));

    seL4_CNode_CapData_ptr->words[0] = 0
        | (guard & 0x3ffffffffffffffull) << 6
        | (guardSize & 0x3full) << 0;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_CNode_CapData_get_guard(seL4_CNode_CapData_t seL4_CNode_CapData) {
    seL4_Uint64 ret;
    ret = (seL4_CNode_CapData.words[0] & 0xffffffffffffffc0ull) >> 6;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_CNode_CapData_t CONST
seL4_CNode_CapData_set_guard(seL4_CNode_CapData_t seL4_CNode_CapData, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffc0ull >> 6 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_CNode_CapData.words[0] &= ~0xffffffffffffffc0ull;
    seL4_CNode_CapData.words[0] |= (v64 << 6) & 0xffffffffffffffc0ull;
    return seL4_CNode_CapData;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_CNode_CapData_ptr_get_guard(seL4_CNode_CapData_t *seL4_CNode_CapData_ptr) {
    seL4_Uint64 ret;
    ret = (seL4_CNode_CapData_ptr->words[0] & 0xffffffffffffffc0ull) >> 6;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_CNode_CapData_ptr_set_guard(seL4_CNode_CapData_t *seL4_CNode_CapData_ptr, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xffffffffffffffc0ull >> 6) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_CNode_CapData_ptr->words[0] &= ~0xffffffffffffffc0ull;
    seL4_CNode_CapData_ptr->words[0] |= (v64 << 6) & 0xffffffffffffffc0;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_CNode_CapData_get_guardSize(seL4_CNode_CapData_t seL4_CNode_CapData) {
    seL4_Uint64 ret;
    ret = (seL4_CNode_CapData.words[0] & 0x3full) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_CNode_CapData_t CONST
seL4_CNode_CapData_set_guardSize(seL4_CNode_CapData_t seL4_CNode_CapData, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0x3full >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_CNode_CapData.words[0] &= ~0x3full;
    seL4_CNode_CapData.words[0] |= (v64 << 0) & 0x3full;
    return seL4_CNode_CapData;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_CNode_CapData_ptr_get_guardSize(seL4_CNode_CapData_t *seL4_CNode_CapData_ptr) {
    seL4_Uint64 ret;
    ret = (seL4_CNode_CapData_ptr->words[0] & 0x3full) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_CNode_CapData_ptr_set_guardSize(seL4_CNode_CapData_t *seL4_CNode_CapData_ptr, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0x3full >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_CNode_CapData_ptr->words[0] &= ~0x3full;
    seL4_CNode_CapData_ptr->words[0] |= (v64 << 0) & 0x3f;
}

struct seL4_CapRights {
    seL4_Uint64 words[1];
};
typedef struct seL4_CapRights seL4_CapRights_t;

LIBSEL4_INLINE_FUNC seL4_CapRights_t CONST
seL4_CapRights_new(seL4_Uint64 capAllowGrantReply, seL4_Uint64 capAllowGrant, seL4_Uint64 capAllowRead, seL4_Uint64 capAllowWrite) {
    seL4_CapRights_t seL4_CapRights;

    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert((capAllowGrantReply & ~0x1ull) == ((0 && (capAllowGrantReply & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((capAllowGrant & ~0x1ull) == ((0 && (capAllowGrant & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((capAllowRead & ~0x1ull) == ((0 && (capAllowRead & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((capAllowWrite & ~0x1ull) == ((0 && (capAllowWrite & (1ull << 63))) ? 0x0 : 0));

    seL4_CapRights.words[0] = 0
        | (capAllowGrantReply & 0x1ull) << 3
        | (capAllowGrant & 0x1ull) << 2
        | (capAllowRead & 0x1ull) << 1
        | (capAllowWrite & 0x1ull) << 0;

    return seL4_CapRights;
}

LIBSEL4_INLINE_FUNC void
seL4_CapRights_ptr_new(seL4_CapRights_t *seL4_CapRights_ptr, seL4_Uint64 capAllowGrantReply, seL4_Uint64 capAllowGrant, seL4_Uint64 capAllowRead, seL4_Uint64 capAllowWrite) {
    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert((capAllowGrantReply & ~0x1ull) == ((0 && (capAllowGrantReply & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((capAllowGrant & ~0x1ull) == ((0 && (capAllowGrant & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((capAllowRead & ~0x1ull) == ((0 && (capAllowRead & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((capAllowWrite & ~0x1ull) == ((0 && (capAllowWrite & (1ull << 63))) ? 0x0 : 0));

    seL4_CapRights_ptr->words[0] = 0
        | (capAllowGrantReply & 0x1ull) << 3
        | (capAllowGrant & 0x1ull) << 2
        | (capAllowRead & 0x1ull) << 1
        | (capAllowWrite & 0x1ull) << 0;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_CapRights_get_capAllowGrantReply(seL4_CapRights_t seL4_CapRights) {
    seL4_Uint64 ret;
    ret = (seL4_CapRights.words[0] & 0x8ull) >> 3;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_CapRights_t CONST
seL4_CapRights_set_capAllowGrantReply(seL4_CapRights_t seL4_CapRights, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0x8ull >> 3 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_CapRights.words[0] &= ~0x8ull;
    seL4_CapRights.words[0] |= (v64 << 3) & 0x8ull;
    return seL4_CapRights;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_CapRights_ptr_get_capAllowGrantReply(seL4_CapRights_t *seL4_CapRights_ptr) {
    seL4_Uint64 ret;
    ret = (seL4_CapRights_ptr->words[0] & 0x8ull) >> 3;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_CapRights_ptr_set_capAllowGrantReply(seL4_CapRights_t *seL4_CapRights_ptr, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0x8ull >> 3) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_CapRights_ptr->words[0] &= ~0x8ull;
    seL4_CapRights_ptr->words[0] |= (v64 << 3) & 0x8;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_CapRights_get_capAllowGrant(seL4_CapRights_t seL4_CapRights) {
    seL4_Uint64 ret;
    ret = (seL4_CapRights.words[0] & 0x4ull) >> 2;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_CapRights_t CONST
seL4_CapRights_set_capAllowGrant(seL4_CapRights_t seL4_CapRights, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0x4ull >> 2 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_CapRights.words[0] &= ~0x4ull;
    seL4_CapRights.words[0] |= (v64 << 2) & 0x4ull;
    return seL4_CapRights;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_CapRights_ptr_get_capAllowGrant(seL4_CapRights_t *seL4_CapRights_ptr) {
    seL4_Uint64 ret;
    ret = (seL4_CapRights_ptr->words[0] & 0x4ull) >> 2;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_CapRights_ptr_set_capAllowGrant(seL4_CapRights_t *seL4_CapRights_ptr, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0x4ull >> 2) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_CapRights_ptr->words[0] &= ~0x4ull;
    seL4_CapRights_ptr->words[0] |= (v64 << 2) & 0x4;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_CapRights_get_capAllowRead(seL4_CapRights_t seL4_CapRights) {
    seL4_Uint64 ret;
    ret = (seL4_CapRights.words[0] & 0x2ull) >> 1;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_CapRights_t CONST
seL4_CapRights_set_capAllowRead(seL4_CapRights_t seL4_CapRights, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0x2ull >> 1 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_CapRights.words[0] &= ~0x2ull;
    seL4_CapRights.words[0] |= (v64 << 1) & 0x2ull;
    return seL4_CapRights;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_CapRights_ptr_get_capAllowRead(seL4_CapRights_t *seL4_CapRights_ptr) {
    seL4_Uint64 ret;
    ret = (seL4_CapRights_ptr->words[0] & 0x2ull) >> 1;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_CapRights_ptr_set_capAllowRead(seL4_CapRights_t *seL4_CapRights_ptr, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0x2ull >> 1) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_CapRights_ptr->words[0] &= ~0x2ull;
    seL4_CapRights_ptr->words[0] |= (v64 << 1) & 0x2;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_CapRights_get_capAllowWrite(seL4_CapRights_t seL4_CapRights) {
    seL4_Uint64 ret;
    ret = (seL4_CapRights.words[0] & 0x1ull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_CapRights_t CONST
seL4_CapRights_set_capAllowWrite(seL4_CapRights_t seL4_CapRights, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0x1ull >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_CapRights.words[0] &= ~0x1ull;
    seL4_CapRights.words[0] |= (v64 << 0) & 0x1ull;
    return seL4_CapRights;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_CapRights_ptr_get_capAllowWrite(seL4_CapRights_t *seL4_CapRights_ptr) {
    seL4_Uint64 ret;
    ret = (seL4_CapRights_ptr->words[0] & 0x1ull) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_CapRights_ptr_set_capAllowWrite(seL4_CapRights_t *seL4_CapRights_ptr, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0x1ull >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_CapRights_ptr->words[0] &= ~0x1ull;
    seL4_CapRights_ptr->words[0] |= (v64 << 0) & 0x1;
}

struct seL4_MessageInfo {
    seL4_Uint64 words[1];
};
typedef struct seL4_MessageInfo seL4_MessageInfo_t;

LIBSEL4_INLINE_FUNC seL4_MessageInfo_t CONST
seL4_MessageInfo_new(seL4_Uint64 label, seL4_Uint64 capsUnwrapped, seL4_Uint64 extraCaps, seL4_Uint64 length) {
    seL4_MessageInfo_t seL4_MessageInfo;

    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert((label & ~0xfffffffffffffull) == ((0 && (label & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((capsUnwrapped & ~0x7ull) == ((0 && (capsUnwrapped & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((extraCaps & ~0x3ull) == ((0 && (extraCaps & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((length & ~0x7full) == ((0 && (length & (1ull << 63))) ? 0x0 : 0));

    seL4_MessageInfo.words[0] = 0
        | (label & 0xfffffffffffffull) << 12
        | (capsUnwrapped & 0x7ull) << 9
        | (extraCaps & 0x3ull) << 7
        | (length & 0x7full) << 0;

    return seL4_MessageInfo;
}

LIBSEL4_INLINE_FUNC void
seL4_MessageInfo_ptr_new(seL4_MessageInfo_t *seL4_MessageInfo_ptr, seL4_Uint64 label, seL4_Uint64 capsUnwrapped, seL4_Uint64 extraCaps, seL4_Uint64 length) {
    /* fail if user has passed bits that we will override */  
    seL4_DebugAssert((label & ~0xfffffffffffffull) == ((0 && (label & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((capsUnwrapped & ~0x7ull) == ((0 && (capsUnwrapped & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((extraCaps & ~0x3ull) == ((0 && (extraCaps & (1ull << 63))) ? 0x0 : 0));  
    seL4_DebugAssert((length & ~0x7full) == ((0 && (length & (1ull << 63))) ? 0x0 : 0));

    seL4_MessageInfo_ptr->words[0] = 0
        | (label & 0xfffffffffffffull) << 12
        | (capsUnwrapped & 0x7ull) << 9
        | (extraCaps & 0x3ull) << 7
        | (length & 0x7full) << 0;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_MessageInfo_get_label(seL4_MessageInfo_t seL4_MessageInfo) {
    seL4_Uint64 ret;
    ret = (seL4_MessageInfo.words[0] & 0xfffffffffffff000ull) >> 12;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_MessageInfo_t CONST
seL4_MessageInfo_set_label(seL4_MessageInfo_t seL4_MessageInfo, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xfffffffffffff000ull >> 12 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_MessageInfo.words[0] &= ~0xfffffffffffff000ull;
    seL4_MessageInfo.words[0] |= (v64 << 12) & 0xfffffffffffff000ull;
    return seL4_MessageInfo;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_MessageInfo_ptr_get_label(seL4_MessageInfo_t *seL4_MessageInfo_ptr) {
    seL4_Uint64 ret;
    ret = (seL4_MessageInfo_ptr->words[0] & 0xfffffffffffff000ull) >> 12;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_MessageInfo_ptr_set_label(seL4_MessageInfo_t *seL4_MessageInfo_ptr, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xfffffffffffff000ull >> 12) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_MessageInfo_ptr->words[0] &= ~0xfffffffffffff000ull;
    seL4_MessageInfo_ptr->words[0] |= (v64 << 12) & 0xfffffffffffff000;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_MessageInfo_get_capsUnwrapped(seL4_MessageInfo_t seL4_MessageInfo) {
    seL4_Uint64 ret;
    ret = (seL4_MessageInfo.words[0] & 0xe00ull) >> 9;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_MessageInfo_t CONST
seL4_MessageInfo_set_capsUnwrapped(seL4_MessageInfo_t seL4_MessageInfo, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xe00ull >> 9 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_MessageInfo.words[0] &= ~0xe00ull;
    seL4_MessageInfo.words[0] |= (v64 << 9) & 0xe00ull;
    return seL4_MessageInfo;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_MessageInfo_ptr_get_capsUnwrapped(seL4_MessageInfo_t *seL4_MessageInfo_ptr) {
    seL4_Uint64 ret;
    ret = (seL4_MessageInfo_ptr->words[0] & 0xe00ull) >> 9;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_MessageInfo_ptr_set_capsUnwrapped(seL4_MessageInfo_t *seL4_MessageInfo_ptr, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0xe00ull >> 9) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_MessageInfo_ptr->words[0] &= ~0xe00ull;
    seL4_MessageInfo_ptr->words[0] |= (v64 << 9) & 0xe00;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_MessageInfo_get_extraCaps(seL4_MessageInfo_t seL4_MessageInfo) {
    seL4_Uint64 ret;
    ret = (seL4_MessageInfo.words[0] & 0x180ull) >> 7;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_MessageInfo_t CONST
seL4_MessageInfo_set_extraCaps(seL4_MessageInfo_t seL4_MessageInfo, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0x180ull >> 7 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_MessageInfo.words[0] &= ~0x180ull;
    seL4_MessageInfo.words[0] |= (v64 << 7) & 0x180ull;
    return seL4_MessageInfo;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_MessageInfo_ptr_get_extraCaps(seL4_MessageInfo_t *seL4_MessageInfo_ptr) {
    seL4_Uint64 ret;
    ret = (seL4_MessageInfo_ptr->words[0] & 0x180ull) >> 7;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_MessageInfo_ptr_set_extraCaps(seL4_MessageInfo_t *seL4_MessageInfo_ptr, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0x180ull >> 7) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_MessageInfo_ptr->words[0] &= ~0x180ull;
    seL4_MessageInfo_ptr->words[0] |= (v64 << 7) & 0x180;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 CONST
seL4_MessageInfo_get_length(seL4_MessageInfo_t seL4_MessageInfo) {
    seL4_Uint64 ret;
    ret = (seL4_MessageInfo.words[0] & 0x7full) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC seL4_MessageInfo_t CONST
seL4_MessageInfo_set_length(seL4_MessageInfo_t seL4_MessageInfo, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0x7full >> 0 ) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_MessageInfo.words[0] &= ~0x7full;
    seL4_MessageInfo.words[0] |= (v64 << 0) & 0x7full;
    return seL4_MessageInfo;
}

LIBSEL4_INLINE_FUNC seL4_Uint64 PURE
seL4_MessageInfo_ptr_get_length(seL4_MessageInfo_t *seL4_MessageInfo_ptr) {
    seL4_Uint64 ret;
    ret = (seL4_MessageInfo_ptr->words[0] & 0x7full) >> 0;
    /* Possibly sign extend */
    if (__builtin_expect(!!(0 && (ret & (1ull << (63)))), 0)) {
        ret |= 0x0;
    }
    return ret;
}

LIBSEL4_INLINE_FUNC void
seL4_MessageInfo_ptr_set_length(seL4_MessageInfo_t *seL4_MessageInfo_ptr, seL4_Uint64 v64) {
    /* fail if user has passed bits that we will override */
    seL4_DebugAssert((((~0x7full >> 0) | 0x0) & v64) == ((0 && (v64 & (1ull << (63)))) ? 0x0 : 0));
    seL4_MessageInfo_ptr->words[0] &= ~0x7full;
    seL4_MessageInfo_ptr->words[0] |= (v64 << 0) & 0x7f;
}

