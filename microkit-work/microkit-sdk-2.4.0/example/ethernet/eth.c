/*
 * Copyright 2021, Breakaway Consulting Pty. Ltd.
 *
 * SPDX-License-Identifier: BSD-2-Clause
 */
#include <stdbool.h>
#include <stdint.h>
#include <microkit.h>

#define OUTPUT_CH 1 /* output from this PD -- becomes input for peer */
#define INPUT_CH 2 /* input to this PD -- comes from peer output */
#define IRQ_CH 3

uintptr_t ring_buffer_vaddr;
uintptr_t packet_buffer_vaddr;

uintptr_t ring_buffer_paddr;
uintptr_t packet_buffer_paddr;

/* Note: in theory 256 should be allowed, but it doesn't work for some reason */
#define RBD_COUNT 128
#define TBD_COUNT 128

#define BUFFER_MAX 1024
#define BUFFER_SIZE (2 * 1024)
#define DATA_OFFSET 64

static unsigned rbd_index = 0;
static unsigned tbd_index = 0;

static uint8_t mac[6];

static uint8_t broadcast_mac[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
static uint8_t my_ip[4] = { 10, 141, 2, 80 };

/* Make the minimum frame buffer 2k. This is a bit of a waste of memory, but ensure alignment */
#define PACKET_BUFFER_SIZE (2 * 1024)

static uint8_t temp_packet[PACKET_BUFFER_SIZE] __attribute__((aligned(64)));

static unsigned output_index = 0;
static unsigned input_index = 0;


/* A small selection of ehtertype that we might see
 * by no means exhaustive, but probably only ever
 * going to see IPv4 and ARP anyway!
 */

#define ETHERTYPE_IPV4 0x0800
#define ETHERTYPE_ARP 0x0806
#define ETHERTYPE_WOL 0x842
#define ETHERTYPE_RARP 0x8035
#define ETHERTYPE_IPV6 0x86DD

uint64_t output_buffer_vaddr;
uint64_t input_buffer_vaddr;

#define OUTPUT_BUFFER output_buffer_vaddr
#define INPUT_BUFFER input_buffer_vaddr


static inline uint64_t
get_sys_counter(void)
{
    uint64_t r;
    /* FIXME: If FEAT_ECV is implemented ISB can be avoided
     * if we used cntpctss_el0 instead.
     */
    asm volatile ("isb sy" : : : "memory");
    asm volatile("mrs %0, cntpct_el0" : "=r" (r));
    return r;
}

struct buffer_descriptor {
    uint16_t data_length;
    uint16_t flags;
};

struct rbd {
    uint16_t data_length;
    uint16_t flags;
    uint32_t addr;
};

struct tbd {
    uint16_t data_length;
    uint16_t flags;
    uint32_t addr;
};

struct eth_header {
    uint8_t dest_mac[6];
    uint8_t src_mac[6];
    uint16_t ethertype;
    uint8_t payload[];
} __attribute__((aligned(2), __packed__));

struct arp {
    uint16_t htype;
    uint16_t ptype;
    uint8_t hlen;
    uint8_t plen;
    uint16_t oper;
    uint8_t sha[6];
    uint8_t spa[4];
    uint8_t tha[6];
    uint8_t tpa[4];
} __attribute__((aligned(2), __packed__));

struct ip {
    uint8_t ver_ihl;
    uint8_t tos;
    uint16_t len;

    uint16_t ident;
    uint16_t flags_frag;

    uint8_t ttl;
    uint8_t protocol;
    uint16_t checksum;

    uint8_t source_address[4];
    uint8_t dest_address[4];
} __attribute__((aligned(2), __packed__));

struct icmp {
    uint8_t type;
    uint8_t code;
    uint16_t checksum;
    uint16_t rest_of_header;
} __attribute__((aligned(2), __packed__));


struct regs {
	/* [10:2]addr = 00 */

	/*  Control and status Registers (offset 000-1FF) */
	uint32_t res0[1];		/* MBAR_ETH + 0x000 */
	uint32_t eir;		/* MBAR_ETH + 0x004 */
	uint32_t eimr;			/* MBAR_ETH + 0x008 */

	uint32_t res1[1];		/* MBAR_ETH + 0x00C */
	uint32_t rdar;		/* MBAR_ETH + 0x010 */
	uint32_t tdar;		/* MBAR_ETH + 0x014 */
	uint32_t res2[3];		/* MBAR_ETH + 0x018-20 */
	uint32_t ecr;		/* MBAR_ETH + 0x024 */

	uint32_t res3[6];		/* MBAR_ETH + 0x028-03C */
	uint32_t mii_data;		/* MBAR_ETH + 0x040 */
	uint32_t mii_speed;		/* MBAR_ETH + 0x044 */
	uint32_t res4[7];		/* MBAR_ETH + 0x048-60 */
	uint32_t mib_control;		/* MBAR_ETH + 0x064 */

	uint32_t res5[7];		/* MBAR_ETH + 0x068-80 */
	uint32_t rcr;		/* MBAR_ETH + 0x084 */
	uint32_t res6[15];		/* MBAR_ETH + 0x088-C0 */
	uint32_t tcr;		/* MBAR_ETH + 0x0C4 */
	uint32_t res7[7];		/* MBAR_ETH + 0x0C8-E0 */
	uint32_t paddr1;		/* MBAR_ETH + 0x0E4 */
	uint32_t paddr2;		/* MBAR_ETH + 0x0E8 */
	uint32_t op_pause;		/* MBAR_ETH + 0x0EC */

	uint32_t res8[10];		/* MBAR_ETH + 0x0F0-114 */
	uint32_t iaddr1;		/* MBAR_ETH + 0x118 */
	uint32_t iaddr2;		/* MBAR_ETH + 0x11C */
	uint32_t gaddr1;		/* MBAR_ETH + 0x120 */
	uint32_t gaddr2;		/* MBAR_ETH + 0x124 */
	uint32_t res9[7];		/* MBAR_ETH + 0x128-140 */

	uint32_t x_wmrk;		/* MBAR_ETH + 0x144 */
	uint32_t res10[1];		/* MBAR_ETH + 0x148 */
	uint32_t r_bound;		/* MBAR_ETH + 0x14C */
	uint32_t r_fstart;		/* MBAR_ETH + 0x150 */
	uint32_t res11[11];		/* MBAR_ETH + 0x154-17C */
	uint32_t erdsr;			/* MBAR_ETH + 0x180 */
	uint32_t etdsr;			/* MBAR_ETH + 0x184 */
	uint32_t emrbr;			/* MBAR_ETH + 0x188 */
	uint32_t res12[29];		/* MBAR_ETH + 0x18C-1FC */

	/*  MIB COUNTERS (Offset 200-2FF) */
	uint32_t rmon_t_drop;		/* MBAR_ETH + 0x200 */
	uint32_t rmon_t_packets;	/* MBAR_ETH + 0x204 */
	uint32_t rmon_t_bc_pkt;		/* MBAR_ETH + 0x208 */
	uint32_t rmon_t_mc_pkt;		/* MBAR_ETH + 0x20C */
	uint32_t rmon_t_crc_align;	/* MBAR_ETH + 0x210 */
	uint32_t rmon_t_undersize;	/* MBAR_ETH + 0x214 */
	uint32_t rmon_t_oversize;	/* MBAR_ETH + 0x218 */
	uint32_t rmon_t_frag;		/* MBAR_ETH + 0x21C */
	uint32_t rmon_t_jab;		/* MBAR_ETH + 0x220 */
	uint32_t rmon_t_col;		/* MBAR_ETH + 0x224 */
	uint32_t rmon_t_p64;		/* MBAR_ETH + 0x228 */
	uint32_t rmon_t_p65to127;	/* MBAR_ETH + 0x22C */
	uint32_t rmon_t_p128to255;	/* MBAR_ETH + 0x230 */
	uint32_t rmon_t_p256to511;	/* MBAR_ETH + 0x234 */
	uint32_t rmon_t_p512to1023;	/* MBAR_ETH + 0x238 */
	uint32_t rmon_t_p1024to2047;	/* MBAR_ETH + 0x23C */
	uint32_t rmon_t_p_gte2048;	/* MBAR_ETH + 0x240 */
	uint32_t rmon_t_octets;		/* MBAR_ETH + 0x244 */
	uint32_t ieee_t_drop;		/* MBAR_ETH + 0x248 */
	uint32_t ieee_t_frame_ok;	/* MBAR_ETH + 0x24C */
	uint32_t ieee_t_1col;		/* MBAR_ETH + 0x250 */
	uint32_t ieee_t_mcol;		/* MBAR_ETH + 0x254 */
	uint32_t ieee_t_def;		/* MBAR_ETH + 0x258 */
	uint32_t ieee_t_lcol;		/* MBAR_ETH + 0x25C */
	uint32_t ieee_t_excol;		/* MBAR_ETH + 0x260 */
	uint32_t ieee_t_macerr;		/* MBAR_ETH + 0x264 */
	uint32_t ieee_t_cserr;		/* MBAR_ETH + 0x268 */
	uint32_t ieee_t_sqe;		/* MBAR_ETH + 0x26C */
	uint32_t t_fdxfc;		/* MBAR_ETH + 0x270 */
	uint32_t ieee_t_octets_ok;	/* MBAR_ETH + 0x274 */

	uint32_t res13[2];		/* MBAR_ETH + 0x278-27C */
	uint32_t rmon_r_drop;		/* MBAR_ETH + 0x280 */
	uint32_t rmon_r_packets;	/* MBAR_ETH + 0x284 */
	uint32_t rmon_r_bc_pkt;		/* MBAR_ETH + 0x288 */
	uint32_t rmon_r_mc_pkt;		/* MBAR_ETH + 0x28C */
	uint32_t rmon_r_crc_align;	/* MBAR_ETH + 0x290 */
	uint32_t rmon_r_undersize;	/* MBAR_ETH + 0x294 */
	uint32_t rmon_r_oversize;	/* MBAR_ETH + 0x298 */
	uint32_t rmon_r_frag;		/* MBAR_ETH + 0x29C */
	uint32_t rmon_r_jab;		/* MBAR_ETH + 0x2A0 */

	uint32_t rmon_r_resvd_0;	/* MBAR_ETH + 0x2A4 */

	uint32_t rmon_r_p64;		/* MBAR_ETH + 0x2A8 */
	uint32_t rmon_r_p65to127;	/* MBAR_ETH + 0x2AC */
	uint32_t rmon_r_p128to255;	/* MBAR_ETH + 0x2B0 */
	uint32_t rmon_r_p256to511;	/* MBAR_ETH + 0x2B4 */
	uint32_t rmon_r_p512to1023;	/* MBAR_ETH + 0x2B8 */
	uint32_t rmon_r_p1024to2047;	/* MBAR_ETH + 0x2BC */
	uint32_t rmon_r_p_gte2048;	/* MBAR_ETH + 0x2C0 */
	uint32_t rmon_r_octets;		/* MBAR_ETH + 0x2C4 */
	uint32_t ieee_r_drop;		/* MBAR_ETH + 0x2C8 */
	uint32_t ieee_r_frame_ok;	/* MBAR_ETH + 0x2CC */
	uint32_t ieee_r_crc;		/* MBAR_ETH + 0x2D0 */
	uint32_t ieee_r_align;		/* MBAR_ETH + 0x2D4 */
	uint32_t r_macerr;		/* MBAR_ETH + 0x2D8 */
	uint32_t r_fdxfc;		/* MBAR_ETH + 0x2DC */
	uint32_t ieee_r_octets_ok;	/* MBAR_ETH + 0x2E0 */

	uint32_t res14[7];		/* MBAR_ETH + 0x2E4-2FC */
#if 0
#if defined(CONFIG_MX25) || defined(CONFIG_MX53) || defined(CONFIG_MX6SL)
	uint16_t miigsk_cfgr;		/* MBAR_ETH + 0x300 */
	uint16_t res15[3];		/* MBAR_ETH + 0x302-306 */
	uint16_t miigsk_enr;		/* MBAR_ETH + 0x308 */
	uint16_t res16[3];		/* MBAR_ETH + 0x30a-30e */
	uint32_t res17[60];		/* MBAR_ETH + 0x300-3FF */
#else
	uint32_t res15[64];		/* MBAR_ETH + 0x300-3FF */
#endif
#endif
};


_Static_assert((sizeof(struct rbd) * RBD_COUNT + sizeof(struct tbd) * TBD_COUNT) <= 0x1000, "Expect rx+tx ring to fit in single 4K page");
_Static_assert((RBD_COUNT + TBD_COUNT) * PACKET_BUFFER_SIZE <= 0x200000, "Expect rx+tx buffers to fit in single 2MB page");

volatile uint64_t *shared_counter = (uint64_t *)(uintptr_t)0x1600000;
volatile uint32_t *eth_raw = (uint32_t *)(uintptr_t)0x2000000;
volatile struct regs *eth = (void *)(uintptr_t)0x2000000;

volatile struct rbd *rbd;
volatile struct tbd *tbd;

static char
hexchar(unsigned int v)
{
    return v < 10 ? '0' + v : ('a' - 10) + v;
}

static char
decchar(unsigned int v) {
    return '0' + v;
}

unsigned int slen(const char *c)
{
    unsigned int i = 0;
    while (*c != 0) {
        i++;
        c++;
    }

    return i;
}

static void
dump_reg(const char *name, uint32_t val)
{

    char buffer[8 + 3 + 1];
    buffer[0] = '0';
    buffer[1] = 'x';
    buffer[8 + 3 - 1] = 0;
    for (unsigned i = 8 + 1 + 1; i > 1; i--) {
        if (i == 6) {
            buffer[i] = '_';
        } else {
            buffer[i] = hexchar(val & 0xf);
            val >>= 4;
        }
    }
    microkit_dbg_puts(name);
    // unsigned int l = 10 - slen(name);
    // for (unsigned i = 0; i < l; i++) {
    //     microkit_dbg_putc(' ');
    // }
    microkit_dbg_puts(": ");
    microkit_dbg_puts(buffer);
    microkit_dbg_puts("\n");
}

static void
puthex64(uint64_t x)
{
    char buffer[19];
    buffer[0] = '0';
    buffer[1] = 'x';
    buffer[2] = hexchar((x >> 60) & 0xf);
    buffer[3] = hexchar((x >> 56) & 0xf);
    buffer[4] = hexchar((x >> 52) & 0xf);
    buffer[5] = hexchar((x >> 48) & 0xf);
    buffer[6] = hexchar((x >> 44) & 0xf);
    buffer[7] = hexchar((x >> 40) & 0xf);
    buffer[8] = hexchar((x >> 36) & 0xf);
    buffer[9] = hexchar((x >> 32) & 0xf);
    buffer[10] = hexchar((x >> 28) & 0xf);
    buffer[11] = hexchar((x >> 24) & 0xf);
    buffer[12] = hexchar((x >> 20) & 0xf);
    buffer[13] = hexchar((x >> 16) & 0xf);
    buffer[14] = hexchar((x >> 12) & 0xf);
    buffer[15] = hexchar((x >> 8) & 0xf);
    buffer[16] = hexchar((x >> 4) & 0xf);
    buffer[17] = hexchar(x & 0xf);
    buffer[18] = 0;
    microkit_dbg_puts(buffer);
}

static void
puthex32(uint32_t x)
{
    char buffer[11];
    buffer[0] = '0';
    buffer[1] = 'x';
    buffer[2] = hexchar((x >> 28) & 0xf);
    buffer[3] = hexchar((x >> 24) & 0xf);
    buffer[4] = hexchar((x >> 20) & 0xf);
    buffer[5] = hexchar((x >> 16) & 0xf);
    buffer[6] = hexchar((x >> 12) & 0xf);
    buffer[7] = hexchar((x >> 8) & 0xf);
    buffer[8] = hexchar((x >> 4) & 0xf);
    buffer[9] = hexchar(x & 0xf);
    buffer[10] = 0;
    microkit_dbg_puts(buffer);
}


static void
puthex16(uint16_t x)
{
    char buffer[7];
    buffer[0] = '0';
    buffer[1] = 'x';
    buffer[2] = hexchar((x >> 12) & 0xf);
    buffer[3] = hexchar((x >> 8) & 0xf);
    buffer[4] = hexchar((x >> 4) & 0xf);
    buffer[5] = hexchar(x & 0xf);
    buffer[6] = 0;
    microkit_dbg_puts(buffer);
}

static void
put8(uint8_t x)
{
    char tmp[4];
    unsigned i = 3;
    tmp[3] = 0;
    do {
        uint8_t c = x % 10;
        tmp[--i] = decchar(c);
        x /= 10;
    } while (x);
    microkit_dbg_puts(&tmp[i]);
}

static void
dump_eth(const char *name, unsigned int x)
{
    dump_reg(name, eth_raw[x / 4]);
}

static uint16_t
swap16(uint16_t v)
{
    return ((v & 0xff) << 8) | (v >> 8);
}


static bool
ip_match(uint8_t *a, uint8_t *b)
{
    return (
        (a[0] == b[0]) &&
        (a[1] == b[1]) &&
        (a[2] == b[2]) &&
        (a[3] == b[3])
    );
}

static void
set_mac(uint8_t *dst, uint8_t *src)
{
    for (unsigned i = 0; i < 6; i++) {
        dst[i] = src[i];
    }

}

static void
set_ip(uint8_t *dst, uint8_t *src)
{
    for (unsigned i = 0; i < 4; i++) {
        dst[i] = src[i];
    }

}

static char *
ethertype_to_str(uint16_t ethertype)
{
    switch (ethertype) {
        case ETHERTYPE_IPV4: return "IPv4";
        case ETHERTYPE_ARP: return "ARP";
        case ETHERTYPE_WOL: return "Wake-on-LAN";
        case ETHERTYPE_RARP: return "Reverse-ARP";
        case ETHERTYPE_IPV6: return "IPv6";
    }
    return "<unknown ether type>";
}

uint16_t
cksum(uint8_t *d, int len)
{
    uint32_t sum = 0;  /* assume 32 bit long, 16 bit short */

    while (len > 1) {
        sum += * ((uint16_t *) d);
        d += 2;
        len -= 2;
    }

    if (len > 0) {       /* take care of left over byte */
        sum += (uint16_t) *d;
    }

    while (sum >> 16) {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    return ~sum;
}

static void
get_mac_addr(volatile struct regs *reg, uint8_t *mac)
{
    uint32_t l, h;
    l = reg->paddr1;
    h = reg->paddr2;

    mac[0] = l >> 24;
    mac[1] = l >> 16 & 0xff;
    mac[2] = l >> 8 & 0xff;
    mac[3] = l & 0xff;
    mac[4] = h >> 24;
    mac[5] = h >> 16 & 0xff;
}

static int
mac_match(uint8_t *m1, uint8_t *m2)
{
    return (
        (m1[0] == m2[0]) &&
        (m1[1] == m2[1]) &&
        (m1[2] == m2[2]) &&
        (m1[3] == m2[3]) &&
        (m1[4] == m2[4]) &&
        (m1[5] == m2[5])
    );
}


static void
dump_mac(uint8_t *mac)
{
    for (unsigned i = 0; i < 6; i++) {
        microkit_dbg_putc(hexchar((mac[i] >> 4) & 0xf));
        microkit_dbg_putc(hexchar(mac[i] & 0xf));
        if (i < 5) {
            microkit_dbg_putc(':');
        }
    }
}

static void
dump_ip(uint8_t *ip)
{
    for (unsigned i = 0; i < 4; i++) {
        put8(ip[i]);
        if (i < 3) {
            microkit_dbg_putc('.');
        }
    }
}

static void
dump_hex(const uint8_t *d, unsigned int length)
{
    unsigned int i = 0;
    while (length) {
        puthex16(i);
        microkit_dbg_puts(": ");
        while (length) {
            microkit_dbg_putc(hexchar((d[i] >> 4) & 0xf));
            microkit_dbg_putc(hexchar(d[i] & 0xf));
            length--;
            i++;
            if (i % 16 == 0) {
                microkit_dbg_putc('\n');
                break;
            } else {
                microkit_dbg_putc(' ');
            }
        }
    }
    if (i % 16) {
        microkit_dbg_putc('\n');
    }
}

/* NOTE: This assumes we are copying to / from packet buffers, so we know we can doing unrolled copies */
static void
mycpy(void *dst, void *src, unsigned int length)
{
    uint64_t *d = dst;
    uint64_t *s = src;
    int i = 0;
    int l = length / 64;
    if (length % 64) {
        l++;
    }
    while (l) {
        d[i] = s[i];
        d[i + 1] = s[i + 1];
        d[i + 2] = s[i + 2];
        d[i + 3] = s[i + 3];
        d[i + 4] = s[i + 4];
        d[i + 5] = s[i + 5];
        d[i + 6] = s[i + 6];
        d[i + 7] = s[i + 7];
        l--;
        i += 8;
    }
}

static int
mycmp(char *a, char *b) {
    int i = 0;
    do {
        if (a[i] != b[i]) {
            return -1;
        }
        i++;
    } while (a[i] != 0);
    return 0;
}


static void
send_frame(uint8_t *d, unsigned int length)
{
    uint16_t flags;
    void *packet;

    flags = tbd[tbd_index].flags;

    if (flags & (1 << 15)) {
        microkit_dbg_puts(microkit_name);
        microkit_dbg_puts(": ran out of tx buffers!!\n");
        return;
    }

#if 0
    microkit_dbg_puts(microkit_name);
    microkit_dbg_puts(": sending -- tbd_index: ");
    puthex16(tbd_index);
    microkit_dbg_puts(" - ");
    {
        uint64_t t;
        asm volatile("mrs %0, cntpct_el0" : "=r"(t));
        puthex64(t);
    }
    microkit_dbg_puts("\n");
#endif

    packet = (void *)(uintptr_t)(packet_buffer_vaddr + ((RBD_COUNT + tbd_index) * PACKET_BUFFER_SIZE));
    mycpy(packet, d, length);
    seL4_ARM_VSpace_CleanInvalidate_Data(3, (uintptr_t)packet, ((uintptr_t)packet) + length);

    flags = (
        (1 << 15) | /* ready */
        (1 << 11) | /* last in frame */
        (1 << 10) /* transmit crc */
    );
    if (tbd_index == TBD_COUNT - 1) {
        flags |= (1 << 13) /* wrap */;
    }

    tbd[tbd_index].data_length = length;
    tbd[tbd_index].flags = flags;

    /* read back flags */
    flags = tbd[tbd_index].flags;

    /* SEND */
    eth->tdar = (1 << 24);

    tbd_index++;
    if (tbd_index == TBD_COUNT) {
        tbd_index = 0;
    }
}


static void
eth_setup(void)
{
    get_mac_addr(eth, mac);
    microkit_dbg_puts("MAC: ");
    dump_mac(mac);
    microkit_dbg_puts("\n");

    rbd = (void *)ring_buffer_vaddr;
    tbd = (void *)(ring_buffer_vaddr + (sizeof(struct rbd) * RBD_COUNT));

    for (unsigned i = 0; i < RBD_COUNT; i++) {
        rbd[i].data_length = 0;
        rbd[i].flags = (1UL << 15);
        rbd[i].addr = packet_buffer_paddr + (i * PACKET_BUFFER_SIZE);
#if 0
        microkit_dbg_puts("ETH: ");
        microkit_dbg_puts(microkit_name);
        microkit_dbg_puts("  rbd[i].addr ");
        puthex32(rbd[i].addr);
        microkit_dbg_puts("\n");
#endif
    }

    for (unsigned i = 0; i < TBD_COUNT; i++) {
        tbd[i].data_length = 0;
        tbd[i].flags = 0;
        tbd[i].addr = packet_buffer_paddr + ((RBD_COUNT + i) * PACKET_BUFFER_SIZE);
#if 0
        microkit_dbg_puts("ETH: ");
        microkit_dbg_puts(microkit_name);
        microkit_dbg_puts("  tbd[i].addr ");
        puthex32(rbd[i].addr);
        microkit_dbg_puts("\n");
#endif
    }

    rbd[RBD_COUNT-1].flags |= (1UL << 13);
    tbd[TBD_COUNT-1].flags |= (1UL << 13);

    eth->eir = eth->eir;
    eth->eimr = 0xffffffffUL;

    /* Set RDSR */
    get_mac_addr(eth, mac);
    microkit_dbg_puts("RING BUFFER ADDR=: ");
    puthex64((uintptr_t)ring_buffer_paddr);
    microkit_dbg_puts("\n");

    eth->erdsr = ring_buffer_paddr;
    eth->etdsr = ring_buffer_paddr + (sizeof(struct rbd) * RBD_COUNT);

    eth->emrbr = 1536;

    eth->ecr |= (1 << 8) | (1 << 5);
    eth->rcr = 0x05f20064 | (1 << 3); /* promiscuous mode */
    eth->tcr = (1 << 2); /* full-duplex */

    /* Set Enable  in ECR */
    eth->ecr |= 2;
    dump_reg("rcr", eth->rcr);
    dump_reg("ecr", eth->ecr);

    eth->rdar = (1 << 24);

    microkit_dbg_puts(microkit_name);
    microkit_dbg_puts(": init complete -- waiting for interrupt\n");
}

static void
handle_rx(microkit_channel ch, volatile struct regs *eth)
{
    uint16_t flags;
    int r;

    /* received at least one frame, iterate through all receive descriptor buffers */
    for (;;) {
        void *packet;
        uint16_t packet_length;
        bool pass_through = true;

        flags = rbd[rbd_index].flags;
        packet_length = rbd[rbd_index].data_length;

//        uint64_t c = get_sys_counter();
//        microkit_dbg_puts("counter: ");
//        puthex64(c);
//        microkit_dbg_puts("\n");

        if ((flags & (1 << 15))) {
            /* buffer is empty, can stop */
            break;
        }


#if 0
        microkit_dbg_puts("rbd_index: ");
        puthex16(rbd_index);
        dump_reg("  flags", flags);
        microkit_dbg_puts(" -- length ");
        puthex16(rbd[rbd_index].data_length);
        microkit_dbg_puts("\n");
#endif
#if 0
        if (mycmp(microkit_name, "eth_inner") == 0) {
            microkit_dbg_puts("XX BUFFER\n");
            for (unsigned xx=0; xx < 16; xx++) {
                microkit_dbg_puts("XX rbd_index: ");
                puthex16(xx);
                dump_reg("  flags", rbd[xx].flags);
                microkit_dbg_puts(" -- length ");
                puthex16(rbd[xx].data_length);
                microkit_dbg_puts("\n");
            }
        }
#endif

        if (packet_length == 0) {
            microkit_dbg_puts("ETH: ");
            microkit_dbg_puts(microkit_name);
            microkit_dbg_puts(" UNEXPECTED ZERO LENGTH RX PACKET rbd_index: ");
            puthex16(rbd_index);
            microkit_dbg_puts("\n");
            for (;;) { }
            goto make_avail;
        }


        packet = (void *)(packet_buffer_vaddr + (rbd_index * PACKET_BUFFER_SIZE));
        r = seL4_ARM_VSpace_Invalidate_Data(3, (uintptr_t)packet, ((uintptr_t)packet) + packet_length);
        if (r != 0) {
            microkit_dbg_puts("ERR: I\n");
            microkit_dbg_puts("ETH: ");
            microkit_dbg_puts(microkit_name);
            microkit_dbg_puts("  --  invalidate with: packet ");
            puthex64((uint64_t)packet);
            microkit_dbg_puts("    length: ");
            puthex16(packet_length);
            microkit_dbg_puts("   rbd_index: ");
            puthex16(rbd_index);
            microkit_dbg_puts("\n");
            for (;;) {

            }
        }

#if 1
        if (mycmp(microkit_name, "eth_outer") == 0) {
                struct eth_header *hdr = packet;

                if (mac_match(hdr->dest_mac, mac) || mac_match(hdr->dest_mac, broadcast_mac)) {
                    pass_through = false;
        #if 0
                    microkit_dbg_puts("DEST MAC: ");
                    dump_mac(hdr->dest_mac);
                    microkit_dbg_puts("\n");
                    microkit_dbg_puts("SRC MAC: ");
                    dump_mac(hdr->src_mac);
                    microkit_dbg_puts("\n");
                    microkit_dbg_puts("Ethertype: ");
                    puthex16(swap16(hdr->ethertype));
                    microkit_dbg_puts("  (");
                    microkit_dbg_puts(ethertype_to_str(swap16(hdr->ethertype)));
                    microkit_dbg_puts(")\n");
                    if (mac_match(hdr->dest_mac, mac)) {
                        microkit_dbg_puts("exact match\n");
                    }

                    if (mac_match(hdr->dest_mac, broadcast_mac)) {
                        microkit_dbg_puts("broadcast match\n");
                    }
        #endif
                    if (swap16(hdr->ethertype) == ETHERTYPE_ARP) {
                        struct arp *a = (struct arp *)&hdr->payload[0];
        #if 0
                        microkit_dbg_puts("   arp.htype: ");
                        puthex16(swap16(a->htype));
                        microkit_dbg_puts("\n   arp.ptype: ");
                        puthex16(swap16(a->ptype));
                        microkit_dbg_puts("\n   arp.plen: ");
                        put8(a->plen);
                        microkit_dbg_puts("\n   arp.hlen: ");
                        put8(a->hlen);
                        microkit_dbg_puts("\n   arp.spa: ");
                        dump_ip(a->spa);
                        microkit_dbg_puts("\n   arp.tpa: ");
                        dump_ip(a->tpa);
                        microkit_dbg_puts("\n   arp.sha: ");
                        dump_mac(a->sha);
                        microkit_dbg_puts("\n   arp.tha: ");
                        dump_mac(a->tha);
                        microkit_dbg_puts("\n");
        #endif
                        if (
                            (swap16(a->htype) == 1) &&
                            (swap16(a->ptype) == 0x0800) &&
                            (a->hlen == 6) &&
                            (a->plen == 4) &&
                            (ip_match(a->tpa, my_ip))
                        ) {
        #if 0
                            microkit_dbg_puts("HELP: ARP packet we should reply to\n");
        #endif
                            mycpy(temp_packet, packet,  rbd[rbd_index].data_length);

                            struct eth_header *snd_hdr = (struct eth_header *)&temp_packet;
                            /* set the MAC addresses */
                            set_mac(snd_hdr->dest_mac, hdr->src_mac);
                            set_mac(snd_hdr->src_mac, mac);
                            struct arp *snd_a = (struct arp *)&snd_hdr->payload[0];
                            snd_a->oper = swap16(2);
                            set_mac(snd_a->sha, mac);
                            set_ip(snd_a->spa, my_ip);

                            set_mac(snd_a->tha, a->sha);
                            set_ip(snd_a->tpa, a->spa);

                            send_frame(temp_packet, rbd[rbd_index].data_length);
                        }

                    }

                    if (swap16(hdr->ethertype) == ETHERTYPE_IPV4) {
                        struct ip *i = (struct ip *)&hdr->payload[0];
                        uint8_t header_len = (i->ver_ihl & 0xf) * 4;
        #if 0
                        uint8_t version = i->ver_ihl >> 4;
                        microkit_dbg_puts("IP\n");
                        microkit_dbg_puts("   ip.version: ");
                        put8(version);
                        microkit_dbg_puts("   ip.header_length: ");
                        put8(header_len);
                        microkit_dbg_puts("   ip.protocol: ");
                        put8(i->protocol);
                        microkit_dbg_puts("\n   ip.tos: ");
                        put8(i->tos);
                        microkit_dbg_puts("\n   ip.len: ");
                        puthex16(swap16(i->len));
                        microkit_dbg_puts("\n   ip.src: ");
                        dump_ip(i->source_address);
                        microkit_dbg_puts("\n   ip.dst: ");
                        dump_ip(i->dest_address);
                        microkit_dbg_puts("\n");
        #endif
                        if (i->protocol == 1) {
                            struct icmp *icmp = (struct icmp *)(&hdr->payload[header_len]);
        #if 0
                            microkit_dbg_puts("ICMP\n");
                            microkit_dbg_puts("   icmp.type: ");
                            put8(icmp->type);
                            microkit_dbg_puts("   icmp.code: ");
                            put8(icmp->code);
                            microkit_dbg_puts("   icmp.rest_of_header: ");
                            puthex16(swap16(icmp->rest_of_header));
                            microkit_dbg_puts("\n");
        #endif

                            if (icmp->type == 8) {
        #if 0
                                microkit_dbg_puts("ICMP ECHO REQUEST\n");
        #endif
                                mycpy(temp_packet, packet,  rbd[rbd_index].data_length);

                                struct eth_header *snd_hdr = (struct eth_header *)&temp_packet;
                                /* set the MAC addresses */
                                set_mac(snd_hdr->dest_mac, hdr->src_mac);
                                set_mac(snd_hdr->src_mac, mac);
                                struct ip *snd_ip = (struct ip *)&snd_hdr->payload[0];

                                set_mac(snd_ip->source_address, i->dest_address);
                                set_ip(snd_ip->dest_address, i->source_address);

                                struct icmp *snd_icmp = (struct icmp *)(&snd_hdr->payload[header_len]);

                                /* Set reply */
                                snd_icmp->type = 0;

                                snd_icmp->checksum = 0;
                                snd_icmp->checksum = cksum((uint8_t *) snd_icmp, swap16(i->len) - header_len);//sizeof(struct icmp));
        #if 0
                                microkit_dbg_puts("CHECKSUM: ");
                                puthex16(snd_icmp->checksum);
                                microkit_dbg_puts("\n");
        #endif
                                send_frame(temp_packet, rbd[rbd_index].data_length);
                            }
                        }


                    }
        #if 0
                    microkit_dbg_puts("\n");
        #endif
                }
        }
#endif

        if (pass_through) {
            /* Try and send */
            volatile struct buffer_descriptor *bd = (void *)(uintptr_t)(OUTPUT_BUFFER + (BUFFER_SIZE * output_index));
            volatile void *output_packet = (void *)(uintptr_t)(OUTPUT_BUFFER + (BUFFER_SIZE * output_index) + DATA_OFFSET);
            if (bd->flags == 1) {
                microkit_dbg_puts("ETH: ");
                microkit_dbg_puts(microkit_name);
                microkit_dbg_puts("dropping packet, no space in channel buffer\n");
            } else {
                bd->data_length = rbd[rbd_index].data_length - 4; /* For the frame check sequence */
                mycpy((void *)output_packet, packet, bd->data_length);
                bd->flags = 1;
                output_index++;
                if (output_index == BUFFER_MAX) {
                    output_index = 0;
                }
                microkit_notify(OUTPUT_CH);
            }
        }

make_avail:
        /* make it available */
        flags = (1 << 15);
        if (rbd_index == RBD_COUNT - 1) {
            flags |= (1 << 13);
        }
        rbd[rbd_index].flags = flags;

        rbd_index++;
        if (rbd_index == RBD_COUNT) {
            rbd_index = 0;
        }
    }

    /* kick the rx engine if necessary */
    eth->rdar = (1 << 24);
}

static void
handle_eth(microkit_channel ch, volatile struct regs *eth)
{
    uint32_t eir = eth->eir;
    eth->eir = eir;

    /* handle all the events of interest -- see 14.6.5.1 for details.

        ignore all of the following:
            babbling errors (tx & rx)
            graceful stop complete
            tx buffer (just handle the frame interrupt)
            rx buffer (just handle the frame interrupt)
            mii interrupt (FIXME: probably need to handle this)
            bus error (FIXME: should treat this as an error condition and recover)
            late collision
            collision retry limit
            tx underrun -- FIXME: maybe we want to handle this?
            payload receive error
            node wakeup request indication (not using magic packets)
            transmit timestamp available (not using timestampping)
            timestamp timer (not using timestamps)
            rx dma ring 0/1/2 (not applicable)
            parser error (not applicable)
            tx/rx buffer/frame / class 1/2/3 (not using QoS).
     */
    if (eir & (1 << 25)) {
        handle_rx(ch, eth);
    }

    if (eir & (1 << 27)) {
    }

    microkit_irq_ack(ch);
}


static void example_constructor(void) __attribute__ ((constructor));

static void
example_constructor(void)
{
    microkit_dbg_puts("Example constructor\n");
}


void
init(void)
{
    microkit_dbg_puts(microkit_name);
    microkit_dbg_puts(": elf PD init function running\n");

    eth_setup();
}

void
notified(microkit_channel ch)
{
    switch (ch) {

        case IRQ_CH:
            handle_eth(ch, eth);
            break;

        case INPUT_CH:
#if 0
            microkit_dbg_puts("ETH: ");
            microkit_dbg_puts(microkit_name);
            microkit_dbg_puts("  got input notification\n");
#endif
            for (;;) {
                volatile struct buffer_descriptor *bd = (void *)(uintptr_t)(INPUT_BUFFER + (BUFFER_SIZE * input_index));
                volatile void *pkt = (void *)(uintptr_t)(INPUT_BUFFER + (BUFFER_SIZE * input_index) + DATA_OFFSET);
                if (bd->flags == 0) {
                    break;
                }
#if 0
                microkit_dbg_puts("ETH: packet: ");
                puthex16(input_index);
                microkit_dbg_puts("packet length: ");
                puthex16(bd->data_length);
                microkit_dbg_puts("\n");
#endif
                send_frame((void*)pkt, bd->data_length);
                bd->flags = 0;

                input_index++;
                if (input_index == BUFFER_MAX) {
                    input_index = 0;
                }
            }
            break;

        case OUTPUT_CH:
#if 0
            microkit_dbg_puts("ETH: ");
            microkit_dbg_puts(microkit_name);
            microkit_dbg_puts("got output ack (is this needed?)\n");
#endif
            break;

        default:
            microkit_dbg_puts("hello: received notification on unexpected channel\n");
            dump_reg("CH", ch);
            break;
    }
}
