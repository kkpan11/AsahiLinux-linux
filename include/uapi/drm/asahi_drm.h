/* SPDX-License-Identifier: MIT */
/*
 * Copyright (C) The Asahi Linux Contributors
 * Copyright © 2014-2018 Broadcom
 * Copyright © 2019 Collabora ltd.
 */
/* clang-format off */
#ifndef _ASAHI_DRM_H_
#define _ASAHI_DRM_H_

#include "drm.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define DRM_ASAHI_GET_PARAMS			0x00
#define DRM_ASAHI_VM_CREATE			0x01
#define DRM_ASAHI_VM_DESTROY			0x02
#define DRM_ASAHI_GEM_CREATE			0x03
#define DRM_ASAHI_GEM_MMAP_OFFSET		0x04
#define DRM_ASAHI_GEM_BIND			0x05
#define DRM_ASAHI_QUEUE_CREATE			0x06
#define DRM_ASAHI_QUEUE_DESTROY			0x07
#define DRM_ASAHI_SUBMIT			0x08
#define DRM_ASAHI_GET_TIME			0x09
/* TODO: Maybe merge with DRM_ASAHI_GEM_BIND? (Becomes IOWR) */
#define DRM_ASAHI_GEM_BIND_OBJECT		0x0a

#define DRM_ASAHI_MAX_CLUSTERS	64

struct drm_asahi_params_global {
	/** @features: Feature bits from drm_asahi_feature */
	__u64 features;

	/** @gpu_generation: GPU generation, e.g. 13 for G13G */
	__u32 gpu_generation;

	/** @gpu_variant: GPU variant as a character, e.g. 'G' for G13G */
	__u32 gpu_variant;

	/** @gpu_revision: GPU revision in BCD, e.g. 0x00 for 'A0' or
	 * 0x21 for 'C1'
	 */
	__u32 gpu_revision;

	/** @chip_id: Chip ID in BCD, e.g. 0x8103 for T8103 */
	__u32 chip_id;

	/** @num_dies: Number of dies in the SoC */
	__u32 num_dies;

	/** @num_clusters_total: Number of GPU clusters (across all dies) */
	__u32 num_clusters_total;

	/** @num_cores_per_cluster: Number of logical cores per cluster
	 *  (including inactive/nonexistent) */
	__u32 num_cores_per_cluster;

	/** @num_frags_per_cluster: Number of frags per cluster */
	__u32 num_frags_per_cluster;

	/** @num_gps_per_cluster: Number of GPs per cluster */
	__u32 num_gps_per_cluster;

	/** @core_masks: Bitmask of present/enabled cores per cluster */
	__u64 core_masks[DRM_ASAHI_MAX_CLUSTERS];

	/** @timer_frequency_hz: Clock frequency for timestamps */
	__u64 timer_frequency_hz;

	/** @min_frequency_khz: Minimum GPU core clock frequency */
	__u32 min_frequency_khz;

	/** @max_frequency_khz: Maximum GPU core clock frequency */
	__u32 max_frequency_khz;

	/** @max_power_mw: Maximum GPU power consumption */
	__u32 max_power_mw;

	/** @vm_page_size: GPU VM page size */
	__u32 vm_page_size;

	/** @vm_user_start: VM user range start VMA */
	__u64 vm_user_start;

	/** @vm_user_end: VM user range end VMA */
	__u64 vm_user_end;

	/** @vm_kernel_min_size: Minimum kernel VMA window size within user
	 * range
	 */
	__u64 vm_kernel_min_size;

	/** @max_commands_per_submission: Maximum number of supported commands
	 * per submission
	 */
	__u32 max_commands_per_submission;

	/** @max_attachments: Maximum number of drm_asahi_attachment's per
	 * command
	 */
	__u32 max_attachments;

	/** @firmware_version: GPU firmware version, as 4 integers */
	__u32 firmware_version[4];

	/** @user_timestamp_frequency_hz: Timebase frequency for user timestamps 
	 */
	__u64 user_timestamp_frequency_hz;
};

/** Feature bits.
 *
 * This covers only features that userspace cannot infer from the architecture
 * version. Most features don't need to be here.
 */
enum drm_asahi_feature {
	/** GPU has "soft fault" enabled. Shader loads of unmapped memory will
	 * return zero. Shader stores to unmapped memory will be silently
	 * discarded. Note that only shader load/store is affected. Other
	 * hardware units are not affected, notably including texture sampling.
	 */
	DRM_ASAHI_FEATURE_SOFT_FAULTS = (1UL) << 0,
};

/** Get driver/GPU parameters */
struct drm_asahi_get_params {
	/** @param: Parameter group to fetch (MBZ) */
	__u32 param_group;

	/** @pad: MBZ */
	__u32 pad;

	/** @value: User pointer to write parameter struct */
	__u64 pointer;

	/** @value: Size of user buffer, max size supported on return */
	__u64 size;
};

/** Create a GPU VM address space */
struct drm_asahi_vm_create {
	/** @kernel_start: Start of the kernel-reserved address range */
	__u64 kernel_start;

	/** @kernel_end: End of the kernel-reserved address range */
	__u64 kernel_end;

	/** @value: Returned VM ID */
	__u32 vm_id;

	/** @pad: MBZ */
	__u32 pad;
};

/** Destroy a GPU VM address space */
struct drm_asahi_vm_destroy {
	/** @value: VM ID to be destroyed */
	__u32 vm_id;

	/** @pad: MBZ */
	__u32 pad;
};

/** BO should be CPU-mapped as writeback, not write-combine. This optimizes for
 * CPU reads.
 */
#define ASAHI_GEM_WRITEBACK	(1L << 0)

/** BO is private to this GPU VM (no exports) */
#define ASAHI_GEM_VM_PRIVATE	(1L << 1)

/** Destroy a GPU VM address space */
struct drm_asahi_gem_create {
	/** @size: Size of the BO */
	__u64 size;

	/** @flags: BO creation flags */
	__u32 flags;

	/** @handle: VM ID to assign to the BO, if ASAHI_GEM_VM_PRIVATE is set
	 */
	__u32 vm_id;

	/** @handle: Returned GEM handle for the BO */
	__u32 handle;

	/** @pad: MBZ */
	__u32 pad;
};

/** Get BO mmap offset */
struct drm_asahi_gem_mmap_offset {
	/** @handle: Handle for the object being mapped. */
	__u32 handle;

	/** @flags: Must be zero */
	__u32 flags;

	/** @offset: The fake offset to use for subsequent mmap call */
	__u64 offset;
};

/** VM_BIND operations */
enum drm_asahi_bind_op {
	/** Bind a BO to a GPU VMA range */
	ASAHI_BIND_OP_BIND = 0,

	/** Unbind a GPU VMA range */
	ASAHI_BIND_OP_UNBIND = 1,

	/** Unbind all mappings of a given BO */
	ASAHI_BIND_OP_UNBIND_ALL = 2,
};

/** Map BO with GPU read permission */
#define ASAHI_BIND_READ		(1L << 0)

/** Map BO with GPU write permission */
#define ASAHI_BIND_WRITE	(1L << 1)

/** Map a single page of the BO repeatedly across the VA range */
#define ASAHI_BIND_SINGLE_PAGE	(1L << 2)

/** BO VM_BIND operations */
struct drm_asahi_gem_bind {
	/** @obj: Bind operation (enum drm_asahi_bind_op) */
	__u32 op;

	/** @flags: One or more of ASAHI_BIND_* (BIND only) */
	__u32 flags;

	/** @obj: GEM object to bind/unbind (BIND or UNBIND_ALL) */
	__u32 handle;

	/** @vm_id: The ID of the VM to operate on */
	__u32 vm_id;

	/** @offset: Offset into the object (BIND only) */
	__u64 offset;

	/** @range: Number of bytes to bind/unbind to addr (BIND or UNBIND only)
	 */
	__u64 range;

	/** @addr: Address to bind to (BIND or UNBIND only) */
	__u64 addr;
};

/** VM_BIND operations */
enum drm_asahi_bind_object_op {
	/** Bind a BO as a special GPU object */
	ASAHI_BIND_OBJECT_OP_BIND = 0,

	/** Unbind a special GPU object */
	ASAHI_BIND_OBJECT_OP_UNBIND = 1,
};

/** Map a BO as a timestamp buffer */
#define ASAHI_BIND_OBJECT_USAGE_TIMESTAMPS	(1L << 0)

/** BO special object operations */
struct drm_asahi_gem_bind_object {
	/** @obj: Bind operation (enum drm_asahi_bind_object_op) */
	__u32 op;

	/** @flags: One or more of ASAHI_BIND_OBJECT_* */
	__u32 flags;

	/** @obj: GEM object to bind/unbind (BIND) */
	__u32 handle;

	/** @vm_id: The ID of the VM to operate on (MBZ currently) */
	__u32 vm_id;

	/** @offset: Offset into the object (BIND only) */
	__u64 offset;

	/** @range: Number of bytes to bind/unbind (BIND only) */
	__u64 range;

	/** @addr: Object handle (out for BIND, in for UNBIND) */
	__u32 object_handle;

	/** @pad: MBZ */
	__u32 pad;
};

/** Command type */
enum drm_asahi_cmd_type {
	/** Render command (Render subqueue, Vert+Frag) */
	DRM_ASAHI_CMD_RENDER = 0,

	/** Compute command (Compute subqueue) */
	DRM_ASAHI_CMD_COMPUTE = 1,
};

/** Queue capabilities */
/* Note: this is an enum so that it can be resolved by Rust bindgen. */
enum drm_asahi_queue_cap {
	/** Supports render commands */
	DRM_ASAHI_QUEUE_CAP_RENDER	= (1UL << DRM_ASAHI_CMD_RENDER),

	/** Supports compute commands */
	DRM_ASAHI_QUEUE_CAP_COMPUTE	= (1UL << DRM_ASAHI_CMD_COMPUTE),
};

/** Create a queue */
struct drm_asahi_queue_create {
	/** @flags: MBZ */
	__u32 flags;

	/** @vm_id: The ID of the VM this queue is bound to */
	__u32 vm_id;

	/** @type: Bitmask of DRM_ASAHI_QUEUE_CAP_* */
	__u32 queue_caps;

	/** @priority: Queue priority, 0-3 */
	__u32 priority;

	/** @queue_id: The returned queue ID */
	__u32 queue_id;

	/** @pad: MBZ */
	__u32 pad;
};

/** Destroy a queue */
struct drm_asahi_queue_destroy {
	/** @queue_id: The queue ID to be destroyed */
	__u32 queue_id;

	/** @pad: MBZ */
	__u32 pad;
};

/** Sync item types */
enum drm_asahi_sync_type {
	/** Binary sync object */
	DRM_ASAHI_SYNC_SYNCOBJ = 0,

	/** Timeline sync object */
	DRM_ASAHI_SYNC_TIMELINE_SYNCOBJ = 1,
};

/** Sync item */
struct drm_asahi_sync {
	/** @sync_type: One of drm_asahi_sync_type */
	__u32 sync_type;

	/** @handle: The sync object handle */
	__u32 handle;

	/** @timeline_value: Timeline value for timeline sync objects */
	__u64 timeline_value;
};

/** Sub-queues within a queue */
enum drm_asahi_subqueue {
	/** Render subqueue */
	DRM_ASAHI_SUBQUEUE_RENDER = 0,

	/** Compute subqueue */
	DRM_ASAHI_SUBQUEUE_COMPUTE = 1,

	/** Queue count, must remain multiple of 2 for struct alignment */
	DRM_ASAHI_SUBQUEUE_COUNT = 2,
};

/** Command index for no barrier */
#define DRM_ASAHI_BARRIER_NONE ~(0U)

/** Top level command structure */
struct drm_asahi_command {
	/** @type: One of drm_asahi_cmd_type */
	__u32 cmd_type;

	/** @flags: Flags for command submission */
	__u32 flags;

	/** @cmdbuf: Pointer to the appropriate command buffer structure */
	__u64 cmd_buffer;

	/** @cmdbuf: Size of the command buffer structure */
	__u64 cmd_buffer_size;

	/** @barriers: Array of command indices per subqueue to wait on */
	__u32 barriers[DRM_ASAHI_SUBQUEUE_COUNT];
};

/** Submit an array of commands to a queue */
struct drm_asahi_submit {
	/** @in_syncs: An optional array of drm_asahi_sync to wait on before
	 * starting this job.
	 */
	__u64 in_syncs;

	/** @out_syncs: An optional array of drm_asahi_sync objects to signal
	 * upon completion.
	 */
	__u64 out_syncs;

	/** @commands: Pointer to the drm_asahi_command array of commands to
	 * submit.
	 */
	__u64 commands;

	/** @flags: Flags for command submission (MBZ) */
	__u32 flags;

	/** @queue_id: The queue ID to be submitted to */
	__u32 queue_id;

	/** @in_sync_count: Number of sync objects to wait on before starting
	 * this job.
	 */
	__u32 in_sync_count;

	/** @out_sync_count: Number of sync objects to signal upon completion of
	 * this job.
	 */
	__u32 out_sync_count;

	/** @command_count: Number of commands to be submitted */
	__u32 command_count;
};

/** An attachment definition. Attachments are any memory written by shaders,
 * notably including render target attachments written by the end-of-tile
 * program. This is purely a hint about the accessed memory regions. It is
 * optional to specify, which is fortunate as it cannot be specified precisely
 * with bindless access anyway. But where possible, it's probably a good idea
 * for userspace to include these hints, forwarded to the firmware.
 */
struct drm_asahi_attachment {
	/** @pointer: Base address of the attachment */
	__u64 pointer;

	/** @size: Size of the attachment in bytes */
	__u64 size;

	/** @pad: MBZ */
	__u32 pad;

	/** @flags: MBZ */
	__u32 flags;
};

/** Vertex stage shader spills */
#define ASAHI_RENDER_VERTEX_SPILLS (1UL << 0)

/** Process empty tiles through the fragment load/store */
#define ASAHI_RENDER_PROCESS_EMPTY_TILES (1UL << 1)

/** Run vertex stage on a single cluster (on multicluster GPUs) */
#define ASAHI_RENDER_NO_VERTEX_CLUSTERING (1UL << 2)

/** Use integer (unorm) formula for depth bias instead of float. This
 * corresponds to bit 18 of the hardware register, so we match that here.
 */
#define ASAHI_RENDER_DBIAS_IS_INT (1UL << 18)

struct drm_asahi_zls_buffer {
	/** @load: Base address of the buffer to load at the start */
	__u64 load;

	/** @store: Base address of the buffer to store at the end */
	__u64 store;

	/** @partial: Base address of the buffer to load and store during a
	 * partial render operation.
	 */
	__u64 partial;

	/** @comp_load, @comp_store, @comp_partial: If the respective buffer is
	 * compressed, address of the compression metadata section.
	 */
	__u64 comp_load;
	__u64 comp_store;
	__u64 comp_partial;

	/** @load_stride, @store_stride, @partial_stride, @comp_load_stride,
	 * @comp_store_stride. @comp_partial_stride: If layered rendering is
	 * enabled, the number of bytes between each layers of the respective
	 * buffer.
	 */
	__u32 load_stride;
	__u32 store_stride;
	__u32 partial_stride;
	__u32 comp_load_stride;
	__u32 comp_store_stride;
	__u32 comp_partial_stride;
};

struct drm_asahi_timestamp {
	/** @handle: Handle of the timestamp buffer */
	__u32 handle;

	/** @offset: Offset to write into the timestamp buffer */
	__u32 offset;
};

struct drm_asahi_timestamps {
	/** @start: Timestamp recorded at the start of the operation */
	struct drm_asahi_timestamp start;

	/** @end: Timestamp recorded at the end of the operation */
	struct drm_asahi_timestamp end;
};

/** The helper program is a compute-like kernel required for various
 * hardware functionality. Its most important role is dynamically allocating
 * scratch/stack memory for individual subgroups, by partitioning a static
 * allocation shared for the whole device. It is supplied by userspace via
 * drm_asahi_helper_program and internally dispatched by the hardware as needed.
 */
struct drm_asahi_helper_program {
	/** @binary: USC address to the helper program binary */
	__u32 binary;

	/** @cfg: Configuration bits for the helper program. */
	__u32 cfg;

	/** @data: Data passed to the helper program. This value is not
	 * interpreted by the kernel, firmware, or hardware in any way. It is
	 * simply a sideband for userspace, set with the submit ioctl and read
	 * via special registers inside the helper program.
	 *
	 * In practice, userspace will pass a 64-bit GPU VA here pointing to the
	 * actual arguments, which presumably don't fit in 64-bits.
	 */
	__u64 data;
};

/** The background and end-of-tile programs are dispatched by the hardware at
 * the beginning and end of rendering. As the hardware "tilebuffer" is simply
 * local memory, these programs are necessary to implement API-level render
 * targets. The fragment-like background program is responsible for loading
 * either the clear colour or the existing render target contents, while the
 * compute-like end-of-tile program stores the tilebuffer contents to memory.
 */
struct drm_asahi_bg_eot {
	/* @usc: USC address of the hardware USC words binding resources
	 * (including images and uniforms) and the program itself. Note this is
	 * an additional layer of indirection compared to the helper program,
	 * avoiding the need for a sideband for data.
	 */
	__u32 usc;

	/* @rsrc_spec: Resource specifier for the program. This is a packed
	 * hardware data structure describing the required number of registers,
	 * uniforms, bound textures, and bound samplers.
	 */
	__u32 rsrc_spec;
};

/** Render command submission data */
struct drm_asahi_cmd_render {
	/** @flags: Zero or more of ASAHI_RENDER_* */
	__u64 flags;

	/* @encoder_ptr: GPU base address to the hardware control stream */
	__u64 encoder_ptr;

	/* @usc_base: GPU base address for all USC binaries (shaders) used in
	 * this command. USC addresses are 32-bit relative to this 64-bit base.
	 */
	__u64 usc_base;

	/* @vertex_attachments: Pointer to drm_asahi_attachment array used for
	 * the vertex portion of this command.
	 */
	__u64 vertex_attachments;

	/* @fragment_attachments: Pointer to drm_asahi_attachment array used for
	 * the fragment portion of this command. This includes the end-of-tile
	 * shader, in addition to the fragment shaders themselves.
	 */
	__u64 fragment_attachments;

	/* @vertex_attachment_count: Number of drm_asahi_attachment's pointed to
	 * by vertex_attachments
	 */
	__u32 vertex_attachment_count;

	/* @fragment_attachment_count: Number of drm_asahi_attachment's pointed
	 * to by fragment_attachmenst
	 */
	__u32 fragment_attachment_count;

	/* @vertex_helper: Helper program used for the vertex shader */
	struct drm_asahi_helper_program vertex_helper;

	/* @fragment_helper: Helper program used for the fragment shader */
	struct drm_asahi_helper_program fragment_helper;

	/* @isp_scissor_base: ISP_SCISSOR_BASE register value. GPU address of an
	 * array of scissor descriptors indexed in the render pass.
	 */
	__u64 isp_scissor_base;

	/* @isp_dbias_base: ISP_DBIAS_BASE register value. GPU address of an
	 * array of depth bias values indexed in the render pass.
	 */
	__u64 isp_dbias_base;

	/* @isp_oclqry_base: ISP_OCLQRY_BASE register value. GPU addrss of an
	 * array of occlusion query results written by the render pass.
	 */
	__u64 isp_oclqry_base;

	/** @depth: Physical buffers backing the logical depth buffer */
	struct drm_asahi_zls_buffer depth;

	/** @stencil: Physical buffers backing the logical stencil buffer */
	struct drm_asahi_zls_buffer stencil;

	/** @zls_ctrl: ZLS_CTRL register value */
	__u64 zls_ctrl;

	/** @ppp_multisamplectl: PPP_MULTISAMPLECTL register value */
	__u64 ppp_multisamplectl;

	/** @sampler_heap: Base address of the sampler heap. This heap is used
	 * for both vertex shaders and fragment shaders. The registers are
	 * per-stage, but there is no known use case for separate heaps.
	 */
	__u64 sampler_heap;

	/** @ppp_ctrl: PPP_CTRL register value */
	__u32 ppp_ctrl;

	/** @width: Framebuffer width in pixels */
	__u16 width;

	/** @height: Framebuffer height in pixels */
	__u16 height;

	/** @layers: Number of layers in the framebuffer */
	__u16 layers;

	/** @sampler_count: Number of samplers in the sampler heap. */
	__u16 sampler_count;

	/** @utile_width: Width of a logical tilebuffer tile in pixels */
	__u8 utile_width;

	/** @utile_height: Height of a logical tilebuffer tile in pixels */
	__u8 utile_height;

	/* @samples: # of samples in the framebuffer. Must be 1, 2, or 4. */
	__u8 samples;

	/* @sample_size: # of bytes in the tilebuffer allocated per sample. */
	__u8 sample_size;

	/** @encoder_id: Opaque handle identifying what encoded this command. */
	__u32 encoder_id;

	/** @cmd_ta_id: Unique identifier for the Tiling Accelerator (TA)
	 * portion of this command.
	 */
	__u32 cmd_ta_id;

	/** @cmd_3d_id: Unique identifier for the 3D
	 * portion of this command.
	 */
	__u32 cmd_3d_id;

	/* @isp_merge_upper_x, @isp_merge_upper_y: 32-bit floats used in the
	 * hardware triangle merging. Calculate as:
	 *
	 *	isp_merge_upper_x = tan(60 deg) * width
	 *	isp_merge_upper_y = tan(60 deg) * height
	 *
	 * Making these values UAPI avoids requiring floating-point calculations
	 * in the kernel in the hot path.
	 */
	__u32 isp_merge_upper_x;
	__u32 isp_merge_upper_y;

	/* @bg: Background program ran at the start of each tile at the start of
	 * the render pass.
	 */
	struct drm_asahi_bg_eot bg;

	/* @eot: End-of-tile program ran at the end of each tile at the end of
	 * the render pass.
	 */
	struct drm_asahi_bg_eot eot;

	/* @partial_bg: Background program ran at the start of each tile when
	 * resuming the render pass during a partial render.
	 */
	struct drm_asahi_bg_eot partial_bg;

	/* @partial_eot: End-of-tile program ran at the end of each tile when
	 * pausing the render pass during a partial render.
	 */
	struct drm_asahi_bg_eot partial_eot;

	/* @isp_zls_pixels: ISP_ZLS_PIXELS register value. This contains the
	 * depth buffer width/height, which is allowed to differ from the
	 * framebuffer width/height.
	 */
	__u32 isp_zls_pixels;

	/* @isp_bgobjdepth: ISP_BGOBJDEPTH register value. This is the depth
	 * buffer clear value, encoded in the depth buffer's format: either a
	 * 32-bit float or a 16-bit unorm (with upper bits zeroed).
	 */
	__u32 isp_bgobjdepth;

	/* @isp_bgobjvals: ISP_BGOBJVALS register value. The bottom 8-bits
	 * contain the stencil buffer clear value.
	 */
	__u32 isp_bgobjvals;

	/* @ts_vtx: Timestamps for the vertex portion of the render */
	struct drm_asahi_timestamps ts_vtx;

	/* @ts_frag: Timestamps for the fragment portion of the render */
	struct drm_asahi_timestamps ts_frag;
};

/** Compute command submission data */
struct drm_asahi_cmd_compute {
	/** @flags: MBZ */
	__u64 flags;

	/* @encoder_ptr: GPU base address to the hardware control stream */
	__u64 encoder_ptr;

	/* @encoder_end: GPU base address to the end of the hardware control
	 * stream. Note this only considers the first contiguous segment of the
	 * control stream, as the stream might jump elsewhere.
	 */
	__u64 encoder_end;

	/* @usc_base: GPU Base address for all USC binaries (shaders) used in
	 * this command. USC addresses are 32-bit relative to this 64-bit base.
	 */
	__u64 usc_base;

	/* @attachments: Pointer to drm_asahi_attachment array used for
	 * this command
	 */
	__u64 attachments;

	/** @sampler_heap: Base address of the sampler heap. This heap is used
	 * for both vertex shaders and fragment shaders. The registers are
	 * per-stage, but there is no known use case for separate heaps.
	 */
	__u64 sampler_heap;

	/* @attachment_count: Number of drm_asahi_attachments pointed to
	 * by attachments
	 */
	__u32 attachment_count;

	/** @sampler_count: Number of samplers in the sampler heap. */
	__u32 sampler_count;

	/* @helper: Helper program used for this compute shader */
	struct drm_asahi_helper_program helper;

	/** @encoder_id: Opaque handle identifying what encoded this command. */
	__u32 encoder_id;

	/** @cmd_id: Unique identifier for this command. */
	__u32 cmd_id;

	/* @ts: Timestamps for the compute command */
	struct drm_asahi_timestamps ts;
};

/** Fetch the current GPU timestamp time */
struct drm_asahi_get_time {
	/** @flags: MBZ. */
	__u64 flags;

	/** @gpu_timestamp: On return, the current GPU timestamp */
	__u64 gpu_timestamp;
};

/* Note: this is an enum so that it can be resolved by Rust bindgen. */
enum {
	DRM_IOCTL_ASAHI_GET_PARAMS       = DRM_IOWR(DRM_COMMAND_BASE + DRM_ASAHI_GET_PARAMS, struct drm_asahi_get_params),
	DRM_IOCTL_ASAHI_VM_CREATE        = DRM_IOWR(DRM_COMMAND_BASE + DRM_ASAHI_VM_CREATE, struct drm_asahi_vm_create),
	DRM_IOCTL_ASAHI_VM_DESTROY       = DRM_IOW(DRM_COMMAND_BASE + DRM_ASAHI_VM_DESTROY, struct drm_asahi_vm_destroy),
	DRM_IOCTL_ASAHI_GEM_CREATE       = DRM_IOWR(DRM_COMMAND_BASE + DRM_ASAHI_GEM_CREATE, struct drm_asahi_gem_create),
	DRM_IOCTL_ASAHI_GEM_MMAP_OFFSET  = DRM_IOWR(DRM_COMMAND_BASE + DRM_ASAHI_GEM_MMAP_OFFSET, struct drm_asahi_gem_mmap_offset),
	DRM_IOCTL_ASAHI_GEM_BIND         = DRM_IOW(DRM_COMMAND_BASE + DRM_ASAHI_GEM_BIND, struct drm_asahi_gem_bind),
	DRM_IOCTL_ASAHI_QUEUE_CREATE     = DRM_IOWR(DRM_COMMAND_BASE + DRM_ASAHI_QUEUE_CREATE, struct drm_asahi_queue_create),
	DRM_IOCTL_ASAHI_QUEUE_DESTROY    = DRM_IOW(DRM_COMMAND_BASE + DRM_ASAHI_QUEUE_DESTROY, struct drm_asahi_queue_destroy),
	DRM_IOCTL_ASAHI_SUBMIT           = DRM_IOW(DRM_COMMAND_BASE + DRM_ASAHI_SUBMIT, struct drm_asahi_submit),
	DRM_IOCTL_ASAHI_GET_TIME         = DRM_IOWR(DRM_COMMAND_BASE + DRM_ASAHI_GET_TIME, struct drm_asahi_get_time),
	DRM_IOCTL_ASAHI_GEM_BIND_OBJECT  = DRM_IOWR(DRM_COMMAND_BASE + DRM_ASAHI_GEM_BIND_OBJECT, struct drm_asahi_gem_bind_object),
};

#if defined(__cplusplus)
}
#endif

#endif /* _ASAHI_DRM_H_ */
