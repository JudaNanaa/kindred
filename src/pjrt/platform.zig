//! Minimal PJRT runtime: `dlopen` a plugin, create a client, compile a StableHLO
//! module, execute it, read the results back. See `docs/design.md` §6 and
//! Milestone 0.
//!
//! The plugin is never linked at compile time: it is opened at runtime and
//! resolved through `GetPjrtApi` plus the `PJRT_Api` table.

const std = @import("std");

/// PJRT table translated from the vendored header by `b.addTranslateC` in build.zig.
const c = @import("pjrt_c");

const Api = @import("api.zig").Api;

/// A host f32 tensor. `data` is not copied: the caller must keep it alive until
/// `execute` returns.
pub const Tensor = struct {
    data: []const f32,
    dims: []const i64,
};

pub const Executable = struct {
    api: *const c.PJRT_Api,
    loaded: *c.PJRT_LoadedExecutable,
    /// Owns the underlying `PJRT_Executable`, destroyed after `loaded`.
    inner: *c.PJRT_Executable,
    num_outputs: usize,

    pub fn destroy(self: *Executable) void {
        var loaded_args: c.PJRT_LoadedExecutable_Destroy_Args = .{
            .struct_size = @sizeOf(c.PJRT_LoadedExecutable_Destroy_Args),
            .extension_start = null,
            .executable = self.loaded,
        };
        _ = self.api.PJRT_LoadedExecutable_Destroy.?(&loaded_args);

        var inner_args: c.PJRT_Executable_Destroy_Args = .{
            .struct_size = @sizeOf(c.PJRT_Executable_Destroy_Args),
            .extension_start = null,
            .executable = self.inner,
        };
        _ = self.api.PJRT_Executable_Destroy.?(&inner_args);

        self.* = undefined;
    }
};

pub const Client = struct {
    allocator: std.mem.Allocator,
    api: Api,
    client: *c.PJRT_Client,
    /// v1 is single-process: the first addressable device is enough.
    device: *c.PJRT_Device,

    pub fn init(allocator: std.mem.Allocator, plugin_path: []const u8) !Client {
        var apio = try Api.open(allocator, plugin_path);
        errdefer apio.deinit();
        const api = apio.api;

        var create: c.PJRT_Client_Create_Args = .{
            .struct_size = @sizeOf(c.PJRT_Client_Create_Args),
            .extension_start = null,
            .create_options = null,
            .num_options = 0,
            .kv_get_callback = null,
            .kv_get_user_arg = null,
            .kv_put_callback = null,
            .kv_put_user_arg = null,
            .client = null,
            .kv_try_get_callback = null,
            .kv_try_get_user_arg = null,
        };
        try check(api, api.PJRT_Client_Create.?(&create), "PJRT_Client_Create", Error.ClientCreateFailed);
        const client = create.client orelse return Error.ClientCreateFailed;
        var destroy_args: c.PJRT_Client_Destroy_Args = .{
            .struct_size = @sizeOf(c.PJRT_Client_Destroy_Args),
            .extension_start = null,
            .client = client,
        };
        errdefer _ = api.PJRT_Client_Destroy.?(&destroy_args);

        var devices: c.PJRT_Client_AddressableDevices_Args = .{
            .struct_size = @sizeOf(c.PJRT_Client_AddressableDevices_Args),
            .extension_start = null,
            .client = client,
            .addressable_devices = null,
            .num_addressable_devices = 0,
        };
        try check(
            api,
            api.PJRT_Client_AddressableDevices.?(&devices),
            "PJRT_Client_AddressableDevices",
            Error.NoAddressableDevice,
        );
        const list = devices.addressable_devices orelse return Error.NoAddressableDevice;
        if (list[0] == null) return Error.NoAddressableDevice;

        return .{
            .allocator = allocator,
            .api = plugin,
            .client = client,
            .device = list[0].?,
        };
    }

    pub fn deinit(self: *Client) void {
        var args: c.PJRT_Client_Destroy_Args = .{
            .struct_size = @sizeOf(c.PJRT_Client_Destroy_Args),
            .extension_start = null,
            .client = self.client,
        };
        _ = self.plugin.api.PJRT_Client_Destroy.?(&args);
        self.plugin.deinit();
        self.* = undefined;
    }

    /// Compiles a StableHLO module written as MLIR text.
    pub fn compile(self: *Client, mlir: []const u8) !Executable {
        const api = self.plugin.api;

        var program: c.PJRT_Program = .{
            .struct_size = @sizeOf(c.PJRT_Program),
            .extension_start = null,
            .code = @constCast(mlir.ptr),
            .code_size = mlir.len,
            .format = "mlir",
            .format_size = 4,
        };
        var args: c.PJRT_Client_Compile_Args = .{
            .struct_size = @sizeOf(c.PJRT_Client_Compile_Args),
            .extension_start = null,
            .client = self.client,
            .program = &program,
            // See `compile_options_one_replica`: an empty proto means `num_replicas`
            // 0, which dies in XLA with `Check failed: replica_count > 0`.
            .compile_options = compile_options_one_replica,
            .compile_options_size = compile_options_one_replica.len,
            .executable = null,
        };
        try check(api, api.PJRT_Client_Compile.?(&args), "PJRT_Client_Compile", Error.CompileFailed);
        const loaded = args.executable orelse return Error.CompileFailed;
        var destroy_args: c.PJRT_LoadedExecutable_Destroy_Args = .{
            .struct_size = @sizeOf(c.PJRT_LoadedExecutable_Destroy_Args),
            .extension_start = null,
            .executable = loaded,
        };
        errdefer _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);

        // `PJRT_Client_Compile` only hands back a LoadedExecutable; the output count
        // is read off the Executable it contains.
        var get: c.PJRT_LoadedExecutable_GetExecutable_Args = .{
            .struct_size = @sizeOf(c.PJRT_LoadedExecutable_GetExecutable_Args),
            .extension_start = null,
            .loaded_executable = loaded,
            .executable = null,
        };
        try check(
            api,
            api.PJRT_LoadedExecutable_GetExecutable.?(&get),
            "PJRT_LoadedExecutable_GetExecutable",
            Error.CompileFailed,
        );
        const inner = get.executable orelse return Error.CompileFailed;
        var inner_args: c.PJRT_Executable_Destroy_Args = .{
            .struct_size = @sizeOf(c.PJRT_Executable_Destroy_Args),
            .extension_start = null,
            .executable = inner,
        };
        errdefer _ = api.PJRT_Executable_Destroy.?(&inner_args);

        var n_out: c.PJRT_Executable_NumOutputs_Args = .{
            .struct_size = @sizeOf(c.PJRT_Executable_NumOutputs_Args),
            .extension_start = null,
            .executable = inner,
            .num_outputs = 0,
        };
        try check(api, api.PJRT_Executable_NumOutputs.?(&n_out), "PJRT_Executable_NumOutputs", Error.CompileFailed);

        return .{
            .api = api,
            .loaded = loaded,
            .inner = inner,
            .num_outputs = n_out.num_outputs,
        };
    }

    /// Executes `exe` and copies every output buffer back to the host.
    ///
    /// The returned slices belong to the client's allocator: release them with
    /// `freeTensors`.
    pub fn execute(self: *Client, exe: *const Executable, inputs: []const Tensor) ![][]f32 {
        const api = self.plugin.api;
        const n_out = exe.num_outputs;

        // H2D
        const in_buffers = try self.allocator.alloc(?*c.PJRT_Buffer, inputs.len);
        defer self.allocator.free(in_buffers);
        const in_events = try self.allocator.alloc(?*c.PJRT_Event, inputs.len);
        defer self.allocator.free(in_events);

        for (inputs, 0..) |t, i| {
            in_buffers[i] = null;
            in_events[i] = null;
            var args: c.PJRT_Client_BufferFromHostBuffer_Args = .{
                .struct_size = @sizeOf(c.PJRT_Client_BufferFromHostBuffer_Args),
                .extension_start = null,
                .client = self.client,
                .data = t.data.ptr,
                .type = @intCast(c.PJRT_Buffer_Type_F32),
                .dims = t.dims.ptr,
                .num_dims = t.dims.len,
                .byte_strides = null,
                .num_byte_strides = 0,
                // The header only knows `...OnlyDuringCall`, `...UntilTransferCompletes`
                // and the zero-copy variants. `execute` awaits the H2D event before
                // returning, so the input slices may live that long.
                .host_buffer_semantics = @intCast(c.PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes),
                .device = self.device,
                .memory = null,
                .device_layout = null,
                .done_with_host_buffer = null,
                .buffer = null,
            };
            try check(
                api,
                api.PJRT_Client_BufferFromHostBuffer.?(&args),
                "PJRT_Client_BufferFromHostBuffer",
                Error.ExecuteFailed,
            );
            in_buffers[i] = args.buffer;
            in_events[i] = args.done_with_host_buffer;
        }
        // If we fail midway, do not leave the buffers behind.
        defer for (in_buffers) |b| destroyBuffer(api, b);

        const out_cells = try self.allocator.alloc(?*c.PJRT_Buffer, n_out);
        defer self.allocator.free(out_cells);
        @memset(out_cells, null);
        defer for (out_cells) |b| destroyBuffer(api, b);

        const done_events = try self.allocator.alloc(?*c.PJRT_Event, 1);
        defer self.allocator.free(done_events);
        done_events[0] = null;
        defer for (done_events) |e| destroyEvent(api, e);

        // `PJRT_LoadedExecutable_Execute` reads `options->struct_size` without
        // checking `options`: passing null dereferences 0 and kills the process.
        // All-zero is enough, and means "default options".
        var exec_options: c.PJRT_ExecuteOptions = .{
            .struct_size = @sizeOf(c.PJRT_ExecuteOptions),
            .extension_start = null,
        };

        var args: c.PJRT_LoadedExecutable_Execute_Args = .{
            .struct_size = @sizeOf(c.PJRT_LoadedExecutable_Execute_Args),
            .extension_start = null,
            .executable = exe.loaded,
            .options = &exec_options,
            .argument_lists = &[_][*]?*c.PJRT_Buffer{in_buffers.ptr},
            .num_devices = 1,
            .num_args = inputs.len,
            // One device, therefore one row of outputs.
            .output_lists = &[_][*c]?*c.PJRT_Buffer{out_cells.ptr},
            .device_complete_events = done_events.ptr,
            .execute_device = null,
        };
        try check(
            api,
            api.PJRT_LoadedExecutable_Execute.?(&args),
            "PJRT_LoadedExecutable_Execute",
            Error.ExecuteFailed,
        );

        // Until these two are ready, `inputs` has to stay alive (the H2D is still
        // reading it) and the outputs have not been computed yet.
        for (in_events) |e| try awaitEvent(api, e, "PJRT_Event_Await (input)");
        try awaitEvent(api, done_events[0], "PJRT_Event_Await (output)");

        const results = try self.allocator.alloc([]f32, n_out);
        errdefer self.allocator.free(results);
        for (results, out_cells) |*r, buf| {
            r.* = try self.toHost(buf orelse return Error.ExecuteFailed);
        }
        return results;
    }

    /// D2H. The size is queried first with `dst = nullptr`.
    fn toHost(self: *Client, buffer: *c.PJRT_Buffer) ![]f32 {
        const api = self.plugin.api;
        var args: c.PJRT_Buffer_ToHostBuffer_Args = .{
            .struct_size = @sizeOf(c.PJRT_Buffer_ToHostBuffer_Args),
            .extension_start = null,
            .src = buffer,
            .host_layout = null,
            .dst = null,
            .dst_size = 0,
            .event = null,
        };
        try check(api, api.PJRT_Buffer_ToHostBuffer.?(&args), "PJRT_Buffer_ToHostBuffer", Error.ExecuteFailed);
        const event = args.event;
        defer destroyEvent(api, event);

        const count = args.dst_size / @sizeOf(f32);
        // `alignedAlloc` demands a comptime alignment; the XLA CPU plugin is happy
        // with f32's natural alignment.
        const out = try self.allocator.alloc(f32, count);
        errdefer self.allocator.free(out);

        args.dst = @ptrCast(out.ptr);
        args.dst_size = count * @sizeOf(f32);
        args.event = null;
        try check(api, api.PJRT_Buffer_ToHostBuffer.?(&args), "PJRT_Buffer_ToHostBuffer", Error.ExecuteFailed);
        try awaitEvent(api, args.event, "PJRT_Event_Await (D2H)");

        return out;
    }

    /// Frees the slices returned by `execute`.
    pub fn freeTensors(self: *Client, results: [][]f32) void {
        for (results) |r| self.allocator.free(r);
        self.allocator.free(results);
    }
};

/// Minimal `CompileOptionsProto` the CPU plugin accepts, serialized the way XLA
/// does it itself: `executable_build_options { num_replicas: 1 num_partitions: 1 }`.
///
/// The format is the **binary** protobuf, not the text format — on the XLA side it
/// is a `ParseFromArray`, hence `failed to deserialize CompileOptionsProto` if you
/// pass text. The six bytes read as:
///
///     1a 04   CompileOptionsProto field 3 (`executable_build_options`), length 4
///     20 01   ExecutableBuildOptionsProto field 4 (`num_replicas`), varint 1
///     28 01   ExecutableBuildOptionsProto field 5 (`num_partitions`), varint 1
///
/// v1 is single-process, hence one replica and one partition: both counters must
/// be non-zero, `DeviceAssignment` refuses 0 for either.
const compile_options_one_replica = "\x1a\x04\x20\x01\x28\x01";

/// Test program: `main(%a, %b) = a + b` on two `tensor<2x2xf32>`.
const smoke_mlir =
    \\module @kindred_smoke {
    \\  func.func @main(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>) -> tensor<2x2xf32> {
    \\    %0 = stablehlo.add %a, %b : tensor<2x2xf32>
    \\    return %0 : tensor<2x2xf32>
    \\  }
    \\}
;

test "Milestone 0: compile and execute a stablehlo.add f32[2,2]" {
    const config = @import("config");

    // The plugin is only bundled for a native target; elsewhere there is nothing
    // to load, so this test has nothing to check.
    const plugin_path = config.pjrt_plugin orelse return error.SkipZigTest;

    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 10, 20, 30, 40 };

    var client = try Client.init(std.testing.allocator, plugin_path);
    defer client.deinit();

    var exe = try client.compile(smoke_mlir);
    defer exe.destroy();

    const out = try client.execute(&exe, &.{
        .{ .data = &a, .dims = &.{ 2, 2 } },
        .{ .data = &b, .dims = &.{ 2, 2 } },
    });
    defer client.freeTensors(out);

    try std.testing.expectEqual(@as(usize, 1), out.len);
    // Sum of two floats: the accuracy here is bit for bit.
    try std.testing.expectEqualSlices(f32, &.{ 11, 22, 33, 44 }, out[0]);
}
