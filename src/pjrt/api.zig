const std = @import("std");

const c = @import("pjrt_c");

const GetPjrtApiFn = *const fn () callconv(.c) *const c.PJRT_Api;

/// Lowest API version the vendored header can describe. Must stay in sync with
/// the `#define PJRT_API_MINOR` of `pjrt_c_api.h` (see `third_party/pjrt/PINNED`).
const min_api_minor = 81;

pub const Error = error{
    /// The .so does not export `GetPjrtApi`: it is not a PJRT plugin.
    MissingGetPjrtApi,
    /// The plugin speaks an API version our header does not cover.
    UnsupportedApiVersion,
    PluginInitializeFailed,
    ClientCreateFailed,
    NoAddressableDevice,
    CompileFailed,
    ExecuteFailed,
};

comptime {
    if (c.PJRT_API_MINOR != min_api_minor) @compileError(
        "PJRT header and src/pjrt/client.zig disagree on PJRT_API_MINOR",
    );
}

/// Prints the PJRT error message to stderr and frees it.
///
/// PJRT signals a failure with a non-null `PJRT_Error*` returned from the call:
/// it has to be printed before being destroyed, otherwise the diagnostic is lost.
fn report(api: *const c.PJRT_Api, err: *c.PJRT_Error, comptime what: []const u8) void {
    var msg: c.PJRT_Error_Message_Args = .{
        .struct_size = @sizeOf(c.PJRT_Error_Message_Args),
        .extension_start = null,
        .@"error" = err,
        .message = null,
        .message_size = 0,
    };
    api.PJRT_Error_Message.?(&msg);
    std.debug.print("PJRT {s}: {s}\n", .{
        what,
        if (msg.message) |m| m[0..msg.message_size] else "<message absent>",
    });
    var destroy_args: c.PJRT_Error_Destroy_Args = .{
        .struct_size = @sizeOf(c.PJRT_Error_Destroy_Args),
        .extension_start = null,
        .@"error" = err,
    };
    _ = api.PJRT_Error_Destroy.?(&destroy_args);
}

/// Propagates a `PJRT_Error*` as a Zig error.
///
/// `target` is the `Error` to raise: `report` has already printed the PJRT
/// message, all that is left is to translate it for the caller.
fn check(
    api: *const c.PJRT_Api,
    err: ?*c.PJRT_Error,
    comptime what: []const u8,
    comptime target: Error,
) Error!void {
    const e = err orelse return;
    report(api, e, what);
    return target;
}

fn destroyBuffer(api: *const c.PJRT_Api, buffer: ?*c.PJRT_Buffer) void {
    const b = buffer orelse return;
    var destroy_args: c.PJRT_Buffer_Destroy_Args = .{
        .struct_size = @sizeOf(c.PJRT_Buffer_Destroy_Args),
        .extension_start = null,
        .buffer = b,
    };
    _ = api.PJRT_Buffer_Destroy.?(&destroy_args);
}

fn destroyEvent(api: *const c.PJRT_Api, event: ?*c.PJRT_Event) void {
    const e = event orelse return;
    var destroy_args: c.PJRT_Event_Destroy_Args = .{
        .struct_size = @sizeOf(c.PJRT_Event_Destroy_Args),
        .extension_start = null,
        .event = e,
    };
    _ = api.PJRT_Event_Destroy.?(&destroy_args);
}

/// Blocks until `event` is ready, and propagates its error if any.
fn awaitEvent(api: *const c.PJRT_Api, event: ?*c.PJRT_Event, comptime what: []const u8) !void {
    const e = event orelse return;
    var args: c.PJRT_Event_Await_Args = .{
        .struct_size = @sizeOf(c.PJRT_Event_Await_Args),
        .extension_start = null,
        .event = e,
    };
    try check(api, api.PJRT_Event_Await.?(&args), what, Error.ExecuteFailed);
}

/// The loaded plugin: `dlopen` handle plus `PJRT_Api` table.
pub const Api = struct {
    allocator: std.mem.Allocator,
    /// Never closed: see `deinit`.
    lib: std.DynLib,
    api: *const c.PJRT_Api,
    path: []u8,

    pub fn open(allocator: std.mem.Allocator, path: []const u8) !Api {
        // No `errdefer lib.close()`: the plugin registers an `atexit` handler that
        // `dlclose` does not unregister, so the process dies on a dangling pointer
        // at exit — including after a failure, which would make the diagnostic
        // unreadable. See `deinit`.
        var lib = try std.DynLib.open(path);

        const get_api = lib.lookup(GetPjrtApiFn, "GetPjrtApi") orelse
            return error.MissingGetPjrtApi;
        const api = get_api();

        // A plugin newer than the header appends fields at the end of the structs:
        // we never read them, which is the whole point of PJRT's forward
        // compatibility. An older plugin, on the other hand, hands back a shorter
        // `PJRT_Api` table in which every slot added since is garbage, so we
        // refuse to run at all.
        const v = api.pjrt_api_version;
        if (v.major_version != c.PJRT_API_MAJOR or v.minor_version < min_api_minor) {
            std.debug.print(
                "PJRT: plugin API {d}.{d}, header requires at least {d}.{d}\n",
                .{ v.major_version, v.minor_version, c.PJRT_API_MAJOR, min_api_minor },
            );
            return error.UnsupportedApiVersion;
        }

        var init: c.PJRT_Plugin_Initialize_Args = .{
            .struct_size = @sizeOf(c.PJRT_Plugin_Initialize_Args),
            .extension_start = null,
        };
        try check(
            api,
            api.PJRT_Plugin_Initialize.?(&init),
            "PJRT_Plugin_Initialize",
            Error.PluginInitializeFailed,
        );

        return .{
            .allocator = allocator,
            .lib = lib,
            .api = api,
            .path = try allocator.dupe(u8, path),
        };
    }

    /// Does not close `lib`: `dlclose` unloads the plugin while an `atexit` handler
    /// it registered is still in libc's list, so the process jumps into unloaded
    /// code on exit and dies of SIGSEGV — after the test, turning a success into a
    /// build failure. The handle is deliberately leaked: a process loads the plugin
    /// only once, and `dlopen` on an already-loaded path is a no-op.
    pub fn deinit(self: *Api) void {
        self.allocator.free(self.path);
        self.* = undefined;
    }
};
