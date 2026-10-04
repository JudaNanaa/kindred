const std = @import("std");
const builtin = @import("builtin");

const c = @import("pjrt_c");

const GetPjrtApiFn = *const fn () callconv(.c) *const c.PJRT_Api;

const log = std.log.scoped(.@"pjrt/api");

/// Lowest API version the vendored header can describe. Must stay in sync with
/// the `#define PJRT_API_MINOR` of `pjrt_c_api.h` (see `third_party/pjrt/PINNED`).
const min_api_minor = 81;

pub const meta = struct {
    // We could calculate it like PJRT does, but it turns out that some of those
    // were wrong in PJRT itself [1], which gets propagated to binary plugins. In
    // order to mirror that, we just the value as computed by PJRT itself, through
    // comptime reflection. We could make the argument to remove that one day since
    // [1] has been fixed. The problem is that this problem could happen again in
    // as the way PJRT does it is not very robust.
    //
    // 1. https://github.com/openxla/xla/issues/10032
    pub fn structSize(comptime T: type) usize {
        // unsafe on purpose, we want this to fail if that ever changes
        const typedef_name = comptime blk: {
            const needle = ".struct_";
            const idx = std.mem.indexOf(u8, @typeName(T), needle).?;
            break :blk @typeName(T)[idx + needle.len ..];
        };
        return @field(c, typedef_name ++ "_STRUCT_SIZE");
    }

    pub fn Struct(comptime T: type) type {
        const fields = std.meta.fields(T);
        var names: [fields.len][]const u8 = undefined;
        var types: [fields.len]type = undefined;
        var attributes: [fields.len]std.builtin.Type.StructField.Attributes = undefined;
        for (fields, &names, &types, &attributes) |field, *name, *type_, *attr| {
            name.* = field.name;
            type_.* = field.type;
            attr.* = .{
                .default_value_ptr = @ptrCast(if (std.mem.eql(u8, field.name, "struct_size"))
                    &structSize(T)
                else
                    &std.mem.zeroes(field.type)),
            };
        }
        return @Struct(
            .@"extern",
            null,
            &names,
            &types,
            &attributes,
        );
    }
};

inline fn interpretPjrtError(api: *const Api, pjrt_error: *Error, context: []const u8) ApiError {
    defer pjrt_error.deinit(api);
    const err_code = pjrt_error.getCode(api).toApiError();
    log.warn("[{s}] {t}: {s}", .{ context, err_code, pjrt_error.getMessage(api) });
    return err_code;
}

fn InnerMixin(comptime innerT: type) type {
    return struct {
        fn inner(self: anytype) *innerT {
            return @ptrCast(@alignCast(@constCast(self)));
        }
    };
}

comptime {
    if (c.PJRT_API_MINOR != min_api_minor) @compileError(
        "PJRT header and src/pjrt/client.zig disagree on PJRT_API_MINOR",
    );
}

pub const ApiError = error{
    Cancelled,
    Unknown,
    InvalidArgument,
    DeadlineExceeded,
    NotFound,
    AlreadyExists,
    PermissionDenied,
    ResourceExhausted,
    FailedPrecondition,
    Aborted,
    OutOfRange,
    Unimplemented,
    Internal,
    Unavailable,
    DataLoss,
    Unauthenticated,
};

// The loaded plugin: `dlopen` handle plus `PJRT_Api` table.

pub const Api = struct {
    pub const Version = struct {
        major: i64,
        minor: i64,

        pub fn format(self: Version, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            try writer.print("{d}.{d}", .{ self.major, self.minor });
        }
    };

    const Funcs = std.meta.FieldEnum(c.PJRT_Api);

    inner: c.PJRT_Api,

    pub fn loadFrom(library: [:0]const u8) !*const Api {
        const basename = std.Io.Dir.path.basename(library);
        log.info("Loading: {s}...", .{basename});

        var lib: std.DynLib = switch (builtin.os.tag) {
            .linux, .macos => blk: {
                const rtld: std.c.RTLD = switch (builtin.os.tag) {
                    // We use RTLD_GLOBAL so that symbols from NEEDED libraries are available in the global namespace.
                    .linux => .{ .LAZY = true, .GLOBAL = true, .NODELETE = true },
                    .macos => .{ .LAZY = true, .LOCAL = true },
                    else => unreachable,
                };
                break :blk .{
                    .inner = .{
                        .handle = std.c.dlopen(library, rtld) orelse {
                            log.err("Unable to dlopen plugin {s}\n{s}", .{ library, std.c.dlerror() orelse "" });
                            return error.FileNotFound;
                        },
                    },
                };
            },
            else => std.DynLib.open(library) catch |err| {
                log.err("Unable to dlopen plugin {s}: {}", .{ library, err });
                return err;
            },
        };

        const api = fromDynLib(&lib) catch |err| {
            log.err("Unable to load PJRT API from plugin {s}: {}", .{ library, err });
            return err;
        };
        log.info("Loaded: {s}", .{basename});
        return api;
    }

    pub fn fromDynLib(lib: *std.DynLib) !*const Api {
        const DynGetPjrtApi = lib.lookup(*const fn () callconv(.c) *const Api, "GetPjrtApi") orelse {
            return error.MissingGetPjrtApi;
        };

        const api = DynGetPjrtApi();
        _ = try api.call(.PJRT_Plugin_Initialize, .{});

        return api;
    }

    fn PJRTFnArg(comptime func: Funcs) type {
        const fti = @typeInfo(@FieldType(c.PJRT_Api, @tagName(func)));
        const fn_ptr = @typeInfo(fti.optional.child);
        const fn_type_info = @typeInfo(fn_ptr.pointer.child);
        const arg_array_type_info = @typeInfo(fn_type_info.@"fn".params[0].type.?);
        return arg_array_type_info.pointer.child;
    }

    fn PJRTFnArgWithDefault(comptime func: Funcs) type {
        const argT = PJRTFnArg(func);
        return switch (@typeInfo(argT)) {
            .@"struct" => meta.Struct(argT),
            else => argT,
        };
    }

    inline fn innerCall(self: *const Api, comptime method: Funcs, arg: *PJRTFnArg(method)) ApiError!void {
        if (@offsetOf(c.PJRT_Api, @tagName(method)) > self.inner.struct_size) {
            std.debug.panic("PJRT Api method {s} not available in this plugin", .{@tagName(method)});
        }
        const fn_ptr = @field(&self.inner, @tagName(method)).?;
        const result = fn_ptr(arg);
        if (@TypeOf(result) == void) {
            return;
        }
        if (result) |pjrt_c_error| {
            const pjrt_error: *Error = @ptrCast(pjrt_c_error);
            return interpretPjrtError(self, pjrt_error, @tagName(method));
        }
    }

    inline fn call(self: *const Api, comptime method: Funcs, arg: PJRTFnArgWithDefault(method)) ApiError!PJRTFnArgWithDefault(method) {
        var ret = arg;
        try innerCall(self, method, @ptrCast(&ret));
        return ret;
    }
};

pub const ErrorCode = enum(c.PJRT_Error_Code) {
    cancelled = c.PJRT_Error_Code_CANCELLED,
    unknown = c.PJRT_Error_Code_UNKNOWN,
    invalid_argument = c.PJRT_Error_Code_INVALID_ARGUMENT,
    deadline_exceeded = c.PJRT_Error_Code_DEADLINE_EXCEEDED,
    not_found = c.PJRT_Error_Code_NOT_FOUND,
    already_exists = c.PJRT_Error_Code_ALREADY_EXISTS,
    permission_denied = c.PJRT_Error_Code_PERMISSION_DENIED,
    resource_exhausted = c.PJRT_Error_Code_RESOURCE_EXHAUSTED,
    failed_precondition = c.PJRT_Error_Code_FAILED_PRECONDITION,
    aborted = c.PJRT_Error_Code_ABORTED,
    out_of_range = c.PJRT_Error_Code_OUT_OF_RANGE,
    unimplemented = c.PJRT_Error_Code_UNIMPLEMENTED,
    internal = c.PJRT_Error_Code_INTERNAL,
    unavailable = c.PJRT_Error_Code_UNAVAILABLE,
    data_loss = c.PJRT_Error_Code_DATA_LOSS,
    unauthenticated = c.PJRT_Error_Code_UNAUTHENTICATED,

    pub fn toApiError(code: ErrorCode) ApiError {
        return switch (code) {
            .cancelled => ApiError.Cancelled,
            .unknown => ApiError.Unknown,
            .invalid_argument => ApiError.InvalidArgument,
            .deadline_exceeded => ApiError.DeadlineExceeded,
            .not_found => ApiError.NotFound,
            .already_exists => ApiError.AlreadyExists,
            .permission_denied => ApiError.PermissionDenied,
            .resource_exhausted => ApiError.ResourceExhausted,
            .failed_precondition => ApiError.FailedPrecondition,
            .aborted => ApiError.Aborted,
            .out_of_range => ApiError.OutOfRange,
            .unimplemented => ApiError.Unimplemented,
            .internal => ApiError.Internal,
            .unavailable => ApiError.Unavailable,
            .data_loss => ApiError.DataLoss,
            .unauthenticated => ApiError.Unauthenticated,
        };
    }
};

pub const Error = opaque {
    const inner = InnerMixin(c.PJRT_Error).inner;

    pub fn deinit(self: *Error, api: *const Api) void {
        _ = api.call(.PJRT_Error_Destroy, .{
            .@"error" = self.inner(),
        }) catch unreachable;
    }

    pub fn getCode(self: *Error, api: *const Api) ErrorCode {
        const ret = api.call(.PJRT_Error_GetCode, .{
            .@"error" = self.inner(),
        }) catch unreachable;
        return @enumFromInt(ret.code);
    }

    pub fn getMessage(self: *Error, api: *const Api) []const u8 {
        const ret = api.call(.PJRT_Error_Message, .{
            .@"error" = self.inner(),
        }) catch unreachable;
        return ret.message[0..ret.message_size];
    }
};
