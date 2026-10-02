const std = @import("std");

pub const c = @cImport({
    @cInclude("pjrt_c_api.h");
});
