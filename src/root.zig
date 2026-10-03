pub const pjrt = @import("pjrt/client.zig");

test {
    // `zig test` only collects the tests of the module's root file: without this
    // reference, the tests in `pjrt/client.zig` are never analysed.
    _ = @import("pjrt/client.zig");
}
