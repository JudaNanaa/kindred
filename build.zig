const std = @import("std");

const PluginSpec = struct { dep: []const u8, file: []const u8 };

fn pluginFor(t: std.Target) ?PluginSpec {
    return switch (t.os.tag) {
        .linux => switch (t.cpu.arch) {
            .x86_64 => .{ .dep = "pjrt_cpu_linux_x86_64", .file = "libpjrt_cpu.so" },
            .aarch64 => .{ .dep = "pjrt_cpu_linux_aarch64", .file = "libpjrt_cpu.so" },
            else => null,
        },
        else => null,
    };
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Vendored PJRT header -> Zig module.
    const pjrt_c = b.addTranslateC(.{
        .root_source_file = b.path("third_party/pjrt/include/pjrt_c_api.h"),
        .target = target,
        .optimize = optimize,
    });

    // Default plugin: an explicit path wins over the bundled one.
    const user_plugin = b.option([]const u8, "pjrt_plugin", "Path to the PJRT plugin (overrides the default)");
    const bundle = b.option(bool, "bundle_pjrt", "Download the PJRT CPU plugin (default: true)") orelse true;

    var default_plugin: ?[]const u8 = null;
    // When cross-compiling, the resolved path would be the build machine's cache
    // path, which is meaningless on the target, so only bake it in for a native
    // target.
    if (bundle and target.query.isNative()) {
        if (pluginFor(target.result)) |spec| {
            // Null on the first pass: Zig downloads, then the build is re-run.
            if (b.lazyDependency(spec.dep, .{})) |dep| {
                default_plugin = dep.builder.pathFromRoot(spec.file);
            }
        }
    }

    const opts = b.addOptions();
    opts.addOption(?[]const u8, "pjrt_plugin", user_plugin orelse default_plugin);

    // Zig module, for Zig consumers.
    const mod = b.addModule("kindred", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true, // required for dlopen on Linux/macOS
        .imports = &.{
            .{ .name = "pjrt_c", .module = pjrt_c.createModule() },
        },
    });
    mod.addOptions("config", opts);

    // Tests
    const tests = b.addTest(.{ .root_module = mod });
    const test_step = b.step("test", "Run the tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);
}
