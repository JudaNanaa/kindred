const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Header PJRT -> module Zig (translate-c fait par le build system)
    const pjrt_c = b.addTranslateC(.{
        .root_source_file = b.path("third_party/pjrt/include/pjrt_c_api.h"),
        .target = target,
        .optimize = optimize,
    });
    // si le header en inclut d'autres : pjrt_c.addIncludePath(b.path("vendor/pjrt"));

    const mod = b.addModule("kindred", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true, // nécessaire pour dlopen
        .imports = &.{
            .{ .name = "pjrt_c", .module = pjrt_c.createModule() },
        },
    });

    // Artefact pour les consommateurs C
    const lib = b.addLibrary(.{
        .name = "kindred",
        .linkage = .static, // ou .dynamic
        .root_module = mod,
    });
    lib.installHeader(b.path("include/kindred.h"), "kindred.h");
    b.installArtifact(lib);

    // Tests
    const tests = b.addTest(.{ .root_module = mod });
    const test_step = b.step("test", "Lance les tests");
    test_step.dependOn(&b.addRunArtifact(tests).step);
}
