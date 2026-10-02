const std = @import("std");
pub const c = @import("c.zig").c;
pub const loader = @import("loader.zig");
pub const Library = loader.Library;
pub const LoadError = loader.LoadError;
pub const FindOptions = loader.FindOptions;

pub const Error = error{
    LoadFailed,
    NotLoaded,
    SymbolLookupFailed,
    ApiError,
};

/// Contexte PJRT chargé (handle + lib)
pub const Context = struct {
    allocator: std.mem.Allocator,
    lib: ?Library = null,

    pub fn init(allocator: std.mem.Allocator) Context {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Context) void {
        if (self.lib) |*l| {
            l.close(self.allocator);
            self.lib = null;
        }
    }

    /// Charge le .so PJRT automatiquement (scanne chemins + env vars)
    pub fn loadAuto(self: *Context, opts: FindOptions) !void {
        if (self.lib != null) return;
        self.lib = try Library.openAuto(self.allocator, opts);
    }

    /// Charge un .so PJRT via chemin explicite
    pub fn load(self: *Context, path: []const u8) !void {
        if (self.lib != null) return;
        self.lib = try Library.open(self.allocator, path);
    }

    /// Vérifie si le runtime PJRT est chargé
    pub fn isLoaded(self: *const Context) bool {
        return self.lib != null;
    }

    /// Récupère un symbole PJRT (sans erreur si absent)
    pub fn getSymbol(self: *Context, comptime T: type, name: [:0]const u8) ?T {
        if (self.lib) |*l| return l.getSymbol(T, name);
        return null;
    }

    /// Récupère un symbole PJRT requis
    pub fn getRequiredSymbol(self: *Context, comptime T: type, name: [:0]const u8) !T {
        if (self.lib) |*l| {
            return l.getRequiredSymbol(T, name) catch error.SymbolLookupFailed;
        }
        return error.NotLoaded;
    }
};
