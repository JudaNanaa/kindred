const std = @import("std");
const c = @import("c.zig").c;

pub const LoadError = error{
    LibraryNotFound,
    SymbolNotFound,
    DlOpenFailed,
};

/// Options pour chercher la lib PJRT
pub const FindOptions = struct {
    /// Nom/chemin explicite (prioritaire)
    explicit_path: ?[]const u8 = null,
    /// Variables d'env à tester (dans l'ordre)
    env_vars: []const []const u8 = &.{
        "PJRT_LIBRARY",
        "PJRT_LIB",
        "PJRT_PLUGIN",
        "PJRT_PLUGIN_PATH",
    },
    /// Noms de librairies à essayer (sans préfixe lib/extension)
    lib_names: []const []const u8 = &.{
        "pjrt_c_api",
        "pjrt_plugin",
        "tpu_pjrt",
        "pjrt",
    },
    /// Dossiers additionnels à chercher
    extra_dirs: []const []const u8 = &.{},
};

/// Cherche le chemin d'une lib PJRT sur le système
pub fn findLibraryPath(allocator: std.mem.Allocator, opts: FindOptions) ![]u8 {
    // 1. Chemin explicite
    if (opts.explicit_path) |p| {
        if (std.fs.path.isAbsolute(p) or std.mem.startsWith(u8, p, "./") or std.mem.startsWith(u8, p, "../")) {
            return allocator.dupe(u8, p);
        }
        // Si juste un nom, on cherche quand même plus bas
        return try searchInCommonDirs(allocator, p, opts);
    }

    // 2. Variables d'environnement
    var env_map = try std.process.getEnvMap(allocator);
    defer env_map.deinit();
    for (opts.env_vars) |ev| {
        if (env_map.get(ev)) |v| {
            if (v.len > 0) {
                return allocator.dupe(u8, v);
            }
        }
    }

    // 3. Essayer les noms de lib dans les dossiers communs
    for (opts.lib_names) |name| {
        if (searchInCommonDirs(allocator, name, opts)) |path| {
            return path;
        } else |err| switch (err) {
            error.LibraryNotFound => continue,
            else => return err,
        }
    }

    return error.LibraryNotFound;
}

fn searchInCommonDirs(allocator: std.mem.Allocator, name: []const u8, opts: FindOptions) ![]u8 {
    // Si `name` contient déjà un séparateur ou une extension .so, tester direct
    if (std.mem.indexOf(u8, name, "/") != null or std.mem.endsWith(u8, name, ".so")) {
        if (fileExists(name)) {
            return allocator.dupe(u8, name);
        }
    }

    // Construire candidats: name.so, libname.so
    var candidates_buf: [2][]const u8 = undefined;
    var ccount: usize = 0;

    // name.so
    candidates_buf[ccount] = try std.fmt.allocPrint(allocator, "{s}.so", .{name});
    ccount += 1;

    // lib{name}.so (si name ne commence pas déjà par lib)
    if (!std.mem.startsWith(u8, name, "lib")) {
        candidates_buf[ccount] = try std.fmt.allocPrint(allocator, "lib{s}.so", .{name});
        ccount += 1;
    }
    defer {
        for (0..ccount) |i| allocator.free(candidates_buf[i]);
    }

    // Dossiers à tester (priorité raisonnable)
    const dirs = [_][]const u8{
        ".",
        "./lib",
        "./libs",
        "../lib",
        "../libs",
        "/usr/local/lib",
        "/usr/local/lib64",
        "/usr/lib",
        "/usr/lib64",
        "/lib",
        "/lib64",
    };

    // + extra_dirs
    for (opts.extra_dirs) |d| {
        for (candidates_buf[0..ccount]) |cand| {
            const full = try std.fs.path.join(allocator, &.{ d, cand });
            defer allocator.free(full);
            if (fileExists(full)) return allocator.dupe(u8, full);
        }
    }

    for (&dirs) |d| {
        for (candidates_buf[0..ccount]) |cand| {
            const full = try std.fs.path.join(allocator, &.{ d, cand });
            defer allocator.free(full);
            if (fileExists(full)) return allocator.dupe(u8, full);
        }
    }

    return error.LibraryNotFound;
}

fn fileExists(path: []const u8) bool {
    std.fs.accessAbsolute(path, .{}) catch return false;
    return true;
}

/// Handle sur la lib PJRT chargée dynamiquement
pub const Library = struct {
    lib: std.DynLib,
    path: []u8,

    pub fn open(allocator: std.mem.Allocator, path: []const u8) !Library {
        var dl = std.DynLib.open(path) catch |err| switch (err) {
            error.FileNotFound => return error.LibraryNotFound,
            else => return error.DlOpenFailed,
        };
        errdefer dl.close();

        return .{
            .lib = dl,
            .path = try allocator.dupe(u8, path),
        };
    }

    pub fn openAuto(allocator: std.mem.Allocator, opts: FindOptions) !Library {
        const p = try findLibraryPath(allocator, opts);
        errdefer allocator.free(p);
        return try open(allocator, p);
    }

    pub fn close(self: *Library, allocator: std.mem.Allocator) void {
        self.lib.close();
        allocator.free(self.path);
        self.* = undefined;
    }

    /// Récupère un symbole (fonction/variable). Retourne pointeur nul si absent.
    pub fn getSymbol(self: *Library, comptime T: type, name: [:0]const u8) ?T {
        return self.lib.lookup(T, name);
    }

    /// Récupère un symbole requis (erreur si absent)
    pub fn getRequiredSymbol(self: *Library, comptime T: type, name: [:0]const u8) !T {
        return self.lib.lookup(T, name) orelse error.SymbolNotFound;
    }
};
