const std = @import("std");
const zigml = @import("zigml");

pub fn main() !void {
    var gpa_state: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    var threaded: std.Io.Threaded = .init(gpa, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const text = try std.Io.Dir.cwd().readFileAlloc(io, "data/the-verdict.txt", gpa, .limited(1 << 20));
    defer gpa.free(text);

    var bpe = try zigml.tokenizer.Bpe.init(gpa);
    defer bpe.deinit(gpa);

    try bpe.train(gpa, text, 256);

    std.debug.print("Last 20 learned tokens:\n", .{});
    const items = bpe.vocab.items;
    for (items[items.len - 20 ..], items.len - 20..) |bytes, id| {
        std.debug.print("\t{d:>4} -> \"{s}\"\n", .{ id, bytes });
    }

    const sample = "I had always thought Jack Gisburn rather a cheap genius";
    const ids = try bpe.encode(gpa, sample);
    defer gpa.free(ids);
    std.debug.print("\n\"{s}\"\n{d} bytes -> {d} tokens \n", .{ sample, sample.len, ids.len });
}
