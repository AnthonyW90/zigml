const std = @import("std");

pub const Bpe = struct {
    /// id -> bytes. Index is the token id. Owns every slice.
    vocab: std.ArrayList([]const u8),

    /// bytes -> id. Keys are the same slices vocab owns (no double-free: free via vocab only).
    vocab_lookup: std.StringArrayHashMapUnmanaged(u32),

    /// (left_id, right_id) packed in u64 -> merged_id
    merges: std.AutoArrayHashMapUnmanaged(u64, u32),

    pub fn init(allocator: std.mem.Allocator) !Bpe {
        var self: Bpe = .{
            .vocab = .empty,
            .vocab_lookup = .empty,
            .merges = .empty,
        };
        errdefer self.deinit(allocator);

        //Base vocabulary: all 256 bytes. Token id == byte value.
        try self.vocab.ensureTotalCapacity(allocator, 256);
        for (0..256) |b| {
            const bytes = try allocator.alloc(u8, 1);
            bytes[0] = @intCast(b);
            self.vocab.appendAssumeCapacity(bytes);
            try self.vocab_lookup.put(allocator, bytes, @intCast(b));
        }
        return self;
    }

    pub fn deinit(self: *Bpe, allocator: std.mem.Allocator) void {
        for (self.vocab.items) |byte| allocator.free(byte);
        self.vocab.deinit(allocator);
        self.vocab_lookup.deinit(allocator);
        self.merges.deinit(allocator);
        self.* = undefined;
    }

    fn pairKey(left: u32, right: u32) u64 {
        return (@as(u64, left) << 32) | right;
    }

    pub fn train(
        self: *Bpe,
        allocator: std.mem.Allocator,
        text: []const u8,
        num_merges: u32,
    ) !void {
        // All scratch lives in an arena: one free at the end, no bookkeeping.
        var arena_state = std.heap.ArenaAllocator.init(allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        // Working corpus: token ids, starts as raw bytes.
        var ids = try arena.alloc(u32, text.len);
        for (text, 0..) |byte, i| ids[i] = byte;
        var len: usize = ids.len; // live length; we compact in place

        var pair_counts: std.AutoHashMapUnmanaged(u64, u32) = .empty;

        var merge_i: u32 = 0;
        while (merge_i < num_merges) : (merge_i += 1) {
            // 1. Count adjacent pairs.
            pair_counts.clearRetainingCapacity();
            for (0..len - 1) |i| {
                const key = pairKey(ids[i], ids[i + 1]);
                const gop = try pair_counts.getOrPut(arena, key);
                if (!gop.found_existing) gop.value_ptr.* = 0;
                gop.value_ptr.* += 1;
            }

            // 2. Find the most frequent pair (tie-break: lower key, for dterminism).
            var best_key: u64 = 0;
            var best_count: u32 = 0;
            var it = pair_counts.iterator();
            while (it.next()) |entry| {
                const count = entry.value_ptr.*;
                const key = entry.key_ptr.*;

                if (count > best_count or (count == best_count and key < best_key)) {
                    best_count = count;
                    best_key = key;
                }
            }

            if (best_count < 2) break; // nothing repeats; further merges are pointless

            const left: u32 = @intCast(best_key >> 32);
            const right: u32 = @truncate(best_key);

            // 3. Mint the new token (long-lived allocator, not the arena).
            const new_id: u32 = @intCast(self.vocab.items.len);
            const left_bytes = self.vocab.items[left];
            const right_bytes = self.vocab.items[right];
            const merged = try allocator.alloc(u8, left_bytes.len + right_bytes.len);
            @memcpy(merged[0..left_bytes.len], left_bytes);
            @memcpy(merged[left_bytes.len..], right_bytes);

            try self.vocab.append(allocator, merged);
            try self.vocab_lookup.put(allocator, merged, new_id);
            try self.merges.put(allocator, best_key, new_id);

            // 4. Rewrite the corpus in place, replacing (left, right) with new_id
            var write: usize = 0;
            var read: usize = 0;
            while (read < len) {
                if (read + 1 < len and ids[read] == left and ids[read + 1] == right) {
                    ids[write] = new_id;
                    read += 2;
                } else {
                    ids[write] = ids[read];
                    read += 1;
                }
                write += 1;
            }
            len = write;
        }
    }

    pub fn decode(self: *const Bpe, allocator: std.mem.Allocator, ids: []const u32) ![]u8 {
        var out: std.ArrayList(u8) = .empty;
        errdefer out.deinit(allocator);
        for (ids) |id| {
            try out.appendSlice(allocator, self.vocab.items[id]);
        }
        return out.toOwnedSlice(allocator);
    }

    pub fn encode(self: *const Bpe, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        var ids: std.ArrayList(u32) = .empty;
        errdefer ids.deinit(allocator);

        try ids.ensureTotalCapacity(allocator, text.len);
        for (text) |byte| {
            ids.appendAssumeCapacity(byte);
        }

        while (ids.items.len > 1) {
            var best_key: u64 = 0;
            var best_merged_id: u32 = std.math.maxInt(u32);
            var found_merge = false;

            // Find the earliest-trained merge available in the current token stream.
            // Token ids are minted in merge order, so the lowest merged id has highest priority.
            for (0..ids.items.len - 1) |i| {
                const key = pairKey(ids.items[i], ids.items[i + 1]);
                if (self.merges.get(key)) |merged_id| {
                    if (!found_merge or merged_id < best_merged_id) {
                        found_merge = true;
                        best_key = key;
                        best_merged_id = merged_id;
                    }
                }
            }

            if (!found_merge) break;

            const left: u32 = @intCast(best_key >> 32);
            const right: u32 = @truncate(best_key);

            // Rewrite the token stream in place, replacing all non-overlapping
            // occurrences of the selected pair with its merged token id.
            var write: usize = 0;
            var read: usize = 0;
            while (read < ids.items.len) {
                if (read + 1 < ids.items.len and ids.items[read] == left and ids.items[read + 1] == right) {
                    ids.items[write] = best_merged_id;
                    read += 2;
                } else {
                    ids.items[write] = ids.items[read];
                    read += 1;
                }
                write += 1;
            }
            ids.shrinkRetainingCapacity(write);
        }

        return ids.toOwnedSlice(allocator);
    }
};

test "train + roundtrip" {
    const allocator = std.testing.allocator;
    var bpe = try Bpe.init(allocator);
    defer bpe.deinit(allocator);

    const text = "the cat sat on the mat the cat sat";
    try bpe.train(allocator, text, 10);

    // "the " repeats: expect merges to have formed beyond the base 256
    try std.testing.expect(bpe.vocab.items.len > 256);

    const ids = try bpe.encode(allocator, text);
    defer allocator.free(ids);
    try std.testing.expect(ids.len < text.len); // compression happened

    const back = try bpe.decode(allocator, ids);
    defer allocator.free(back);
    try std.testing.expectEqualStrings(text, back);
}
