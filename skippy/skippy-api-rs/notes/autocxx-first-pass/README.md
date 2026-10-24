# autocxx first pass (2026-10-17)

`seed.h` is the facade with its shared types written in plain C++ (autocxx
reads headers, so it cannot see types a cxx bridge defines). `seed.rs` is the
`include_cpp!` directive; `extract.rs` is the relevant part of what
`autocxx-gen 0.30.0` produced from them (`-std=c++20`, libclang 18).

What it got right, and `skippy-sys/src/lib.rs` keeps:

- enum variants, values and `repr` (u8 / i8), one for one
- POD struct fields and order (it chose `c_uint`/`c_int`; the bridge uses `u32`/`i32`)
- the method shapes: `self: Pin<&mut Bridge>`, `self: &Bridge`, `&Model`,
  `Pin<&mut Model>`, `new_bridge() -> UniquePtr<Bridge>`

What it could not do, and the hand bridge does instead:

- `ModelResult` / `SessionResult` (a status plus a `unique_ptr<Model>`) became
  opaque blobs with `moveit` placement-return plumbing and no field access;
  the bridge declares them as shared structs with a `UniquePtr<Model>` field
- pointer + length parameters stayed raw pointers in `unsafe fn`s; the bridge
  takes `&str`, `&[i32]`, `&mut [f32]`, `&mut Vec<i32>`
- it exposed `Bridge::impl_` as a public field

autocxx is archived (2026-09); nothing in the build depends on it.
