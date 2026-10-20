(module
  ;; The guest cannot read files, open sockets, or invoke a shell. It only
  ;; schedules the host's bounded model/tool calls. No WASI or memory import.
  (import "probe" "next_action" (func $next_action (result i32)))
  (import "probe" "run_tool" (func $run_tool (param i32) (result i32)))
  (func (export "run") (result i32)
    (local $action i32)
    (block $done
      (loop $again
        call $next_action
        local.tee $action
        i32.eqz
        br_if $done
        local.get $action
        call $run_tool
        i32.eqz
        br_if $done
        br $again))
    i32.const 1)
)
