import Darwin
import Foundation

public enum ParentWatchdog {
  public static func install(parentPID: Int?) throws {
    guard let parentPID else { return }
    guard parentPID > 1, parentPID <= Int(Int32.max) else {
      throw AppleRuntimeFailure(
        code: "invalid_parent_pid",
        message: "--parent-pid must identify a non-system process",
        retryable: false
      )
    }
    let watchedPID = pid_t(parentPID)
    Task.detached(priority: .background) {
      while true {
        if Darwin.kill(watchedPID, 0) == -1, errno == ESRCH {
          Darwin._exit(0)
        }
        try? await Task.sleep(for: .milliseconds(50))
      }
    }
  }
}
