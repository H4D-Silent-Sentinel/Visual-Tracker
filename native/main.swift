import AppKit
import AVFoundation
import Foundation
import Darwin

enum CameraAction {
    case start, request, deny
}

func cameraAction(for status: AVAuthorizationStatus) -> CameraAction {
    switch status {
    case .authorized: return .start
    case .notDetermined: return .request
    case .denied, .restricted: return .deny
    @unknown default: return .deny
    }
}

func backendEnvironment(_ inherited: [String: String]) -> [String: String] {
    var environment = inherited
    environment["VISUAL_TRACKER_NATIVE"] = "1"
    environment["OPENCV_AVFOUNDATION_SKIP_AUTH"] = "1"
    environment["PORT"] = "8001"
    return environment
}

if CommandLine.arguments.dropFirst() == ["--self-check"] {
    precondition(cameraAction(for: .authorized) == .start)
    precondition(cameraAction(for: .notDetermined) == .request)
    precondition(cameraAction(for: .denied) == .deny)
    precondition(cameraAction(for: .restricted) == .deny)
    let environment = backendEnvironment(["PORT": "8000", "OPENCV_AVFOUNDATION_SKIP_AUTH": "0", "PATH": "/bin"])
    precondition(environment["VISUAL_TRACKER_NATIVE"] == "1")
    precondition(environment["OPENCV_AVFOUNDATION_SKIP_AUTH"] == "1")
    precondition(environment["PORT"] == "8001" && environment["PATH"] == "/bin")
    print("Native launcher self-check passed")
    exit(0)
}

@MainActor
final class TrackerDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private let status = NSTextField(wrappingLabelWithString: "Starting Visual Tracker…")
    private var root: URL?
    private var pidFile: URL?
    private var child: Process?
    private var log: FileHandle?
    private var terminating = false
    private var signalSources: [DispatchSourceSignal] = []

    func applicationDidFinishLaunching(_ notification: Notification) {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 520, height: 210),
                          styleMask: [.titled, .closable, .miniaturizable], backing: .buffered, defer: false)
        window.title = "Visual Tracker"
        window.isReleasedWhenClosed = false
        status.frame = NSRect(x: 24, y: 76, width: 472, height: 110)
        status.font = .systemFont(ofSize: 15)
        status.isSelectable = true
        window.contentView?.addSubview(status)
        let quit = NSButton(title: "Quit and stop tracker", target: self, action: #selector(quitTracker(_:)))
        quit.frame = NSRect(x: 24, y: 24, width: 200, height: 32)
        quit.bezelStyle = .rounded
        window.contentView?.addSubview(quit)
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)

        do {
            guard CommandLine.arguments.count == 3 else {
                throw launchError("Launch this app with the project folder and its PID-file path.")
            }
            let project = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true).standardizedFileURL
            let launcher = project.appendingPathComponent("Start Tracker.command")
            guard FileManager.default.fileExists(atPath: launcher.path) else {
                throw launchError("The project folder has no Start Tracker.command.")
            }
            let pid = URL(fileURLWithPath: CommandLine.arguments[2]).standardizedFileURL
            guard pid.deletingLastPathComponent() == project.appendingPathComponent(".venv", isDirectory: true),
                  pid.lastPathComponent.hasPrefix("native-launch-"), pid.pathExtension == "pid" else {
                throw launchError("The PID file must be .venv/native-launch-<launcher PID>.pid in this project.")
            }
            try "\(ProcessInfo.processInfo.processIdentifier)\n".write(to: pid, atomically: true, encoding: .utf8)
            root = project
            pidFile = pid
        } catch {
            status.stringValue = "Cannot start Visual Tracker.\n\(error.localizedDescription)"
            return
        }

        for number in [SIGTERM, SIGINT] {
            signal(number, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
            source.setEventHandler { [weak self] in self?.quitTracker(nil) }
            source.resume()
            signalSources.append(source)
        }
        // The native main run loop must be active before macOS displays the camera prompt.
        DispatchQueue.main.async { self.authorizeCamera() }
    }

    private func launchError(_ message: String) -> NSError {
        NSError(domain: "VisualTracker", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }

    private func authorizeCamera() {
        guard !terminating else { return }
        switch cameraAction(for: AVCaptureDevice.authorizationStatus(for: .video)) {
        case .start:
            startBackend()
        case .request:
            status.stringValue = "Requesting camera access…\nChoose Allow in the macOS camera permission window."
            AVCaptureDevice.requestAccess(for: .video) { allowed in
                DispatchQueue.main.async {
                    guard !self.terminating else { return }
                    if allowed { self.startBackend() } else { self.cameraDenied() }
                }
            }
        case .deny:
            cameraDenied()
        }
    }

    private func cameraDenied() {
        status.stringValue = "Camera access is unavailable. Enable Visual Tracker in Camera settings, then relaunch."
        let alert = NSAlert()
        alert.messageText = "Visual Tracker needs camera access"
        alert.informativeText = "Enable Visual Tracker in System Settings → Privacy & Security → Camera, then quit and relaunch the app. macOS remembers a denial and will not show the permission prompt again automatically. A system restriction may require your administrator."
        alert.addButton(withTitle: "Open Camera Settings")
        alert.addButton(withTitle: "Quit")
        if alert.runModal() == .alertFirstButtonReturn,
           let settings = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera") {
            NSWorkspace.shared.open(settings)
        }
        NSApp.terminate(nil)
    }

    private func startBackend() {
        guard !terminating, child == nil, let root else { return }
        guard cameraAction(for: AVCaptureDevice.authorizationStatus(for: .video)) == .start else {
            cameraDenied()
            return
        }
        status.stringValue = "Camera access allowed. Starting the tracker…\nThe Live dashboard will open in your browser when ready."
        let logURL = root.appendingPathComponent(".venv/visual-tracker-native.log")
        do {
            if !FileManager.default.fileExists(atPath: logURL.path) {
                guard FileManager.default.createFile(atPath: logURL.path, contents: nil) else {
                    throw launchError("Cannot create the tracker log.")
                }
            }
            let handle = try FileHandle(forWritingTo: logURL)
            log = handle
            _ = try handle.seekToEnd()
            try handle.write(contentsOf: Data("\n\(Date()): camera access allowed; launching tracker on port 8001.\n".utf8))
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/zsh")
            process.arguments = [root.appendingPathComponent("Start Tracker.command").path]
            process.currentDirectoryURL = root
            process.environment = backendEnvironment(ProcessInfo.processInfo.environment)
            process.standardInput = FileHandle.nullDevice
            process.standardOutput = handle
            process.standardError = handle
            process.terminationHandler = { process in
                let code = process.terminationStatus
                DispatchQueue.main.async { self.backendStopped(code: code) }
            }
            try process.run()
            child = process
        } catch {
            status.stringValue = "Tracker could not start.\n\(error.localizedDescription)\nLog: \(logURL.path)"
        }
    }

    private func backendStopped(code: Int32) {
        child = nil
        if terminating {
            NSApp.reply(toApplicationShouldTerminate: true)
        } else {
            let logPath = root?.appendingPathComponent(".venv/visual-tracker-native.log").path ?? ""
            status.stringValue = "Tracker stopped (exit \(code)).\nCheck the log for details, then quit and relaunch.\n\(logPath)"
        }
    }

    @objc private func quitTracker(_ sender: Any?) {
        NSApp.terminate(sender)
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if terminating { return child?.isRunning == true ? .terminateLater : .terminateNow }
        terminating = true
        guard let child, child.isRunning else { return .terminateNow }
        status.stringValue = "Stopping the tracker…"
        // The launcher traps TERM and cleans up only the services it started.
        child.terminate()
        return .terminateLater
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        for source in signalSources { source.cancel() }
        try? log?.close()
        if let pidFile, let recorded = try? String(contentsOf: pidFile, encoding: .utf8),
           recorded.trimmingCharacters(in: .whitespacesAndNewlines) == String(ProcessInfo.processInfo.processIdentifier) {
            try? FileManager.default.removeItem(at: pidFile)
        }
    }
}

MainActor.assumeIsolated {
    let application = NSApplication.shared
    let delegate = TrackerDelegate()
    application.delegate = delegate
    application.setActivationPolicy(.regular)
    let menu = NSMenu()
    let appMenuItem = NSMenuItem()
    let appMenu = NSMenu()
    appMenu.addItem(withTitle: "Quit Visual Tracker", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
    appMenuItem.submenu = appMenu
    menu.addItem(appMenuItem)
    application.mainMenu = menu
    withExtendedLifetime(delegate) { application.run() }
}
